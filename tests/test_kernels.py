import itertools
import unittest

import torch


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() in {(8, 6), (8, 9), (9, 0), (10, 0), (10, 3), (12, 0)},
    "Requires Ampere, Ada, Hopper or Blackwell",
)
class CuTeAttentionTests(unittest.TestCase):
    def test_sm89_split_forward_compiled_dynamic_query_boundaries(self):
        if torch.cuda.get_device_capability() not in {(8, 9), (9, 0)}:
            self.skipTest("Requires Ada or Hopper short-query forward kernels.")
        from benchmarks.reference import fp64_reference
        from kohakufa_cute import attention

        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        function = torch.compile(attention, fullgraph=True, dynamic=True)
        for queries in (31, 33, 65, 257, 12):
            with self.subTest(queries=queries):
                torch.manual_seed(817)
                q = torch.randn(1, queries, 16, 64, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
                k, v = [torch.randn(1, 257, 16, 64, device="cuda", dtype=q.dtype).transpose(1, 2) for _ in range(2)]
                dout = torch.randn_like(q)
                expected = fp64_reference((q, k, v), dout)
                inputs = [tensor.requires_grad_() for tensor in (q, k, v)]
                output = function(*inputs)
                gradients = torch.autograd.grad(output, inputs, dout)
                for got, want in zip((output, *gradients), expected):
                    torch.testing.assert_close(got.double(), want, atol=0.02, rtol=0.02)

    def test_sm89_short_query_split_forward_matches_fp64_and_saved_statistics(self):
        if torch.cuda.get_device_capability() not in {(8, 9), (9, 0)}:
            self.skipTest("Requires Ada or Hopper short-query forward kernels.")
        from benchmarks.reference import fp64_reference
        from kohakufa_cute import native

        extension = native.extension()
        extension.clear_cache()
        try:
            for dtype, queries, keys, scale in itertools.product(
                (torch.float16, torch.bfloat16), (1, 3, 12, 32, 64, 128, 129, 256), (256, 787), (0.125, -0.125, 0.0)
            ):
                with self.subTest(dtype=dtype, queries=queries, keys=keys, scale=scale):
                    torch.manual_seed(813)
                    q = torch.randn(1, queries, 16, 64, device="cuda", dtype=dtype).transpose(1, 2)
                    k, v = [torch.randn(1, keys, 16, 64, device="cuda", dtype=dtype).transpose(1, 2) for _ in range(2)]
                    dout = torch.randn_like(q)

                    def run():
                        out, maximum, logsum, blocks = extension.forward(q, k, v, None, scale, False)
                        gradients = extension.backward(q, k, v, None, maximum, logsum, dout, blocks, out, scale, False)
                        return out, *gradients

                    run()
                    run()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        actual = run()
                    for seed in (814, 815):
                        torch.manual_seed(seed)
                        for tensor in (q, k, v, dout):
                            tensor.copy_(torch.randn_like(tensor))
                        graph.replay()
                        torch.cuda.synchronize()
                        expected = fp64_reference((q, k, v), dout, scale=scale)
                        tolerance = 0.02 if dtype == torch.bfloat16 else 0.003
                        for got, want in zip(actual, expected):
                            torch.testing.assert_close(got.double(), want, atol=tolerance, rtol=tolerance)
        finally:
            extension.clear_cache()

    def test_sm89_short_query_split_forward_saturated_and_offset_logits(self):
        if torch.cuda.get_device_capability() not in {(8, 9), (9, 0)}:
            self.skipTest("Requires Ada or Hopper short-query forward kernels.")
        from benchmarks.precision import make_inputs
        from benchmarks.reference import fp64_reference
        from kohakufa_cute import attention, native

        native.extension().clear_cache()
        try:
            for dtype, logit_scale, offset in itertools.product(
                (torch.float16, torch.bfloat16), (10, 1000, 1e5, 1e7), (0, 100)
            ):
                with self.subTest(dtype=dtype, logit_scale=logit_scale, offset=offset):
                    q, k, v, dout = make_inputs(logit_scale, device="cuda", dtype=dtype, seq=787, dim=64, seed=0, heads=16)
                    q, dout = q[:, :, :3].contiguous(), dout[:, :, :3].contiguous()
                    k = (k.float() + offset).to(dtype)
                    inputs = [tensor.requires_grad_() for tensor in (q, k, v)]
                    output = attention(*inputs)
                    actual = (output, *torch.autograd.grad(output, inputs, dout))
                    expected = fp64_reference(inputs, dout)
                    tolerance = 0.02 if dtype == torch.bfloat16 else 0.003
                    torch.testing.assert_close(actual[0].double(), expected[0], atol=tolerance, rtol=tolerance)
                    for got, want in zip(actual[1:], expected[1:]):
                        torch.testing.assert_close(got.double(), want, atol=tolerance, rtol=tolerance)
            for dtype in (torch.float16, torch.bfloat16):
                q, k, v, dout = make_inputs(1e5, device="cuda", dtype=dtype, seq=787, dim=64, seed=0, heads=16)
                q, dout = q[:, :, :3].contiguous(), torch.ones_like(dout[:, :, :3])
                v.fill_(1)
                inputs = [tensor.requires_grad_() for tensor in (q, k, v)]
                output = attention(*inputs)
                dq, dk, _ = torch.autograd.grad(output, inputs, dout)
                self.assertEqual(dq.count_nonzero().item(), 0)
                self.assertEqual(dk.count_nonzero().item(), 0)
            for dtype, gap in itertools.product((torch.float16, torch.bfloat16), (1, 5)):
                with self.subTest(dtype=dtype, saturated_gap=gap):
                    q = torch.zeros(1, 16, 3, 64, device="cuda", dtype=dtype)
                    k = torch.zeros(1, 16, 787, 64, device="cuda", dtype=dtype)
                    v, dout = torch.zeros_like(k), torch.zeros_like(q)
                    q[..., 0], k[..., 0], k[:, :, 0, 0] = 10, 10 - gap, 10
                    v[..., 0], v[:, :, 0, 0], dout[..., 0] = -1, 1, 1
                    inputs = [tensor.requires_grad_() for tensor in (q, k, v)]
                    output = attention(*inputs, scale=1.0)
                    actual = (output, *torch.autograd.grad(output, inputs, dout))
                    expected = fp64_reference(inputs, dout, scale=1.0)
                    for got, want in zip(actual, expected):
                        torch.testing.assert_close(got.double(), want, atol=1e-7 if dtype == torch.float16 else 0, rtol=0.02)
        finally:
            native.extension().clear_cache()

    def test_sm89_short_query_split_forward_fallback_boundaries(self):
        if torch.cuda.get_device_capability() not in {(8, 9), (9, 0)}:
            self.skipTest("Requires Ada or Hopper short-query forward kernels.")
        from unittest.mock import patch

        from kohakufa_cute import native

        extension = native.extension()
        extension.clear_cache()
        compiler = patch.object(native, "compile_kernel", wraps=native.compile_kernel)
        compiled = compiler.start()
        try:
            # Keep the warm native cache valid while crossing each selection boundary.
            for queries, keys, heads, kvheads, mode in (
                (3, 255, 16, 16, "dense"),
                (3, 256, 16, 16, "dense"),
                (3, 257, 16, 16, "dense"),
                (31, 257, 16, 16, "dense"),
                (32, 257, 16, 16, "dense"),
                (33, 257, 16, 16, "dense"),
                (63, 257, 16, 16, "dense"),
                (64, 257, 16, 16, "dense"),
                (65, 257, 16, 16, "dense"),
                (127, 257, 16, 16, "dense"),
                (128, 257, 16, 16, "dense"),
                (129, 257, 16, 16, "dense"),
                (255, 257, 16, 16, "dense"),
                (256, 257, 16, 16, "dense"),
                (257, 257, 16, 16, "dense"),
                (3, 16384, 16, 16, "dense"),
                (3, 16385, 16, 16, "dense"),
                (3, 257, 16, 4, "dense"),
                (3, 257, 2, 2, "dense"),
                (3, 257, 16, 16, "causal"),
                (3, 257, 16, 16, "masked"),
                (3, 257, 16, 16, "strided"),
                (3, 256, 16, 16, "dense"),
            ):
                with self.subTest(queries=queries, keys=keys, heads=heads, kvheads=kvheads, mode=mode):
                    torch.manual_seed(816)
                    step = 2 if mode == "strided" else 1
                    q = torch.randn(1, queries, heads, 64 * step, device="cuda", dtype=torch.bfloat16)[
                        ..., ::step
                    ].transpose(1, 2)
                    k, v = [
                        torch.randn(1, keys, kvheads, 64 * step, device="cuda", dtype=q.dtype)[..., ::step].transpose(1, 2)
                        for _ in range(2)
                    ]
                    mask = torch.ones(1, 1, queries, keys, device="cuda", dtype=torch.bool) if mode == "masked" else None
                    if mask is not None:
                        mask[..., 0, :] = False
                    previous = compiled.call_count
                    self.compare(q, k, v, mask=mask, causal=mode == "causal")
                    forward_compiles = [
                        call.args for call in compiled.call_args_list[previous:] if call.args[0] == "forward"
                    ]
                    split = (
                        queries <= 256 and 256 <= keys <= 16384 and heads == kvheads == 16 and mode in ("dense", "strided")
                    )
                    for arguments in forward_compiles:
                        self.assertEqual(arguments[1][7].ndim == 5, split)
                    if queries == 3 and keys == 257 and heads == kvheads == 16 and mode == "dense":
                        self.assertEqual(len(forward_compiles), 1)
        finally:
            compiler.stop()
            extension.clear_cache()

    def test_generic_backward_layouts_reuse_warm_programs(self):
        if torch.cuda.get_device_capability() not in ((8, 6), (8, 9), (12, 0)):
            self.skipTest("Requires measured generic warp-MMA layouts.")
        from unittest.mock import patch

        from kohakufa_cute import native

        for dimension, dtype in itertools.product((64, 128, 256, 512), (torch.float16, torch.bfloat16)):
            native.extension().clear_cache()
            try:
                classes = set()
                threshold = {64: 1024, 128: 512, 256: 256, 512: 256}[dimension]
                with patch.object(native, "compile_kernel", wraps=native.compile_kernel) as compiler:
                    for queries, keys in ((257, 261), (513, 517), (273, 261)):
                        for causal in (False, True):
                            with self.subTest(dim=dimension, dtype=dtype, queries=queries, keys=keys, causal=causal):
                                torch.manual_seed(409)
                                q = torch.randn(1, queries, 2, dimension, device="cuda", dtype=dtype).transpose(1, 2)
                                k, v = [
                                    torch.randn(1, keys, 2, dimension, device="cuda", dtype=dtype).transpose(1, 2)
                                    for _ in range(2)
                                ]
                                self.compare(q, k, v, causal=causal, scale=-0.125)
                                classes.add((causal, max(queries, keys) >= threshold))
                                self.assertEqual(compiler.call_count, 2 * len(classes))
                                self.assertEqual(native.extension().cache_size(), compiler.call_count)
            finally:
                native.extension().clear_cache()

    def test_generic_backward_layouts_read_changed_graph_inputs(self):
        if torch.cuda.get_device_capability() not in ((8, 6), (8, 9), (12, 0)):
            self.skipTest("Requires measured generic warp-MMA layouts.")
        from benchmarks.reference import fp64_reference
        from kohakufa_cute import attention, native

        for dimension, dtype, causal in itertools.product(
            (64, 128, 256, 512), (torch.float16, torch.bfloat16), (False, True)
        ):
            with self.subTest(dim=dimension, dtype=dtype, causal=causal):
                native.extension().clear_cache()
                try:
                    inputs = [
                        torch.randn(1, 2, 257, dimension, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)
                    ]
                    dout = torch.randn_like(inputs[0])

                    def run():
                        out = attention(*inputs, causal=causal, scale=-0.125)
                        return out, *torch.autograd.grad(out, inputs, dout)

                    run()
                    run()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        actual = run()
                    for seed in (410, 411):
                        torch.manual_seed(seed)
                        with torch.no_grad():
                            for tensor in (*inputs, dout):
                                tensor.copy_(torch.randn_like(tensor))
                        graph.replay()
                        torch.cuda.synchronize()
                        expected = fp64_reference(inputs, dout, causal=causal, scale=-0.125)
                        tolerance = 0.03 if dtype == torch.bfloat16 else 0.005
                        for got, want in zip(actual, expected):
                            torch.testing.assert_close(got.double(), want, atol=tolerance, rtol=tolerance)
                finally:
                    native.extension().clear_cache()

    def test_sm103_native_products_tails_gqa_and_signed_scale(self):
        if torch.cuda.get_device_capability() != (10, 3):
            self.skipTest("Requires B300 TMEM reductions.")
        from unittest.mock import patch

        from kohakufa_cute import kernels, native

        original = kernels.kernel_configuration
        for dim, rows in itertools.product((64, 128, 256, 512), (64, 128)):
            for dtype in (torch.float16, torch.bfloat16):
                for pipelined in (False, True):
                    if dim == 512 and pipelined:
                        continue

                    def configuration(device, mode, descriptors):
                        if (
                            mode == "forward"
                            or mode == "q"
                            and (rows == 64 or dim <= 256)
                            or mode == "kv"
                            and not pipelined
                            and (rows == 64 or dim <= 256)
                        ):
                            configuration = (32, min(dim, 256), 4, False, mode != "forward", True, pipelined, False, rows)
                            return (*configuration, True) if mode != "forward" and dim >= 256 else configuration
                        return original(device, mode, descriptors)

                    with patch.object(kernels, "kernel_configuration", configuration):
                        native.extension().clear_cache()
                        try:
                            for queries, keys, causal, scale in (
                                (1, 193, False, None),
                                (1, 193, True, None),
                                (65, 1, True, None),
                                (137, 193, False, -0.125),
                                (137, 193, True, 0.0),
                            ):
                                with self.subTest(
                                    dim=dim, rows=rows, dtype=dtype, pipelined=pipelined, causal=causal, scale=scale
                                ):
                                    torch.manual_seed(174)
                                    q = torch.randn(1, queries, 4, dim, device="cuda", dtype=dtype).transpose(1, 2)
                                    k, v = [
                                        torch.randn(1, keys, 2, dim, device="cuda", dtype=dtype).transpose(1, 2)
                                        for _ in range(2)
                                    ]
                                    self.compare(q, k, v, causal=causal, scale=scale)
                        finally:
                            native.extension().clear_cache()

    def test_sm103_tmem_reductions_read_changed_graph_inputs(self):
        if torch.cuda.get_device_capability() != (10, 3):
            self.skipTest("Requires B300 TMEM reductions.")
        from unittest.mock import patch

        from kohakufa_cute import attention, kernels, native

        original = kernels.kernel_configuration

        def configuration(device, mode, descriptors):
            if mode == "forward":
                return 64, 64, 4, False, False, True, True, False, rows
            return original(device, mode, descriptors)

        with patch.object(kernels, "kernel_configuration", configuration):
            native.extension().clear_cache()
            try:
                for rows, scale in itertools.product((64, 128), (0.125, -0.125)):
                    native.extension().clear_cache()
                    inputs = [torch.randn(1, 2, 193, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            attention(*inputs, scale=scale)
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        actual = attention(*inputs, scale=scale)
                    for seed in (175, 176, 177):
                        torch.manual_seed(seed)
                        for tensor in inputs:
                            tensor.copy_(torch.randn_like(tensor))
                        graph.replay()
                        # The default fused SDPA backend can return NaNs for negative scales.
                        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                            expected = torch.nn.functional.scaled_dot_product_attention(*inputs, scale=scale)
                        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
            finally:
                native.extension().clear_cache()

    def test_blackwell_probability_tiles_match_checked_reconstruction(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("Requires the Blackwell producer/consumer pipeline.")
        from contextlib import nullcontext
        from unittest.mock import patch

        from kohakufa_cute import attention, blackwell_pipeline, kernels, native

        worker = blackwell_pipeline.Backward
        for dim, columns, dtype, causal in itertools.product(
            (128, 512), (64, 128), (torch.float16, torch.bfloat16), (False, True)
        ):

            def configuration(device, mode, descriptors):
                selected = kernels._BLACKWELL_PIPELINE_CONFIGURATIONS[(dim, mode)]
                return selected if mode == "forward" else (columns, *selected[1:])

            torch.manual_seed(238)
            q = torch.randn(1, 4, 257, dim, device="cuda", dtype=dtype)
            k, v = [torch.randn(1, 2, 385, dim, device="cuda", dtype=dtype) for _ in range(2)]
            dout = torch.randn_like(q)
            with (
                self.subTest(dim=dim, columns=columns, dtype=dtype, causal=causal),
                patch.object(kernels, "kernel_configuration", configuration),
            ):
                results = []
                try:
                    for checked in (False, True):
                        native.extension().clear_cache()
                        kernels._compile_kernel.cache_clear()
                        context = (
                            patch.object(worker, "reconstruct_probabilities", worker.reconstruct_values)
                            if checked
                            else nullcontext()
                        )
                        with context:
                            inputs = [tensor.detach().requires_grad_() for tensor in (q, k, v)]
                            output = attention(*inputs, causal=causal, scale=-0.125)
                            results.append((output.detach(), *torch.autograd.grad(output, inputs, dout)))
                    for actual, expected in zip(*results):
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                finally:
                    native.extension().clear_cache()
                    kernels._compile_kernel.cache_clear()

    def test_blackwell_probability_tiles_read_changed_graph_inputs(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("Requires the Blackwell producer/consumer pipeline.")
        from unittest.mock import patch

        from benchmarks.reference import fp64_reference
        from kohakufa_cute import attention, kernels, native

        for dim, columns, dtype in itertools.product((128, 512), (64, 128), (torch.float16, torch.bfloat16)):

            def configuration(device, mode, descriptors):
                selected = kernels._BLACKWELL_PIPELINE_CONFIGURATIONS[(dim, mode)]
                return selected if mode == "forward" else (columns, *selected[1:])

            with (
                self.subTest(dim=dim, columns=columns, dtype=dtype),
                patch.object(kernels, "kernel_configuration", configuration),
            ):
                native.extension().clear_cache()
                kernels._compile_kernel.cache_clear()
                try:
                    q = torch.randn(1, 4, 257, dim, device="cuda", dtype=dtype).requires_grad_()
                    k, v = [torch.randn(1, 2, 385, dim, device="cuda", dtype=dtype).requires_grad_() for _ in range(2)]
                    dout = torch.randn_like(q)
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            output = attention(q, k, v, scale=1.0)
                            torch.autograd.grad(output, (q, k, v), dout)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=stream):
                            output = attention(q, k, v, scale=1.0)
                            gradients = torch.autograd.grad(output, (q, k, v), dout)
                    torch.cuda.current_stream().wait_stream(stream)
                    for saturated in (False, True, False):
                        with torch.no_grad():
                            if saturated:
                                for tensor in (q, k, v, dout):
                                    tensor.zero_()
                                q[..., 0], k[..., 0], k[:, :, 0, 0] = 10, 5 if dtype == torch.bfloat16 else 9, 10
                                v[..., 0], v[:, :, 0, 0], dout[..., 0] = -1, 1, 1
                            else:
                                for tensor in (q, k, v, dout):
                                    tensor.normal_()
                                q.mul_(0.1)
                                k.mul_(0.1)
                        graph.replay()
                        torch.cuda.synchronize()
                        expected = fp64_reference((q, k, v), dout, scale=1.0)
                        for actual, reference in zip((output, *gradients), expected):
                            torch.testing.assert_close(
                                actual.double(),
                                reference,
                                atol=(
                                    0
                                    if saturated and dtype == torch.bfloat16
                                    else 0.03 if dtype == torch.bfloat16 else 0.005
                                ),
                                rtol=0.03 if dtype == torch.bfloat16 else 0.005,
                            )
                        if saturated:
                            self.assertGreater(gradients[0][..., 0].min().item(), 0)
                finally:
                    native.extension().clear_cache()
                    kernels._compile_kernel.cache_clear()

    def test_blackwell_warp_specialized_pipeline(self):
        self._check_blackwell_pipeline(fused=False)

    def test_sm103_fused_dispatch_crosses_sequence_threshold(self):
        if torch.cuda.get_device_capability() != (10, 3):
            self.skipTest("Requires B300 dispatch.")
        from benchmarks.reference import fp64_reference
        from kohakufa_cute import attention, kernels, native

        for dim, dtype in itertools.product((64, 128, 256, 512), (torch.float16, torch.bfloat16)):
            threshold = 1024 if dim == 64 else 512 if dim == 128 else 256
            native.extension().clear_cache()
            try:
                for keys in (threshold - 1, threshold + 1, threshold - 1):
                    with self.subTest(dim=dim, dtype=dtype, keys=keys):
                        q = torch.zeros(1, 4, 17, dim, device="cuda", dtype=dtype)
                        k = torch.zeros(1, 2, keys, dim, device="cuda", dtype=dtype)
                        v, dout = torch.zeros_like(k), torch.zeros_like(q)
                        q[..., 0], k[..., 0], k[:, :, 0, 0] = 10, 5 if dtype == torch.bfloat16 else 9, 10
                        v[..., 0], v[:, :, 0, 0], dout[..., 0] = -1, 1, 1
                        descriptors = tuple(kernels.descriptor(t) for t in (q, k, v, q, q, q, q, dout))
                        for mode in ("q", "kv"):
                            configuration = kernels.kernel_configuration(q.device.index, mode, descriptors)
                            if keys >= threshold:
                                self.assertEqual(configuration[0], 128)
                                self.assertEqual(configuration[2], 12)
                            else:
                                self.assertNotEqual(configuration[2], 12)
                        expected = fp64_reference((q, k, v), dout, scale=1.0)
                        inputs = [tensor.requires_grad_() for tensor in (q, k, v)]
                        out = attention(*inputs, scale=1.0)
                        gradients = torch.autograd.grad(out, inputs, dout)
                        self.assertGreater(gradients[0][..., 0].min().item(), 0)
                        for actual, reference in zip((out, *gradients), expected):
                            torch.testing.assert_close(
                                actual.double(), reference, atol=1e-7 if dtype == torch.float16 else 0, rtol=0.03
                            )
            finally:
                native.extension().clear_cache()

    def test_blackwell_fused_probability_and_ds_pipeline(self):
        self._check_blackwell_pipeline(fused=True)

    def test_blackwell_fused_pipeline_retains_saturated_gradients(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("Requires the fused Blackwell pipeline.")
        from unittest.mock import patch

        from benchmarks.reference import fp64_reference
        from kohakufa_cute import attention, kernels, native

        for dim, dtype in itertools.product((64, 128, 256, 512), (torch.float16, torch.bfloat16)):

            def configuration(device, mode, descriptors):
                return kernels._SM100_FUSED_CONFIGURATIONS[(dim, mode)]

            with self.subTest(dim=dim, dtype=dtype), patch.object(kernels, "kernel_configuration", configuration):
                native.extension().clear_cache()
                try:
                    q = torch.zeros(1, 4, 17, dim, device="cuda", dtype=dtype)
                    k = torch.zeros(1, 2, 19, dim, device="cuda", dtype=dtype)
                    v, dout = torch.zeros_like(k), torch.zeros_like(q)
                    q[..., 0], k[..., 0], k[:, :, 0, 0] = 10, 5 if dtype == torch.bfloat16 else 9, 10
                    v[..., 0], v[:, :, 0, 0], dout[..., 0] = -1, 1, 1
                    expected = fp64_reference((q, k, v), dout, scale=1.0)
                    inputs = [tensor.requires_grad_() for tensor in (q, k, v)]
                    output = attention(*inputs, scale=1.0)
                    actual = output, *torch.autograd.grad(output, inputs, dout)
                    self.assertGreater(actual[1][..., 0].min().item(), 0)
                    for got, want in zip(actual, expected):
                        torch.testing.assert_close(got.double(), want, atol=1e-7 if dtype == torch.float16 else 0, rtol=0.03)
                finally:
                    native.extension().clear_cache()

    def _check_blackwell_pipeline(self, *, fused):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("Requires the measured datacentre Blackwell architecture.")
        from unittest.mock import patch

        from kohakufa_cute import kernels, native

        for dim, dtype in itertools.product((64, 128, 256, 512), (torch.float16, torch.bfloat16)):

            def configuration(device, mode, descriptors):
                table = kernels._SM100_FUSED_CONFIGURATIONS if fused else kernels._BLACKWELL_PIPELINE_CONFIGURATIONS
                return table[(dim, mode)]

            with patch.object(kernels, "kernel_configuration", configuration):
                native.extension().clear_cache()
                try:
                    # Long key tails exercise both the five-slot fused ring and the legacy eight-slot ring.
                    for queries, keys, causal, scale in (
                        (257, 641, False, None),
                        (65, 129, True, -0.125),
                        (1, 641, True, 0.0),
                        (65, 1, True, None),
                    ):
                        with self.subTest(dim=dim, dtype=dtype, queries=queries, keys=keys, causal=causal, scale=scale):
                            torch.manual_seed(178)
                            q = torch.randn(1, queries, 4, dim, device="cuda", dtype=dtype).transpose(1, 2)
                            k, v = [
                                torch.randn(1, keys, 2, dim, device="cuda", dtype=dtype).transpose(1, 2) for _ in range(2)
                            ]
                            self.compare(q, k, v, causal=causal, scale=scale)
                finally:
                    native.extension().clear_cache()

    def test_blackwell_pipeline_reads_changed_graph_inputs(self):
        self._check_blackwell_graph_inputs(fused=False)

    def test_blackwell_fused_pipeline_reads_changed_graph_inputs(self):
        self._check_blackwell_graph_inputs(fused=True)

    def _check_blackwell_graph_inputs(self, *, fused):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("Requires the measured datacentre Blackwell architecture.")
        from unittest.mock import patch

        from kohakufa_cute import attention, kernels, native

        for dim, scale in itertools.product((64, 128, 512) if fused else (128, 512), (0.125, -0.125)):

            def configuration(device, mode, descriptors):
                table = kernels._SM100_FUSED_CONFIGURATIONS if fused else kernels._BLACKWELL_PIPELINE_CONFIGURATIONS
                return table[(dim, mode)]

            with self.subTest(dim=dim, scale=scale), patch.object(kernels, "kernel_configuration", configuration):
                native.extension().clear_cache()
                try:
                    inputs = [
                        torch.randn(1, 2, 257, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                        for _ in range(3)
                    ]
                    dout = torch.randn_like(inputs[0])

                    def run():
                        output = attention(*inputs, scale=scale)
                        return output, *torch.autograd.grad(output, inputs, dout)

                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            run()
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        actual = run()
                    for seed in (179, 180):
                        torch.manual_seed(seed)
                        with torch.no_grad():
                            for tensor in (*inputs, dout):
                                tensor.copy_(torch.randn_like(tensor))
                        graph.replay()
                        refs = [tensor.detach().double().requires_grad_() for tensor in inputs]
                        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                            expected = torch.nn.functional.scaled_dot_product_attention(*refs, scale=scale)
                        gradients = torch.autograd.grad(expected, refs, dout.double())
                        for got, want in zip(actual, (expected, *gradients)):
                            torch.testing.assert_close(got.double(), want, atol=0.03, rtol=0.03)
                finally:
                    native.extension().clear_cache()

    def test_sm100_wide_tma_layouts_and_signed_scale(self):
        if torch.cuda.get_device_capability() != (10, 0):
            self.skipTest("Requires measured sm_100 wide-head configurations.")
        for dtype in (torch.float16, torch.bfloat16):
            for queries, keys, bshd, causal, scale in (
                (1, 65, True, False, None),
                (65, 1, True, True, None),
                (73, 129, True, True, -(512**-0.5)),
                (73, 129, False, False, 0.0),
                (73, 129, False, False, None),
            ):
                with self.subTest(dtype=dtype, queries=queries, keys=keys, bshd=bshd, causal=causal, scale=scale):
                    torch.manual_seed(131)
                    q = torch.randn(1, 4, queries, 512, device="cuda", dtype=dtype)
                    k, v = [torch.randn(1, 2, keys, 512, device="cuda", dtype=dtype) for _ in range(2)]
                    if bshd:
                        q, k, v = [x.transpose(1, 2).contiguous().transpose(1, 2) for x in (q, k, v)]
                    self.compare(q, k, v, causal=causal, scale=scale, compiled=queries == 73 and scale is None)

    def test_sm100_tcgen_statistics_across_key_tiles(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("Requires datacenter Blackwell tcgen05 instructions.")
        from unittest.mock import patch

        from kohakufa_cute import kernels, native

        for dtype in (torch.float16, torch.bfloat16):
            for tma, pipelined in ((False, False), (True, False), (True, True)):

                def configuration(device, mode, descriptors):
                    return 32, 64, 4, False, mode != "forward", tma, pipelined, True

                with patch.object(kernels, "kernel_configuration", configuration):
                    native.extension().clear_cache()
                    try:
                        for visibility in ("dense", "causal", "mask"):
                            with self.subTest(dtype=dtype, tma=tma, pipelined=pipelined, visibility=visibility):
                                torch.manual_seed(129)
                                q = torch.randn(1, 4, 137, 64, device="cuda", dtype=dtype)
                                k, v = [torch.randn(1, 2, 193, 64, device="cuda", dtype=dtype) for _ in range(2)]
                                q, k, v = [x.transpose(1, 2).contiguous().transpose(1, 2) for x in (q, k, v)]
                                mask = None
                                if visibility == "mask":
                                    mask = (
                                        torch.arange(137, device="cuda")[:, None] // 31
                                        == torch.arange(193, device="cuda")[None, :] // 31
                                    )
                                    mask[0] = False
                                self.compare(q, k, v, mask=mask, causal=visibility == "causal")
                    finally:
                        native.extension().clear_cache()

    def test_centered_backward_statistics_across_multiple_key_tiles(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(127)
                q = torch.randn(1, 16, 137, 64, device="cuda", dtype=dtype)
                k, v = [torch.randn(1, 16, 193, 64, device="cuda", dtype=dtype) for _ in range(2)]
                self.compare(q, k, v)

    def test_unseen_sequence_lengths_reuse_native_programs(self):
        from unittest.mock import patch

        from kohakufa_cute import native

        for mode in ("dense", "mask", "padding", "causal", "gqa", "projection"):
            with self.subTest(mode=mode):
                native.extension().clear_cache()
                with patch.object(native, "compile_kernel", wraps=native.compile_kernel) as compiler:
                    for queries, keys in ((137, 193), (144, 208), (201, 269)):
                        q = torch.randn(2, 4, queries, 64, device="cuda", dtype=torch.bfloat16)
                        k, v = [
                            torch.randn(2, 2 if mode == "gqa" else 4, keys, 64, device="cuda", dtype=q.dtype)
                            for _ in range(2)
                        ]
                        if mode == "projection":
                            q, k, v = [x.transpose(1, 2).contiguous().transpose(1, 2) for x in (q, k, v)]
                        mask = None
                        if mode == "mask":
                            mask = torch.rand(1, 1, queries, keys, device="cuda") > 0.25
                            mask[..., 0, :] = False
                        elif mode == "padding":
                            mask = torch.arange(keys, device="cuda") < keys - 7
                        self.compare(q, k, v, mask=mask, causal=mode == "causal")
                        self.assertEqual(compiler.call_count, 2)
                        self.assertEqual(native.extension().cache_size(), 2)

    def test_dynamic_lengths_cross_blackwell_pipeline_threshold(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("Requires datacentre Blackwell TMEM.")
        from unittest.mock import patch

        from kohakufa_cute import native

        native.extension().clear_cache()
        with patch.object(native, "compile_kernel", wraps=native.compile_kernel) as compiler:
            for index, queries in enumerate((511, 513, 519)):
                q = torch.randn(1, 2, queries, 128, device="cuda", dtype=torch.bfloat16)
                k, v = [torch.randn(1, 2, 193, 128, device="cuda", dtype=q.dtype) for _ in range(2)]
                self.compare(q, k, v)
                self.assertEqual(compiler.call_count, 2 if index == 0 else 4)

    def test_sm103_wide_reuses_cache_across_length_dispatch(self):
        if torch.cuda.get_device_capability() != (10, 3):
            self.skipTest("Requires the SM103 wide backward.")
        from unittest.mock import patch

        from kohakufa_cute import native

        for causal in (False, True):
            with self.subTest(causal=causal):
                native.extension().clear_cache()
                try:
                    with patch.object(native, "compile_kernel", wraps=native.compile_kernel) as compiler:
                        cases = (
                            (511, 511, False),
                            (512, 512, False),
                            (513, 513, False),
                            (2048, 2048, False),
                            (1024, 1024, False),
                            (1025, 1025, False),
                            (2049, 2049, False),
                            (2049, 2051, False),
                            (2048, 2048, True),
                            (2048, 2048, False),
                        )
                        for queries, keys, projection in cases:
                            with self.subTest(queries=queries, keys=keys, projection=projection):
                                torch.manual_seed(371)
                                q = torch.randn(1, 8, queries, 512, device="cuda", dtype=torch.bfloat16)
                                k, v = [torch.randn(1, 8, keys, 512, device="cuda", dtype=q.dtype) for _ in range(2)]
                                if projection:
                                    q, k, v = [tensor.transpose(1, 2).contiguous().transpose(1, 2) for tensor in (q, k, v)]
                                self.compare(q, k, v, causal=causal)
                                self.assertEqual(compiler.call_count, 2)
                                self.assertEqual(native.extension().cache_size(), 2)
                finally:
                    native.extension().clear_cache()

    def test_sm103_wide_graph_reads_changed_inputs(self):
        if torch.cuda.get_device_capability() != (10, 3):
            self.skipTest("Requires the SM103 wide backward.")
        from benchmarks.reference import fp64_reference
        from kohakufa_cute import attention, native

        for causal, sequence in ((False, 512), (True, 512), (False, 2048), (True, 2048)):
            with self.subTest(causal=causal, sequence=sequence):
                native.extension().clear_cache()
                try:
                    torch.manual_seed(372)
                    inputs = [
                        torch.randn(1, 8, sequence, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                        for _ in range(3)
                    ]
                    dout = torch.randn_like(inputs[0])

                    def run():
                        out = attention(*inputs, causal=causal, scale=-0.125)
                        return out, *torch.autograd.grad(out, inputs, dout)

                    run()
                    run()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        actual = run()
                    for _ in range(2):
                        with torch.no_grad():
                            for tensor in (*inputs, dout):
                                tensor.copy_(torch.randn_like(tensor))
                        graph.replay()
                        torch.cuda.synchronize()
                        expected = fp64_reference(inputs, dout, causal=causal, scale=-0.125)
                        for got, want in zip(actual, expected):
                            torch.testing.assert_close(got.double(), want, atol=0.03, rtol=0.03)
                finally:
                    native.extension().clear_cache()

    def test_sm103_wide_preserves_zero_covariance(self):
        if torch.cuda.get_device_capability() != (10, 3):
            self.skipTest("Requires the SM103 wide backward.")
        from benchmarks.reference import fp64_reference
        from kohakufa_cute import attention, native

        for causal, sequence in ((False, 513), (True, 513), (False, 2049), (True, 2049)):
            with self.subTest(causal=causal, sequence=sequence):
                native.extension().clear_cache()
                try:
                    q = torch.zeros(1, 8, sequence, 512, device="cuda", dtype=torch.bfloat16)
                    k, v, dout = torch.zeros_like(q), torch.zeros_like(q), torch.zeros_like(q)
                    k[..., :3, 0] = torch.tensor([-1, 0, 1], device="cuda", dtype=q.dtype)
                    v[..., 0], dout[..., 0] = 1, 1
                    v[..., 1, 0] = 1 + torch.finfo(q.dtype).eps
                    for scale in (-0.125, 0.0, 1.0):
                        with self.subTest(scale=scale):
                            expected = fp64_reference((q, k, v), dout, causal=causal, scale=scale)
                            inputs = [tensor.detach().requires_grad_(True) for tensor in (q, k, v)]
                            out = attention(*inputs, causal=causal, scale=scale)
                            actual = out, *torch.autograd.grad(out, inputs, dout)
                            self.assertEqual(actual[1][..., 2:, :].count_nonzero().item(), 0)
                            for got, want in zip(actual, expected):
                                torch.testing.assert_close(got.double(), want, atol=0, rtol=0.03)
                finally:
                    native.extension().clear_cache()

    def test_dynamic_lengths_preserve_forward_configuration_thresholds(self):
        from unittest.mock import patch

        from kohakufa_cute import native

        cases = (
            (2, 16, ((1023, 17), (1025, 17), (1027, 17))),
            (1, 2, ((1535, 1031), (1537, 1031), (1551, 1043))),
        )
        for batch, heads, lengths in cases:
            with self.subTest(batch=batch, heads=heads):
                native.extension().clear_cache()
                with patch.object(native, "compile_kernel", wraps=native.compile_kernel) as compiler:
                    for index, (queries, keys) in enumerate(lengths):
                        q = torch.randn(batch, heads, queries, 128, device="cuda", dtype=torch.bfloat16)
                        k, v = [torch.randn(batch, heads, keys, 128, device="cuda", dtype=q.dtype) for _ in range(2)]
                        self.compare(q, k, v)
                        self.assertEqual(compiler.call_count, 2 if index == 0 else 3)

    def test_warm_dispatch_does_not_reenter_python_kernels(self):
        import cProfile
        from pathlib import Path
        from types import CodeType

        from kohakufa_cute import api

        q, k, v = [torch.randn(1, 2, 17, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
        dout = torch.randn_like(q)
        mask = torch.ones(17, 17, device="cuda", dtype=torch.bool)
        for fn in (api.scaled_dot_product_attention, api.automatic_scaled_dot_product_attention):
            for selected_mask in (None, mask):
                with self.subTest(backend=fn.__name__, masked=selected_mask is not None):

                    def run():
                        out = fn(q, k, v, attn_mask=selected_mask)
                        return torch.autograd.grad(out, (q, k, v), dout)

                    run()
                    torch.cuda.synchronize()
                    profile = cProfile.Profile()
                    with profile:
                        run()
                    torch.cuda.synchronize()
                    callbacks = [
                        entry.code.co_name
                        for entry in profile.getstats()
                        if isinstance(entry.code, CodeType) and "cutlass" in Path(entry.code.co_filename).parts
                    ]
                    self.assertEqual(callbacks, [])

    def test_automatic_unsupported_first_call_does_not_build_extension(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from kohakufa_cute import api

        original = torch.nn.functional.scaled_dot_product_attention
        with patch.object(api, "_extension", side_effect=RuntimeError("Unexpected extension build")) as builder:
            builder.cache_info.return_value = SimpleNamespace(currsize=0)
            for dtype, dropout in [(torch.float32, 0.0), (torch.bfloat16, 0.1)]:
                q = torch.randn(1, 2, 17, 64, device="cuda", dtype=dtype, requires_grad=True)
                torch.manual_seed(101)
                actual = api.automatic_scaled_dot_product_attention(q, q, q, dropout_p=dropout)
                torch.manual_seed(101)
                expected = original(q, q, q, dropout_p=dropout)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                torch.testing.assert_close(
                    torch.autograd.grad(actual.sum(), q)[0], torch.autograd.grad(expected.sum(), q)[0], atol=0, rtol=0
                )
                builder.assert_not_called()

    def test_native_sdpa_rejects_unsupported_features(self):
        from kohakufa_cute import scaled_dot_product_attention

        q, k, v = [torch.randn(1, 2, 17, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        mask = torch.ones(17, 17, device="cuda", dtype=torch.bool)
        cases = [
            ((q, k, v), {"dropout_p": 0.1}, "dropout"),
            ((q, k, v), {"attn_mask": mask, "is_causal": True}, "folded"),
            ((q, k, v), {"attn_mask": mask.float()}, "boolean"),
            ((q, k[:, :1], v[:, :1]), {}, "enable_gqa"),
            ((q.float(), k.float(), v.float()), {}, "FP32"),
            ((q.cpu(), k.cpu(), v.cpu()), {}, "CUDA"),
            ((q.squeeze(0), k, v), {}, "4D"),
            ((q, k, v[:, :, :16]), {}, "K/V shapes"),
            ((q[:, :, :0], k, v), {}, "nonempty"),
        ]
        for inputs, kwargs, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                scaled_dot_product_attention(*inputs, **kwargs)

    def test_native_dispatch_concurrent_streams_and_cache_release(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier

        import tvm_ffi

        from kohakufa_cute import attention
        from kohakufa_cute.native import extension

        tensors = [torch.randn(1, 2, 17, 64, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
        inputs = [tensor.detach().requires_grad_() for tensor in tensors[:3]]
        output = attention(*inputs, scale=-0.125)
        expected = (output, *torch.autograd.grad(output, inputs, tensors[3]))
        torch.cuda.synchronize()
        native = extension()
        native.clear_cache()
        barrier = Barrier(2)

        def run():
            stream = torch.cuda.Stream()
            barrier.wait()
            with torch.cuda.stream(stream):
                inputs = [tensor.detach().clone().requires_grad_() for tensor in tensors[:3]]
                output = attention(*inputs, scale=-0.125)
                gradients = torch.autograd.grad(output, inputs, tensors[3])
            stream.synchronize()
            return output, *gradients

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: run(), range(2)))
        for result in results:
            for got, want in zip(result, expected):
                torch.testing.assert_close(got, want, atol=0, rtol=0)
        self.assertGreater(native.cache_size(), 0)
        self.assertLessEqual(native.cache_size(), 128)
        self.assertFalse(any(name.startswith("kohakufa_cute.native.") for name in tvm_ffi.registry.list_global_func_names()))
        native.clear_cache()
        self.assertEqual(native.cache_size(), 0)

    def test_compile_after_backend_selection_in_fresh_process(self):
        import subprocess
        import sys
        import textwrap

        code = textwrap.dedent(
            """
            from types import SimpleNamespace
            import torch
            from benchmarks.precision import fp64_reference
            from kohakufa_cute import scaled_dot_product_attention

            torch.overrides.get_testing_overrides.cache_clear()
            q = torch.zeros(1, 1, 1, 128, device="cuda", dtype=torch.bfloat16)
            k = torch.zeros(1, 1, 2, 128, device="cuda", dtype=q.dtype)
            v = torch.zeros_like(k)
            q[..., 0] = 10
            k[0, 0, :, 0] = torch.tensor([10, 5], device="cuda", dtype=q.dtype)
            v[0, 0, :, 0] = torch.tensor([1, -1], device="cuda", dtype=q.dtype)
            dout = torch.zeros_like(q)
            dout[..., 0] = 1
            expected = fp64_reference((q, k, v), dout, scale=1.0)
            inputs = [t.requires_grad_() for t in (q, k, v)]
            fn = torch.compile(scaled_dot_product_attention, fullgraph=True)
            out = fn(*inputs, scale=1.0)
            gradients = torch.autograd.grad(out, inputs, dout)
            assert gradients[0][..., 0].item() > 0
            for actual, reference in zip((out, *gradients), expected):
                torch.testing.assert_close(actual.double(), reference, rtol=0.03, atol=0)
            """
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def compare(self, q, k, v, mask=None, causal=False, scale=None, compiled=False):
        from kohakufa_cute import attention

        refs = [x.detach().double().requires_grad_() for x in (q, k, v)]
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            expected = torch.nn.functional.scaled_dot_product_attention(
                *refs, attn_mask=mask, is_causal=causal, scale=scale, enable_gqa=q.shape[1] != k.shape[1]
            )
        dout = torch.randn_like(q)
        ref_grads = torch.autograd.grad(expected, refs, dout.double())
        inputs = [x.detach().requires_grad_() for x in (q, k, v)]
        if compiled:
            torch.compiler.reset()
        function = torch.compile(attention, fullgraph=True) if compiled else attention
        actual = function(*inputs, mask=mask, causal=causal, scale=scale)
        grads = torch.autograd.grad(actual, inputs, dout)
        tolerance = 0.02 if q.dtype == torch.bfloat16 else 0.003
        for got, want in zip((actual, *grads), (expected, *ref_grads)):
            torch.testing.assert_close(got.double(), want, atol=tolerance, rtol=tolerance)
        if mask is not None and mask.ndim == 4 and not mask[..., 0, :].any():
            self.assertEqual(actual[..., 0, :].count_nonzero().item(), 0)
            self.assertEqual(grads[0][..., 0, :].count_nonzero().item(), 0)

    def test_forward_backward_tails_masks_gqa_and_compile(self):
        for dtype in (torch.float16, torch.bfloat16):
            for mode in ("dense", "mask", "padding", "causal", "gqa", "strided"):
                with self.subTest(dtype=dtype, mode=mode):
                    torch.manual_seed(41)
                    q = torch.randn(2, 4, 33, 64, device="cuda", dtype=dtype)
                    k, v = [torch.randn(2, 2 if mode == "gqa" else 4, 41, 64, device="cuda", dtype=dtype) for _ in range(2)]
                    mask = None
                    if mode == "mask":
                        mask = torch.rand(1, 1, 33, 41, device="cuda") > 0.25
                        mask[..., 0, :] = False
                    if mode == "padding":
                        mask = torch.arange(41, device="cuda") < 29
                    if mode == "strided":
                        q, k, v = [x.transpose(1, 2).contiguous().transpose(1, 2) for x in (q, k, v)]
                    self.compare(q, k, v, mask, mode == "causal", scale=0.3, compiled=True)

    def test_head_dimensions(self):
        for dtype in (torch.float16, torch.bfloat16):
            for dim in (1, 17, 96, 128, 256, 512):
                with self.subTest(dtype=dtype, dim=dim):
                    torch.manual_seed(42)
                    q = torch.randn(1, 1, 17, dim, device="cuda", dtype=dtype)
                    k, v = [torch.randn(1, 1, 35, dim, device="cuda", dtype=dtype) for _ in range(2)]
                    self.compare(q, k, v)

    def test_preserves_projection_layout_with_compiled_backward(self):
        from kohakufa_cute import attention

        dimensions = (
            (64, 128, 256, 512) if torch.cuda.get_device_capability() in ((8, 6), (8, 9), (12, 0)) else (64, 128, 256)
        )
        for dim, causal in itertools.product(dimensions, (False, True)):
            with self.subTest(dim=dim, causal=causal):
                torch.manual_seed(87)
                inputs = [
                    torch.randn(1, 137, 2, dim, device="cuda", dtype=torch.bfloat16).transpose(1, 2).requires_grad_()
                    for _ in range(3)
                ]
                torch.compiler.reset()
                output = torch.compile(attention, fullgraph=True)(*inputs, causal=causal)
                gradients = torch.autograd.grad(output, inputs, torch.randn_like(output))
                for tensor in (output, *gradients):
                    self.assertEqual(tensor.stride(), inputs[0].stride())
                self.compare(*inputs, causal=causal, compiled=True)

    def test_wide_head_tails_masks_gqa_and_strides(self):
        for dim in (129, 192, 257, 384, 511):
            with self.subTest(dim=dim):
                torch.manual_seed(81)
                q = torch.randn(1, 4, 19, dim * 2, device="cuda", dtype=torch.bfloat16)[..., ::2]
                k, v = [torch.randn(1, 2, 37, dim + 1, device="cuda", dtype=torch.bfloat16)[..., 1:] for _ in range(2)]
                mask = torch.rand(1, 1, 19, 37, device="cuda") > 0.2
                mask[..., 0, :] = False
                self.compare(q, k, v, mask=mask, scale=-0.1)

    def test_causal_keys_beyond_query_length(self):
        for dim in (64, 128):
            for queries, keys in ((32, 193), (64, 193), (65, 193)):
                with self.subTest(dim=dim, queries=queries, keys=keys):
                    torch.manual_seed(88)
                    q = torch.randn(1, 2, queries, dim, device="cuda", dtype=torch.bfloat16)
                    k, v = [torch.randn(1, 2, keys, dim, device="cuda", dtype=q.dtype) for _ in range(2)]
                    self.compare(q, k, v, causal=True)

    def test_large_logits_key_and_value_gradients(self):
        from benchmarks.precision import fp64_reference
        from kohakufa_cute import attention

        for dtype in (torch.float16, torch.bfloat16):
            generator = torch.Generator(device="cuda").manual_seed(2)
            centers = torch.randn(1, 2, 8, 64, device="cuda", generator=generator)
            pick = torch.randint(8, (1, 2, 269), device="cuda", generator=generator)
            base = centers.gather(2, pick[..., None].expand(-1, -1, -1, 64))
            q, k = [
                (400 * (base + 1e-3 * torch.randn(base.shape, device="cuda", generator=generator))).to(dtype)
                for _ in range(2)
            ]
            v, dout = [torch.randn(base.shape, device="cuda", generator=generator).to(dtype) for _ in range(2)]
            for causal in (False, True):
                with self.subTest(dtype=dtype, causal=causal):
                    _, *expected_grads = fp64_reference((q, k, v), dout, causal=causal)
                    inputs = [x.detach().requires_grad_() for x in (q, k, v)]
                    actual = attention(*inputs, causal=causal)
                    grads = torch.autograd.grad(actual, inputs, dout)
                    for got, want, bound in zip(grads[1:], expected_grads[1:], (0.995, 0.9999)):
                        cosine = torch.nn.functional.cosine_similarity(got.double().flatten(), want.flatten(), dim=0)
                        self.assertGreater(cosine.item(), bound)

    def test_wide_head_gradient_cancellation(self):
        from benchmarks.precision import fp64_reference, make_inputs
        from kohakufa_cute import attention

        dimensions = (
            (128, 256, 512)
            if torch.cuda.get_device_capability() == (12, 0)
            else (256, 512) if torch.cuda.get_device_capability() in ((8, 6), (8, 9)) else (256,)
        )
        cases = ((torch.float16, 1e5), (torch.float16, 1e6), (torch.bfloat16, 1e6))
        for dimension, (dtype, logit_scale) in itertools.product(dimensions, cases):
            with self.subTest(dim=dimension, dtype=dtype, logit_scale=logit_scale):
                q, k, v, dout = make_inputs(logit_scale, device="cuda", dtype=dtype, seq=256, dim=dimension, seed=0)
                _, *expected_grads = fp64_reference((q, k, v), dout)
                inputs = [tensor.requires_grad_() for tensor in (q, k, v)]
                output = attention(*inputs)
                grads = torch.autograd.grad(output, inputs, dout)
                for got, want in zip(grads[:2], expected_grads[:2]):
                    self.assertTrue(torch.isfinite(got).all())
                    rounded_reference = want.to(dtype)
                    if rounded_reference.count_nonzero().item() == 0:
                        # Classify underflow from this device's rounded FP64 reference.
                        self.assertEqual(got.count_nonzero().item(), 0)
                        continue
                    self.assertGreater(rounded_reference.count_nonzero().item(), 0)
                    relative_error = torch.linalg.vector_norm(got.double() - want) / torch.linalg.vector_norm(want)
                    cosine = torch.nn.functional.cosine_similarity(
                        got.double().flatten(), want.flatten(), dim=0, eps=torch.finfo(want.dtype).tiny
                    )
                    self.assertLess(relative_error.item(), 0.3)
                    self.assertGreater(cosine.item(), 0.995)

    def test_blackwell_wide_query_keeps_zero_covariance(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("Requires the Blackwell query pipeline.")
        from unittest.mock import patch

        from benchmarks.reference import fp64_reference
        from kohakufa_cute import attention, kernels, native

        for dim, dtype in itertools.product((256, 512), (torch.float16, torch.bfloat16)):

            def configuration(device, mode, descriptors):
                return kernels._SM100_FUSED_CONFIGURATIONS[(dim, mode)]

            with self.subTest(dim=dim, dtype=dtype), patch.object(kernels, "kernel_configuration", configuration):
                native.extension().clear_cache()
                try:
                    q = torch.zeros(1, 2, 257, dim, device="cuda", dtype=dtype)
                    k = torch.zeros(1, 2, 3, dim, device="cuda", dtype=dtype)
                    k[..., 0] = torch.tensor([-1, 0, 1], device="cuda", dtype=dtype)
                    v, dout = torch.zeros_like(k), torch.zeros_like(q)
                    v[..., 0] = 1
                    v[:, :, 1, 0] = 1 + torch.finfo(dtype).eps
                    dout[..., 0] = 1
                    expected = fp64_reference((q, k, v), dout, scale=1.0)
                    inputs = [t.requires_grad_() for t in (q, k, v)]
                    out = attention(*inputs, scale=1.0)
                    actual = out, *torch.autograd.grad(out, inputs, dout)
                    self.assertEqual(actual[1].count_nonzero().item(), 0)
                    for got, want in zip(actual, expected):
                        torch.testing.assert_close(got.double(), want, atol=1e-7 if dtype == torch.float16 else 0, rtol=0.03)
                finally:
                    native.extension().clear_cache()

    def test_saturated_softmax_gradient_sign_and_magnitude(self):
        from benchmarks.precision import fp64_reference
        from kohakufa_cute import attention

        if torch.cuda.get_device_capability() not in {(8, 6), (8, 9), (9, 0), (10, 0), (10, 3), (12, 0)}:
            self.skipTest("Centered backward requires Ampere, Ada, Hopper or Blackwell.")
        for dtype in (torch.bfloat16, torch.float16):
            for dim in (1, 17, 64, 80, 96, 128, 160, 192, 224, 256, 257, 384, 511, 512):
                with self.subTest(dtype=dtype, dim=dim):
                    q = torch.zeros(1, 1, 1, dim, device="cuda", dtype=dtype)
                    k = torch.zeros(1, 1, 2, dim, device="cuda", dtype=dtype)
                    v = torch.zeros_like(k)
                    dout = torch.zeros_like(q)
                    q[..., 0] = 10
                    k[..., 0] = torch.tensor([10, 5 if dtype == torch.bfloat16 else 9], device="cuda", dtype=dtype)
                    v[..., 0] = torch.tensor([1, -1], device="cuda", dtype=dtype)
                    dout[..., 0] = 1
                    reference = fp64_reference((q, k, v), dout, scale=1.0)
                    inputs = [tensor.requires_grad_() for tensor in (q, k, v)]
                    output = attention(*inputs, scale=1.0)
                    gradients = torch.autograd.grad(output, inputs, dout)
                    self.assertGreater(gradients[0][..., 0].item(), 0)
                    for actual, expected in zip((output, *gradients), reference):
                        torch.testing.assert_close(
                            actual.double(), expected, atol=1e-7 if dtype == torch.float16 else 0, rtol=0.02
                        )

    def test_long_dense_tails_gqa_and_compile(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(21)
                q = torch.randn(1, 1537, 4, 128, device="cuda", dtype=dtype).transpose(1, 2)
                k, v = [torch.randn(1, 1601, 2, 128, device="cuda", dtype=dtype).transpose(1, 2) for _ in range(2)]
                self.compare(q, k, v, scale=-0.125, compiled=True)

    def test_long_dense_saturated_gradient(self):
        from benchmarks.precision import fp64_reference
        from kohakufa_cute import attention

        q = torch.zeros(1, 1537, 2, 128, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
        k = torch.zeros(1, 1601, 2, 128, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
        v, dout = torch.zeros_like(k), torch.zeros_like(q)
        q[..., 0], k[..., 0], k[:, :, 0, 0] = 10, 5, 10
        v[..., 0], v[:, :, 0, 0], dout[..., 0] = -1, 1, 1
        expected = fp64_reference((q, k, v), dout, scale=1.0)
        inputs = [tensor.requires_grad_() for tensor in (q, k, v)]
        output = attention(*inputs, scale=1.0)
        actual = (output, *torch.autograd.grad(output, inputs, dout))
        self.assertGreater(actual[1][..., 0].min().item(), 0)
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got.double(), want, atol=0, rtol=0.03)

    def test_long_masked_runs_keep_tma_stages_synchronized(self):
        from benchmarks.precision import fp64_reference
        from kohakufa_cute import attention

        for dim in (64, 128):
            with self.subTest(dim=dim):
                torch.manual_seed(100)
                q, k, v, dout = [torch.randn(1, 2, 4096, dim, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
                segments = torch.arange(4096, device="cuda") // 256
                mask = segments[:, None] == segments[None, :]
                inputs = [tensor.requires_grad_() for tensor in (q, k, v)]
                expected = fp64_reference(inputs, dout, mask)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        output = attention(*inputs, mask=mask)
                        gradients = torch.autograd.grad(output, inputs, dout)
                torch.cuda.current_stream().wait_stream(stream)
                for actual, reference in zip((output, *gradients), expected):
                    torch.testing.assert_close(actual.double(), reference, atol=0.02, rtol=0.02)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    output = attention(*inputs, mask=mask)
                    gradients = torch.autograd.grad(output, inputs, dout)
                for _ in range(10):
                    graph.replay()
                torch.cuda.synchronize()
                for actual, reference in zip((output, *gradients), expected):
                    torch.testing.assert_close(actual.double(), reference, atol=0.02, rtol=0.02)

    def test_single_query_and_key_tma_layouts(self):
        for dim in (64, 128, 256):
            for queries, keys, batch, heads, kvheads, bshd, causal in (
                (1, 2, 1, 1, 1, False, False),
                (2, 1, 1, 1, 1, False, False),
                (1, 1, 2, 2, 1, True, False),
                (1, 65, 1, 2, 2, True, True),
            ):
                with self.subTest(dim=dim, queries=queries, keys=keys, bshd=bshd, causal=causal):
                    torch.manual_seed(97)
                    q = torch.randn(batch, heads, queries, dim, device="cuda", dtype=torch.bfloat16)
                    k, v = [torch.randn(batch, kvheads, keys, dim, device="cuda", dtype=q.dtype) for _ in range(2)]
                    if bshd:
                        q, k, v = [tensor.transpose(1, 2).contiguous().transpose(1, 2) for tensor in (q, k, v)]
                    self.compare(q, k, v, causal=causal)

    def test_centered_backward_compilation(self):
        for dtype in (torch.float16, torch.bfloat16):
            for dim in (64, 80, 128, 192, 256, 512):
                with self.subTest(dtype=dtype, dim=dim):
                    torch.manual_seed(85)
                    q = torch.randn(1, 4, 67, dim, device="cuda", dtype=dtype)
                    k, v = [torch.randn(1, 2, 83, dim, device="cuda", dtype=dtype) for _ in range(2)]
                    mask = torch.arange(67, device="cuda")[:, None] // 19 == torch.arange(83, device="cuda")[None, :] // 19
                    self.compare(q, k, v, mask=mask, compiled=True)

    def test_block_masks_and_broadcast_axes(self):
        torch.manual_seed(82)
        q = torch.randn(2, 4, 137, 64, device="cuda", dtype=torch.bfloat16)
        k, v = [torch.randn(2, 2, 149, 64, device="cuda", dtype=q.dtype) for _ in range(2)]
        queries = torch.arange(137, device="cuda")[:, None]
        keys = torch.arange(149, device="cuda")[None, :]
        masks = {
            "segments": (queries // 43 == keys // 43)[None, None],
            "padding": (keys < 91)[None, None],
            "query": (queries < 81)[None, None],
            "full": torch.ones(1, 1, 1, 1, device="cuda", dtype=torch.bool),
            "empty": torch.zeros(1, 1, 1, 1, device="cuda", dtype=torch.bool),
            "strided": (torch.rand(2, 4, 274, 298, device="cuda") > 0.4)[..., ::2, ::2],
        }
        for name, mask in masks.items():
            with self.subTest(mask=name):
                self.compare(q, k, v, mask=mask, compiled=True)

    def test_zero_negative_scale_and_offset_views(self):
        q = torch.randn(1, 2, 17, 65, device="cuda", dtype=torch.bfloat16)[..., 1:]
        k, v = [torch.randn(1, 2, 33, 65, device="cuda", dtype=torch.bfloat16)[..., 1:] for _ in range(2)]
        for scale in (0.0, -0.25):
            with self.subTest(scale=scale):
                self.compare(q, k, v, scale=scale)

    def test_cuda_graph_current_stream_and_gradients(self):
        from kohakufa_cute import attention

        inputs = [torch.randn(1, 2, 33, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                output = attention(*inputs)
                grads = torch.autograd.grad(output.sum(), inputs)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = attention(*inputs)
            grads = torch.autograd.grad(output.sum(), inputs)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(all(torch.isfinite(x).all() for x in (output, *grads)))
        reference = torch.nn.functional.scaled_dot_product_attention(*inputs)
        torch.testing.assert_close(output, reference, atol=0.01, rtol=0.01)

    def test_cuda_graph_reclassifies_changed_masks(self):
        from kohakufa_cute import attention

        for dim in (64, 128, 256):
            torch.manual_seed(84)
            inputs = [torch.randn(1, 2, 137, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
            dout = torch.randn_like(inputs[0])
            ids = torch.arange(137, device="cuda") // 43
            mask = (ids[:, None] == ids[None, :])[None, None]
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    output = attention(*inputs, mask=mask)
                    grads = torch.autograd.grad(output, inputs, dout)
            torch.cuda.current_stream().wait_stream(stream)
            del output, grads
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = attention(*inputs, mask=mask)
                grads = torch.autograd.grad(output, inputs, dout)
            for visible in (False, True):
                mask.fill_(visible)
                graph.replay()
                reference_inputs = [x.detach().double().requires_grad_() for x in inputs]
                with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                    expected = torch.nn.functional.scaled_dot_product_attention(*reference_inputs, attn_mask=mask)
                ref_grads = torch.autograd.grad(expected, reference_inputs, dout.double())
                for got, want in zip((output, *grads), (expected, *ref_grads)):
                    torch.testing.assert_close(got.double(), want, atol=0.02, rtol=0.02)


if __name__ == "__main__":
    unittest.main()
