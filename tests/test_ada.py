"""Qualify the complete RTX 4090 attention schedules on live inputs."""

import itertools
import math
import unittest
from unittest.mock import patch

import torch


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_name() == "NVIDIA GeForce RTX 4090",
    "Requires a full RTX 4090",
)
class AdaScheduleTests(unittest.TestCase):
    def setUp(self):
        from kohakufa_cute import kernels, native

        native.extension().clear_cache()
        kernels._compile_kernel.cache_clear()
        self.addCleanup(native.extension().clear_cache)
        self.addCleanup(kernels._compile_kernel.cache_clear)

    def assert_reference(self, actual, expected, dtype):
        from benchmarks.reference import metrics

        for name, got, want in zip(("out", "dq", "dk", "dv"), actual, expected):
            error = metrics(got, want)
            with self.subTest(component=name):
                self.assertTrue(error["finite"])
                if error["reference_norm"]:
                    self.assertLess(error["relative_error"], 0.01 if dtype == torch.bfloat16 else 0.005)
                else:
                    self.assertEqual(got.count_nonzero().item(), 0)

    def test_long_graphs_read_changed_inputs_and_partial_tiles(self):
        from benchmarks.reference import fp64_reference
        from kohakufa_cute import attention

        for dtype, dimension, causal, (queries, keys) in itertools.product(
            (torch.float16, torch.bfloat16), (16, 64, 128, 256), (False, True), ((1024, 1024), (1025, 1031))
        ):
            with self.subTest(dtype=dtype, dimension=dimension, causal=causal, queries=queries, keys=keys):
                torch.manual_seed(892)
                q = torch.randn(1, 16, queries, dimension, device="cuda", dtype=dtype, requires_grad=True)
                k, v = [
                    torch.randn(1, 16, keys, dimension, device="cuda", dtype=dtype, requires_grad=True) for _ in range(2)
                ]
                dout = torch.randn_like(q)

                def run():
                    out = attention(q, k, v, causal=causal)
                    return out, *torch.autograd.grad(out, (q, k, v), dout)

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    run()
                    run()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        actual = run()
                torch.cuda.current_stream().wait_stream(stream)
                for replay in range(2):
                    if replay:
                        with torch.no_grad():
                            for tensor in (q, k, v):
                                tensor.copy_(torch.randn_like(tensor) * 0.75)
                    graph.replay()
                    expected = fp64_reference((q, k, v), dout, causal=causal)
                    self.assert_reference(actual, expected, dtype)

    def test_runtime_length_boundaries_preserve_cache_and_exact_results(self):
        from kohakufa_cute import attention, kernels, native

        for dimension, causal in itertools.product((16, 64, 128, 256), (False, True)):
            native.extension().clear_cache()
            kernels._compile_kernel.cache_clear()
            saved = []
            with patch.object(native, "compile_kernel", wraps=native.compile_kernel) as compiler:
                counts = []
                for queries, keys in ((1023, 1023), (1025, 1023), (1025, 1031), (1023, 1031), (1025, 1031)):
                    with self.subTest(dimension=dimension, causal=causal, queries=queries, keys=keys):
                        torch.manual_seed(893)
                        q = torch.randn(1, 16, queries, dimension, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                        k, v = [
                            torch.randn(1, 16, keys, dimension, device="cuda", dtype=q.dtype, requires_grad=True)
                            for _ in range(2)
                        ]
                        dout = torch.randn_like(q)
                        out = attention(q, k, v, causal=causal)
                        actual = (out, *torch.autograd.grad(out, (q, k, v), dout))
                        saved.append((q, k, v, dout, actual))
                        counts.append(compiler.call_count)
                self.assertEqual(counts, [2, 4, 4, 4, 4] if dimension == 64 else [2] * 5)
            # Force separate controls, preserving warm reuse within each policy.
            native.extension().clear_cache()
            kernels._compile_kernel.cache_clear()
            with patch.object(kernels, "_rtx4090_tuning", return_value=False):
                for q, k, v, dout, actual in saved:
                    control_out = attention(q, k, v, causal=causal)
                    expected = (control_out, *torch.autograd.grad(control_out, (q, k, v), dout))
                    if dimension == 16:
                        from benchmarks.reference import fp64_reference

                        self.assert_reference(actual, fp64_reference((q, k, v), dout, causal=causal), q.dtype)
                    else:
                        for got, want in zip(actual, expected):
                            self.assertTrue(torch.equal(got.view(torch.uint8), want.view(torch.uint8)))

    def test_signed_scales_and_saturated_gradients(self):
        from benchmarks.reference import fp64_reference
        from kohakufa_cute import attention

        for dtype, dimension, causal, scale in itertools.product(
            (torch.float16, torch.bfloat16), (16, 64, 128, 256), (False, True), (-0.25, 0.0)
        ):
            with self.subTest(dtype=dtype, dimension=dimension, causal=causal, scale=scale):
                torch.manual_seed(894)
                q = torch.randn(1, 16, 1025, dimension, device="cuda", dtype=dtype, requires_grad=True)
                k, v = [torch.randn_like(q, requires_grad=True) for _ in range(2)]
                dout = torch.randn_like(q)
                out = attention(q, k, v, causal=causal, scale=scale)
                actual = (out, *torch.autograd.grad(out, (q, k, v), dout))
                self.assert_reference(actual, fp64_reference((q, k, v), dout, causal=causal, scale=scale), dtype)
        for dtype, dimension in itertools.product((torch.float16, torch.bfloat16), (16, 64)):
            q = torch.zeros(1, 16, 1025, dimension, device="cuda", dtype=dtype)
            k, v = torch.zeros_like(q), torch.zeros_like(q)
            dout = torch.zeros_like(q)
            q[..., 0], k[..., 0], k[:, :, 0, 0] = 10, 9 if dtype == torch.float16 else 5, 10
            v[..., 0], v[:, :, 0, 0], dout[..., 0] = -1, 1, 1
            q, k, v = [tensor.requires_grad_() for tensor in (q, k, v)]
            out = attention(q, k, v, scale=1.0)
            actual = (out, *torch.autograd.grad(out, (q, k, v), dout))
            self.assert_reference(actual, fp64_reference((q, k, v), dout, scale=1.0), dtype)

    def test_negative_maximum_and_lse_on_partial_tiles(self):
        from benchmarks.reference import fp64_reference
        from kohakufa_cute import attention, native

        for dtype, dimension, causal in itertools.product(
            (torch.float16, torch.bfloat16), (16, 64, 128, 256), (False, True)
        ):
            with self.subTest(dtype=dtype, dimension=dimension, causal=causal):
                torch.manual_seed(895)
                q = torch.randn(1, 16, 1025, dimension, device="cuda", dtype=dtype) * 0.3
                k = torch.randn(1, 16, 1031, dimension, device="cuda", dtype=dtype) * 0.3
                q[..., 0], k[..., 0] = 20, -20
                v, dout = torch.randn_like(k), torch.randn_like(q)
                q, k, v = [tensor.requires_grad_() for tensor in (q, k, v)]
                scale = dimension**-0.5
                _, maximum, logsum, _ = native.extension().forward(q, k, v, None, scale, causal)
                self.assertTrue((maximum < 0).all())
                self.assertTrue((maximum * scale + logsum * math.log(2) < 0).all())
                out = attention(q, k, v, causal=causal)
                actual = (out, *torch.autograd.grad(out, (q, k, v), dout))
                self.assert_reference(actual, fp64_reference((q, k, v), dout, causal=causal), dtype)

    def test_compiled_calls_cross_the_runtime_boundary(self):
        from benchmarks.reference import fp64_reference
        from kohakufa_cute import attention

        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        function = torch.compile(attention, fullgraph=True, dynamic=True)
        for dimension, sequence in itertools.product((16, 64, 128), (1023, 1025, 1023)):
            with self.subTest(dimension=dimension, sequence=sequence):
                torch.manual_seed(896)
                q, k, v, dout = [
                    torch.randn(1, 16, sequence, dimension, device="cuda", dtype=torch.bfloat16) for _ in range(4)
                ]
                inputs = [tensor.requires_grad_() for tensor in (q, k, v)]
                output = function(*inputs, causal=True)
                actual = (output, *torch.autograd.grad(output, inputs, dout))
                self.assert_reference(actual, fp64_reference(inputs, dout, causal=True), q.dtype)

    def test_complete_tile_guards_survive_native_and_compiled_cache_reuse(self):
        from benchmarks.reference import fp64_reference
        from kohakufa_cute import attention, kernels, native

        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        compiled = torch.compile(attention, fullgraph=True, dynamic=True)
        for dimension, causal in ((16, False), (64, False), (128, False), (128, True)):
            native.extension().clear_cache()
            kernels._compile_kernel.cache_clear()
            with patch.object(native, "compile_kernel", wraps=native.compile_kernel) as compiler:
                for index, (queries, keys) in enumerate(
                    ((1024, 1024), (1025, 1024), (1024, 1025), (1088, 1152), (1024, 1024))
                ):
                    with self.subTest(dimension=dimension, causal=causal, queries=queries, keys=keys):
                        torch.manual_seed(902)
                        q = torch.randn(1, 16, queries, dimension, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                        k, v = [
                            torch.randn(1, 16, keys, dimension, device="cuda", dtype=q.dtype, requires_grad=True)
                            for _ in range(2)
                        ]
                        dout = torch.randn_like(q)
                        function = compiled if index % 2 else attention
                        out = function(q, k, v, causal=causal)
                        actual = (out, *torch.autograd.grad(out, (q, k, v), dout))
                        self.assert_reference(actual, fp64_reference((q, k, v), dout, causal=causal), q.dtype)
                self.assertEqual(compiler.call_count, 2)

    def test_packed_statistics_preserve_signed_scales_negative_maxima_and_tiny_gradients(self):
        from benchmarks.reference import fp64_reference
        from kohakufa_cute import attention, native

        for dtype, case in itertools.product((torch.float16, torch.bfloat16), ("negative", "zero", "maxima", "sharp")):
            with self.subTest(dtype=dtype, case=case):
                torch.manual_seed(957)
                q, k, v, dout = [torch.randn(1, 16, 1024, 128, device="cuda", dtype=dtype) for _ in range(4)]
                scale = -0.25 if case == "negative" else 0.0 if case == "zero" else 128**-0.5
                if case == "maxima":
                    q.mul_(0.3)
                    k.mul_(0.3)
                    q[..., 0], k[..., 0] = 20, -20
                    _, maximum, logsum, _ = native.extension().forward(q, k, v, None, scale, True)
                    self.assertTrue((maximum < 0).all())
                    self.assertTrue((maximum * scale + logsum * math.log(2) < 0).all())
                elif case == "sharp":
                    for tensor in (q, k, v, dout):
                        tensor.zero_()
                    q[..., 0], k[..., 0], k[:, :, 0, 0] = 10, 9 if dtype == torch.float16 else 5, 10
                    v[..., 0], v[:, :, 0, 0], dout[..., 0] = -1, 1, 1
                    scale = 1.0
                inputs = [tensor.requires_grad_() for tensor in (q, k, v)]
                out = attention(*inputs, causal=True, scale=scale)
                actual = (out, *torch.autograd.grad(out, inputs, dout))
                self.assert_reference(actual, fp64_reference(inputs, dout, causal=True, scale=scale), dtype)
                if case == "sharp":
                    self.assertGreater(actual[1].double().norm().item(), 0.0)
                    self.assertGreater(actual[2].double().norm().item(), 0.0)

    def test_short_graphs_read_changed_inputs_and_strided_partial_tiles(self):
        from benchmarks.reference import fp64_reference
        from kohakufa_cute import attention

        for dtype, dimension, causal in itertools.product((torch.float16, torch.bfloat16), (128, 256, 512), (False, True)):
            with self.subTest(dtype=dtype, dimension=dimension, causal=causal):
                torch.manual_seed(897)
                # Keep aligned row copies eligible while leaving a gap between heads.
                q = torch.randn(1, 16, 131, dimension, device="cuda", dtype=dtype)[:, :, :127].requires_grad_()
                k, v = [
                    torch.randn(1, 16, 119, dimension, device="cuda", dtype=dtype)[:, :, :113].requires_grad_()
                    for _ in range(2)
                ]
                dout = torch.randn_like(q)

                def run():
                    out = attention(q, k, v, causal=causal)
                    return out, *torch.autograd.grad(out, (q, k, v), dout)

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    run()
                    run()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        actual = run()
                torch.cuda.current_stream().wait_stream(stream)
                for replay in range(2):
                    if replay:
                        with torch.no_grad():
                            for tensor in (q, k, v):
                                tensor.copy_(torch.randn_like(tensor) * 0.75)
                    graph.replay()
                    self.assert_reference(actual, fp64_reference((q, k, v), dout, causal=causal), dtype)

    def test_short_runtime_boundaries_and_compiled_calls(self):
        from benchmarks.reference import fp64_reference
        from kohakufa_cute import attention

        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        compiled = torch.compile(attention, fullgraph=True, dynamic=True)
        for dimension, causal in itertools.product((128, 256, 512), (False, True)):
            for queries, keys in ((1, 1), (17, 31), (128, 128), (129, 128), (128, 129), (129, 129), (128, 128)):
                with self.subTest(dimension=dimension, causal=causal, queries=queries, keys=keys):
                    torch.manual_seed(898)
                    q = torch.randn(1, 16, queries, dimension, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                    k, v = [
                        torch.randn(1, 16, keys, dimension, device="cuda", dtype=q.dtype, requires_grad=True)
                        for _ in range(2)
                    ]
                    dout = torch.randn_like(q)
                    function = compiled if dimension == 128 and causal else attention
                    out = function(q, k, v, causal=causal)
                    actual = (out, *torch.autograd.grad(out, (q, k, v), dout))
                    self.assert_reference(actual, fp64_reference((q, k, v), dout, causal=causal), q.dtype)

    def test_short_signed_scales_negative_statistics_and_tiny_gradients(self):
        from benchmarks.reference import fp64_reference
        from kohakufa_cute import attention, native

        for dtype, dimension, causal, scale in itertools.product(
            (torch.float16, torch.bfloat16), (128, 256, 512), (False, True), (-0.25, 0.0)
        ):
            with self.subTest(dtype=dtype, dimension=dimension, causal=causal, scale=scale):
                torch.manual_seed(899)
                q = torch.randn(1, 16, 127, dimension, device="cuda", dtype=dtype, requires_grad=True)
                k, v = [torch.randn(1, 16, 113, dimension, device="cuda", dtype=dtype, requires_grad=True) for _ in range(2)]
                dout = torch.randn_like(q)
                out = attention(q, k, v, causal=causal, scale=scale)
                actual = (out, *torch.autograd.grad(out, (q, k, v), dout))
                self.assert_reference(actual, fp64_reference((q, k, v), dout, causal=causal, scale=scale), dtype)
        for dtype, dimension in itertools.product((torch.float16, torch.bfloat16), (128, 256, 512)):
            with self.subTest(dtype=dtype, dimension=dimension):
                torch.manual_seed(900)
                q = torch.randn(1, 16, 127, dimension, device="cuda", dtype=dtype) * 0.3
                k = torch.randn(1, 16, 113, dimension, device="cuda", dtype=dtype) * 0.3
                q[..., 0], k[..., 0] = 20, -20
                v, dout = torch.randn_like(k), torch.randn_like(q)
                q, k, v = [tensor.requires_grad_() for tensor in (q, k, v)]
                _, maximum, logsum, _ = native.extension().forward(q, k, v, None, dimension**-0.5, True)
                self.assertTrue((maximum < 0).all())
                self.assertTrue((maximum * dimension**-0.5 + logsum * math.log(2) < 0).all())
                out = attention(q, k, v, causal=True)
                actual = (out, *torch.autograd.grad(out, (q, k, v), dout))
                self.assert_reference(actual, fp64_reference((q, k, v), dout, causal=True), dtype)

                q = torch.zeros(1, 16, 128, dimension, device="cuda", dtype=dtype)
                k, v, dout = [torch.zeros_like(q) for _ in range(3)]
                q[..., 0], k[..., 0], k[:, :, 0, 0] = 10, 9 if dtype == torch.float16 else 5, 10
                v[..., 0], v[:, :, 0, 0], dout[..., 0] = -1, 1, 1
                q, k, v = [tensor.requires_grad_() for tensor in (q, k, v)]
                out = attention(q, k, v, scale=1.0)
                actual = (out, *torch.autograd.grad(out, (q, k, v), dout))
                self.assert_reference(actual, fp64_reference((q, k, v), dout, scale=1.0), dtype)


if __name__ == "__main__":
    unittest.main()
