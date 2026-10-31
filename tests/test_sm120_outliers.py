"""Qualify the RTX 5090 short-forward and logical D16 dispatch boundaries."""

import itertools
import unittest
from unittest.mock import patch

import torch


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_name() == "NVIDIA GeForce RTX 5090",
    "Requires a full RTX 5090",
)
class SM120OutlierTests(unittest.TestCase):
    def setUp(self):
        from kohakufa_cute import kernels, native

        native.extension().clear_cache()
        kernels._compile_kernel.cache_clear()
        self.addCleanup(native.extension().clear_cache)
        self.addCleanup(kernels._compile_kernel.cache_clear)

    def assert_reference(self, actual, inputs, dout, causal, scale=None, *, norm_only=False):
        from benchmarks.reference import metrics
        from tests.attention_reference import attention_reference

        reference_fn = attention_reference
        if inputs[0].shape[1] == inputs[1].shape[1] and max(t.shape[2] for t in inputs) > 512:
            parts = [
                reference_fn(
                    tuple(t[:, h : h + 2] for t in inputs),
                    dout[:, h : h + 2],
                    causal=causal,
                    scale=scale,
                    output=actual[0][:, h : h + 2],
                )
                for h in range(0, inputs[0].shape[1], 2)
            ]
            expected = tuple(torch.cat([part[i] for part in parts], dim=1) for i in range(4))
        else:
            expected = reference_fn(inputs, dout, causal=causal, scale=scale, output=actual[0])
        for name, got, want in zip(("out", "dq", "dk", "dv"), actual, expected):
            error = metrics(got, want)
            with self.subTest(component=name):
                tolerance = 0.02 if got.dtype == torch.bfloat16 else 0.003
                if not norm_only:
                    torch.testing.assert_close(got.double(), want, atol=tolerance, rtol=tolerance)
                # Large common score offsets can move a cancelled coordinate
                # across a P/dS rounding boundary. These cases retain the
                # existing component norm gate below; they do not require
                # compensated coordinate accuracy.
                self.assertTrue(error["finite"])
                if error["reference_norm"]:
                    self.assertLess(error["relative_error"], 0.01 if got.dtype == torch.bfloat16 else 0.005)
                else:
                    self.assertEqual(got.count_nonzero().item(), 0)

    def run_attention(self, inputs, dout, causal, scale=None, mask=None):
        from kohakufa_cute import attention

        out = attention(*inputs, mask=mask, causal=causal, scale=scale)
        return out, *torch.autograd.grad(out, inputs, dout)

    def test_short_graphs_read_changed_strided_operands(self):
        for dtype, dimension, causal in itertools.product((torch.float16, torch.bfloat16), (128, 256, 512), (False, True)):
            with self.subTest(dtype=dtype, dimension=dimension, causal=causal):
                torch.manual_seed(1181)
                q = torch.randn(1, 16, 131, dimension, device="cuda", dtype=dtype)[:, :, :127].requires_grad_()
                k, v = [
                    torch.randn(1, 16, 119, dimension, device="cuda", dtype=dtype)[:, :, :113].requires_grad_()
                    for _ in range(2)
                ]
                dout = torch.randn_like(q)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    self.run_attention((q, k, v), dout, causal)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        actual = self.run_attention((q, k, v), dout, causal)
                torch.cuda.current_stream().wait_stream(stream)
                for replay in range(2):
                    if replay:
                        with torch.no_grad():
                            for tensor in (q, k, v, dout):
                                tensor.copy_(torch.randn_like(tensor) * 0.75)
                    graph.replay()
                    self.assert_reference(actual, (q, k, v), dout, causal)

    def test_short_cache_preserves_both_length_axes(self):
        from kohakufa_cute import kernels, native

        for dimension, causal in itertools.product((128, 256, 512), (False, True)):
            native.extension().clear_cache()
            kernels._compile_kernel.cache_clear()
            with patch.object(native, "compile_kernel", wraps=native.compile_kernel) as compiler:
                for queries, keys in ((128, 128), (129, 128), (128, 129), (127, 113), (128, 128)):
                    with self.subTest(dimension=dimension, causal=causal, queries=queries, keys=keys):
                        torch.manual_seed(1183)
                        q = torch.randn(1, 16, queries, dimension, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                        k, v = [
                            torch.randn(1, 16, keys, dimension, device="cuda", dtype=q.dtype, requires_grad=True)
                            for _ in range(2)
                        ]
                        dout = torch.randn_like(q)
                        actual = self.run_attention((q, k, v), dout, causal)
                        self.assert_reference(actual, (q, k, v), dout, causal)
                self.assertEqual(compiler.call_count, 2)

    def test_long_d16_boundaries_and_partial_rectangles(self):
        for dtype, causal, (queries, keys) in itertools.product(
            (torch.float16, torch.bfloat16),
            (False, True),
            ((1023, 1025), (1024, 1024), (1025, 1031), (1088, 1152), (4097, 4103)),
        ):
            with self.subTest(dtype=dtype, causal=causal, queries=queries, keys=keys):
                torch.manual_seed(1187)
                q = torch.randn(1, 16, queries, 16, device="cuda", dtype=dtype, requires_grad=True)
                k, v = [torch.randn(1, 16, keys, 16, device="cuda", dtype=dtype, requires_grad=True) for _ in range(2)]
                dout = torch.randn_like(q)
                actual = self.run_attention((q, k, v), dout, causal)
                self.assert_reference(actual, (q, k, v), dout, causal)

    def test_d64_column_cache_preserves_both_length_axes(self):
        from kohakufa_cute import native

        shapes = ((127, 127), (128, 128), (129, 129), (128, 129), (129, 128), (511, 511), (512, 512), (513, 513))
        for dtype, causal in itertools.product((torch.float16, torch.bfloat16), (False, True)):
            native.extension().clear_cache()
            with patch.object(native, "compile_kernel", wraps=native.compile_kernel) as compiler:
                for queries, keys in shapes:
                    with self.subTest(dtype=dtype, causal=causal, queries=queries, keys=keys):
                        torch.manual_seed(1471)
                        q = torch.randn(1, 16, queries, 64, device="cuda", dtype=dtype, requires_grad=True)
                        k, v = [
                            torch.randn(1, 16, keys, 64, device="cuda", dtype=dtype, requires_grad=True) for _ in range(2)
                        ]
                        dout = torch.randn_like(q)
                        self.assert_reference(self.run_attention((q, k, v), dout, causal), (q, k, v), dout, causal)
                self.assertEqual(compiler.call_count, 2)

    def test_d64_column_causal_upper_boundary_preserves_rectangular_fallbacks(self):
        from kohakufa_cute import native

        for dtype in (torch.float16, torch.bfloat16):
            native.extension().clear_cache()
            with patch.object(native, "compile_kernel", wraps=native.compile_kernel) as compiler:
                for queries, keys in ((4095, 4095), (4096, 4096), (4097, 4097), (4096, 4097), (4097, 4096)):
                    with self.subTest(dtype=dtype, queries=queries, keys=keys):
                        torch.manual_seed(1481)
                        q = torch.randn(1, 16, queries, 64, device="cuda", dtype=dtype, requires_grad=True)
                        k, v = [
                            torch.randn(1, 16, keys, 64, device="cuda", dtype=dtype, requires_grad=True) for _ in range(2)
                        ]
                        dout = torch.randn_like(q)
                        self.assert_reference(self.run_attention((q, k, v), dout, True), (q, k, v), dout, True)
                self.assertEqual(compiler.call_count, 2)

    def test_d64_column_graph_reads_changed_strided_inputs(self):
        from benchmarks.compare_sm120 import graph_nodes

        for dtype, causal in itertools.product((torch.float16, torch.bfloat16), (False, True)):
            with self.subTest(dtype=dtype, causal=causal):
                torch.manual_seed(1483)
                q, k, v = [
                    torch.randn(1, 16, 137, 64, device="cuda", dtype=dtype)[:, :, :129].requires_grad_() for _ in range(3)
                ]
                dout = torch.randn_like(q)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    self.run_attention((q, k, v), dout, causal)
                    graph = torch.cuda.CUDAGraph(keep_graph=True)
                    with torch.cuda.graph(graph, stream=stream):
                        actual = self.run_attention((q, k, v), dout, causal)
                torch.cuda.current_stream().wait_stream(stream)
                self.assertTrue(any("sm120_backwardD64Values" in n.get("symbol", "") for n in graph_nodes(graph)))
                for changed in (False, True):
                    if changed:
                        with torch.no_grad():
                            for tensor in (q, k, v, dout):
                                tensor.copy_(torch.randn_like(tensor) * 0.75)
                    with torch.no_grad():
                        for tensor in actual:
                            tensor.fill_(float("nan"))
                    graph.replay()
                    self.assert_reference(actual, (q, k, v), dout, causal)

    def test_d64_column_preserves_strong_causal_key_coordinate(self):
        from tests.test_key_precision import key_cancellation

        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                inputs, dout = key_cancellation(64, dtype, "cuda", square=True)
                q, k, v = inputs
                # Every active query now sees all 1031 non-negligible keys,
                # preserving the original closed form under causal visibility.
                q, dout = q.roll(1031, 2), dout.roll(1031, 2)
                inputs = tuple(t.requires_grad_() for t in (q, k, v))
                actual = self.run_attention(inputs, dout, True)
                self.assert_reference(actual, inputs, dout, True)

    def test_d64_column_preserves_signed_scales_and_negative_maxima(self):
        for dtype, causal, scale in itertools.product((torch.float16, torch.bfloat16), (False, True), (-0.25, 0.0, None)):
            with self.subTest(dtype=dtype, causal=causal, scale=scale):
                torch.manual_seed(1487)
                q, k, v, dout = [torch.randn(1, 16, 129, 64, device="cuda", dtype=dtype) for _ in range(4)]
                q.mul_(0.3)
                k.mul_(0.3)
                q[..., 0], k[..., 0] = 20, -20
                inputs = tuple(t.requires_grad_() for t in (q, k, v))
                self.assert_reference(
                    self.run_attention(inputs, dout, causal, scale), inputs, dout, causal, scale, norm_only=True
                )

    def test_d16_signed_scales_negative_maxima_and_tiny_gradients(self):
        for dtype, causal, case in itertools.product(
            (torch.float16, torch.bfloat16), (False, True), ("negative", "zero", "maxima", "sharp")
        ):
            with self.subTest(dtype=dtype, causal=causal, case=case):
                torch.manual_seed(1193)
                q, k, v, dout = [torch.randn(1, 16, 1025, 16, device="cuda", dtype=dtype) for _ in range(4)]
                scale = -0.25 if case == "negative" else 0.0 if case == "zero" else 16**-0.5
                if case == "maxima":
                    q.mul_(0.3)
                    k.mul_(0.3)
                    q[..., 0], k[..., 0] = 20, -20
                elif case == "sharp":
                    for tensor in (q, k, v, dout):
                        tensor.zero_()
                    q[..., 0], k[..., 0], k[:, :, 0, 0] = 10, 9 if dtype == torch.float16 else 5, 10
                    v[..., 0], v[:, :, 0, 0], dout[..., 0] = -1, 1, 1
                    scale = 1.0
                inputs = [tensor.requires_grad_() for tensor in (q, k, v)]
                actual = self.run_attention(inputs, dout, causal, scale)
                self.assert_reference(actual, inputs, dout, causal, scale, norm_only=case == "maxima")

    def test_short_backward_preserves_signed_scales_and_tiny_gradients(self):
        for dtype, dimension, causal, case in itertools.product(
            (torch.float16, torch.bfloat16), (128, 256), (False, True), ("negative", "zero", "maxima", "sharp")
        ):
            with self.subTest(dtype=dtype, dimension=dimension, causal=causal, case=case):
                torch.manual_seed(1207)
                q, k, v, dout = [torch.randn(1, 16, 128, dimension, device="cuda", dtype=dtype) for _ in range(4)]
                scale = -0.25 if case == "negative" else 0.0 if case == "zero" else dimension**-0.5
                if case == "maxima":
                    q.mul_(0.3)
                    k.mul_(0.3)
                    q[..., 0], k[..., 0] = 20, -20
                elif case == "sharp":
                    for tensor in (q, k, v, dout):
                        tensor.zero_()
                    q[..., 0], k[..., 0], k[:, :, 0, 0] = 10, 9 if dtype == torch.float16 else 5, 10
                    v[..., 0], v[:, :, 0, 0], dout[..., 0] = -1, 1, 1
                    scale = 1.0
                inputs = [tensor.requires_grad_() for tensor in (q, k, v)]
                actual = self.run_attention(inputs, dout, causal, scale)
                self.assert_reference(actual, inputs, dout, causal, scale, norm_only=case == "maxima")

    def test_key_gradients_preserve_common_and_invisible_query_rows(self):
        # Row zero is an irrelevant anchor for causal key columns after zero.
        # A scalar missing-mass correction times Q[0] creates large false dK.
        for dtype, causal in itertools.product((torch.float16, torch.bfloat16), (False, True)):
            with self.subTest(dtype=dtype, causal=causal):
                torch.manual_seed(1277)
                q, k, v, dout = [torch.randn(1, 16, 128, 128, device="cuda", dtype=dtype) for _ in range(4)]
                q[..., 0], k[..., 0] = 20, -20
                q[:, :, 0].zero_()
                q[:, :, 0, 0] = 32768
                inputs = [tensor.requires_grad_() for tensor in (q, k, v)]
                actual = self.run_attention(inputs, dout, causal)
                self.assert_reference(actual, inputs, dout, causal, norm_only=True)

    def test_unsupported_families_preserve_untuned_fallback(self):
        from kohakufa_cute import kernels, native, sm120_schedule

        for batch, heads, kvheads, dimension, masked in (
            (2, 16, 16, 16, False),
            (1, 16, 4, 128, False),
            (1, 16, 16, 256, True),
            (1, 16, 4, 64, False),
            (1, 16, 16, 64, True),
        ):
            with self.subTest(batch=batch, kvheads=kvheads, dimension=dimension, masked=masked):
                torch.manual_seed(1199)
                q = torch.randn(batch, heads, 129, dimension, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                k, v = [
                    torch.randn(batch, kvheads, 131, dimension, device="cuda", dtype=q.dtype, requires_grad=True)
                    for _ in range(2)
                ]
                dout = torch.randn_like(q)
                mask = torch.rand(batch, 1, 129, 131, device="cuda") > 0.2 if masked else None
                actual = self.run_attention((q, k, v), dout, False, mask=mask)
                native.extension().clear_cache()
                kernels._compile_kernel.cache_clear()
                with patch.object(sm120_schedule, "eligible", return_value=False):
                    expected = self.run_attention((q, k, v), dout, False, mask=mask)
                for got, want in zip(actual, expected):
                    self.assertTrue(torch.equal(got.view(torch.uint8), want.view(torch.uint8)))

    def test_compiled_calls_cross_short_and_long_boundaries(self):
        from kohakufa_cute import attention

        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        compiled = torch.compile(attention, fullgraph=True, dynamic=True)
        for dimension, sequence in ((128, 128), (128, 129), (16, 1023), (16, 1025), (128, 127)):
            with self.subTest(dimension=dimension, sequence=sequence):
                torch.manual_seed(1201)
                q, k, v, dout = [
                    torch.randn(1, 16, sequence, dimension, device="cuda", dtype=torch.bfloat16) for _ in range(4)
                ]
                inputs = [tensor.requires_grad_() for tensor in (q, k, v)]
                out = compiled(*inputs, causal=True)
                actual = (out, *torch.autograd.grad(out, inputs, dout))
                self.assert_reference(actual, inputs, dout, True)
