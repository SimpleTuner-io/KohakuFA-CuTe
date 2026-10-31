"""Qualify L40S schedules against the declared rounded attention equations."""

import itertools
import math
import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_name() == "NVIDIA L40S", "Requires a full L40S")
class L40SScheduleTests(unittest.TestCase):
    def setUp(self):
        from kohakufa_cute import kernels, native

        native.extension().clear_cache()
        kernels._compile_kernel.cache_clear()
        torch.manual_seed(1253)

    def compare(self, q, k, v, dout, *, causal=False, scale=None, mask=None, norm_only=False):
        from benchmarks.reference import metrics
        from kohakufa_cute import attention
        from tests.attention_reference import attention_reference

        inputs = [tensor.detach().requires_grad_(True) for tensor in (q, k, v)]
        out = attention(*inputs, causal=causal, scale=scale, mask=mask)
        actual = (out, *torch.autograd.grad(out, inputs, dout))
        expected = attention_reference((q, k, v), dout, causal=causal, scale=scale, mask=mask, output=out)
        tolerance = 0.02 if q.dtype == torch.bfloat16 else 0.003
        norm_limit = 0.01 if q.dtype == torch.bfloat16 else 0.005
        for component, got, want in zip(("out", "dq", "dk", "dv"), actual, expected):
            with self.subTest(component=component):
                error = metrics(got, want)
                self.assertTrue(error["finite"])
                if not norm_only:
                    torch.testing.assert_close(got.double(), want, atol=tolerance, rtol=tolerance)
                # Cancellation-sensitive fixtures retain the component norm gate;
                # single-rounded operands do not promise compensated accuracy
                # at every cancelled coordinate.
                if error["reference_norm"]:
                    self.assertLess(error["relative_error"], norm_limit)
                else:
                    self.assertEqual(error["max_absolute_error"], 0)
        return actual

    def inputs(self, queries, keys, dimension, dtype):
        q = torch.randn(1, 16, queries, dimension, device="cuda", dtype=dtype)
        k, v = [torch.randn(1, 16, keys, dimension, device="cuda", dtype=dtype) for _ in range(2)]
        return q, k, v, torch.randn_like(q)

    def test_short_lengths_preserve_both_runtime_axes(self):
        for dtype, dimension, causal in itertools.product((torch.bfloat16, torch.float16), (128, 256, 512), (False, True)):
            for queries, keys in ((64, 64), (128, 128), (129, 128), (128, 129), (128, 17)):
                with self.subTest(dtype=dtype, dimension=dimension, causal=causal, queries=queries, keys=keys):
                    self.compare(*self.inputs(queries, keys, dimension, dtype), causal=causal)

    def test_long_boundaries_and_partial_tiles(self):
        for dtype, dimension, causal in itertools.product((torch.bfloat16, torch.float16), (16, 64, 128), (False, True)):
            for queries, keys in ((1023, 1024), (1024, 1024), (1024, 1152), (1025, 1031), (1024, 1023)):
                with self.subTest(dtype=dtype, dimension=dimension, causal=causal, queries=queries, keys=keys):
                    self.compare(*self.inputs(queries, keys, dimension, dtype), causal=causal)

    def test_transposed_views_and_masked_fallbacks(self):
        for dimension, sequence, masked in (
            (16, 1025, False),
            (64, 1025, False),
            (128, 128, False),
            (128, 1025, False),
            (256, 4096, False),
            (256, 128, True),
        ):
            q, k, v, dout = [
                torch.randn(1, sequence, 16, dimension, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
                for _ in range(4)
            ]
            mask = None
            if masked:
                mask = (torch.arange(sequence, device="cuda") % 5 != 1).reshape(1, 1, 1, sequence)
            with self.subTest(dimension=dimension, sequence=sequence, masked=masked):
                self.compare(q, k, v, dout, causal=True, mask=mask)

    def test_signed_scales_and_tiny_gradients(self):
        for dtype, (dimension, sequence), causal, scale in itertools.product(
            (torch.bfloat16, torch.float16),
            ((16, 1025), (64, 1025), (128, 128), (128, 1025), (256, 128), (512, 128)),
            (False, True),
            (-0.25, 0.0),
        ):
            with self.subTest(dtype=dtype, dimension=dimension, sequence=sequence, causal=causal, scale=scale):
                self.compare(
                    *self.inputs(sequence, sequence, dimension, dtype), causal=causal, scale=scale, norm_only=scale < 0
                )
        for dtype, (dimension, sequence) in itertools.product(
            (torch.bfloat16, torch.float16), ((16, 1025), (64, 1025), (128, 128), (128, 1025))
        ):
            q = torch.zeros(1, 16, sequence, dimension, device="cuda", dtype=dtype)
            k, v, dout = [torch.zeros_like(q) for _ in range(3)]
            q[..., 0], k[..., 0], k[:, :, 0, 0] = 10, 9 if dtype == torch.float16 else 5, 10
            v[..., 0], v[:, :, 0, 0], dout[..., 0] = -1, 1, 1
            with self.subTest(dtype=dtype, dimension=dimension, tiny=True):
                self.compare(q, k, v, dout, scale=1.0)

    def test_short_rectangular_backward_uses_one_rounded_probability(self):
        for dtype in (torch.bfloat16, torch.float16):
            q = torch.zeros(1, 16, 128, 128, device="cuda", dtype=dtype)
            k = torch.zeros(1, 16, 17, 128, device="cuda", dtype=dtype)
            v, dout = torch.zeros_like(k), torch.zeros_like(q)
            q[:, :, :64, 0], q[:, :, 64:, 0], k[:, :, 0, 0] = 12, 4, 1
            dout[:, :, :64, 0], dout[:, :, 64:, 0] = 1, -1.875
            actual = self.compare(q, k, v, dout)
            a, b = math.exp(12 / math.sqrt(128)), math.exp(4 / math.sqrt(128))
            pa, pb = torch.tensor([a / (a + 16), b / (b + 16)], dtype=dtype).double().tolist()
            expected = 64 * (pa - 1.875 * pb)
            with self.subTest(dtype=dtype):
                torch.testing.assert_close(
                    actual[3][:, :, 0, 0].double(),
                    torch.full((1, 16), expected, device="cuda", dtype=torch.float64),
                    atol=0,
                    rtol=0.02,
                )

    def test_key_gradients_preserve_common_and_invisible_query_rows(self):
        for dtype, causal, outlier in itertools.product((torch.bfloat16, torch.float16), (False, True), (False, True)):
            torch.manual_seed(1207)
            q, k, v, dout = self.inputs(128, 128, 128, dtype)
            q.mul_(0.3)
            k.mul_(0.3)
            q[..., 0], k[..., 0] = 20, -20
            if outlier:
                q[:, :, 0].zero_()
                q[:, :, 0, 0] = 32768
            with self.subTest(dtype=dtype, causal=causal, outlier=outlier):
                self.compare(q, k, v, dout, causal=causal, norm_only=True)

    def test_long_dense_key_owner_boundaries_and_fallback_cache(self):
        from unittest.mock import patch

        from kohakufa_cute import kernels, native

        with patch.object(native, "compile_kernel", wraps=native.compile_kernel) as compiler:
            counts = []
            for queries, keys in ((4095, 4095), (4096, 4096), (4097, 4097), (4097, 4096), (4096, 4097), (4096, 4096)):
                with self.subTest(queries=queries, keys=keys):
                    self.compare(*self.inputs(queries, keys, 256, torch.bfloat16))
                    counts.append(compiler.call_count)
            self.assertEqual(counts, [2] * len(counts))
        for scale in (-0.125, 0.0):
            with self.subTest(scale=scale):
                self.compare(*self.inputs(4096, 4096, 256, torch.bfloat16), scale=scale)
        self.assertTrue(kernels._compile_kernel.cache_info().currsize)

    def test_masks_grouped_heads_and_feature_strides_preserve_fallbacks(self):
        from unittest.mock import patch

        from kohakufa_cute import kernels, l40s, native

        for case in ("mask", "grouped", "feature_stride"):
            dimension, queries, keys = 128, 129, 131
            physical = dimension * (2 if case == "feature_stride" else 1)
            q = torch.randn(1, 16, queries, physical, device="cuda", dtype=torch.bfloat16)
            k, v = [
                torch.randn(1, 4 if case == "grouped" else 16, keys, physical, device="cuda", dtype=q.dtype)
                for _ in range(2)
            ]
            if case == "feature_stride":
                q, k, v = [tensor[..., ::2] for tensor in (q, k, v)]
            dout = torch.randn_like(q)
            mask = (torch.arange(keys, device="cuda") % 5 != 1).reshape(1, 1, 1, keys) if case == "mask" else None
            with self.subTest(case=case):
                actual = self.compare(q, k, v, dout, causal=True, mask=mask)
                native.extension().clear_cache()
                kernels._compile_kernel.cache_clear()
                with patch.object(l40s, "eligible", return_value=False):
                    control = self.compare(q, k, v, dout, causal=True, mask=mask)
                for got, want in zip(actual, control):
                    self.assertTrue(torch.equal(got, want))

    def test_compiled_calls_cross_both_forward_runtime_thresholds(self):
        from benchmarks.reference import metrics
        from kohakufa_cute import attention
        from tests.attention_reference import attention_reference

        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        operation = torch.compile(attention, fullgraph=True, dynamic=True)
        for dimension, sequence in ((16, 1023), (16, 1025), (64, 1023), (64, 1025), (128, 128), (128, 129), (128, 1025)):
            q, k, v, dout = self.inputs(sequence, sequence, dimension, torch.bfloat16)
            q, k, v = [tensor.requires_grad_() for tensor in (q, k, v)]
            out = operation(q, k, v, causal=True)
            actual = (out, *torch.autograd.grad(out, (q, k, v), dout))
            expected = attention_reference((q, k, v), dout, causal=True)
            for got, want in zip(actual, expected):
                torch.testing.assert_close(got.double(), want, atol=0.02, rtol=0.02)
                error = metrics(got, want)
                self.assertLess(error["relative_error"], 0.01)

    def test_negative_maxima_on_partial_long_tiles(self):
        from kohakufa_cute import native

        for dtype, dimension, causal in itertools.product((torch.bfloat16, torch.float16), (16, 64, 128), (False, True)):
            q, k, v, dout = self.inputs(1025, 1031, dimension, dtype)
            q.mul_(0.3)
            k.mul_(0.3)
            q[..., 0], k[..., 0] = 20, -20
            scale = dimension**-0.5
            _, maximum, logsum, _ = native.extension().forward(q, k, v, None, scale, causal)
            self.assertTrue((maximum < 0).all())
            self.assertTrue((maximum * scale + logsum * math.log(2) < 0).all())
            with self.subTest(dtype=dtype, dimension=dimension, causal=causal):
                self.compare(q, k, v, dout, causal=causal, norm_only=True)

    def test_graph_replay_reads_changed_inputs_after_poisoning_outputs(self):
        from kohakufa_cute import attention

        for dimension, sequence, causal in ((16, 1025, False), (128, 128, True), (256, 128, False), (512, 128, True)):
            q, k, v, dout = self.inputs(sequence, sequence, dimension, torch.bfloat16)
            q, k, v = [tensor.requires_grad_(True) for tensor in (q, k, v)]

            def run():
                out = attention(q, k, v, causal=causal)
                return out, *torch.autograd.grad(out, (q, k, v), dout)

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    run()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    actual = run()
            torch.cuda.current_stream().wait_stream(stream)
            with torch.no_grad():
                for tensor in (q, k, v, dout):
                    tensor.copy_(torch.randn_like(tensor))
                for tensor in actual:
                    tensor.fill_(float("nan"))
            graph.replay()
            expected = self.compare(q, k, v, dout, causal=causal)
            for got, want in zip(actual, expected):
                torch.testing.assert_close(got, want, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
