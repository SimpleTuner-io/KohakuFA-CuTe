import unittest
from unittest.mock import patch

import torch


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0),
    "Requires Hopper SM90",
)
class ClefHopperForwardTests(unittest.TestCase):
    def test_pipeline_matches_serial_statistics_outputs_and_gradients(self):
        from kohakufa_cute import attention, hopper_forward, kernels, native

        def serial(mode, causal, masked, *configuration, vector):
            return kernels.Attention(mode, causal, masked, 128, *configuration[1:], vector=vector)

        def evaluate(inputs, dout, scale):
            native.extension().clear_cache()
            kernels._compile_kernel.cache_clear()
            q, k, v = [tensor.detach().requires_grad_(True) for tensor in inputs]
            out = attention(q, k, v, causal=True, scale=scale)
            gradients = torch.autograd.grad(out, (q, k, v), dout)
            statistics = native.extension().forward(q, k, v, None, scale, True)[1:3]
            return out, *gradients, *statistics

        cases = ((1, 1, 257), (1, 63, 63), (1, 127, 127), (2, 129, 129), (1, 257, 257), (1, 129, 257), (1, 257, 129))
        try:
            for batch, queries, keys in cases:
                for scale in (0.0625, -0.125, 0.0):
                    with self.subTest(batch=batch, queries=queries, keys=keys, scale=scale):
                        torch.manual_seed(509)
                        q = torch.randn(batch, 24, queries, 256, device="cuda", dtype=torch.bfloat16)
                        k = torch.randn(batch, 4, keys, 256, device="cuda", dtype=torch.bfloat16)
                        v = torch.randn(batch, keys, 4, 256, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
                        dout = torch.randn_like(q)
                        with patch.object(hopper_forward, "HopperForwardSpecialized", serial):
                            expected = evaluate((q, k, v), dout, scale)
                        actual = evaluate((q, k, v), dout, scale)
                        # The denominator now reduces across lanes in the
                        # epilogue. Its FP32 reassociation can change BF16 ties.
                        for got, want in zip(actual[:4], expected[:4]):
                            torch.testing.assert_close(got, want, atol=0.02, rtol=0.02)
                        self.assertTrue(torch.equal(actual[4], expected[4]))
                        torch.testing.assert_close(actual[5], expected[5], atol=1e-5, rtol=1e-5)
        finally:
            native.extension().clear_cache()
            kernels._compile_kernel.cache_clear()

    def test_infinite_values_preserve_serial_output(self):
        from kohakufa_cute import attention, hopper_forward, kernels, native

        def serial(mode, causal, masked, *configuration, vector):
            return kernels.Attention(mode, causal, masked, 128, *configuration[1:], vector=vector)

        def evaluate(q, k, v):
            native.extension().clear_cache()
            kernels._compile_kernel.cache_clear()
            return attention(q, k, v, causal=True)

        try:
            for queries, keys in ((1, 1), (129, 3)):
                with self.subTest(queries=queries, keys=keys):
                    q = torch.zeros(1, 24, queries, 256, device="cuda", dtype=torch.bfloat16)
                    k = torch.zeros(1, 4, keys, 256, device="cuda", dtype=torch.bfloat16)
                    v = torch.zeros(1, keys, 4, 256, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
                    v[:, :, 0, 0] = float("inf")
                    v[:, :, 0, 1] = -float("inf")
                    v[:, :, 0, 2] = float("nan")
                    v[:, :, 0, 3] = 1
                    with patch.object(hopper_forward, "HopperForwardSpecialized", serial):
                        expected = evaluate(q, k, v)
                    actual = evaluate(q, k, v)
                    self.assertTrue(torch.isposinf(expected[..., 0]).all().item())
                    self.assertTrue(torch.isneginf(expected[..., 1]).all().item())
                    self.assertTrue(torch.isnan(expected[..., 2]).all().item())
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
        finally:
            native.extension().clear_cache()
            kernels._compile_kernel.cache_clear()

    def test_causal_gqa_matches_fp64(self):
        from benchmarks.reference import fp64_reference
        from kohakufa_cute import attention

        for scale in (0.0625, 0.125, -0.125, 0.0, 1e-30, -1e-30):
            with self.subTest(scale=scale):
                torch.manual_seed(510)
                q = torch.randn(1, 24, 137, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                k = torch.randn(1, 4, 193, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                v = torch.randn(1, 193, 4, 256, device="cuda", dtype=torch.bfloat16).transpose(1, 2).requires_grad_()
                dout = torch.randn_like(q)
                expected = fp64_reference((q, k, v), dout, causal=True, scale=scale)
                out = attention(q, k, v, causal=True, scale=scale)
                actual = out, *torch.autograd.grad(out, (q, k, v), dout)
                for got, want in zip(actual, expected):
                    torch.testing.assert_close(got.double(), want, atol=0.02, rtol=0.02)

    def test_causal_gqa_graph_reads_changed_inputs(self):
        from benchmarks.clef_forward import fp64_forward_chunks
        from kohakufa_cute import attention, kernels, native

        native.extension().clear_cache()
        kernels._compile_kernel.cache_clear()
        try:
            q = torch.randn(1, 24, 257, 256, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(1, 4, 269, 256, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(1, 269, 4, 256, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
            attention(q, k, v, causal=True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = attention(q, k, v, causal=True)
            for seed in (511, 512):
                torch.manual_seed(seed)
                for tensor in (q, k, v):
                    tensor.copy_(torch.randn_like(tensor))
                graph.replay()
                torch.cuda.synchronize()
                eager = attention(q, k, v, causal=True)
                self.assertTrue(torch.equal(actual, eager))
                expected = fp64_forward_chunks(q, k, v, causal=True, scale=0.0625)
                torch.testing.assert_close(actual.double(), expected, atol=0.02, rtol=0.02)

        finally:
            native.extension().clear_cache()
            kernels._compile_kernel.cache_clear()


if __name__ == "__main__":
    unittest.main()
