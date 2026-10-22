import unittest
from unittest.mock import patch

import torch


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 9),
    "Requires Ada SM89",
)
class ClefForwardOverlapTests(unittest.TestCase):
    def test_causal_gqa_tails_strides_statistics_and_gradients_match_serial(self):
        from kohakufa_cute import attention, kernels, native

        original = kernels.Attention.__init__

        def serial(program, *args, **kwargs):
            kwargs["forward_overlap"] = False
            original(program, *args, **kwargs)

        selected = []

        def observe(program, *args, **kwargs):
            selected.append(kwargs.get("forward_overlap", False))
            original(program, *args, **kwargs)

        def evaluate(inputs, dout, scale):
            native.extension().clear_cache()
            kernels._compile_kernel.cache_clear()
            q, k, v = [t.detach().requires_grad_(True) for t in inputs]
            out = attention(q, k, v, causal=True, scale=scale)
            gradients = torch.autograd.grad(out, (q, k, v), dout)
            statistics = native.extension().forward(q, k, v, None, scale, True)[1:3]
            return out, *gradients, *statistics

        cases = ((1, 127, 127), (1, 128, 128), (2, 129, 129), (1, 257, 257), (1, 129, 257), (1, 257, 129))
        try:
            for batch, queries, keys in cases:
                for scale in (0.0625, -0.125, 0.0):
                    with self.subTest(batch=batch, queries=queries, keys=keys, scale=scale):
                        torch.manual_seed(419)
                        q = torch.randn(batch, 16, queries, 256, device="cuda", dtype=torch.bfloat16)
                        k = torch.randn(batch, 4, keys, 256, device="cuda", dtype=torch.bfloat16)
                        v = torch.randn(batch, keys, 4, 256, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
                        dout = torch.randn_like(q)
                        with patch.object(kernels.Attention, "__init__", serial):
                            expected = evaluate((q, k, v), dout, scale)
                        selected.clear()
                        with patch.object(kernels.Attention, "__init__", observe):
                            actual = evaluate((q, k, v), dout, scale)
                        self.assertIn(True, selected)
                        for got, want in zip(actual, expected):
                            self.assertTrue(torch.equal(got, want))
        finally:
            native.extension().clear_cache()
            kernels._compile_kernel.cache_clear()

    def test_causal_gqa_matches_fp64(self):
        from benchmarks.reference import fp64_reference
        from kohakufa_cute import attention

        torch.manual_seed(422)
        q = torch.randn(1, 16, 137, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(1, 4, 193, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(1, 193, 4, 256, device="cuda", dtype=torch.bfloat16).transpose(1, 2).requires_grad_()
        dout = torch.randn_like(q)
        expected = fp64_reference((q, k, v), dout, causal=True, scale=-0.125)
        out = attention(q, k, v, causal=True, scale=-0.125)
        actual = out, *torch.autograd.grad(out, (q, k, v), dout)
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got.double(), want, atol=0.02, rtol=0.02)

    def test_causal_gqa_graph_reads_changed_inputs(self):
        from kohakufa_cute import attention, kernels, native

        original = kernels.Attention.__init__

        def serial(program, *args, **kwargs):
            kwargs["forward_overlap"] = False
            original(program, *args, **kwargs)

        native.extension().clear_cache()
        kernels._compile_kernel.cache_clear()
        try:
            q = torch.randn(1, 16, 257, 256, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(1, 4, 269, 256, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(1, 269, 4, 256, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
            attention(q, k, v, causal=True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = attention(q, k, v, causal=True)
            for seed in (420, 421):
                torch.manual_seed(seed)
                for tensor in (q, k, v):
                    tensor.copy_(torch.randn_like(tensor))
                graph.replay()
                torch.cuda.synchronize()
                with patch.object(kernels.Attention, "__init__", serial):
                    native.extension().clear_cache()
                    kernels._compile_kernel.cache_clear()
                    expected = attention(q, k, v, causal=True)
                self.assertTrue(torch.equal(actual, expected))
        finally:
            native.extension().clear_cache()
            kernels._compile_kernel.cache_clear()


if __name__ == "__main__":
    unittest.main()
