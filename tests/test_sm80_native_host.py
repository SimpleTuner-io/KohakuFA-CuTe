import unittest

import torch

from kohakufa_cute import fa4_sm80


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 0) and fa4_sm80.available(),
    "Requires the qualified optional SM80 runtime",
)
class Sm80NativeHostTests(unittest.TestCase):
    def check_case(self, inputs, dout, *, causal=False, scale=None, directed=False, pointwise=False):
        from benchmarks.sm80_native_host import attention
        from benchmarks.validate_sm80_static import assess, statistics

        saved = []

        def pack(tensor):
            saved.append(tensor)
            return tensor

        operands = [t.detach().requires_grad_(True) for t in inputs]
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            out = attention(*operands, is_causal=causal, scale=scale)
        self.assertEqual(len(saved), 6)
        actual_scale = inputs[0].shape[-1] ** -0.5 if scale is None else scale
        statistics(inputs, saved[-2], saved[-1], causal, actual_scale)
        actual = out, *torch.autograd.grad(out, operands, dout)
        assess(actual, inputs, dout, causal=causal, scale=scale, directed=directed, pointwise=pointwise)
        return actual

    def test_tails_saved_statistics_and_cache_reuse(self):
        for dtype in (torch.bfloat16, torch.float16):
            for dimension in (64, 128):
                for sequence in (129, 257):
                    for causal in (False, True):
                        with self.subTest(dtype=dtype, dimension=dimension, sequence=sequence, causal=causal):
                            torch.manual_seed(221)
                            q, k, v, dout = [
                                torch.randn(1, 2, sequence, dimension, device="cuda", dtype=dtype) for _ in range(4)
                            ]
                            self.check_case((q, k, v), dout, causal=causal)
                            q.mul_(0.3)
                            k.mul_(0.3)
                            q[..., 0], k[..., 0] = 20, -20
                            self.check_case((q, k, v), dout, causal=causal, directed=True)
                            self.check_case((q, k, v), dout, causal=causal, scale=0.125, directed=True)

    def test_rounded_cancellation_and_underflow(self):
        from tests.test_value_precision import square_value_cancellation, value_residual_underflow

        for dimension in (64, 128):
            for dtype in (torch.bfloat16, torch.float16):
                inputs, dout, scale, _ = square_value_cancellation(dimension, dtype, "cuda", length=1024, heads=2)
                self.check_case(inputs, dout, scale=scale, directed=True, pointwise=True)
            inputs, dout, scale, _ = value_residual_underflow(dimension, "cuda", length=1024, heads=2)
            self.check_case(inputs, dout, scale=scale, directed=True, pointwise=True)

    def test_changed_poisoned_graph(self):
        from benchmarks.compare_ada import capture
        from benchmarks.compare_l40s import graph_kernels
        from benchmarks.sm80_native_host import attention
        from benchmarks.validate_sm80_static import assess

        for dtype, causal in ((torch.bfloat16, True), (torch.float16, False)):
            inputs = [torch.randn(1, 2, 129, 128, device="cuda", dtype=dtype) for _ in range(3)]
            dout = torch.randn_like(inputs[0])
            graph, storage = capture(attention, inputs, dout, causal, "joint", keep_graph=True)
            symbols = [record["symbol"] for record in graph_kernels(graph)]
            self.assertEqual(len(symbols), 4)
            self.assertTrue(any("fa4_sm80" in symbol and "Forward" in symbol for symbol in symbols))
            self.assertTrue(any("fa4_sm80" in symbol and "BackwardSm80" in symbol for symbol in symbols))
            operands, _, _, actual = storage
            for seed in (249, 251):
                torch.manual_seed(seed)
                with torch.no_grad():
                    for tensor in (*operands, dout):
                        tensor.copy_(torch.randn_like(tensor))
                    for tensor in actual:
                        tensor.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                assess(actual, operands, dout, causal=causal, scale=None, directed=False, pointwise=False)

    def test_expanded_dout_and_rejected_inputs(self):
        from benchmarks.reference import fp64_reference, metrics
        from benchmarks.sm80_native_host import attention

        for dtype in (torch.bfloat16, torch.float16):
            inputs = [torch.randn(1, 2, 257, 128, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)]
            out = attention(*inputs, is_causal=True)
            actual = out, *torch.autograd.grad(out.sum(), inputs)
            expected = fp64_reference(inputs, torch.ones_like(out), causal=True)
            for value, reference in zip(actual, expected):
                self.assertLess(metrics(value, reference)["relative_error"], 0.01 if dtype == torch.bfloat16 else 0.005)
            for scale in (0, -0.1, float("inf"), float("nan")):
                with self.assertRaises(ValueError):
                    attention(*inputs, scale=scale)
            with self.assertRaises(ValueError):
                attention(*(t[..., ::2] for t in inputs))
            offset = torch.empty(inputs[0].numel() + 1, device="cuda", dtype=dtype)[1:].view_as(inputs[0])
            offset.copy_(torch.randn_like(offset))
            with self.assertRaises(ValueError):
                attention(offset, inputs[1], inputs[2])
            self.check_case(inputs, offset, causal=True)


if __name__ == "__main__":
    unittest.main()
