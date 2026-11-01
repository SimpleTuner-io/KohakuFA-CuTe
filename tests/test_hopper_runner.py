"""GPU guards, changed inputs and captured descriptors for Hopper native host."""

import unittest

import torch

from tests.hopper_runtime import full_h200_runtime


@unittest.skipUnless(full_h200_runtime(), "Requires full H200 and the pinned optional runtime")
class HopperRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from benchmarks.configured_hopper import make_attention

        cls.operation = staticmethod(make_attention())

    def operands(self, length=128, dimension=64, heads=2):
        return [
            torch.randn(1, heads, length, dimension, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            for _ in range(3)
        ]

    def test_runtime_scale_and_layout_guards(self):
        inputs = self.operands()
        for scale in (0.0, -0.25, float("nan"), float("inf"), 1e-100):
            with self.subTest(scale=scale), self.assertRaises((RuntimeError, ValueError)):
                self.operation(*inputs, scale=scale)
        with self.assertRaises((RuntimeError, ValueError)):
            self.operation(*(x[..., ::2] for x in self.operands(dimension=128)))

    def test_changed_graph_reads_fresh_inputs_and_preserves_results(self):
        inputs = self.operands()
        dout = torch.randn_like(inputs[0])
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())

        def run():
            output = self.operation(*inputs, is_causal=True)
            return (output, *torch.autograd.grad(output, inputs, dout))

        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                actual = run()
        torch.cuda.current_stream().wait_stream(stream)
        for generation in range(2):
            with torch.no_grad():
                if generation:
                    for item in inputs:
                        item.copy_(torch.randn_like(item) * 0.3)
                    dout.copy_(torch.randn_like(dout))
                for item in actual:
                    item.fill_(torch.nan)
            graph.replay()
            torch.cuda.synchronize()
            expected = run()
            for got, want in zip(actual, expected):
                self.assertTrue(torch.isfinite(got).all())
                torch.testing.assert_close(got, want, atol=0.02, rtol=0.02)

    def test_native_saved_state_survives_saved_tensor_hooks_and_reuse(self):
        inputs = self.operands()
        packed = []

        def pack(tensor):
            packed.append(tensor)
            return tensor.detach().clone()

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            output = self.operation(*inputs, is_causal=True)
        gradients = torch.autograd.grad(output, inputs, torch.randn_like(output))
        self.assertTrue(packed)
        self.assertTrue(all(torch.isfinite(item).all() for item in gradients))
        again = self.operation(*self.operands(length=256), is_causal=False)
        self.assertTrue(torch.isfinite(again).all())


if __name__ == "__main__":
    unittest.main()
