"""Graph benchmarks own fresh leaves rather than caller autograd history."""

import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class BenchmarkCaptureTests(unittest.TestCase):
    def test_external_history_and_inputs_are_independent_of_capture(self):
        from benchmarks.compare_ada import capture
        from kohakufa_cute import scaled_dot_product_attention

        for backend in (torch.nn.functional.scaled_dot_product_attention, scaled_dot_product_attention):
            for dtype in (torch.float16, torch.bfloat16):
                parents = [torch.randn(1, 2, 129, 64, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)]
                inputs = [tensor * 0.75 for tensor in parents]
                originals = [tensor.detach().clone() for tensor in inputs]
                dout = torch.randn_like(inputs[0])
                for causal in (False, True):
                    for phase in ("forward", "backward", "joint"):
                        with self.subTest(backend=backend.__module__, dtype=dtype, causal=causal, phase=phase):
                            graph, storage = capture(backend, inputs, dout, causal, phase, keep_graph=True)
                            operands, _, _, actual = storage
                            self.assertTrue(all(tensor.is_leaf and tensor.grad_fn is None for tensor in operands))
                            self.assertTrue(
                                all(tensor.data_ptr() != original.data_ptr() for tensor, original in zip(operands, inputs))
                            )
                            graph.replay()
                            torch.cuda.synchronize()
                            self.assertTrue(all(torch.isfinite(tensor).all().item() for tensor in actual))
                            for tensor, original in zip(inputs, originals):
                                torch.testing.assert_close(tensor.detach(), original, rtol=0, atol=0)
                            self.assertTrue(all(tensor.grad is None for tensor in parents))


if __name__ == "__main__":
    unittest.main()
