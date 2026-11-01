import unittest
from dataclasses import asdict

import torch

from kohakufa_cute import fa4_sm80


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 0) and fa4_sm80.available(),
    "Requires the qualified A100 runtime",
)
class WarpInferenceTests(unittest.TestCase):
    def runner(self, dimension, sequence, causal):
        from kohakufa_cute.fa4_sm80._schedules import a100_schedule
        from kohakufa_cute.native import extension

        return extension().AttentionRunner("warp", asdict(a100_schedule(dimension, sequence, causal)))

    def test_gradient_and_layout_contracts(self):
        from benchmarks.warp_inference import assess_output, references

        runner = self.runner(64, 129, False)
        inputs = [torch.randn(1, 2, 129, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        for position in range(3):
            operands = list(inputs)
            operands[position] = operands[position].detach().requires_grad_(True)
            with self.assertRaisesRegex(ValueError, "requested input gradients"):
                runner.inference(*operands)
            saved = []
            with (
                torch.no_grad(),
                torch.autograd.graph.saved_tensors_hooks(lambda tensor: saved.append(tensor) or tensor, lambda t: t),
            ):
                output = runner.inference(*operands)
            self.assertFalse(output.requires_grad)
            self.assertIsNone(output.grad_fn)
            self.assertIsNone(output._base)
            self.assertEqual(saved, [])
            assess_output(output, references(inputs))
        for position in range(3):
            with torch.autograd.forward_ad.dual_level():
                operands = list(inputs)
                operands[position] = torch.autograd.forward_ad.make_dual(inputs[position], torch.ones_like(inputs[position]))
                with torch.no_grad(), self.assertRaisesRegex(ValueError, "forward-AD"):
                    runner.inference(*operands)
        for scale in (0, -0.1, 1e-50, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                runner.inference(*inputs, scale=scale)
        noncontiguous = torch.randn(1, 2, 129, 128, device="cuda", dtype=torch.bfloat16)[..., ::2]
        self.assertFalse(noncontiguous.is_contiguous())
        with self.assertRaises(ValueError):
            runner.inference(noncontiguous, inputs[1], inputs[2])
        unaligned = torch.empty(inputs[0].numel() + 1, device="cuda", dtype=inputs[0].dtype)[1:].view_as(inputs[0])
        unaligned.copy_(inputs[0])
        with self.assertRaises(ValueError):
            runner.inference(unaligned, inputs[1], inputs[2])
        negative_view = torch.ops.aten._neg_view.default(inputs[0])
        self.assertTrue(negative_view.is_contiguous())
        with self.assertRaises(ValueError):
            runner.inference(negative_view, inputs[1], inputs[2])
        with torch.inference_mode():
            output = runner.inference(*inputs)
        assess_output(output, references(inputs))
        for sequence in (1, 7, 63, 65):
            small = [torch.randn(1, 2, sequence, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
            assess_output(runner.inference(*small, causal=True), references(small, True))

    def test_output_and_training_cache_alternation(self):
        from benchmarks.validate_sm80_static import assess, statistics
        from benchmarks.warp_inference import assess_output, references

        for dtype in (torch.bfloat16, torch.float16):
            for dimension in (64, 128):
                for sequence in (128, 129, 257):
                    for causal in (False, True):
                        with self.subTest(dtype=dtype, dimension=dimension, sequence=sequence, causal=causal):
                            torch.manual_seed(1013)
                            inputs = [torch.randn(1, 2, sequence, dimension, device="cuda", dtype=dtype) for _ in range(3)]
                            dout = torch.randn_like(inputs[0])
                            runner = self.runner(dimension, sequence, causal)
                            for scale in (0.17, 0.41, 0.17):
                                expected = references(inputs, causal, scale)
                                with torch.no_grad():
                                    inferred = runner.inference(*inputs, scale=scale, causal=causal)
                                assess_output(inferred, expected)
                                operands = [tensor.detach().requires_grad_(True) for tensor in inputs]
                                saved = []
                                with torch.autograd.graph.saved_tensors_hooks(
                                    lambda tensor: saved.append(tensor) or tensor, lambda tensor: tensor
                                ):
                                    trained = runner(*operands, scale=scale, causal=causal)
                                self.assertEqual(len(saved), 6)
                                statistics(operands, saved[-2], saved[-1], causal, scale)
                                actual = trained, *torch.autograd.grad(trained, operands, dout)
                                assess(actual, operands, dout, causal=causal, scale=scale, directed=False, pointwise=False)
                                assess_output(runner.inference(*inputs, scale=scale, causal=causal), expected)
                            if sequence == 129:
                                with torch.no_grad():
                                    inputs[0].mul_(0.3)
                                    inputs[1].mul_(0.3)
                                    inputs[0][..., 0] = 20
                                    inputs[1][..., 0] = -20
                                expected = references(inputs, causal, 0.17)
                                assess_output(runner.inference(*inputs, scale=0.17, causal=causal), expected)

    def test_poisoned_graphs_and_current_stream(self):
        from benchmarks.compare_ada import capture
        from benchmarks.compare_l40s import graph_kernels
        from benchmarks.validate_sm80_static import assess
        from benchmarks.warp_inference import assess_output, references

        for dtype, dimension, causal in ((torch.bfloat16, 64, True), (torch.float16, 128, False)):
            torch.manual_seed(1019)
            inputs = [torch.randn(1, 2, 129, dimension, device="cuda", dtype=dtype) for _ in range(3)]
            dout = torch.randn_like(inputs[0])
            runner = self.runner(dimension, 129, causal)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.no_grad():
                for _ in range(3):
                    runner.inference(*inputs, causal=causal)
                inference = torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(inference, stream=stream):
                    output = runner.inference(*inputs, causal=causal)
            torch.cuda.current_stream().wait_stream(stream)
            self.assertEqual(len(graph_kernels(inference)), 1)

            def training(q, k, v, *, is_causal=False):
                return runner(q, k, v, causal=is_causal)

            trained, storage = capture(
                training, [tensor.detach() for tensor in inputs], dout, causal, "joint", keep_graph=True
            )
            self.assertEqual(len(graph_kernels(trained)), 4)
            operands, _, _, actual = storage
            for _ in range(2):
                with torch.no_grad():
                    for source, destination in zip(inputs, operands):
                        fresh = torch.randn_like(source)
                        source.copy_(fresh)
                        destination.copy_(fresh)
                    dout.copy_(torch.randn_like(dout))
                    output.fill_(float("nan"))
                    for tensor in actual:
                        tensor.fill_(float("nan"))
                inference.replay()
                trained.replay()
                torch.cuda.synchronize()
                assess_output(output, references(inputs, causal))
                assess(actual, operands, dout, causal=causal, scale=None, directed=False, pointwise=False)


if __name__ == "__main__":
    unittest.main()
