"""Verify that adversarial fixtures exercise the claimed numerical conditions."""

import unittest

import torch

from benchmarks.adversarial import cases, logits_statistics, make_case
from benchmarks.reference import fp64_reference, metrics


class AdversarialFixtureTests(unittest.TestCase):
    def test_negative_statistics_and_tiny_reported_gradient(self):
        for case in cases(extended=False):
            with self.subTest(case=case.name):
                inputs, dout, mask = make_case(case, device="cpu", dtype=torch.bfloat16)
                stats = logits_statistics(inputs, mask, scale=case.scale)
                if case.name != "zero_shift_control":
                    self.assertLess(stats["lse_min"], -88.7)
                else:
                    self.assertGreater(stats["lse_min"], 0)
                expected = fp64_reference(inputs, dout, mask, scale=case.scale)
                self.assertTrue(all(torch.isfinite(value).all() for value in expected))
                if case.literal:
                    norm = metrics(expected[1], expected[1])["reference_norm"]
                    self.assertGreater(norm, 1e-12)
                    self.assertLess(norm, 1e-7)

    def test_constant_logit_shift_preserves_forward_and_gradients(self):
        # Inputs with an exactly representable, constant key component isolate
        # softmax shift invariance from input-rounding and cancellation effects.
        q = torch.tensor([[[[8.0, 0.25], [8.0, -0.5]]]], dtype=torch.float64)
        k = torch.tensor([[[[0.0, 1.0], [0.0, -1.0], [0.0, 0.5]]]], dtype=torch.float64)
        v = torch.tensor([[[[1.0, 0.0], [-1.0, 0.5], [0.0, -0.25]]]], dtype=torch.float64)
        dout = torch.ones_like(q)
        expected = fp64_reference((q, k, v), dout, scale=0.125)
        shifted = k.clone()
        shifted[..., 0] = -128
        for actual, want in zip(fp64_reference((q, shifted, v), dout, scale=0.125), expected):
            torch.testing.assert_close(actual, want, atol=2e-13, rtol=2e-13)

    def test_comparator_rejects_lost_tiny_gradients_and_lost_tail_rows(self):
        from benchmarks.cudnn_regressions import assess

        reference = torch.full((1, 1, 2, 8), 1e-22, dtype=torch.float64)
        actual = torch.zeros_like(reference)
        result = assess((actual,) * 4, (reference,) * 4, torch.bfloat16)
        self.assertEqual(result["status"], "numerical_failure")
        self.assertEqual(result["tensors"]["dq"]["relative_error"], 1)
        reference = torch.ones(1, 1, 10001, 1, dtype=torch.float64)
        actual = reference.clone()
        actual[..., -1, :] = 0
        result = assess((actual,) * 4, (reference,) * 4, torch.bfloat16)
        self.assertLess(result["tensors"]["dq"]["relative_error"], 0.05)
        self.assertEqual(result["status"], "numerical_failure")
        self.assertFalse(result["tensors"]["dq"]["edge_rows"]["last"]["passed"])


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.cuda.get_device_capability() in {(8, 6), (8, 9), (9, 0), (10, 0), (10, 3), (12, 0), (12, 1)},
    "Requires supported CuTe CUDA kernels",
)
class AdversarialKernelTests(unittest.TestCase):
    def assert_reference(self, actual, expected, dtype, case):
        for name, got, want in zip(("out", "dq", "dk", "dv"), actual, expected):
            for region in (slice(None), slice(0, 1), slice(-1, None)):
                value, reference = got[..., region, :], want[..., region, :]
                result = metrics(value, reference)
                self.assertTrue(result["finite"], (case, name, region, result))
                if result["reference_norm"]:
                    self.assertLess(
                        result["relative_error"], 0.05 if dtype == torch.bfloat16 else 0.015, (case, name, region, result)
                    )
                else:
                    self.assertEqual(value.count_nonzero().item(), 0)

    def test_reported_cudnn_inputs_and_negative_lse_tails(self):
        from kohakufa_cute import attention

        for dtype in (torch.bfloat16, torch.float16):
            for case in cases(extended=False):
                with self.subTest(dtype=dtype, case=case.name):
                    inputs, dout, mask = make_case(case, device="cuda", dtype=dtype)
                    expected = fp64_reference(inputs, dout, mask, scale=case.scale)
                    variables = [value.detach().requires_grad_() for value in inputs]
                    output = attention(*variables, mask=mask, scale=case.scale)
                    actual = output, *torch.autograd.grad(output, variables, dout)
                    self.assert_reference(actual, expected, dtype, case.name)

    def test_partial_tiles_preserve_negative_unscaled_maxima(self):
        import math

        from kohakufa_cute import native

        selected = {"issue1072", "kv_tail_64", "kv_tail_192"}
        for dtype in (torch.bfloat16, torch.float16):
            for case in (case for case in cases(extended=False) if case.name in selected):
                with self.subTest(dtype=dtype, case=case.name):
                    inputs, _, _ = make_case(case, device="cuda", dtype=dtype)
                    scale = case.dim**-0.5
                    _, maximum, logsum, _ = native.extension().forward(*inputs, None, scale, False)
                    scores = inputs[0].double() @ inputs[1].double().transpose(-1, -2)
                    self.assertLess(maximum.max().item(), 0)
                    self.assertTrue(torch.isfinite(logsum).all())
                    torch.testing.assert_close(maximum.double(), scores.amax(-1), atol=1e-3, rtol=1e-6)
                    reconstructed = maximum.double() * scale + logsum.double() * math.log(2)
                    torch.testing.assert_close(reconstructed, (scores * scale).logsumexp(-1), atol=1e-3, rtol=1e-6)

    def test_compiled_and_graph_replay_read_changed_negative_lse_inputs(self):
        from kohakufa_cute import attention

        dtype = torch.bfloat16
        for case in (case for case in cases(extended=False) if case.name in {"issue1072", "kv_tail_192"}):
            with self.subTest(case=case.name):
                inputs, dout, _ = make_case(case, device="cuda", dtype=dtype)
                variables = [value.detach().requires_grad_() for value in inputs]

                def run(operation=attention):
                    output = operation(*variables)
                    return output, *torch.autograd.grad(output, variables, dout)

                compiled = torch.compile(attention, fullgraph=True)
                expected = fp64_reference(variables, dout)
                self.assert_reference(run(compiled), expected, dtype, case.name)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        run()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    actual = run()
                for seed in (124, 125):
                    changed, changed_dout, _ = make_case(case, device="cuda", dtype=dtype, seed=seed)
                    with torch.no_grad():
                        for target, source in zip((*variables, dout), (*changed, changed_dout)):
                            target.copy_(source)
                    graph.replay()
                    torch.cuda.synchronize()
                    self.assert_reference(actual, fp64_reference(variables, dout), dtype, case.name)


if __name__ == "__main__":
    unittest.main()
