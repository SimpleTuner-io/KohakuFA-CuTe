import unittest
from unittest.mock import Mock, patch

import torch

from kohakufa_cute import api, fa4_sm80
from kohakufa_cute.fa4_sm80 import rtx4090
from tests.test_rtx4090_shared import operand


class Rtx4090ShortTests(unittest.TestCase):
    def setUp(self):
        self.patches = [
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(rtx4090, "_short_device", return_value=True),
            patch.object(fa4_sm80, "available", return_value=True),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def tensor(self, **kwargs):
        value = operand((1, 16, 128, 16), **kwargs)
        value.requires_grad = True
        return value

    def test_training_and_live_gradient_boundaries(self):
        q = self.tensor()
        for causal in (False, True):
            self.assertTrue(rtx4090._short_compatible(q, q, q, None, 0, causal, None, False))
        q.requires_grad = False
        self.assertFalse(rtx4090._short_compatible(q, q, q, None, 0, False, None, False))
        k = self.tensor()
        self.assertTrue(rtx4090._short_compatible(q, k, q, None, 0, False, None, False))
        with torch.no_grad():
            self.assertFalse(rtx4090._short_compatible(q, k, q, None, 0, False, None, False))

    def test_metadata_and_policy_fallbacks(self):
        q = self.tensor()
        for shape in ((1, 16, 127, 16), (1, 16, 129, 16), (1, 16, 128, 64), (1, 8, 128, 16), (2, 16, 128, 16)):
            bad = self.tensor()
            bad.shape = shape
            self.assertFalse(rtx4090._short_compatible(bad, bad, bad, None, 0, True, None, False))
        for index in range(3):
            for bad in (self.tensor(pointer=4098), self.tensor(contiguous=False), self.tensor(dtype=torch.float32)):
                values = [q, q, q]
                values[index] = bad
                self.assertFalse(rtx4090._short_compatible(*values, None, 0, True, None, False))
        for scale in (0, -0.25, 1e-40, float("inf"), float("nan")):
            self.assertFalse(rtx4090._short_compatible(q, q, q, None, 0, True, scale, False))
        for mask, dropout, gqa in ((object(), 0, False), (None, 0.1, False), (None, 0, True)):
            self.assertFalse(rtx4090._short_compatible(q, q, q, mask, dropout, True, None, gqa))
        for target, attribute in ((torch.compiler, "is_compiling"), (torch, "are_deterministic_algorithms_enabled")):
            with patch.object(target, attribute, return_value=True):
                self.assertFalse(rtx4090._short_compatible(q, q, q, None, 0, True, None, False))
        with patch.object(fa4_sm80, "available", return_value=False):
            self.assertFalse(rtx4090._short_compatible(q, q, q, None, 0, True, None, False))

    def test_public_selection_error_and_no_grad_fallback(self):
        q = self.tensor()
        runner = Mock(return_value="short")
        with patch.object(rtx4090, "_short_runner", return_value=runner):
            self.assertEqual(rtx4090.scaled_dot_product_attention(q, q, q, is_causal=True), "short")
        runner.assert_called_once_with(q, q, q, scale=None, causal=True)
        with patch.object(rtx4090, "_short_runner", side_effect=RuntimeError("short failed")):
            with self.assertRaisesRegex(RuntimeError, "short failed"):
                rtx4090.scaled_dot_product_attention(q, q, q)
        with torch.no_grad(), patch.object(rtx4090, "compatible", return_value=False):
            with patch.object(api, "scaled_dot_product_attention", return_value="fallback") as fallback:
                self.assertEqual(rtx4090.scaled_dot_product_attention(q, q, q, is_causal=True), "fallback")
            fallback.assert_called_once_with(q, q, q, None, 0.0, True, scale=None, enable_gqa=False)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class Rtx4090ShortGpuTests(unittest.TestCase):
    def test_actual_gradient_metadata_and_no_grad_fallback(self):
        if torch.cuda.get_device_name() != "NVIDIA GeForce RTX 4090" or not fa4_sm80.available():
            self.skipTest("Qualified full RTX 4090 runtime required")
        from benchmarks.reference import fp64_reference, metrics

        torch.manual_seed(71053)
        for dtype in (torch.bfloat16, torch.float16):
            for causal in (False, True):
                for selected in range(3):
                    values = [torch.randn(1, 16, 128, 16, device="cuda", dtype=dtype) for _ in range(3)]
                    values[selected] = torch.nn.Parameter(values[selected])
                    dout = torch.randn_like(values[0])
                    with patch.object(rtx4090, "_short_runner", wraps=rtx4090._short_runner) as selected_factory:
                        out = rtx4090.scaled_dot_product_attention(*values, is_causal=causal)
                    selected_factory.assert_called_once()
                    grad = torch.autograd.grad(out, values[selected], dout)[0]
                    expected = fp64_reference(tuple(t.cpu() for t in values), dout.cpu(), causal=causal)
                    for got, want in ((out.cpu(), expected[0]), (grad.cpu(), expected[selected + 1])):
                        for head in range(16):
                            error = metrics(got[:, head : head + 1], want[:, head : head + 1])
                            self.assertTrue(error["finite"])
                            self.assertLess(error["relative_error"], 0.01 if dtype == torch.bfloat16 else 0.005)
                    with (
                        torch.no_grad(),
                        patch.object(rtx4090, "_short_runner", side_effect=AssertionError("short selected")),
                    ):
                        result = rtx4090.scaled_dot_product_attention(*values, is_causal=causal)
                    self.assertTrue(torch.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
