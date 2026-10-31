"""The default arithmetic includes operand rounding, not residual recovery."""

import unittest

import torch

from benchmarks.reference import fp64_reference, rounded_reference
from tests.test_value_precision import value_residual_underflow


class BaselinePrecisionTests(unittest.TestCase):
    def test_single_key_reference_has_exact_zero_score_gradients(self):
        from tests.attention_reference import attention_reference

        torch.manual_seed(257)
        q = torch.randn(1, 4, 1, 128, dtype=torch.bfloat16)
        k, v = [torch.randn_like(q) for _ in range(2)]
        reference = attention_reference((q, k, v), torch.randn_like(q))
        self.assertEqual(reference[1].count_nonzero().item(), 0)
        self.assertEqual(reference[2].count_nonzero().item(), 0)

    def test_fp16_probability_underflow_is_part_of_the_operand_contract(self):
        inputs, dout, scale, _ = value_residual_underflow(64, "cpu", length=1024, heads=1)
        exact = fp64_reference(inputs, dout, scale=scale)
        rounded = rounded_reference(inputs, dout, scale=scale)
        self.assertGreater(exact[3][0, 0, 0, 0].item(), 0.5)
        self.assertEqual(rounded[3][0, 0, 0, 0].item(), 0)

    def test_sharp_attention_rounds_p_before_forming_ds(self):
        q = torch.tensor([[[[10.0]]]], dtype=torch.bfloat16)
        k = torch.tensor([[[[10.0], [5.0]]]], dtype=q.dtype)
        v = torch.tensor([[[[1.0], [-1.0]]]], dtype=q.dtype)
        expected = rounded_reference((q, k, v), torch.ones_like(q), scale=1)
        self.assertGreater(expected[2][0, 0, 1, 0].abs().item(), 0)
        self.assertEqual(expected[2][0, 0, 0, 0].item(), 0)
        self.assertEqual(expected[3][0, 0, 0, 0].item(), 1)

    def test_hidden_rows_and_zero_scale_stay_zero(self):
        torch.manual_seed(17)
        q = torch.randn(1, 4, 5, 8, dtype=torch.bfloat16)
        k, v = [torch.randn(1, 2, 7, 8, dtype=q.dtype) for _ in range(2)]
        mask = torch.ones(5, 7, dtype=torch.bool)
        mask[0] = False
        output = rounded_reference((q, k, v), torch.ones_like(q), mask, scale=0)
        self.assertTrue(all(torch.isfinite(t).all() for t in output))
        self.assertEqual(output[0][:, :, 0].count_nonzero().item(), 0)
        self.assertEqual(output[1].count_nonzero().item(), 0)
        self.assertEqual(output[2].count_nonzero().item(), 0)

    def test_reference_restores_the_callers_matmul_precision(self):
        previous = torch.backends.cuda.matmul.fp32_precision
        try:
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            q = torch.ones(1, 1, 2, 8, dtype=torch.bfloat16)
            rounded_reference((q, q, q), q)
            self.assertEqual(torch.backends.cuda.matmul.fp32_precision, "tf32")
        finally:
            torch.backends.cuda.matmul.fp32_precision = previous


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.cuda.get_device_capability() in {(8, 6), (8, 9), (9, 0), (10, 0), (10, 3), (12, 0), (12, 1)},
    "Requires supported CuTe CUDA kernels",
)
class ModelShapePrecisionTests(unittest.TestCase):
    def test_model_head_counts_match_declared_cancellation_arithmetic(self):
        from benchmarks.reference import metrics
        from kohakufa_cute import attention
        from tests.test_key_precision import key_cancellation

        # Constructor defaults audited from the corresponding model sources.
        for model, heads, dimension in (
            ("flux-moss-qwen", 24, 128),
            ("hunyuan-anima", 16, 128),
            ("qwen21-ltx2-video", 32, 128),
            ("ltxvideo", 32, 64),
            ("sd3", 18, 64),
            ("auraflow", 12, 256),
        ):
            for dtype in (torch.float16, torch.bfloat16):
                with self.subTest(model=model, heads=heads, dimension=dimension, dtype=dtype):
                    inputs, dout = key_cancellation(dimension, dtype, "cuda", heads=heads)
                    operands = [t.requires_grad_() for t in inputs]
                    output = attention(*operands)
                    actual = output, *torch.autograd.grad(output, operands, dout)
                    for h in range(0, heads, 2):
                        reference = rounded_reference(
                            tuple(t[:, h : h + 2] for t in inputs), dout[:, h : h + 2], output=output[:, h : h + 2]
                        )
                        for got, want in zip(actual, reference):
                            got = got[:, h : h + 2]
                            error = metrics(got, want)
                            self.assertTrue(error["finite"])
                            if error["reference_norm"]:
                                self.assertLess(error["relative_error"], 0.01 if dtype == torch.bfloat16 else 0.005)
                            else:
                                self.assertEqual(got.count_nonzero().item(), 0)
                            tolerance = 0.02 if dtype == torch.bfloat16 else 0.003
                            torch.testing.assert_close(got.double(), want, atol=tolerance, rtol=tolerance)


if __name__ == "__main__":
    unittest.main()
