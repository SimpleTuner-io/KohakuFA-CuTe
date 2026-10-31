"""Regression coverage for cancellation in rectangular low-precision value gradients."""

import unittest

import torch

from benchmarks.reference import fp64_reference, metrics
from kohakufa_cute import attention


def qualified_device():
    return torch.cuda.is_available() and torch.cuda.get_device_name() in (
        "NVIDIA GeForce RTX 3090 Ti",
        "NVIDIA L40S",
        "NVIDIA GeForce RTX 5090",
    )


@unittest.skipUnless(qualified_device(), "Requires a qualified rectangular low-precision device")
class RectangularValuePrecisionTests(unittest.TestCase):
    def compare(self, q, k, v, dout, *, mask=None, causal=False):
        operands = [tensor.detach().requires_grad_(True) for tensor in (q, k, v)]
        # This frontend supports combined boolean/causal visibility and implicit GQA.
        out = attention(*operands, mask=mask, causal=causal)
        actual = (out, *torch.autograd.grad(out, operands, dout))
        reference = fp64_reference((q, k, v), dout, mask=mask, causal=causal)
        tolerance = 0.02 if q.dtype == torch.bfloat16 else 0.003
        norm_limit = 0.01 if q.dtype == torch.bfloat16 else 0.005
        for got, want in zip(actual, reference):
            torch.testing.assert_close(got.double(), want, atol=tolerance, rtol=tolerance)
            error = metrics(got, want)
            if error["reference_norm"]:
                self.assertLess(error["relative_error"], norm_limit)
            else:
                self.assertEqual(error["max_absolute_error"], 0)
        return actual, reference

    def test_value_cancellation_preserves_sign_and_small_coordinate(self):
        # A single rounded P produces the wrong sign in both supported dtypes.
        # The complete dV norm passes, so this coordinate needs its own gate.
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                q = torch.zeros(1, 1, 1023, 128, device="cuda", dtype=dtype)
                k = torch.zeros(1, 1, 17, 128, device="cuda", dtype=dtype)
                v, dout = torch.zeros_like(k), torch.zeros_like(q)
                q[:, :, :512, 0], q[:, :, 512:, 0], k[:, :, 0, 0] = 12, 4, 1
                dout[:, :, :512, 0], dout[:, :, 512:, 0] = 1, -1.875
                actual, reference = self.compare(q, k, v, dout)
                got, want = actual[3][0, 0, 0, 0].double(), reference[3][0, 0, 0, 0]
                self.assertLess(float(got), 0)
                self.assertNotEqual(float(got), 0)
                torch.testing.assert_close(got, want, atol=0, rtol=0.02)

    def test_cached_rectangular_and_square_boundaries(self):
        from kohakufa_cute import native

        native.extension().clear_cache()
        torch.manual_seed(1229)
        for dtype in (torch.bfloat16, torch.float16):
            for causal in (False, True):
                for queries, keys in ((129, 128), (127, 128), (128, 128), (130, 129), (65, 17), (17, 65)):
                    with self.subTest(queries=queries, keys=keys, causal=causal):
                        q = torch.randn(1, 2, queries, 128, device="cuda", dtype=dtype)
                        k, v = [torch.randn(1, 2, keys, 128, device="cuda", dtype=q.dtype) for _ in range(2)]
                        self.compare(q, k, v, torch.randn_like(q), causal=causal)

    def test_strided_grouped_inputs_with_padding_full_masks_and_hidden_rows(self):
        torch.manual_seed(1237)
        for dtype in (torch.bfloat16, torch.float16):
            for full in (False, True):
                for causal in (False, True):
                    with self.subTest(full=full, causal=causal):
                        q = torch.randn(1, 4, 129, 256, device="cuda", dtype=dtype)[..., ::2]
                        k, v = [torch.randn(1, 2, 17, 256, device="cuda", dtype=q.dtype)[..., ::2] for _ in range(2)]
                        if full:
                            mask = torch.rand(1, 1, 129, 17, device="cuda") > 0.2
                            mask[:, :, :3] = False
                        else:
                            mask = (torch.arange(17, device="cuda") % 5 != 1).reshape(1, 1, 1, 17)
                        self.compare(q, k, v, torch.randn_like(q), mask=mask, causal=causal)


if __name__ == "__main__":
    unittest.main()
