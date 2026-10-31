"""Regression coverage for cancellation in low-precision value gradients."""

import math
import unittest

import torch

from benchmarks.reference import fp64_reference, metrics, rounded_reference
from kohakufa_cute import attention


def square_value_cancellation(dimension, dtype, device, *, length=None, heads=16):
    """Embed the same probability cancellation in public square dispatches."""
    if length is None:
        length = 4096 if dimension in (256, 512) else 1024
    if length != 128 and length < 1024:
        raise ValueError("The cancellation fixture needs 128 or at least 1024 tokens.")
    first_count, active_queries = (64, 128) if length == 128 else (512, 1023)
    q = torch.zeros(1, heads, length, dimension, device=device, dtype=dtype)
    k, v, dout = torch.zeros_like(q), torch.zeros_like(q), torch.zeros_like(q)
    q[:, :, :first_count, 0], q[:, :, first_count:, 0] = 12, 4
    k[:, :, 0, 0], k[:, :, 17:, 0] = 1, -32768
    dout[:, :, :first_count, 0], dout[:, :, first_count:active_queries, 0] = 1, -1.875
    scale = 128**-0.5
    a, b = (math.exp(x * scale) for x in (12, 4))
    target = first_count * a / (16 + a) - (active_queries - first_count) * 1.875 * b / (16 + b)
    return (q, k, v), dout, scale, target


def value_residual_underflow(dimension, device, *, length=None, heads=16):
    """A representable FP16 gradient needs probabilities below half's range."""
    if length is None:
        length = 4096 if dimension in (256, 512) else 1024
    if length != 128 and length < 1024:
        raise ValueError("The underflow fixture needs 128 or at least 1024 tokens.")
    active = 128 if length == 128 else 1024
    first = active // 2
    q = torch.zeros(1, heads, length, dimension, device=device, dtype=torch.float16)
    k, v, dout = torch.zeros_like(q), torch.zeros_like(q), torch.zeros_like(q)
    q[:, :, :first, 0], q[:, :, first:, 0] = 1, 2
    k[:, :, 0, 0], k[:, :, active:, 0] = -122, -32768
    dout[:, :, :first, 0], dout[:, :, first:active, 0] = 65504, -65472
    scale = 128**-0.5
    a, b = math.exp(-122 * scale), math.exp(-244 * scale)
    target = first * (65504 * a / (active - 1 + a) - 65472 * b / (active - 1 + b))
    return (q, k, v), dout, scale, target


def qualified_device():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in {
        (8, 6),
        (8, 9),
        (9, 0),
        (10, 0),
        (10, 3),
        (12, 0),
        (12, 1),
    }


class ValueCancellationReferenceTests(unittest.TestCase):
    def test_square_embedding_preserves_the_closed_form(self):
        for dimension in (16, 64, 128, 256, 512):
            for length in (128, 1024):
                with self.subTest(dimension=dimension, length=length):
                    inputs, dout, scale, target = square_value_cancellation(
                        dimension, torch.bfloat16, "cpu", length=length, heads=1
                    )
                    reference = fp64_reference(inputs, dout, scale=scale)
                    self.assertEqual(reference[0].abs().max().item(), 0)
                    self.assertEqual(reference[1].abs().max().item(), 0)
                    self.assertEqual(reference[2].abs().max().item(), 0)
                    self.assertAlmostEqual(float(reference[3][0, 0, 0, 0]), target, delta=1e-9)
                    self.assertEqual(reference[3][:, :, 17:].abs().max().item(), 0)

    def test_probability_underflow_is_not_a_final_output_rounding_limit(self):
        for length in (128, 1024):
            with self.subTest(length=length):
                inputs, dout, scale, target = value_residual_underflow(128, "cpu", length=length, heads=1)
                reference = fp64_reference(inputs, dout, scale=scale)
                self.assertAlmostEqual(float(reference[3][0, 0, 0, 0]), target, delta=1e-7)
                self.assertGreater(target, 0.5)
                self.assertLess(metrics(reference[3].half(), reference[3])["relative_error"], 0.005)


@unittest.skipUnless(qualified_device(), "Requires a qualified low-precision value-gradient device")
class RectangularValuePrecisionTests(unittest.TestCase):
    def compare(self, q, k, v, dout, *, mask=None, causal=False, scale=None):
        operands = [tensor.detach().requires_grad_(True) for tensor in (q, k, v)]
        # This frontend supports combined boolean/causal visibility and implicit GQA.
        out = attention(*operands, mask=mask, causal=causal, scale=scale)
        actual = (out, *torch.autograd.grad(out, operands, dout))
        if q.shape[1] == k.shape[1] and q.shape[1] > 2 and mask is None:
            parts = [
                rounded_reference(
                    tuple(tensor[:, h : h + 2] for tensor in (q, k, v)),
                    dout[:, h : h + 2],
                    causal=causal,
                    scale=scale,
                    output=out[:, h : h + 2],
                )
                for h in range(0, q.shape[1], 2)
            ]
            reference = tuple(torch.cat([part[i] for part in parts], dim=1) for i in range(4))
        else:
            reference = rounded_reference((q, k, v), dout, mask=mask, causal=causal, scale=scale, output=out)
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

    def test_value_cancellation_matches_single_rounded_probability(self):
        # The rounded-P result may have a different sign from unrounded FP64.
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                q = torch.zeros(1, 1, 1023, 128, device="cuda", dtype=dtype)
                k = torch.zeros(1, 1, 17, 128, device="cuda", dtype=dtype)
                v, dout = torch.zeros_like(k), torch.zeros_like(q)
                q[:, :, :512, 0], q[:, :, 512:, 0], k[:, :, 0, 0] = 12, 4, 1
                dout[:, :, :512, 0], dout[:, :, 512:, 0] = 1, -1.875
                actual, reference = self.compare(q, k, v, dout)
                got, want = actual[3][0, 0, 0, 0].double(), reference[3][0, 0, 0, 0]
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

    def test_square_embeddings_match_rounded_probability(self):
        for dtype in (torch.bfloat16, torch.float16):
            for dimension in (16, 64, 128, 256, 512):
                for length in (128, 4096 if dimension in (256, 512) else 1024):
                    with self.subTest(dtype=dtype, dimension=dimension, length=length):
                        inputs, dout, scale, target = square_value_cancellation(dimension, dtype, "cuda", length=length)
                        actual, expected = self.compare(*inputs, dout, scale=scale)
                        coordinate = actual[3][:, :, 0, 0].double()
                        torch.testing.assert_close(coordinate, expected[3][:, :, 0, 0], atol=0.02, rtol=0.02)

    def test_fp16_probability_underflow_matches_rounded_contract(self):
        for dimension in (16, 64, 128, 256, 512):
            for length in (128, 4096 if dimension in (256, 512) else 1024):
                with self.subTest(dimension=dimension, length=length):
                    inputs, dout, scale, target = value_residual_underflow(dimension, "cuda", length=length)
                    actual, expected = self.compare(*inputs, dout, scale=scale)
                    coordinate = actual[3][:, :, 0, 0].double()
                    torch.testing.assert_close(coordinate, expected[3][:, :, 0, 0], atol=0.003, rtol=0.005)
                    if length >= 1024:
                        self.assertEqual(coordinate.count_nonzero().item(), 0)

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
