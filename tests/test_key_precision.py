"""Deterministic cancellation checks for rounded key-gradient operands."""

import math
import unittest

import torch

from benchmarks.reference import fp64_reference, metrics
from kohakufa_cute import attention

RECIPES = {
    16: (34, 10, 12.375),
    64: (68, 20, 12.375),
    128: (59, 43, 3.21875),
    256: (136, 40, 12.375),
    512: (195, 43, 20.625),
}


def key_cancellation(dimension, dtype, device, *, heads=16, square=False):
    first, second, ratio = RECIPES[dimension]
    queries, keys = (4096, 4096) if square else (1025, 1031)
    q = torch.zeros(1, heads, queries, dimension, dtype=dtype, device=device)
    k = torch.zeros(1, heads, keys, dimension, dtype=dtype, device=device)
    v, dout = torch.zeros_like(k), torch.zeros_like(q)
    q[..., 0], k[..., 0] = 20, -20
    q[:, :, :512, 1], q[:, :, 512:1025, 1] = first, second
    k[:, :, 0, 1], v[:, :, 0, 0] = 1, 1
    dout[:, :, :512, 0], dout[:, :, 512:1025, 0] = 1, -ratio
    if square:
        # Extra keys have exactly zero probability in FP64 too. Extra query
        # rows have zero dO, preserving the smaller fixture's closed form.
        k[:, :, 1031:, 0] = -32768
    return (q, k, v), dout


def key_coordinate(dimension):
    first, second, ratio = RECIPES[dimension]
    scale = dimension**-0.5
    a = math.exp(first * scale) / (1030 + math.exp(first * scale))
    b = math.exp(second * scale) / (1030 + math.exp(second * scale))
    return (512 * a * (1 - a) - 513 * ratio * b * (1 - b)) * 20 * scale


class KeyCancellationReferenceTests(unittest.TestCase):
    def test_closed_form_matches_centered_reference_on_rounded_inputs(self):
        for dimension in RECIPES:
            with self.subTest(dimension=dimension):
                bf16, dout = key_cancellation(dimension, torch.bfloat16, "cpu", heads=1)
                fp16, half_dout = key_cancellation(dimension, torch.float16, "cpu", heads=1)
                for left, right in zip((*bf16, dout), (*fp16, half_dout)):
                    self.assertTrue(torch.equal(left.double(), right.double()))
                reference = fp64_reference(bf16, dout)
                self.assertTrue(all(torch.isfinite(tensor).all() for tensor in reference))
                self.assertLess(key_coordinate(dimension), 0)
                self.assertAlmostEqual(float(reference[2][0, 0, 0, 0]), key_coordinate(dimension), delta=1e-9)


def qualified_device():
    # Device PRs enable these checks alongside their precision policy.
    return False


@unittest.skipUnless(qualified_device(), "Requires a qualified key-gradient precision device")
class KeyPrecisionTests(unittest.TestCase):
    def setUp(self):
        from kohakufa_cute import kernels, native

        native.extension().clear_cache()
        kernels._compile_kernel.cache_clear()
        self.addCleanup(native.extension().clear_cache)
        self.addCleanup(kernels._compile_kernel.cache_clear)

    def compare(self, inputs, dout, **kwargs):
        operands = [tensor.detach().requires_grad_(True) for tensor in inputs]
        out = attention(*operands, **kwargs)
        actual = (out, *torch.autograd.grad(out, operands, dout))
        if inputs[0].shape[1] == inputs[1].shape[1]:
            parts = [
                fp64_reference(tuple(tensor[:, h : h + 2] for tensor in inputs), dout[:, h : h + 2], **kwargs)
                for h in range(0, inputs[0].shape[1], 2)
            ]
            reference = tuple(torch.cat([part[i] for part in parts], dim=1) for i in range(4))
        else:
            reference = fp64_reference(inputs, dout, **kwargs)
        tolerance = 0.02 if inputs[0].dtype == torch.bfloat16 else 0.003
        norm_limit = 0.01 if inputs[0].dtype == torch.bfloat16 else 0.005
        for name, got, want in zip(("out", "dq", "dk", "dv"), actual, reference):
            with self.subTest(component=name):
                error = metrics(got, want)
                self.assertTrue(error["finite"])
                torch.testing.assert_close(got.double(), want, atol=tolerance, rtol=tolerance)
                if error["reference_norm"]:
                    self.assertLess(error["relative_error"], norm_limit)
                else:
                    self.assertEqual(error["max_absolute_error"], 0)
        return actual

    def test_cancellation_preserves_small_key_coordinate_across_head_widths(self):
        for dtype in (torch.bfloat16, torch.float16):
            for dimension in RECIPES:
                with self.subTest(dtype=dtype, dimension=dimension):
                    inputs, dout = key_cancellation(dimension, dtype, "cuda", square=dimension in (256, 512))
                    actual = self.compare(inputs, dout)
                    coordinate = actual[2][:, :, 0, 0].double()
                    self.assertTrue((coordinate < 0).all())
                    target = torch.full_like(coordinate, key_coordinate(dimension))
                    torch.testing.assert_close(coordinate, target, atol=0, rtol=0.02)

    def test_strided_grouped_inputs_with_full_and_padding_masks(self):
        torch.manual_seed(1319)
        for dtype in (torch.bfloat16, torch.float16):
            for dimension in (16, 64, 128):
                for full in (False, True):
                    with self.subTest(dtype=dtype, dimension=dimension, full=full):
                        q = torch.randn(1, 4, 129, 2 * dimension, device="cuda", dtype=dtype)[..., ::2]
                        k, v = [torch.randn(1, 2, 65, 2 * dimension, device="cuda", dtype=dtype)[..., ::2] for _ in range(2)]
                        mask = (torch.arange(65, device="cuda") % 5 != 1).reshape(1, 1, 1, 65)
                        if full:
                            mask = mask.expand(1, 1, 129, 65).clone()
                            mask[:, :, :3] = False
                        self.compare((q, k, v), torch.randn_like(q), mask=mask)


if __name__ == "__main__":
    unittest.main()
