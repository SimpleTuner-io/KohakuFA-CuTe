"""Actual-object parity for the combined metadata predicate."""

import unittest
from unittest.mock import patch

import torch

from kohakufa_cute import fa4_sm80
from kohakufa_cute.fa4_sm80 import a40


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class A40NativePredicateTests(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_name() != "NVIDIA A40" or not fa4_sm80.available():
            self.skipTest("The qualified full A40 runtime is required")

    def test_qualified_python_parity_on_actual_mutable_metadata(self):
        from benchmarks.a40_predicate_candidates import python_compatible

        checked = 0

        def check(q, k, v, **options):
            nonlocal checked
            expected = python_compatible(q, k, v, **options)
            self.assertEqual(a40.compatible(q, k, v, **options), expected)
            checked += 1

        for dtype in (torch.bfloat16, torch.float16):
            for dimension in (16, 64, 128, 256, 512):
                for sequence in (128, 1024, 1025, 4096):
                    q = torch.empty(1, 16, sequence, dimension, device="cuda", dtype=dtype)
                    for causal in (False, True):
                        check(q, q, q, causal=causal)
            q = torch.empty(1, 16, 1024, 128, device="cuda", dtype=dtype)
            for options in (
                {"mask": torch.ones(1, device="cuda", dtype=torch.bool)},
                {"dropout": 0.1},
                {"gqa": True},
                {"scale": 0.0},
                {"scale": -0.125},
                {"scale": 1e-40},
                {"scale": float("inf")},
                {"scale": float("nan")},
                {"scale": 0.125},
                {"scale": 0.0625},
            ):
                check(q, q, q, **options)
            with torch.no_grad():
                padded = torch.empty(q.numel() + 16, device="cuda", dtype=dtype)
                misaligned = padded[1 : 1 + q.numel()].view_as(q)
                noncontiguous = torch.empty(1, 16, 1024, 129, device="cuda", dtype=dtype)[..., :128]
                for position in range(3):
                    for bad in (
                        misaligned,
                        noncontiguous,
                        q.cpu(),
                        q.float(),
                        q[:, :8],
                        q[:, :, :512],
                    ):
                        values = [q, q, q]
                        values[position] = bad
                        check(*values, causal=True)
                values = [padded[: q.numel()].view_as(q), q, q]
                check(*values, causal=True)
                values[0].set_(padded.untyped_storage(), 1, q.shape, q.stride())
                check(*values, causal=True)
            with patch.object(torch, "are_deterministic_algorithms_enabled", return_value=True):
                check(q, q, q, causal=True)
            with patch.object(torch.compiler, "is_compiling", return_value=True):
                check(q, q, q, causal=True)
            with patch.object(fa4_sm80, "available", return_value=False):
                check(q, q, q, causal=True)
            if torch.cuda.device_count() > 1:
                with torch.cuda.device(1):
                    check(q, q, q, causal=True)
        self.assertGreater(checked, 100)

    def test_native_metadata_rejects_non_tensors_and_cpu_without_cuda_reads(self):
        from kohakufa_cute.native import extension

        predicate = extension().a40_metadata_compatible
        q = torch.empty(1, 16, 1024, 128)
        self.assertFalse(predicate(q, q, q, True))
        self.assertFalse(predicate(None, None, None, True))
        self.assertFalse(predicate(object(), object(), object(), True))


if __name__ == "__main__":
    unittest.main()
