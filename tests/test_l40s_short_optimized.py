"""Independent L40S flag policy and native semantic-cache qualification."""

import unittest
from unittest.mock import patch

import torch

from kohakufa_cute import fa4_sm80
from kohakufa_cute.fa4_sm80 import _l40s


class L40sShortPolicyTests(unittest.TestCase):
    def test_independent_measured_choices(self):
        self.assertEqual(_l40s._short_flags(torch.bfloat16, False), (False, True))
        self.assertEqual(_l40s._short_flags(torch.float16, False), (True, True))
        for dtype in (torch.bfloat16, torch.float16):
            self.assertEqual(_l40s._short_flags(dtype, True), (True, False))
        with self.assertRaises(ValueError):
            _l40s._short_flags(torch.float32, False)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class L40sShortPolicyGpuTests(unittest.TestCase):
    def test_both_semantic_namespaces_and_policy_routing(self):
        if not _l40s.is_device(torch.device("cuda", torch.cuda.current_device())) or not fa4_sm80.available():
            self.skipTest("Qualified full L40S runtime required")
        from benchmarks.experiments.ada.l40s_short_optimized_controls import make_attention
        from kohakufa_cute import native

        module = native.extension()
        module.clear_cache()
        with self.assertRaisesRegex(ValueError, "fixed geometry"):
            module.AttentionRunner("short-optimized", {"query_stages": 2})
        modes = []
        original = native.compile_kernel

        def record(mode, *args, **kwargs):
            modes.append(mode)
            return original(mode, *args, **kwargs)

        q, k, v = [torch.randn(1, 16, 128, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
        dout = torch.randn_like(q)
        with patch.object(native, "compile_kernel", side_effect=record):
            for mode in ("parent", "public"):
                out = make_attention(mode=mode)(q, k, v)
                torch.autograd.grad(out, (q, k, v), dout)
        self.assertEqual(set(modes), {"short_forward", "short_backward", "shortopt_forward", "shortopt_backward"})
        self.assertEqual(len(modes), 4)


if __name__ == "__main__":
    unittest.main()
