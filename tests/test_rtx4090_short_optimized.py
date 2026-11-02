import unittest
from unittest.mock import patch

import torch

from kohakufa_cute import fa4_sm80
from kohakufa_cute.fa4_sm80 import rtx4090


class ShortOptimizedPolicyTests(unittest.TestCase):
    def test_measured_dtype_visibility_choices(self):
        self.assertEqual(rtx4090._short_flags(torch.bfloat16, False), (False, True))
        self.assertEqual(rtx4090._short_flags(torch.float16, False), (True, True))
        for dtype in (torch.bfloat16, torch.float16):
            self.assertEqual(rtx4090._short_flags(dtype, True), (True, False))

    def test_unsupported_precision_is_rejected(self):
        with self.assertRaises(ValueError):
            rtx4090._short_flags(torch.float32, False)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class ShortOptimizedCacheTests(unittest.TestCase):
    def test_actual_distinct_semantic_namespaces_and_fixed_schedules(self):
        if torch.cuda.get_device_name() != "NVIDIA GeForce RTX 4090" or not fa4_sm80.available():
            self.skipTest("Qualified full RTX 4090 runtime required")
        from benchmarks.rtx4090_short_optimized_controls import make_attention
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
            for policy in ("previous", "selected"):
                output = make_attention(policy=policy)(q, k, v)
                torch.autograd.grad(output, (q, k, v), dout)
        self.assertEqual(set(modes), {"short_forward", "short_backward", "shortopt_forward", "shortopt_backward"})
        self.assertEqual(len(modes), 4)


if __name__ == "__main__":
    unittest.main()
