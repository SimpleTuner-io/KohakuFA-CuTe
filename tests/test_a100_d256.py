"""Keep A100 D256 dispatch bounded to qualified runtime families."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from kohakufa_cute import fa4_sm80
from tests.test_fa4_sm80 import cuda_operand


class A100D256PolicyTests(unittest.TestCase):
    def test_selected_shapes_and_fallbacks(self):
        with (
            patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)),
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(multi_processor_count=108)),
            patch.object(fa4_sm80, "available", return_value=True),
        ):
            for heads in (2, 8, 9, 16, 17):
                for sequence in (127, 128, 129, 257, 1024, 2048, 4096):
                    q = cuda_operand((1, heads, sequence, 256))
                    for causal in (False, True):
                        self.assertEqual(
                            fa4_sm80.compatible(q, q, q, None, 0, causal, None, False),
                            (heads in (8, 16) and sequence == 4096) or (heads == 16 and sequence == 1024),
                        )
            q = cuda_operand((1, 16, 1024, 256))
            for scale in (-0.125, 0, 1e-50, float("inf")):
                self.assertFalse(fa4_sm80.compatible(q, q, q, None, 0, False, scale, False))
            with patch.object(fa4_sm80, "available", return_value=False):
                self.assertFalse(fa4_sm80.compatible(q, q, q, None, 0, False, None, False))

    def test_d256_schedule_keeps_other_families_stable(self):
        from kohakufa_cute.fa4_sm80._attention import Schedule
        from kohakufa_cute.fa4_sm80._schedules import a100_schedule

        for heads in (8, 16):
            for sequence in (128, 1024, 4096):
                for causal in (False, True):
                    s = a100_schedule(256, sequence, causal, heads)
                    self.assertEqual(
                        (s.forward_m, s.forward_n, s.backward_m, s.backward_n), (64, 64, 64, 64 if sequence == 4096 else 32)
                    )
                    self.assertEqual(
                        (s.query_stages, s.dout_stages, s.score_m_warps, s.kv_m_warps, s.dq_m_warps), (1, 1, 4, 1, 2)
                    )
                    self.assertEqual(s.v_in_regs, sequence != 4096)
                    self.assertEqual(s.causal_head_group, 4 if causal and sequence == 4096 else 1)
        self.assertEqual(Schedule().backward_n, 128)


if __name__ == "__main__":
    unittest.main()
