"""Select only the measured A100 shared-V long family."""

import unittest

from kohakufa_cute.fa4_sm80._schedules import a100_schedule


class A100SharedValuePolicyTests(unittest.TestCase):
    def test_long_family_selects_shared_value_and_causal_group(self):
        for heads in (8, 16):
            for causal in (False, True):
                s = a100_schedule(256, 4096, causal, heads)
                self.assertEqual((s.backward_m, s.backward_n, s.kv_m_warps, s.dq_m_warps), (64, 64, 1, 2))
                self.assertFalse(s.v_in_regs)
                self.assertEqual(s.causal_head_group, 4 if causal else 1)

    def test_other_experimental_d256_shapes_keep_n32(self):
        for heads, sequence in ((8, 128), (16, 1024), (9, 4096), (16, 4095)):
            s = a100_schedule(256, sequence, True, heads)
            self.assertEqual(s.backward_n, 32)
            self.assertTrue(s.v_in_regs)
            self.assertEqual(s.causal_head_group, 1)


if __name__ == "__main__":
    unittest.main()
