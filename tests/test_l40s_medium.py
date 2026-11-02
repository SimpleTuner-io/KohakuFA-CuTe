"""Bounded dtype-specific selection for the medium causal L40S family."""

import unittest

import torch

from kohakufa_cute.fa4_sm80 import _l40s


class MediumSelectionTests(unittest.TestCase):
    def test_only_target_geometry_changes(self):
        for dtype in (torch.bfloat16, torch.float16):
            s = _l40s.schedule(64, 1024, True, dtype)
            self.assertEqual(
                (
                    s.forward_m,
                    s.forward_n,
                    s.backward_m,
                    s.backward_n,
                    s.query_stages,
                    s.dout_stages,
                    s.causal_head_group,
                ),
                (128, 64, 64, 64, 2 if dtype == torch.bfloat16 else 1, 1, 16),
            )
            self.assertTrue(s.v_in_regs)
            for d, n, c in ((64, 1024, False), (64, 4096, True), (128, 1024, True), (128, 4096, False)):
                self.assertEqual(_l40s.schedule(d, n, c, dtype), _l40s.schedule(d, n, c))
            self.assertIs(s, _l40s.schedule(64, 1024, True, dtype))
