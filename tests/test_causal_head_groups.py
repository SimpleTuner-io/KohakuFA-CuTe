"""Preserve default schedule identity and reject unsupported work orders."""

import unittest
from dataclasses import FrozenInstanceError

import torch

from kohakufa_cute import fa4_sm80
from kohakufa_cute.fa4_sm80._attention import Schedule


class HeadGroupScheduleTests(unittest.TestCase):
    def test_default_and_distinct_immutable_cache_keys(self):
        self.assertEqual(Schedule().causal_head_group, 1)
        schedules = [Schedule(causal_head_group=n) for n in (1, 2, 4, 8, 16)]
        self.assertEqual(len(set(schedules)), 5)
        with self.assertRaises(FrozenInstanceError):
            schedules[0].causal_head_group = 4

    def test_invalid_groups_are_rejected_before_compilation(self):
        for group in (-1, 0, 3, 32):
            with self.subTest(group=group), self.assertRaises(ValueError):
                Schedule(causal_head_group=group)


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.cuda.get_device_capability() in ((8, 0), (8, 6), (8, 9), (12, 0), (12, 1))
    and fa4_sm80.available(),
    "Requires a supported warp device and the qualified optional runtime",
)
class NativeHeadGroupControlsTests(unittest.TestCase):
    def test_constructor_domain_and_hopper_identity(self):
        from kohakufa_cute.native import extension

        module = extension()
        for group in (1, 2, 4, 8, 16):
            module.AttentionRunner("warp", {"causal_head_group": group})
        module.AttentionRunner("hopper", {"causal_head_group": 1, "maximum_multiple": 128})
        for group in (-1, 0, 3, 32):
            with self.subTest(group=group), self.assertRaises(ValueError):
                module.AttentionRunner("warp", {"causal_head_group": group})
        with self.assertRaises(ValueError):
            module.AttentionRunner("hopper", {"causal_head_group": 4})


if __name__ == "__main__":
    unittest.main()
