"""Bound A100 head ordering to the measured public families."""

import sys
import unittest


class A100GroupPolicyTests(unittest.TestCase):
    def test_short_dense_other_heads_and_lengths_keep_original_policy(self):
        from kohakufa_cute.fa4_sm80._schedules import a100_schedule

        for dimension in (64, 128):
            for heads in (8, 9, 15, 16):
                for sequence in (128, 1024, 2048, 4095, 4096, 4097):
                    for causal in (False, True):
                        selected = causal and sequence == 4096 and heads in (8, 16)
                        self.assertEqual(
                            a100_schedule(dimension, sequence, causal, heads).causal_head_group,
                            (4 if heads == 8 else 16) if selected else 1,
                        )

    def test_before_and_after_api_modules_retain_independent_schedule_sources(self):
        from benchmarks.a100_group_api_controls import make_attention
        from kohakufa_cute import api

        previous, candidate = make_attention(previous=True), make_attention(previous=False)
        self.assertEqual(previous.__code__.co_code, candidate.__code__.co_code)
        self.assertEqual(candidate.__code__.co_code, api.scaled_dot_product_attention.__code__.co_code)
        old, new = previous.__globals__, candidate.__globals__
        self.assertIsNot(old["_extension"], new["_extension"])
        self.assertIsNot(old["fa4_sm80"], new["fa4_sm80"])
        packages = [scope["fa4_sm80"].__name__ for scope in (old, new)]
        old_schedule, new_schedule = [sys.modules[name + "._schedules"].a100_schedule for name in packages]
        self.assertEqual(old_schedule(64, 4096, True).causal_head_group, 1)
        self.assertEqual(new_schedule(64, 4096, True, 16).causal_head_group, 16)
        for scope in (old, new):
            self.assertEqual(scope["_extension"].cache_info().currsize, 0)


if __name__ == "__main__":
    unittest.main()
