"""Logical ownership checks for the explicit causal scheduler experiment."""

import unittest
from collections import Counter

from benchmarks.experimental_hopper_lpt import decode_work, head_swizzle


class HopperSchedulerTests(unittest.TestCase):
    def test_measured_shapes_match_fa3_swizzle_despite_l2_budget_difference(self):
        for dim, expected in ((64, 16), (128, 8)):
            for budget in (40 << 20, 50 << 20):
                self.assertEqual(head_swizzle(4096, dim, dim, l2_bytes=budget), expected)

    def test_mapping_is_bijective_including_residual_heads_and_batches(self):
        for key_tiles, heads, batches, swizzle in (
            (32, 16, 1, 16),
            (32, 16, 1, 8),
            (3, 5, 2, 8),
            (3, 2, 1, 8),
            (1, 1, 1, 1),
        ):
            with self.subTest(geometry=(key_tiles, heads, batches, swizzle)):
                actual = [decode_work(i, key_tiles, heads, batches, swizzle) for i in range(key_tiles * heads * batches)]
                expected = {
                    (key, head, batch) for key in range(key_tiles) for head in range(heads) for batch in range(batches)
                }
                self.assertEqual(set(actual), expected)
                self.assertEqual(len(actual), len(expected))

    def test_non_deterministic_mapping_preserves_key_order_and_interleaves_heads(self):
        actual = [decode_work(i, 32, 16, 1, 16) for i in range(32)]
        self.assertEqual(actual[:16], [(0, head, 0) for head in range(16)])
        self.assertEqual(actual[16:], [(1, head, 0) for head in range(16)])

    def test_linear_first_wave_is_only_a_mapping_model(self):
        # Hardware admission is not guaranteed to follow this linear model.
        for dim, maximum_per_head in ((64, 9), (128, 17)):
            swizzle = head_swizzle(4096, dim, dim)
            counts = Counter(decode_work(i, 32, 16, 1, swizzle)[1] for i in range(132))
            self.assertEqual(max(counts.values()), maximum_per_head)

    def test_out_of_grid_index_is_rejected(self):
        for index in (-1, 512):
            with self.assertRaises(ValueError):
                decode_work(index, 32, 16, 1, 16)


if __name__ == "__main__":
    unittest.main()
