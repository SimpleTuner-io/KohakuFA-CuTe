"""Ownership and visibility boundaries of the separate causal mask loops."""

import math
import unittest


class HopperMaskPartitionTests(unittest.TestCase):
    def test_partial_and_aligned_sequences_keep_each_query_tile_once(self):
        tile_m = tile_n = 128
        for length in (1, 63, 128, 129, 257, 1025, 4096):
            query_tiles = math.ceil(length / tile_m)
            for key_tile in range(math.ceil(length / tile_n)):
                first = key_tile * tile_n // tile_m
                diagonal_end = min(query_tiles, math.ceil((key_tile + 1) * tile_n / tile_m))
                masked = list(range(first, diagonal_end))
                steady = list(range(diagonal_end, query_tiles))
                self.assertEqual(masked + steady, list(range(first, query_tiles)))
                self.assertEqual(len(set(masked + steady)), len(masked + steady))
                for query_tile in steady:
                    first_query = query_tile * tile_m
                    last_key = min(length, (key_tile + 1) * tile_n) - 1
                    self.assertGreaterEqual(first_query, last_key)

    def test_steady_loop_still_requires_the_key_tail_predicate(self):
        length = 257
        key_tile = 2
        valid_keys = [key for key in range(key_tile * 128, (key_tile + 1) * 128) if key < length]
        self.assertEqual(valid_keys, [256])


if __name__ == "__main__":
    unittest.main()
