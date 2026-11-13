"""Independent traversal coverage of proposed wider neighborhood tiles."""

import unittest

from benchmarks.neighborhood_reference import neighborhoods
from benchmarks.neighborhood_tuning import (
    BASELINE_COLUMNS,
    COLUMN_EXPERIMENTS,
    SECONDARY_COLUMN_EXPERIMENTS,
    describe_columns,
    normalize_columns,
)
from kohakufa_cute.neighborhood_geometry import (
    full_row_tile_bounds,
    normalize_exact_contiguous,
    spatial_bounds,
    spatial_tile,
)


class NeighborhoodTuningTests(unittest.TestCase):
    def test_experiments_preserve_baseline_and_reject_invalid_widths(self):
        self.assertIsNone(normalize_columns(None))
        self.assertEqual(describe_columns(None)["columns"], "automatic")
        self.assertEqual(COLUMN_EXPERIMENTS["baseline"], (32, 32, 32))
        self.assertEqual(BASELINE_COLUMNS, normalize_columns([32, 32, 32]))
        self.assertEqual(COLUMN_EXPERIMENTS["columns64"], (64, 64, 64))
        self.assertEqual(SECONDARY_COLUMN_EXPERIMENTS["columns128"], (128, 128, 128))
        for value in ((True, 32, 32), (64, 64), (64, 64, 48), (0, 32, 32)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_columns(value)

    def test_per_owner_traversal_flags_preserve_bool_compatibility(self):
        self.assertEqual(normalize_exact_contiguous(False), (False, False, False))
        self.assertEqual(normalize_exact_contiguous(True), (True, True, True))
        self.assertEqual(normalize_exact_contiguous([True, False, True]), (True, False, True))
        for invalid in (None, 1, (True, False), (True, 0, True), (True, False, True, False)):
            with self.subTest(selection=invalid), self.assertRaises(ValueError):
                normalize_exact_contiguous(invalid)

    def test_complete_row_interval_preserves_entire_active_tile_sequence(self):
        for height in (1, 3, 9, 17, 33):
            for width in (1, 2, 4, 8, 16, 32, 64):
                for window_h in {1, min(3, height), min(7, height), height}:
                    for window_w in {1, min(3, width), min(7, width), width}:
                        for columns in (32, 64, 128):
                            for inverse in (False, True):
                                for start in range(0, height * width, 64):
                                    bounds = spatial_bounds(start, 64, height, width, window_h, window_w, inverse)
                                    y0, y1, x0, x1 = bounds
                                    self.assertEqual((x0, x1), (0, width))
                                    slots = (x1 - x0 + 2 * columns - 2) // columns
                                    previous = [
                                        block
                                        for index in range((y1 - y0) * slots)
                                        for block, active in [spatial_tile(index, y0, x0, x1, width, columns)]
                                        if active
                                    ]
                                    self.assertEqual(
                                        list(range(*full_row_tile_bounds(y0, y1, width, columns))),
                                        previous,
                                        (height, width, window_h, window_w, columns, inverse, start),
                                    )

    def test_wider_tiles_cover_forward_and_inverse_edges_without_duplicates(self):
        for spatial, window in (((3, 5), (3, 3)), ((9, 17), (3, 4)), ((17, 131), (7, 5))):
            edges = neighborhoods(spatial, window)
            for columns in (32, 64, 128):
                for inverse in (False, True):
                    for start in range(0, len(edges), 64):
                        y0, y1, x0, x1 = spatial_bounds(start, 64, *spatial, *window, inverse)
                        slots = (x1 - x0 + 2 * columns - 2) // columns
                        selected = [
                            block
                            for index in range((y1 - y0) * slots)
                            for block, active in [spatial_tile(index, y0, x0, x1, spatial[1], columns)]
                            if active
                        ]
                        self.assertEqual(len(selected), len(set(selected)))
                        covered = set(selected)
                        owners = set(range(start, min(start + 64, len(edges))))
                        needed = (
                            {query for query, keys in enumerate(edges) if owners.intersection(keys)}
                            if inverse
                            else {key for query in owners for key in edges[query]}
                        )
                        self.assertTrue({key // columns for key in needed} <= covered)


if __name__ == "__main__":
    unittest.main()
