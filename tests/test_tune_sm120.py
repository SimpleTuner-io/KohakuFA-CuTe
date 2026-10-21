import unittest

from benchmarks.tune_sm120 import candidates


class SM120TuningTests(unittest.TestCase):
    def test_wide_tma_pipeline_fits_query_but_not_four_warp_keyvalue(self):
        limit = 101376
        query = candidates("q", 256, (32, 128, 4, False, True, False, False), limit)
        self.assertIn((16, 128, 4, False, True, True, True), query)

        keyvalue = candidates("kv", 256, (32, 256, 4, False, True, False, False), limit)
        # The shared score/probability tiles also occupy memory in the KV pass.
        self.assertNotIn((32, 256, 4, False, True, True, True), keyvalue)
        self.assertIn((32, 256, 2, False, True, True, True), keyvalue)


if __name__ == "__main__":
    unittest.main()
