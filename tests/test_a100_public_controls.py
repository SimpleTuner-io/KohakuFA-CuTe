"""Keep old/new public-entry benchmark controls independent."""

import unittest


class A100PublicControlsTests(unittest.TestCase):
    def test_exact_api_bodies_have_isolated_selectors_and_loader_caches(self):
        from benchmarks.a100_public_api_controls import make_attention
        from kohakufa_cute import api, fa4_sm80

        previous, candidate = make_attention(previous=True), make_attention(previous=False)
        self.assertEqual(previous.__code__.co_code, candidate.__code__.co_code)
        self.assertEqual(candidate.__code__.co_code, api.scaled_dot_product_attention.__code__.co_code)
        old, new = previous.__globals__, candidate.__globals__
        self.assertIsNot(old, new)
        self.assertIsNot(old["_extension"], new["_extension"])
        self.assertIsNot(old["fa4_sm80"], new["fa4_sm80"])
        for scope in (old, new):
            self.assertEqual(scope["_extension"].cache_info().currsize, 0)
            self.assertIsNot(scope["fa4_sm80"], fa4_sm80)
            self.assertIs(scope["fa4_sm80"].compatible.__globals__, vars(scope["fa4_sm80"]))
        self.assertNotEqual(old["fa4_sm80"].attention.__code__.co_code, new["fa4_sm80"].attention.__code__.co_code)


if __name__ == "__main__":
    unittest.main()
