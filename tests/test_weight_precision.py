"""Check precision-policy ownership without requiring a CUDA runtime."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class Owner:
    constructions = 0

    def __init__(self, mode="kv", causal=False, masked=False, **configuration):
        type(self).constructions += 1
        self.mode, self.causal, self.masked = mode, causal, masked
        defaults = dict(
            centered=True,
            columns=32,
            width=128,
            vector=8,
            copy_cache_global=True,
            reload_center=True,
            logical_width=0,
            narrow_n=False,
            full_visibility=False,
            full_copies=False,
            packed_statistics=False,
            tma=False,
            pipelined=False,
        )
        self.__dict__.update(defaults | configuration)


class UntouchableQuery:
    def __getattr__(self, name):
        raise AssertionError("The precision policy must not inspect query ownership.")


class WeightPrecisionPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).parents[1] / "src/kohakufa_cute/weight_precision.py"
        spec = importlib.util.spec_from_file_location("kohakufa_cute._weight_policy_test", path)
        cls.policy = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"kohakufa_cute.kernels": types.SimpleNamespace(Attention=Owner)}):
            spec.loader.exec_module(cls.policy)

    def test_unsupported_widths_stop_before_rewriting(self):
        for dimension in (0, 32, 96, 256, 512):
            with self.subTest(dimension=dimension), self.assertRaises(ValueError):
                self.policy.schedule(types.SimpleNamespace(), dimension)

    def test_shared_joint_owner_is_rewritten_once_and_query_is_untouched(self):
        query, original = UntouchableQuery(), Owner()
        joint = types.SimpleNamespace(query=query, keyvalue=original)
        wrapper = types.SimpleNamespace(fallback=joint, tuned=joint)
        count = Owner.constructions
        actual = self.policy.schedule(wrapper, 128)
        self.assertIs(actual, wrapper)
        self.assertIs(actual.fallback, actual.tuned)
        self.assertIs(joint.query, query)
        self.assertIsNot(joint.keyvalue, original)
        self.assertEqual(Owner.constructions - count, 1)

    def test_d64_replaces_tma_without_changing_query_or_complete_guards(self):
        query = UntouchableQuery()
        original = Owner(width=64, columns=64, tma=True, pipelined=True, full_visibility=True, full_copies=True)
        complete = types.SimpleNamespace(query=query, keyvalue=original)
        actual = self.policy.schedule(types.SimpleNamespace(complete=complete), 64)
        replacement = actual.complete.keyvalue
        self.assertIs(actual.complete.query, query)
        self.assertFalse(replacement.tma)
        self.assertFalse(replacement.pipelined)
        self.assertTrue(replacement.full_visibility)
        self.assertTrue(replacement.full_copies)
        self.assertTrue(replacement.copy_cache_global)
        self.assertTrue(replacement.reload_center)
        self.assertEqual(replacement.columns, 32)
        self.assertEqual(replacement.kv_column_warps, 1)
        self.assertTrue(replacement.compensated_keys and replacement.compensated_values)
        self.assertTrue(replacement.scaled_key_residual and replacement.scaled_value_residual)

    def test_logical_d16_and_short_d128_ownership_are_preserved(self):
        logical = self.policy.schedule(Owner(width=32, columns=64, logical_width=16), 16)
        self.assertEqual((logical.columns, logical.logical_width), (64, 16))
        short = self.policy.schedule(Owner(narrow_n=True), 128)
        long = self.policy.schedule(Owner(), 128)
        self.assertTrue(short.narrow_n)
        self.assertEqual(short.kv_column_warps, 4)
        self.assertTrue(short.combined_key_products and short.combined_value_products)
        self.assertEqual(long.kv_column_warps, 2)
        self.assertFalse(long.combined_key_products or long.combined_value_products)

    def test_custom_kernel_subclasses_keep_their_qualified_math(self):
        class Custom(Owner):
            pass

        original = Custom()
        self.assertIs(self.policy.schedule(original, 128), original)


if __name__ == "__main__":
    unittest.main()
