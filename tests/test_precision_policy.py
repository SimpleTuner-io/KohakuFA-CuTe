"""Check wide-owner replacement while preserving query and wrapper ownership."""

import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class Owner:
    def __init__(self, mode="kv", *, masked=False, causal=True):
        self.mode, self.masked, self.causal = mode, masked, causal
        self.centered, self.vector, self.copy_cache_global = True, 8, True
        self.compensated_normalizer = False


class Replacement:
    def __init__(self, causal, **options):
        self.causal, self.options = causal, options
        self.scaled_key_residual = False


class UntouchableQuery:
    def __getattr__(self, name):
        raise AssertionError("Replacing a KV owner must not inspect its query sibling.")


class WidePrecisionPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).parents[1] / "src/kohakufa_cute/precision.py"
        spec = importlib.util.spec_from_file_location("kohakufa_cute._wide_precision_test", path)
        cls.policy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.policy)

    def replace(self, program, dimension, **options):
        modules = {
            "kohakufa_cute.kernels": types.SimpleNamespace(Attention=Owner),
            "kohakufa_cute.medium_backward": types.SimpleNamespace(MediumKeyValue=Replacement),
            "kohakufa_cute.wide_backward": types.SimpleNamespace(WideKeyValue=Replacement),
        }
        with patch.dict(sys.modules, modules):
            return self.policy.schedule(program, dimension, **options)

    def test_all_length_branches_receive_one_shared_kv_replacement(self):
        for dimension in (256, 512):
            with self.subTest(dimension=dimension):
                original, query = Owner(), UntouchableQuery()
                short = types.SimpleNamespace(query=query, keyvalue=original)
                long = types.SimpleNamespace(query=query, keyvalue=original)
                program = types.SimpleNamespace(fallback=short, long=long, checked=short, complete=long)
                actual = self.replace(program, dimension)
                self.assertIs(actual, program)
                self.assertIs(short.query, query)
                self.assertIs(long.query, query)
                self.assertIs(short.keyvalue, long.keyvalue)
                self.assertIsInstance(short.keyvalue, Replacement)
                self.assertTrue(short.keyvalue.causal)
                self.assertEqual(short.keyvalue.options, {"vector": 8, "copy_cache_global": True})

    def test_masked_and_custom_owners_preserve_their_qualified_paths(self):
        class Custom(Owner):
            pass

        for dimension in (256, 512):
            for owner in (Owner(masked=True), Custom()):
                with self.subTest(dimension=dimension, owner=type(owner).__name__):
                    self.assertIs(self.replace(owner, dimension), owner)

    def test_unsupported_width_stops_before_replacing_ownership(self):
        for dimension in (0, 32, 96, 257, 513):
            with self.subTest(dimension=dimension), self.assertRaises(ValueError):
                self.policy.schedule(Owner(), dimension)

    def test_packed_wide_selection_uses_the_explicit_compile_device(self):
        capabilities = {0: (8, 9), 1: (12, 0)}
        names = {0: "NVIDIA L40S", 1: "NVIDIA GeForce RTX 5090"}
        with (
            patch("torch.cuda.get_device_capability", side_effect=lambda device: capabilities[device]),
            patch("torch.cuda.get_device_name", side_effect=lambda device: names[device]),
        ):
            packed = self.replace(Owner(), 512, device=1)
            scalar = self.replace(Owner(), 512, device=0)
        self.assertTrue(packed.options["packed_bf16_causal"])
        self.assertNotIn("packed_bf16_causal", scalar.options)

    def test_forward_normalizer_and_medium_scale_share_the_precision_boundary(self):
        capabilities = {0: (8, 9), 1: (12, 0)}
        names = {0: "NVIDIA L40S", 1: "NVIDIA GeForce RTX 5090"}
        with (
            patch("torch.cuda.get_device_capability", side_effect=lambda device: capabilities[device]),
            patch("torch.cuda.get_device_name", side_effect=lambda device: names[device]),
        ):
            for device in (0, 1):
                fallback, short, query = Owner("forward"), object(), UntouchableQuery()
                wrapper = types.SimpleNamespace(fallback=fallback, short=short, query=query)
                actual = self.replace(wrapper, 256, device=device)
                self.assertIs(actual, wrapper)
                self.assertIs(actual.fallback, fallback)
                self.assertIs(actual.short, short)
                self.assertIs(actual.query, query)
                self.assertEqual(fallback.compensated_normalizer, device == 1)
                keyvalue = self.replace(Owner(), 256, device=device)
                self.assertEqual(keyvalue.scaled_key_residual, device == 1)

    def test_full_cta_barriers_stay_outside_partial_thread_guards(self):
        # The wide kernel's 128 score producers share a CTA with 128 other
        # gradient consumers. Every thread must reach each full-CTA barrier.
        source = Path(__file__).parents[1] / "src/kohakufa_cute"
        for filename in ("medium_backward.py", "wide_backward.py"):
            tree = ast.parse((source / filename).read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.If) or not any(
                    isinstance(value, ast.Name) and value.id == "tid" for value in ast.walk(node.test)
                ):
                    continue
                barriers = [
                    value
                    for child in (*node.body, *node.orelse)
                    for value in ast.walk(child)
                    if isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute)
                    and value.func.attr == "sync_threads"
                ]
                with self.subTest(filename=filename, line=node.lineno):
                    self.assertFalse(barriers, "A full-CTA barrier is nested inside a partial-thread branch.")


if __name__ == "__main__":
    unittest.main()
