import types
import unittest
from unittest import mock

import torch

from benchmarks.neighborhood import tune as tuning


class NeighborhoodVariantTests(unittest.TestCase):
    def test_no_fill_bitwise_gate_detects_signed_zero_and_missing_baseline(self):
        baseline = tuple(torch.tensor([0.0, 1.0], dtype=torch.float16) for _ in range(4))
        actual = tuple(tensor.clone() for tensor in baseline)
        equal = tuning.baseline_equivalence(actual, baseline, required=True)
        self.assertTrue(equal["gate_applied"])
        self.assertTrue(equal["all_bitwise_equal"])
        actual[1][0] = -0.0
        changed = tuning.baseline_equivalence(actual, baseline, required=True)
        self.assertTrue(changed["components"]["dq"]["torch_equal"])
        self.assertFalse(changed["components"]["dq"]["bitwise_equal"])
        self.assertFalse(changed["all_bitwise_equal"])
        diagnostic = tuning.baseline_equivalence(actual, baseline, required=False)
        self.assertFalse(diagnostic["gate_applied"])
        missing = tuning.baseline_equivalence(actual, None, required=True)
        self.assertEqual(missing["status"], "unavailable_baseline_not_selected")
        self.assertFalse(missing["gate_applied"])

    def test_exact_candidates_compare_same_column_reference_and_order(self):
        names = ["exactmixed_no_stat_fills", "exact32_no_stat_fills", "baseline", "keyvalue64_no_stat_fills"]
        ordered = tuning.qualification_order(names)
        self.assertEqual(ordered[0], "baseline")
        self.assertLess(ordered.index("keyvalue64_no_stat_fills"), ordered.index("exactmixed_no_stat_fills"))
        values = tuple(torch.tensor([1.0], dtype=torch.bfloat16) for _ in range(4))
        references = {"baseline": tuple(tensor * 2 for tensor in values), "keyvalue64_no_stat_fills": values}
        check = tuning.schedule_equivalence(values, references, "exactmixed_no_stat_fills", names)
        self.assertEqual(check["reference_variant"], "keyvalue64_no_stat_fills")
        self.assertTrue(check["gate_applied"])
        self.assertTrue(check["all_bitwise_equal"])
        check = tuning.schedule_equivalence(values, references, "exactmixed_noq_no_stat_fills", names)
        self.assertEqual(check["reference_variant"], "keyvalue64_no_stat_fills")
        self.assertTrue(check["gate_applied"])
        self.assertTrue(check["all_bitwise_equal"])
        missing = tuning.schedule_equivalence(values, {}, "exactmixed128_no_stat_fills", names)
        self.assertEqual(missing["status"], "unavailable_reference_not_selected")
        self.assertFalse(missing["gate_applied"])

    def test_production_selects_same_column_equivalence_reference(self):
        values = tuple(torch.tensor([1.0], dtype=torch.bfloat16) for _ in range(4))
        references = {"baseline": values, "keyvalue64_no_stat_fills": values}
        for columns, expected in (((32, 32, 32), "baseline"), ((32, 32, 64), "keyvalue64_no_stat_fills")):
            check = tuning.schedule_equivalence(values, references, "production", references, effective={"columns": columns})
            self.assertEqual(check["reference_variant"], expected)
            self.assertTrue(check["all_bitwise_equal"])

    def test_screen_and_full_boundaries(self):
        cases = tuning.selected_workloads("screen")
        self.assertEqual({case.name for case in cases}, tuning.SCREEN_CASES)
        self.assertTrue(all(case.dtype == "bf16" for case in cases))
        self.assertEqual(len(tuning.selected_workloads("full")), 10)
        self.assertNotIn("columns128", tuning.SCREEN_VARIANTS)
        self.assertEqual(len(tuning.SCREEN_VARIANTS), 8)
        with self.assertRaises(ValueError):
            tuning.selected_workloads("screen", ["32sq-w7-fp16"])

    def test_context_wraps_complete_call_and_restores_after_failure(self):
        state = {"columns": (32, 32, 32), "exact": (False, False, False)}

        def setter(columns):
            previous, state["columns"] = state["columns"], columns
            return previous

        def set_exact(enabled):
            previous, state["exact"] = state["exact"], enabled
            return previous

        fake_kernel = types.SimpleNamespace(set_experiment_configuration=setter, set_experiment_exact_contiguous=set_exact)
        fake_ops = types.SimpleNamespace(_MATERIALIZE_OUTPUT_GRADS=False)
        modules = {"kohakufa_cute.neighborhood_kernels": fake_kernel, "kohakufa_cute.neighborhood_ops": fake_ops}
        with mock.patch.dict("sys.modules", modules):
            with tuning.variant_context(tuning.VARIANTS["baseline"]):
                self.assertTrue(fake_ops._MATERIALIZE_OUTPUT_GRADS)
                with self.assertRaisesRegex(RuntimeError, "probe"):
                    with tuning.variant_context(tuning.VARIANTS["exactmixed_no_stat_fills"]):
                        self.assertEqual(state["columns"], (32, 32, 64))
                        self.assertEqual(state["exact"], (True, True, True))
                        self.assertFalse(fake_ops._MATERIALIZE_OUTPUT_GRADS)
                        raise RuntimeError("probe")
                self.assertEqual(state["columns"], (32, 32, 32))
                self.assertEqual(state["exact"], (False, False, False))
                self.assertTrue(fake_ops._MATERIALIZE_OUTPUT_GRADS)
            self.assertFalse(fake_ops._MATERIALIZE_OUTPUT_GRADS)
            self.assertEqual(state["columns"], (32, 32, 32))

            with tuning.variant_context(tuning.VARIANTS["production"]):
                self.assertIsNone(state["columns"])
                self.assertIsNone(state["exact"])
            self.assertEqual(state["columns"], (32, 32, 32))
            self.assertEqual(state["exact"], (False, False, False))

            with tuning.variant_context(tuning.VARIANTS["exactmixed_noq_no_stat_fills"]):
                self.assertEqual(state["columns"], (32, 32, 64))
                self.assertEqual(state["exact"], (True, False, True))
            self.assertEqual(state["exact"], (False, False, False))

            invalid = tuning.Variant("invalid", (64, 64, 64), exact_contiguous=(True, 0, True))
            with self.assertRaises(ValueError):
                with tuning.variant_context(invalid):
                    self.fail("Invalid traversal configuration entered.")
            self.assertEqual(state["columns"], (32, 32, 32))
            self.assertEqual(state["exact"], (False, False, False))

            execution = object.__new__(tuning.VariantExecution)
            execution.variant = tuning.VARIANTS["columns64_no_stat_fills"]
            execution.backend = types.SimpleNamespace(call=lambda *args: None)
            execution.inputs, execution.dout, execution.training = (), None, True

            def complete(function, inputs, dout, training):
                self.assertTrue(training)
                self.assertEqual(state["columns"], (64, 64, 64))
                self.assertFalse(fake_ops._MATERIALIZE_OUTPUT_GRADS)
                return "complete forward and backward"

            with mock.patch.object(tuning.Execution, "eager", autospec=True) as eager:
                eager.side_effect = lambda instance: complete(None, None, None, instance.training)
                self.assertEqual(execution.eager(), "complete forward and backward")
            self.assertEqual(state["columns"], (32, 32, 32))
            self.assertFalse(fake_ops._MATERIALIZE_OUTPUT_GRADS)


if __name__ == "__main__":
    unittest.main()
