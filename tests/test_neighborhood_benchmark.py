"""CPU qualification of sparse reference equations and timing bookkeeping."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from benchmarks.neighborhood.benchmark import (
    assess,
    atomic_save,
    execute,
    paired_order,
    sample_positions,
    sparse_references,
    summarize,
    workloads,
)
from benchmarks.neighborhood.reference import neighborhood_mask
from benchmarks.reference import fp64_reference, rounded_reference


class NeighborhoodBenchmarkTests(unittest.TestCase):
    def test_matrix_preserves_window_dtype_and_tail_boundaries(self):
        matrix = workloads()
        self.assertEqual(len(matrix), 10)
        self.assertEqual(len({row.name for row in matrix}), 10)
        self.assertEqual({row.dtype for row in matrix}, {"fp16", "bf16"})
        self.assertEqual({row.spatial for row in matrix}, {(32, 32), (64, 64), (128, 128), (33, 49)})
        self.assertTrue(any(row.batch == 2 for row in matrix))
        self.assertTrue(any(row.window == (13, 13) for row in matrix))
        self.assertTrue(all(row.shape[-1] == 64 for row in matrix))
        self.assertEqual(len(workloads("quick")), 4)
        with self.assertRaises(ValueError):
            workloads("unknown")

    def test_sparse_references_include_all_inverse_boundary_contributions(self):
        torch.manual_seed(1291)
        for dtype in (torch.float16, torch.bfloat16):
            for spatial, window in (((3, 5), (3, 3)), ((3, 5), (2, 4)), ((1, 5), (1, 3))):
                with self.subTest(dtype=dtype, spatial=spatial, window=window):
                    shape = (1, 2, spatial[0] * spatial[1], 64)
                    inputs = [(torch.randn(shape) * 2).round().div(4).to(dtype) for _ in range(3)]
                    dout = (torch.randn(shape) * 2).round().div(4).to(dtype)
                    mask = neighborhood_mask(spatial, window)
                    positions = sorted({0, shape[2] // 2, shape[2] - 1})
                    for scale in (0.125, -0.125, 0.0):
                        expected = fp64_reference(inputs, dout, mask, scale=scale)
                        output = expected[0].to(dtype)
                        rounded = rounded_reference(inputs, dout, mask, scale=scale, output=output)
                        actual = sparse_references(
                            inputs, dout, output, spatial, window, scale=scale, positions=positions, chunk_size=2
                        )
                        for name, values in (("fp64", expected), ("rounded", rounded)):
                            for got, want in zip(actual[name], values):
                                torch.testing.assert_close(got, want[:, :, positions], atol=1e-10, rtol=1e-10)

    def test_inverse_key_center_receives_all_five_queries(self):
        inputs = [torch.zeros(1, 1, 5, 64, dtype=torch.float16) for _ in range(3)]
        dout = torch.ones_like(inputs[0])
        output = torch.zeros_like(inputs[0])
        reference = sparse_references(inputs, dout, output, (1, 5), (1, 3), positions=[2])
        for policy in ("fp64", "rounded"):
            torch.testing.assert_close(
                reference[policy][3], torch.full((1, 1, 1, 64), 5 / 3, dtype=torch.float64), atol=0.001, rtol=0
            )

    def test_sparse_reference_rejects_empty_or_invalid_selection(self):
        q = torch.zeros(1, 1, 5, 64, dtype=torch.float16)
        for positions in ([], [-1], [5]):
            with self.assertRaises(ValueError):
                sparse_references([q, q, q], q, q, (1, 5), (1, 3), positions=positions)
        self.assertIn(0, sample_positions((33, 49)))
        self.assertIn(33 * 49 - 1, sample_positions((33, 49)))

    def test_accuracy_gates_each_head_and_component_independently(self):
        tensor = torch.ones(1, 2, 3, 64)
        references = dict(positions=[0, 1, 2], fp64=(tensor,) * 4, rounded=(tensor,) * 4)
        actual = [tensor.clone() for _ in range(4)]
        actual[2][0, 1].mul_(1.02)
        rows = assess(actual, references, torch.bfloat16)
        failures = [(row["component"], row["head"]) for row in rows if not row["passed"]]
        self.assertEqual(failures, [("dk", 1)])
        actual[2].fill_(float("nan"))
        self.assertTrue(any(not row["passed"] for row in assess(actual, references, torch.bfloat16)))

    def test_zero_reference_requires_exact_zero(self):
        zero = torch.zeros(1, 1, 1, 64)
        actual = [zero.clone() for _ in range(4)]
        refs = dict(positions=[0], fp64=(zero,) * 4, rounded=(zero,) * 4)
        self.assertTrue(all(row["passed"] for row in assess(actual, refs, torch.float16)))
        actual[0][..., 0] = 1e-20
        self.assertFalse(assess(actual, refs, torch.float16)[0]["passed"])

    def test_forward_backward_includes_canonical_gradient_copies(self):
        inputs = [torch.randn(1, 2, 5, 64, requires_grad=True) for _ in range(3)]
        dout = torch.randn_like(inputs[0])
        outputs = execute(
            lambda q, k, v: (q + k + v).permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3), inputs, dout, True
        )
        self.assertEqual(len(outputs), 4)
        self.assertTrue(all(output.is_contiguous() for output in outputs[1:]))
        for gradient in outputs[1:]:
            torch.testing.assert_close(gradient, dout)

    def test_round_orders_rotate_and_alternate_without_dropping_backends(self):
        names = ["kohakufa", "flex", "natten"]
        orders = [paired_order(names, index) for index in range(6)]
        self.assertTrue(all(set(order) == set(names) for order in orders))
        self.assertEqual({order[0] for order in orders}, set(names))
        self.assertEqual(paired_order([], 0), [])
        self.assertEqual(summarize([1, 2, 3])["median"], 2)
        self.assertGreater(summarize([1, 2, 3])["stdev"], 0)
        with self.assertRaises(ValueError):
            summarize([])

    def test_atomic_report_retains_nonfinite_failure_as_null(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "result.json"
            atomic_save(target, {"finite": False, "error": float("nan"), "status": "failed"})
            self.assertEqual(json.loads(target.read_text()), dict(finite=False, error=None, status="failed"))
            self.assertFalse(target.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
