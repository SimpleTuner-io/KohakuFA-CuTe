"""CPU checks for retained operands and directed accuracy-probe semantics."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from benchmarks.neighborhood_reference import neighborhood_mask
from benchmarks.qualify_neighborhood_accuracy import cases, save, tensor_hash
from benchmarks.reference import fp64_reference, rounded_reference


class NeighborhoodAccuracyProbeTests(unittest.TestCase):
    def test_fixture_matrix_retains_original_common_offset_construction(self):
        fixtures = list(cases("cpu"))
        self.assertEqual(len(fixtures), 11)
        for dtype in (torch.float16, torch.bfloat16):
            common = [fixture for fixture in fixtures if fixture[0] == "common-offset" and fixture[1][0].dtype == dtype]
            self.assertEqual(len(common), 2)
            for _, inputs, gradient, spatial, window, scale, directed, seed in common:
                q, k, v = inputs
                self.assertEqual(spatial, (5, 9))
                self.assertEqual(window, (3, 3))
                self.assertTrue(directed)
                self.assertEqual(seed, 792)
                self.assertTrue(torch.all(q[..., 0] == 64))
                self.assertTrue(torch.all(q[..., 1] == 1))
                self.assertTrue(torch.all(k[..., 0] == (64 if scale > 0 else -64)))
                self.assertEqual(gradient.shape, q.shape)
                self.assertEqual(v.shape, q.shape)

    def test_large_common_offset_preserves_fp32_scores_and_fp64_output(self):
        fixtures = [fixture for fixture in cases("cpu") if fixture[0] == "large-common-offset"]
        self.assertEqual(len(fixtures), 2)
        for _, inputs, gradient, spatial, window, scale, directed, seed in fixtures:
            q, k, v = inputs
            self.assertEqual(seed, 792)
            self.assertTrue(directed)
            scores32 = q.float() @ k.float().transpose(-1, -2)
            scores64 = q.double() @ k.double().transpose(-1, -2)
            torch.testing.assert_close(scores32.double(), scores64, atol=0, rtol=0)
            centered_q = q.clone()
            centered_q[..., 0] = 0
            centered_scores = centered_q.double() @ k.double().transpose(-1, -2)
            torch.testing.assert_close(scores64 - centered_scores, torch.full_like(scores64, 2**20), atol=0, rtol=0)
            mask = neighborhood_mask(spatial, window)
            original = fp64_reference(inputs, gradient, mask, scale=scale)[0]
            centered = fp64_reference([centered_q, k, v], gradient, mask, scale=scale)[0]
            torch.testing.assert_close(original, centered, atol=1e-12, rtol=1e-12)

    def test_probability_underflow_is_different_from_mathematical_fp64(self):
        _, inputs, gradient, spatial, window, scale, directed, _ = next(
            fixture for fixture in cases("cpu") if fixture[0] == "probability-underflow"
        )
        self.assertTrue(directed)
        mask = neighborhood_mask(spatial, window)
        exact = fp64_reference(inputs, gradient, mask, scale=scale)
        rounded = rounded_reference(inputs, gradient, mask, scale=scale, output=torch.zeros_like(inputs[0]))
        self.assertEqual(exact[0].dtype, torch.float64)
        self.assertGreater(float(exact[3][0, 0, 0, 0]), 0)
        self.assertEqual(int(rounded[3][0, 0, 0].count_nonzero()), 0)

    def test_operand_hash_covers_shape_dtype_and_values(self):
        value = torch.arange(16, dtype=torch.bfloat16).reshape(1, 1, 4, 4)
        expected = tensor_hash([value])
        self.assertEqual(tensor_hash([value.clone()]), expected)
        self.assertNotEqual(tensor_hash([value.reshape(1, 1, 2, 8)]), expected)
        self.assertNotEqual(tensor_hash([value.half()]), expected)
        changed = value.clone()
        changed[..., 0, 0] = 1
        self.assertNotEqual(tensor_hash([changed]), expected)

    def test_json_retains_failure_flags_with_nonfinite_metric_as_null(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "accuracy.json"
            save(path, {"finite": False, "relative_error": float("nan")})
            result = json.loads(path.read_text())
        self.assertFalse(result["finite"])
        self.assertIsNone(result["relative_error"])


if __name__ == "__main__":
    unittest.main()
