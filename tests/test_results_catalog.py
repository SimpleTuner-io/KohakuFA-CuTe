"""Protect published timings, cohort selection, and chart coverage during curation."""

import copy
import json
import math
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from benchmarks.plot_performance import TARGETS, graph_cells, render

ROOT = Path(__file__).resolve().parents[1]


class ResultsCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reports = {target: json.loads((ROOT / "results" / f"{target}.json").read_text()) for target in TARGETS}

    def test_one_report_per_target_and_valid_record_ownership(self):
        self.assertEqual({p.stem for p in (ROOT / "results").glob("*.json")}, set(TARGETS))
        for target, report in self.reports.items():
            with self.subTest(target=target):
                self.assertEqual(report["schema_version"], 2)
                self.assertEqual(report["target"], target)
                self.assertEqual(report["precision_contract"], "docs/precision.md")
                self.assertTrue(report["measurements"])
                seen, owners, referenced = set(), {}, set()
                for row in report["measurements"]:
                    case = (
                        tuple(row["shape"]),
                        row["dtype"],
                        row["visibility"],
                        row["phase"],
                        row["boundary"],
                        row["workload"],
                    )
                    self.assertNotIn((*case, row["run"]), seen)
                    seen.add((*case, row["run"]))
                    self.assertEqual(owners.setdefault(case, row["study"]), row["study"])
                    self.assertIn(row["dtype"], ("bf16", "fp16"))
                    self.assertIn(row["phase"], ("forward", "backward", "joint"))
                    self.assertIn(row["boundary"], ("graph", "eager"))
                    study = report["studies"][row["study"]]
                    self.assertGreaterEqual(row["record"], 0)
                    self.assertLess(row["record"], len(study["records"]))
                    referenced.add((row["study"], row["record"]))
                expected = {(name, i) for name, study in report["studies"].items() for i in range(len(study["records"]))}
                self.assertEqual(referenced, expected, "Superseded raw records must not linger in the catalog")

    def test_normalized_medians_match_original_raw_records(self):
        for target, report in self.reports.items():
            for row in report["measurements"]:
                raw = report["studies"][row["study"]]["records"][row["record"]]
                for backend, value in row["median_ms"].items():
                    with self.subTest(target=target, study=row["study"], phase=row["phase"], backend=backend):
                        if "backends" in raw:
                            original = raw["backends"][backend]
                            if row["boundary"] + "_ms" in original:
                                expected = original[row["boundary"] + "_ms"][row["phase"]]
                            else:
                                self.assertEqual(original[row["phase"]]["status"], "pass")
                                expected = original[row["phase"]]["median_ms"]
                        else:
                            original = raw.get("median_ms", raw.get("medians_ms"))[backend]
                            if isinstance(original, dict):
                                expected = original.get(row["metric"], original.get(row["phase"]))
                            else:
                                expected = original
                        self.assertEqual(value, expected, "Curation must not alter timing values or mix metrics")
                        self.assertTrue(math.isfinite(value) and value > 0)

    def test_latest_qualified_families_replace_older_cells(self):
        cases = (
            ("rtx4090", [1, 16, 1024, 64], "bf16", "causal", "rtx4090-s1024-host"),
            ("rtx5090", [1, 16, 1024, 256], "bf16", "causal", "rtx5090-d256-short-groups"),
            ("a100", [1, 8, 4096, 256], "bf16", "dense", "a100-d256-shared-v"),
            ("a100", [1, 8, 2048, 128], "bf16", "causal", "a100-medium-causal-g4"),
            ("l40s", [1, 16, 1024, 64], "fp16", "causal", "l40s-medium-causal"),
        )
        for target, shape, dtype, visibility, study in cases:
            with self.subTest(target=target, study=study):
                rows = [
                    r
                    for r in self.reports[target]["measurements"]
                    if r["shape"] == shape
                    and r["dtype"] == dtype
                    and r["visibility"] == visibility
                    and r["phase"] == "joint"
                ]
                self.assertTrue(rows)
                self.assertEqual({r["study"] for r in rows}, {study})
        self.assertNotIn(
            "gb10", self.reports["gb10"]["studies"], "The former precision policy is not a current backward result"
        )

    def test_chart_covers_every_target_and_preserves_workload_boundaries(self):
        for report in self.reports.values():
            cells = graph_cells(report)
            self.assertTrue(cells)
            for key, (ratio, row) in cells.items():
                self.assertEqual(row["boundary"], "graph")
                self.assertEqual(key[-1], row["workload"])
                reference = row["median_ms"].get("sdpa", row["median_ms"].get("torch_sdpa"))
                self.assertEqual(ratio, reference / row["median_ms"][row["backend"]])
        svg = ET.fromstring(render(list(self.reports.values())))
        links = {element.attrib["href"] for element in svg.iter() if "href" in element.attrib}
        self.assertEqual(links, {target + ".json" for target in TARGETS})

    def test_chart_does_not_invent_sdpa_or_average_independent_runs(self):
        report = copy.deepcopy(self.reports["rtx5090"])
        cells = graph_cells(report)
        multi = next((key, row) for key, (_, row) in cells.items() if row["run"] != "primary")
        key, selected = multi
        matching = [r for r in report["measurements"] if r["study"] == selected["study"] and r["run"] != selected["run"]]
        self.assertTrue(matching)
        for row in matching:
            row["median_ms"][row["backend"]] *= 100
        self.assertEqual(graph_cells(report)[key][0], cells[key][0])
        report["measurements"] = [copy.deepcopy(selected)]
        del report["measurements"][0]["median_ms"]["sdpa"]
        self.assertEqual(graph_cells(report), {})


if __name__ == "__main__":
    unittest.main()
