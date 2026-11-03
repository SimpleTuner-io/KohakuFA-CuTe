"""Protect published timings, cohort selection, and chart coverage during curation."""

import copy
import json
import math
import re
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from benchmarks.plot_performance import TARGETS, comparison_reference, graph_cells, render

ROOT = Path(__file__).resolve().parents[1]


def chart_comparisons(report):
    for series in {r.get("series", "current") for r in report["measurements"]}:
        references = {comparison_reference(r) for r in report["measurements"] if r.get("series", "current") == series}
        for reference in {"sdpa" if r == "torch_sdpa" else r for r in references if r is not None}:
            for key, value in graph_cells(report, series=series, reference=reference).items():
                yield series, reference, key, value


class ResultsCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reports = {target: json.loads((ROOT / "results" / f"{target}.json").read_text()) for target in TARGETS}

    def test_one_report_per_target_and_valid_record_ownership(self):
        self.assertEqual({p.stem for p in (ROOT / "results").glob("*.json")}, set(TARGETS))
        for target, report in self.reports.items():
            with self.subTest(target=target):
                self.assertEqual(report["schema_version"], 3)
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
                        row.get("series", "current"),
                        "sdpa" if comparison_reference(row) == "torch_sdpa" else comparison_reference(row),
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
                            original = raw.get("median_ms", raw.get("medians_ms", raw.get("medians")))[backend]
                            if isinstance(original, dict):
                                phase_key = {
                                    "forward": "forward_ms",
                                    "backward": "backward_ms",
                                    "joint": "forward_backward_ms",
                                }[row["phase"]]
                                expected = original.get(row["metric"], original.get(row["phase"], original.get(phase_key)))
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
                    and r.get("series", "current") == "current"
                ]
                self.assertTrue(rows)
                self.assertEqual({r["study"] for r in rows}, {study})
        self.assertNotIn(
            "gb10",
            {r["study"] for r in self.reports["gb10"]["measurements"] if r.get("series", "current") == "current"},
            "The former precision policy is not a current backward result",
        )
        latest_fa2 = graph_cells(self.reports["gb10"], reference="fa2")
        self.assertEqual(
            latest_fa2[1, 16, 4096, "bf16", "dense", 128, "joint", "training"][1]["study"],
            "gb10-forward-geometry",
        )

    def test_chart_covers_every_target_and_preserves_workload_boundaries(self):
        for report in self.reports.values():
            cells = list(chart_comparisons(report))
            self.assertTrue(cells)
            for _, _, key, (ratio, row) in cells:
                self.assertEqual(row["boundary"], "graph")
                self.assertEqual(key[-1], row["workload"])
                reference = row["median_ms"][comparison_reference(row)]
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

    def test_rendered_chart_shows_every_measurement_once_with_no_placeholders(self):
        svg = ET.fromstring(render(list(self.reports.values())))
        expected = {
            (target, series, reference, key): ratio
            for target, report in self.reports.items()
            for series, reference, key, (ratio, _) in chart_comparisons(report)
        }
        rendered, inference_targets = {}, set()
        for panel in svg.iter():
            if panel.attrib.get("class") != "gpu-panel":
                continue
            target, workload = panel.attrib["data-target"], panel.attrib["data-workload"]
            series, reference = panel.attrib["data-series"], panel.attrib["data-reference"]
            if workload == "inference" and series == "current":
                inference_targets.add(target)
            for cell in panel:
                if cell.attrib.get("class") != "measurement":
                    continue
                key = tuple(json.loads(cell.attrib["data-key"]))
                self.assertEqual(key[-1], workload)
                identity = (target, series, reference, key)
                self.assertNotIn(identity, rendered)
                text = cell.find("{http://www.w3.org/2000/svg}text").text
                rendered[identity] = text
                self.assertEqual(text, f"{expected[identity]:.2f}")
        self.assertEqual(set(rendered), set(expected))
        self.assertEqual(len(rendered), 1275)
        self.assertEqual(inference_targets, {"a100", "a40", "gb10", "rtx4090", "l40s"})
        for target in ("rtx3090ti", "rtx4090", "rtx5090", "l40s"):
            self.assertEqual(
                sum(
                    t == target and series == "current" and ref == "sdpa" and key[-1] == "training"
                    for t, series, ref, key in rendered
                ),
                120,
            )
        texts = [element.text for element in svg.iter() if element.tag.endswith("}text")]
        self.assertNotIn("–", texts)

    def test_sparse_gpu_table_omits_unmeasured_dimensions_and_phases(self):
        report = copy.deepcopy(self.reports["rtx5090"])
        training = next(r for r in report["measurements"] if r["phase"] == "joint" and r["boundary"] == "graph")
        inference = copy.deepcopy(training)
        inference.update(phase="forward", workload="inference", shape=[1, 16, 512, 256])
        report["measurements"] = [training, inference]
        svg = ET.fromstring(render([report]))
        panels = [element for element in svg.iter() if element.attrib.get("class") == "gpu-panel"]
        self.assertEqual(len(panels), 2)
        for panel, expected_dimension, expected_phase in (
            (panels[0], training["shape"][3], "F+B"),
            (panels[1], 256, "I"),
        ):
            texts = [element.text for element in panel if element.tag.endswith("}text")]
            self.assertEqual([s for s in texts if s.startswith("D")], [f"D{expected_dimension}"])
            self.assertEqual([s for s in texts if s in ("F", "I", "F+B")], [expected_phase])
            self.assertEqual(sum(element.attrib.get("class") == "measurement" for element in panel), 1)

    def test_chart_stays_readable_at_github_readme_width(self):
        svg = ET.fromstring(render(list(self.reports.values())))
        width = float(svg.attrib["width"])
        self.assertLessEqual(width, 1000)
        sizes = [float(element.attrib["font-size"]) for element in svg.iter() if "font-size" in element.attrib]
        for element in svg.iter():
            if element.tag.endswith("}style"):
                sizes.extend(float(size) for size in re.findall(r"font-size:\s*([\d.]+)px", element.text))
        self.assertGreaterEqual(min(sizes) * min(1, 830 / width), 12)
        panels = [element for element in svg.iter() if element.attrib.get("class") == "gpu-panel"]
        titles = [panel.find(".//{http://www.w3.org/2000/svg}text") for panel in panels]
        self.assertEqual(len({title.attrib["x"] for title in titles}), 1)
        self.assertTrue(all(float(a.attrib["y"]) < float(b.attrib["y"]) for a, b in zip(titles, titles[1:])))
        previous_section, previous_target = None, -1
        for panel in panels:
            section = panel.attrib["data-series"], panel.attrib["data-workload"]
            target = TARGETS.index(panel.attrib["data-target"])
            if section == previous_section:
                self.assertGreaterEqual(target, previous_target)
            previous_section, previous_target = section, target

    def test_long_gpu_headings_wrap_without_losing_text(self):
        report = copy.deepcopy(self.reports["rtx5090"])
        report["label"] = ("Long GPU name & detailed device policy " * 5).strip()
        svg = ET.fromstring(render([report]))
        panel = next(element for element in svg.iter() if element.attrib.get("class") == "gpu-panel")
        title = panel.find("{http://www.w3.org/2000/svg}a")
        lines = [element.text for element in title]
        self.assertGreater(len(lines), 1)
        self.assertEqual(" ".join(lines), report["label"].strip() + " · vs default SDPA")

    def test_recovered_families_keep_their_actual_precision_hardware_and_reference(self):
        for target, count in (("gb10", 112), ("b200", 64), ("b300", 64), ("h100", 12)):
            with self.subTest(target=target):
                historical = graph_cells(self.reports[target], series="historical")
                self.assertEqual(len(historical), count)
                self.assertTrue(all("previous arithmetic" in row["series_label"] for _, row in historical.values()))
        for target in ("b200", "b300"):
            historical = graph_cells(self.reports[target], series="historical")
            self.assertEqual({key[5] for key in historical}, {64, 128, 256, 512})
            self.assertEqual({key[3] for key in historical}, {"bf16", "fp16"})
        h200 = self.reports["h200"]
        self.assertEqual(graph_cells(h200), {})
        self.assertEqual(len(graph_cells(h200, series="experimental", reference="fa3")), 16)
        self.assertIn("unresolved", h200["qualification_status"])
        mig = graph_cells(self.reports["b300"], series="historical-mig", reference="gluon")
        self.assertEqual(len(mig), 36)
        self.assertTrue(all("18 SMs" in row["series_label"] for _, row in mig.values()))
        self.assertEqual(len(graph_cells(self.reports["h100"], series="captured")), 6)
        for target in ("h100", "l40s"):
            for _, row in graph_cells(self.reports[target], series="captured").values():
                self.assertEqual(row["workload"], "inference")
                raw = self.reports[target]["studies"][row["study"]]["records"][row["record"]]
                self.assertEqual(row["kv_shape"], raw["kv_shape"])

    def test_historical_timings_cannot_replace_current_comparisons(self):
        report = copy.deepcopy(self.reports["gb10"])
        current = {key: ratio for key, (ratio, _) in graph_cells(report).items()}
        for row in report["measurements"]:
            if row.get("series") == "historical":
                row["median_ms"][row["backend"]] *= 100
        self.assertEqual({key: ratio for key, (ratio, _) in graph_cells(report).items()}, current)
        self.assertTrue(graph_cells(report, series="historical"))

    def test_unrecoverable_timings_are_explicit_gaps(self):
        svg = render(list(self.reports.values()))
        self.assertIn("Missing current timing evidence: A100/A40 D512; RTX 3090 D256/D512", svg)
        for target, dimensions in (("a100", {512}), ("a40", {512}), ("rtx3090", {256, 512})):
            with self.subTest(target=target):
                report = self.reports[target]
                self.assertTrue(report["coverage_gaps"])
                self.assertTrue(all(row["shape"][3] not in dimensions for row in report["measurements"]))
        self.assertEqual(self.reports["a100"]["coverage_gaps"][0]["aggregate_matrix"]["cases"], 60)
        for target in ("rtx3090", "rtx3090ti", "rtx5090", "h100", "h200", "b200", "b300"):
            report = self.reports[target]
            self.assertTrue(any(gap["family"] == "Current synthetic NoGrad inference" for gap in report["coverage_gaps"]))
            self.assertFalse(
                any(
                    row["workload"] == "inference" and row.get("series", "current") == "current"
                    for row in report["measurements"]
                )
            )


if __name__ == "__main__":
    unittest.main()
