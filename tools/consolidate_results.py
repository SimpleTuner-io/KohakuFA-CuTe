"""Rebuild the published per-target catalog from its versioned source reports.

This migration reads the frozen pre-cleanup tree without checking it out. New
measurements should update the target report directly using the documented schema.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

SOURCE_REVISION = "30704427809008e83ad38174e7c1780016c44258"
TARGETS = {
    "a100": "A100",
    "a40": "A40",
    "rtx3090": "RTX 3090",
    "rtx3090ti": "RTX 3090 Ti",
    "rtx4090": "RTX 4090",
    "rtx5090": "RTX 5090",
    "l40s": "L40S",
    "gb10": "GB10",
    "h100": "H100 PCIe",
    "b200": "B200",
    "b300": "B300",
}


def source(name):
    path = f"results/{name}.json"
    data = subprocess.check_output(["git", "show", f"{SOURCE_REVISION}:{path}"])
    return json.loads(data), {"revision": SOURCE_REVISION, "path": path, "sha256": hashlib.sha256(data).hexdigest()}


def dtype_name(value):
    return {"bfloat16": "bf16", "float16": "fp16"}.get(value.removeprefix("torch."), value)


def measurement(row, backend, *, phase=None, boundary=None, workload="training", run="primary"):
    """Normalize units and boundaries while retaining the original timing record."""
    phase = phase or row.get("phase", "forward")
    boundary = boundary or row.get("execution", row.get("mode"))
    metric = "cuda_event_ms" if boundary == "graph" else "wall_ms"
    medians = row.get("median_ms", row.get("medians_ms"))
    if medians is None:
        raise ValueError("A timing record must have explicit medians.")
    values = {}
    for name, value in medians.items():
        if isinstance(value, dict):
            value = value[metric]
        if value is not None:
            values[name] = value
    if backend not in values:
        raise ValueError(f"Missing selected backend {backend}")
    shape = row.get("shape", [1, 16, row.get("sequence"), row.get("dimension")])
    visibility = row.get("visibility", "causal" if row.get("causal", row.get("family") == "causal") else "dense")
    if row.get("workload") == "no_grad" or row.get("phase") == "no_grad":
        workload, phase = "inference", "forward"
    return {
        "shape": shape,
        "dtype": dtype_name(row["dtype"]),
        "visibility": visibility,
        "phase": phase,
        "boundary": boundary,
        "workload": workload if phase == "forward" else "training",
        "metric": metric,
        "backend": backend,
        "run": str(run),
        "median_ms": values,
    }


def case_key(row):
    return tuple(row["shape"]), row["dtype"], row["visibility"], row["phase"], row["boundary"], row["workload"]


def format_report(data):
    """Keep each raw record and measurement on one independently reviewable line."""

    def compact(value):
        return json.dumps(value, separators=(",", ":"), allow_nan=False)

    lines = ["{"]
    for key, value in data.items():
        if key not in ("studies", "measurements"):
            lines.append(f'  "{key}": {compact(value)},')
    lines.append('  "studies": {')
    for index, (name, study) in enumerate(data["studies"].items()):
        lines.extend([f'    "{name}": {{', f'      "provenance": {compact(study["provenance"])},', '      "metadata": {'])
        metadata = list(study["metadata"].items())
        for i, (key, value) in enumerate(metadata):
            lines.append(f"        {compact(key)}: {compact(value)}" + ("," if i + 1 < len(metadata) else ""))
        lines.extend(["      },", '      "records": ['])
        records = study["records"]
        lines.extend("        " + compact(row) + ("," if i + 1 < len(records) else "") for i, row in enumerate(records))
        lines.extend(["      ]", "    }" + ("," if index + 1 < len(data["studies"]) else "")])
    lines.extend(["  },", '  "measurements": ['])
    rows = data["measurements"]
    lines.extend("    " + compact(row) + ("," if i + 1 < len(rows) else "") for i, row in enumerate(rows))
    lines.extend(["  ]", "}"])
    return "\n".join(lines) + "\n"


class Catalog:
    def __init__(self, target):
        self.data = {
            "schema_version": 2,
            "target": target,
            "label": TARGETS[target],
            "precision_contract": "docs/precision.md",
            "selection": "Latest retained study for each measured workload; distinct evaluation runs are preserved.",
            "studies": {},
            "measurements": [],
        }

    def add(self, name, tables, backend, *, forward_inference=False, predicate=None):
        report, provenance = source(name)
        rows = []
        for path, defaults in tables:
            value = report
            for key in path.split("/"):
                value = value[int(key)] if isinstance(value, list) else value[key]
            for raw in value:
                if predicate and not predicate(raw):
                    continue
                workload = "inference" if forward_inference and raw.get("phase") == "forward" else "training"
                options = dict(workload=workload, **{k: v for k, v in defaults.items() if k != "workload"})
                options["workload"] = defaults.get("workload", workload)
                row = measurement(raw, backend, **options)
                rows.append((row, raw))
            # Move primary timing records out of metadata; retain every record
            # selected below, including samples, errors and resources, exactly.
        self.add_rows(name, report, provenance, rows, [p for p, _ in tables])

    def add_rows(self, name, report, provenance, rows, table_paths=()):
        if not rows:
            raise ValueError(f"No measurements in {name}")
        # Study order is explicit: later studies supersede matching cases.
        replaced = {case_key(row) for row, _ in rows}
        self.data["measurements"] = [row for row in self.data["measurements"] if case_key(row) not in replaced]
        metadata = json.loads(json.dumps(report))
        for path in table_paths:
            parts = path.split("/")
            parent = metadata
            for part in parts[:-1]:
                parent = parent[int(part)] if isinstance(parent, list) else parent[part]
            parent[parts[-1]] = "See retained records."
        self.data["studies"][name] = {"provenance": provenance, "metadata": metadata, "records": []}
        records = self.data["studies"][name]["records"]
        seen = {}
        for row, raw in rows:
            fingerprint = json.dumps(raw, sort_keys=True)
            if fingerprint not in seen:
                seen[fingerprint] = len(records)
                records.append(raw)
            row.update(study=name, record=seen[fingerprint])
            self.data["measurements"].append(row)

    def grid(self, name):
        report, provenance = source(name)
        rows = []
        for raw in report["measurements"]:
            for phase in ("forward", "backward", "joint"):
                row = dict(raw, phase=phase, visibility=raw["mode"])
                row["medians_ms"] = {key: value[phase] for key, value in raw["medians_ms"].items()}
                rows.append((measurement(row, "cute", boundary="graph"), raw))
        self.add_rows(name, report, provenance, rows, ["measurements"])

    def backend_grid(self, name, backend, predicate=None):
        report, provenance = source(name)
        rows = []
        for raw in report["measurements"]:
            if predicate and not predicate(raw):
                continue
            for phase, selected in raw["backends"][backend].items():
                if not isinstance(selected, dict) or selected.get("status") != "pass":
                    continue
                medians = {
                    key: value[phase]["median_ms"]
                    for key, value in raw["backends"].items()
                    if value.get(phase, {}).get("status") == "pass"
                }
                row = dict(raw, phase=phase, median_ms=medians, visibility=raw.get("visibility", raw.get("mode")))
                rows.append((measurement(row, backend, boundary="graph"), raw))
        self.add_rows(name, report, provenance, rows, ["measurements"])

    def finish(self, output):
        # Drop superseded raw timing records, then repair their indices.
        studies = self.data["studies"]
        for name in list(studies):
            active = [row for row in self.data["measurements"] if row["study"] == name]
            if not active:
                del studies[name]
                continue
            retained = sorted({row["record"] for row in active})
            remap = {old: new for new, old in enumerate(retained)}
            studies[name]["records"] = [studies[name]["records"][old] for old in retained]
            for row in active:
                row["record"] = remap[row["record"]]
        output.write_text(format_report(self.data))


def table(path="rows", **defaults):
    return [(path, defaults)]


def build(target):
    c = Catalog(target)
    if target in ("rtx3090ti", "rtx4090", "rtx5090", "l40s", "h100", "b300"):
        c.grid("h100-pcie" if target == "h100" else target)
    if target == "a100":
        c.add("a100-public-host", table(), "candidate_public", forward_inference=True)
        c.add("a100-causal-order", table(), "candidate_public", forward_inference=True)
        c.add(
            "a100-d256-training",
            table(),
            "candidate_public",
            forward_inference=True,
            predicate=lambda r: (r["shape"][1], r["shape"][2]) in ((16, 1024), (8, 4096), (16, 4096)),
        )
        c.add("a100-d256-shared-v", table(), "n64_public", forward_inference=True)
        c.add("a100-medium-causal", table(), "public")
        c.add("a100-medium-causal-g4", table(), "public")
    elif target == "a40":
        c.add("a40-native-predicate", table("causal/rows") + table("dense/rows"), "native_predicate")
        c.add("a40-d256", table(), "public")
        c.add("a40-short-d256", table(), "public")
    elif target == "rtx3090":
        report, provenance = source("rtx3090-optin")
        rows = []
        for raw in report["rows"]:
            for boundary in ("graph", "eager"):
                for phase in ("forward", "backward", "joint"):
                    values = {k: v[f"{boundary}_ms"][phase] for k, v in raw["backends"].items()}
                    row = dict(raw, phase=phase, median_ms=values)
                    rows.append((measurement(row, "public", boundary=boundary), raw))
        c.add_rows("rtx3090-optin", report, provenance, rows, ["rows"])
        c.add("rtx3090-short", table("timing_rows"), "public")
        c.add("rtx3090-short-d64", table("timing_rows"), "public")
        c.add("rtx3090-short-causal-stats", table("timing_rows"), "public")
    elif target == "rtx4090":
        c.add("rtx4090-short-optimized", table("training_rows"), "selected")
        c.add("rtx4090-s1024-host", table("public_rows"), "public")
    elif target == "rtx5090":
        c.add("rtx5090-configured-host", table(), "public-cute")
        c.add("rtx5090-causal-g16", table(), "public-cute")
        paths = table("holdout_runs/0/rows", run="seed-32452843") + table("holdout_runs/1/rows", run="seed-32452867")
        c.add("rtx5090-d256-short-groups", paths, "public-d256")
        paths = []
        report, _ = source("rtx5090-medium-causal")
        for i, run in enumerate(report["holdout_runs"]):
            paths += table(f"holdout_runs/{i}/rows", run=f"seed-{run['seed']}")
        c.add("rtx5090-medium-causal", paths, "public-cute", predicate=lambda r: r.get("selected", False))
    elif target == "l40s":
        c.add("l40s-causal-groups", table(), "public")
        c.add("l40s-short-optimized", table("primary_rows"), "public")
        c.add("l40s-medium-causal", table("heldout/rows"), "public")
    elif target == "gb10":
        # The old gb10.json uses the former compensated arithmetic and cannot
        # qualify current backward. Only current-policy studies enter this file.
        c.backend_grid("gb10-query-ownership", "selected")
        c.add("gb10-shared-summary", table(), "public")
        c.add("gb10-causal-groups-summary", table("studies/1/reports/selected-pair.json/rows"), "selected")
        c.add(
            "gb10-long-causal-summary",
            table("public_pairs/0/rows", run="first") + table("public_pairs/1/rows", run="second"),
            "selected",
        )
        c.add("gb10-forward-geometry", table(), "selected")
        c.add(
            "gb10-inference",
            table("measurements/graph", boundary="graph", workload="inference")
            + table("measurements/eager", boundary="eager", workload="inference"),
            "selected",
        )
        for name in ("gb10-native-metadata", "gb10-short-forward", "gb10-long-d256", "gb10-long-d64"):
            c.add(
                name,
                table("measurements/graph", boundary="graph", workload="inference")
                + table("measurements/eager", boundary="eager", workload="inference"),
                "selected",
            )
    elif target == "b200":
        c.backend_grid("b200-wide-backward", "current_default")
    return c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Directory for the eleven consolidated reports.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        build(target).finish(args.output / f"{target}.json")


if __name__ == "__main__":
    main()
