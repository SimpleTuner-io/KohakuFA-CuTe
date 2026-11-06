"""Rebuild the published per-target catalog from its versioned source reports.

This migration reads the frozen pre-cleanup tree without checking it out. New
measurements should update the target report directly using the documented schema.
"""

import argparse
import hashlib
import json
import re
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
    "h200": "H200",
    "b200": "B200",
    "b300": "B300",
}


def source(name, revision=SOURCE_REVISION):
    revision = subprocess.check_output(["git", "rev-parse", "--verify", f"{revision}^{{commit}}"], text=True).strip()
    path = f"results/{name}.json"
    data = subprocess.check_output(["git", "show", f"{revision}:{path}"])
    return json.loads(data), {"revision": revision, "path": path, "sha256": hashlib.sha256(data).hexdigest()}


def dtype_name(value):
    return {"bfloat16": "bf16", "float16": "fp16"}.get(value.removeprefix("torch."), value)


def measurement(row, backend, *, phase=None, boundary=None, workload="training", run="primary"):
    """Normalize units and boundaries while retaining the original timing record."""
    phase = phase or row.get("phase", "forward")
    boundary = boundary or row.get("execution", row.get("mode"))
    metric = "cuda_event_ms" if boundary == "graph" else "wall_ms"
    medians = row.get("median_ms", row.get("medians_ms", row.get("medians")))
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
    reference = next(
        (
            name
            for name in (row.get("reference_backend"), "sdpa", "torch_sdpa", "fa3", "fa2", "gluon", "preceding_control")
            if name and name in row["median_ms"] and name != row["backend"]
        ),
        None,
    )
    reference = "sdpa" if reference == "torch_sdpa" else reference
    return (
        tuple(row["shape"]),
        row["dtype"],
        row["visibility"],
        row["phase"],
        row["boundary"],
        row["workload"],
        row.get("series", "current"),
        reference,
    )


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
            "schema_version": 3,
            "target": target,
            "label": TARGETS[target],
            "precision_contract": "docs/precision.md",
            "selection": "Latest retained study for each measured workload, evidence series and reference; distinct evaluation runs are preserved.",
            "series_contract": "current uses docs/precision.md; historical is previous arithmetic or hardware; captured is historical model-input forward only; experimental retains unresolved qualification limits.",
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
        # A newer SDPA study can also contain the latest FA2 comparison. Refresh
        # already retained reference families from that same-process record,
        # rather than leaving an older FA2 schedule beside the newer SDPA row.
        expanded = []
        for row, raw in rows:
            expanded.append((row, raw))
            key = case_key(row)
            previous_references = {case_key(old)[-1] for old in self.data["measurements"] if case_key(old)[:-1] == key[:-1]}
            for reference in sorted(ref for ref in previous_references if ref and ref != key[-1]):
                actual = "torch_sdpa" if reference == "sdpa" and "sdpa" not in row["median_ms"] else reference
                if actual in row["median_ms"]:
                    expanded.append((dict(row, reference_backend=actual), raw))
        rows = expanded
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
        if name in self.data["studies"]:
            previous = self.data["studies"][name]
            assert previous["provenance"] == provenance
            metadata = previous["metadata"]
            for path in table_paths:
                parts = path.split("/")
                parent = metadata
                for part in parts[:-1]:
                    parent = parent[int(part)] if isinstance(parent, list) else parent[part]
                parent[parts[-1]] = "See retained records."
        else:
            self.data["studies"][name] = {"provenance": provenance, "metadata": metadata, "records": []}
        records = self.data["studies"][name]["records"]
        seen = {json.dumps(raw, sort_keys=True): i for i, raw in enumerate(records)}
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

    def backend_grid(
        self, name, backend, predicate=None, *, path="measurements", workload="training", series=None, label=None
    ):
        report, provenance = source(name)
        rows = []
        raw_rows = report
        for key in path.split("/"):
            raw_rows = raw_rows[key]
        for raw in raw_rows:
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
                normalized = measurement(row, backend, boundary="graph", workload=workload)
                if series:
                    normalized.update(series=series, series_label=label)
                rows.append((normalized, raw))
        self.add_rows(name, report, provenance, rows, [path])

    def legacy(
        self,
        name,
        *,
        revision=SOURCE_REVISION,
        path="rows",
        series="historical",
        label=None,
        workload="training",
        study_id=None,
    ):
        report, provenance = source(name, revision)
        table = report
        for key in path.split("/"):
            table = table[int(key)] if isinstance(table, list) else table[key]
        rows = []
        for raw in table:
            medians = raw.get("medians_ms", raw.get("medians"))
            for legacy_phase, phase in (
                ("forward_ms", "forward"),
                ("backward_ms", "backward"),
                ("forward_backward_ms", "joint"),
            ):
                values = {
                    backend: times[legacy_phase]
                    for backend, times in medians.items()
                    if isinstance(times, dict) and times.get(legacy_phase) is not None
                }
                if "cute" not in values:
                    continue
                row = dict(
                    raw,
                    phase=phase,
                    dtype=raw.get("dtype", report.get("dtype", "bf16")),
                    visibility=raw.get("mask", "causal" if report.get("causal") else "dense"),
                    median_ms=values,
                )
                normalized = measurement(row, "cute", boundary="graph", workload=workload)
                normalized.update(series=series, series_label=label)
                if "kv_shape" in raw:
                    normalized["kv_shape"] = raw["kv_shape"]
                rows.append((normalized, raw))
        metadata, metadata_path = report, path
        if path.startswith("datasets/"):
            dataset = report["datasets"][int(path.split("/")[1])]
            metadata = {"dataset": dataset, "timing": report.get("timing"), "source_report_status": report.get("status")}
            metadata_path = "dataset/rows"
        self.add_rows(study_id or name, metadata, provenance, rows, [metadata_path])

    def hopper(self):
        report, provenance = source("h200-experimental-summary")
        rows = []
        for raw in report["experiments"]["native_host"]["rows"]:
            for phase, timing in raw["backends"]["candidate"].items():
                if timing.get("status") != "pass":
                    continue
                values = {
                    backend: v[phase]["median_ms"]
                    for backend, v in raw["backends"].items()
                    if v.get(phase, {}).get("status") == "pass"
                }
                row = dict(raw, phase=phase, visibility=raw["mode"], median_ms=values)
                normalized = measurement(row, "candidate", boundary="graph")
                normalized.update(series="experimental", series_label="H200 NVL · experimental", reference_backend="fa3")
                rows.append((normalized, raw))
        for raw in report["complete_eager"]["rows"]:
            row = dict(raw, shape=[1, 16, 4096, raw["dimension"]], median_ms=raw["medians"])
            normalized = measurement(row, "runner", boundary="eager")
            normalized.update(series="experimental", series_label="H200 NVL · experimental", reference_backend="fa3")
            rows.append((normalized, raw))
        self.add_rows(
            "h200-experimental-summary", report, provenance, rows, ["experiments/native_host/rows", "complete_eager/rows"]
        )
        self.data["qualification_status"] = "Experimental; unmodified initialization checking remains unresolved."

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
        c.add(
            "a100-explicit-inference",
            table("boundaries/graph/rows", boundary="graph", workload="inference")
            + table("boundaries/eager/rows", boundary="eager", workload="inference"),
            "public",
        )
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
        c.add("rtx4090-native-metadata", table("no_grad_rows", workload="inference"), "native")
        c.add("rtx4090-short-optimized", table("no_grad_rows", workload="inference"), "selected")
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
        c.add("l40s-short-optimized", table("primary_rows") + table("no_grad_controls", workload="inference"), "public")
        c.add("l40s-medium-causal", table("heldout/rows") + table("nograd/rows", workload="inference"), "public")
    elif target == "gb10":
        # The old gb10.json uses the former compensated arithmetic and cannot
        # qualify current backward. Only current-policy studies enter this file.
        c.backend_grid("gb10-query-ownership", "selected")
        c.add("gb10-shared-summary", table(), "public")
        c.add("gb10-device-cache-summary", table(), "cached")
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
    elif target == "h200":
        c.hopper()
    elif target == "b200":
        c.backend_grid("b200-wide-backward", "current_default")
    # Older arithmetic and hardware variants are evidence, not requalification
    # of current dispatch. Keep them visible in explicitly separate series.
    if target == "gb10":
        for path in ("short_sequence_rows", "long_sequence_rows"):
            c.legacy("gb10", path=path, label="GB10 · previous arithmetic")
    if target in ("b200", "b300"):
        for dtype in ("bf16", "fp16"):
            for causal in (False, True):
                name = f"{target}-speed-{dtype}" + ("-causal" if causal else "")
                c.legacy(
                    name,
                    revision="cce86ea8b7e9cfecf6effb10a458613e90a9fa33^",
                    label=f"{TARGETS[target]} · previous arithmetic",
                )
                environment, provenance = source(
                    f"{target}-environment", revision="cce86ea8b7e9cfecf6effb10a458613e90a9fa33^"
                )
                public_environment = {
                    key: environment[key]
                    for key in (
                        "python",
                        "torch",
                        "triton",
                        "cuda",
                        "cudnn",
                        "device",
                        "capability",
                        "multiprocessors",
                        "total_memory",
                        "shared_per_block_optin",
                        "packages",
                    )
                }
                telemetry = environment.get("nvidia_smi", "")
                driver = re.search(r"Driver Version:\s*([\d.]+)", telemetry)
                limits = sorted({int(value) for value in re.findall(r"\d+W\s*/\s*(\d+)W", telemetry)})
                if driver:
                    public_environment["driver"] = driver[1]
                if limits:
                    public_environment["power_limit_w"] = limits
                c.data["studies"][name]["metadata"]["environment_summary"] = public_environment
                c.data["studies"][name]["metadata"]["environment_provenance"] = provenance
    if target in ("h100", "l40s"):
        report, _ = source("performance")
        for index, dataset in enumerate(report["datasets"]):
            if dataset["id"].startswith(target + "-clef"):
                c.legacy(
                    "performance",
                    path=f"datasets/{index}/rows",
                    series="captured",
                    label=f"{TARGETS[target].replace('PCIe', 'SXM')} · captured model inputs",
                    workload="inference",
                    study_id=dataset["id"],
                )
    if target == "b300":
        report, _ = source("performance")
        index = next(i for i, d in enumerate(report["datasets"]) if d["id"] == "b300-mig")
        c.legacy(
            "performance",
            path=f"datasets/{index}/rows",
            series="historical-mig",
            label="B300 MIG 1g.34gb · 18 SMs · previous arithmetic",
            study_id="b300-mig-historical",
        )
    if target == "h100":
        report, _ = source("performance")
        index = next(i for i, d in enumerate(report["datasets"]) if d["id"] == "h100")
        c.legacy(
            "performance",
            path=f"datasets/{index}/rows",
            label="H100 SXM · previous arithmetic",
            study_id="h100-sxm-historical",
        )
    gaps = []
    if target == "a100":
        path = "docs/a100-support.md"
        evidence = subprocess.check_output(["git", "show", f"{SOURCE_REVISION}:{path}"])
        gaps.append(
            {
                "family": "D512",
                "status": "Aggregate qualification only; no retained per-case timing samples found.",
                "source": {"path": path, "revision": SOURCE_REVISION, "sha256": hashlib.sha256(evidence).hexdigest()},
                "aggregate_matrix": {
                    "cases": 60,
                    "dimensions": [16, 64, 128, 256, 512],
                    "sequences": [128, 1024, 4096],
                    "dtype_limits": {"bf16": 0.01, "fp16": 0.005},
                },
                "timing_status": "Aggregate win counts cannot supply D512 latency values.",
            }
        )
    if target == "a40":
        gaps.append(
            {
                "family": "D512",
                "status": "No retained per-case timing samples found in repository history or local archives.",
            }
        )
    if target == "rtx3090":
        gaps.append(
            {
                "family": "D256/D512",
                "status": "No retained per-case timing samples found for the ordinary 82-SM RTX 3090; Ti results are a different target.",
            }
        )
    if not any(r.get("series", "current") == "current" and r["workload"] == "inference" for r in c.data["measurements"]):
        gaps.append(
            {
                "family": "Current synthetic NoGrad inference",
                "status": "No retained timings; training forward and captured historical model inputs do not qualify this workload.",
            }
        )
    if gaps:
        c.data["coverage_gaps"] = gaps
    return c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Directory for the twelve consolidated reports.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        build(target).finish(args.output / f"{target}.json")


if __name__ == "__main__":
    main()
