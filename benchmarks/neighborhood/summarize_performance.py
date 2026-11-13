"""Sanitize private performance reports into reproducible comparison tables.

Inputs are explicit LABEL=PATH pairs. Publication output contains labels,
repository-relative source references, hashes and compact metrics, never input
paths, device UUIDs, raw kernel symbols or exception messages. Raw reports remain
outside the repository. Comparisons stay within one report/process and preserve
phase, graph policy, workload, input seed and source/runtime configuration.
"""

import argparse
import collections
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path, PurePosixPath

BACKENDS = ("kohakufa", "flex", "natten")
PHASES = ("forward", "forward_backward")
MODES = ("eager", "graph")
LAYOUT = "contiguous BHSD Q/K/V/output/dQ/dK/dV"


def finite(value, *, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else True)


def public_token(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.+ -]+", value):
        raise ValueError("Unsupported public metadata token.")
    return value


def artifact_name(value):
    name = Path(value).name if isinstance(value, str) else ""
    return name if re.fullmatch(r"[A-Za-z0-9_.-]+", name) else None


def distribution(values):
    if not values or not all(finite(value, positive=True) for value in values):
        raise ValueError("Timing samples must be finite and positive.")
    return dict(
        median=statistics.median(values),
        minimum=min(values),
        maximum=max(values),
        stdev=statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def source_metadata(report):
    revision = report.get("source_revision", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Expected a full source revision.")
    hashes = {}
    for name, digest in report.get("source_sha256", {}).items():
        path = PurePosixPath(name)
        allowed = str(path).startswith(("src/kohakufa_cute/", "benchmarks/", "docs/"))
        if path.is_absolute() or ".." in path.parts or not allowed or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Expected repository-relative source hashes.")
        hashes[name] = digest
    if not hashes:
        raise ValueError("Source hashes are required.")
    return dict(
        base_revision=revision,
        source_sha256=hashes,
        versions={public_token(name): public_token(value) for name, value in report.get("versions", {}).items()},
    )


def kernel_families(profile):
    names = profile.get("launched_cuda_kernels", []) if isinstance(profile, dict) else []
    counts = collections.Counter()
    for name in names:
        lowered = str(name).lower()
        if "neighborhoodattention" in lowered:
            family = "candidate attention"
        elif "outputdelta" in lowered:
            family = "candidate delta"
        elif "fna" in lowered:
            family = "NATTEN FNA"
        elif "fmha" in lowered:
            family = "NATTEN FMHA"
        elif "flex_attention" in lowered or "flexattention" in lowered:
            family = "FlexAttention"
        elif "copy" in lowered or "memcpy" in lowered:
            family = "copy"
        elif "fill" in lowered or "memset" in lowered:
            family = "fill"
        else:
            family = "other"
        counts[family] += 1
    return dict(available=bool(names), kernel_events=len(names), families=dict(sorted(counts.items())))


def accuracy_summary(row):
    accuracy = row.get("accuracy") or {}
    errors = accuracy.get("errors") or []
    passed = accuracy.get("passed") is True and bool(errors) and all(error.get("passed") is True for error in errors)
    components = {}
    for component in ("out", "dq", "dk", "dv"):
        selected = [error for error in errors if error.get("component") == component]
        values = {}
        for reference in ("fp64", "rounded"):
            records = [error.get(reference) or {} for error in selected]
            numeric = [record.get("relative_error") for record in records]
            values[reference] = max(numeric) if records and all(finite(value) for value in numeric) else None
        components[component] = dict(
            heads=len(selected),
            failed_heads=sum(error.get("passed") is not True for error in selected),
            worst_head_relative_error=values,
        )
        passed &= bool(selected)
    return dict(passed=passed, positions=accuracy.get("positions", []), components=components)


def workload_metadata(row):
    workload = row["workload"]
    public_token(workload["name"])
    if workload["dtype"] not in ("fp16", "bf16"):
        raise ValueError("Unsupported workload dtype.")
    for key in ("batch", "heads"):
        if type(workload[key]) is not int or workload[key] < 1:
            raise ValueError("Invalid workload dimensions.")
    spatial, window = workload["spatial"], workload["window"]
    if len(spatial) != 2 or len(window) != 2 or any(type(value) is not int or value < 1 for value in (*spatial, *window)):
        raise ValueError("Invalid spatial dimensions.")
    if any(size > extent for size, extent in zip(window, spatial)) or not finite(workload["scale"]):
        raise ValueError("Invalid window or scale.")
    expected_shape = [workload["batch"], workload["heads"], math.prod(spatial), 64]
    if row["shape"] != expected_shape or type(row.get("seed")) is not int:
        raise ValueError("Invalid canonical shape or input seed.")
    digest = hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest()
    if row.get("workload_sha256") != digest:
        raise ValueError("Workload hash mismatch.")
    return dict(**workload, shape=expected_shape, seed=row["seed"], sha256=digest)


def timing_summary(row, key, expected_samples, capture_batch):
    raw = row.get("timings", {}).get(key) or {}
    reasons = []
    if row.get("status") not in ("pass", "partial-failure", "ready"):
        reasons.append("backend qualification or execution failed")
    if not accuracy_summary(row)["passed"]:
        reasons.append("independent accuracy gate unavailable or failed")
    if raw.get("status") != "pass":
        reasons.append("timing incomplete or failed")
    samples = raw.get("samples") or []
    rounds = [sample.get("round") for sample in samples]
    complete = len(samples) == expected_samples and sorted(rounds) == list(range(expected_samples))
    if not complete:
        reasons.append("expected paired rounds missing")
    phase, mode = key.split("/")
    graph_check = row.get("graph_checks", {}).get(phase) or {}
    if mode == "graph" and graph_check.get("passed") is not True:
        reasons.append("changed-input graph check unavailable or failed")
    trace = kernel_families(row.get("profiles", {}).get(key))
    if not trace["available"]:
        reasons.append("actual kernel trace unavailable")
    values = {"device_us": [], "wall_us": []}
    for sample in samples:
        valid = all(finite(sample.get(metric), positive=True) for metric in ("device_ms", "wall_ms"))
        calls = sample.get("complete_calls")
        valid &= type(calls) is int and calls > 0
        if mode == "graph":
            replays = sample.get("graph_replays")
            valid &= type(replays) is int and replays > 0 and calls == replays * capture_batch
        if not valid:
            reasons.append("invalid timing sample or normalization")
            continue
        for source, target in (("device_ms", "device_us"), ("wall_ms", "wall_us")):
            values[target].append(sample[source] * 1000)
    for source, target in (("device_ms", "device_us"), ("wall_ms", "wall_us")):
        recorded = raw.get(source, {}).get("median")
        if values[target] and (
            not finite(recorded, positive=True)
            or not math.isclose(recorded * 1000, statistics.median(values[target]), rel_tol=1e-9)
        ):
            reasons.append("stored median does not match retained samples")
    error_type = row.get("error", {}).get("type") or raw.get("error", {}).get("type")
    result = dict(
        status=raw.get("status", "unavailable"),
        eligible_for_ratio=not reasons,
        exclusions=list(dict.fromkeys(reasons)),
        sample_count=len(samples),
        rounds=rounds,
        kernels=trace,
    )
    if error_type:
        result["error_type"] = public_token(error_type)
    for key, numbers in values.items():
        if numbers:
            result[key] = dict(**distribution(numbers), samples=numbers)
    return result


def prepare_summary(row):
    preparation = row.get("preparation") or {}
    metadata = row.get("backend_metadata") or {}
    result = dict(storage_bytes=metadata.get("preparation_storage_bytes"))
    for key in ("host_ms", "device_interval_ms"):
        samples = [sample.get(key) for sample in preparation.get("samples", [])]
        if samples and all(finite(value, positive=True) for value in samples):
            result[key] = dict(**distribution(samples), sample_count=len(samples))
    for key in ("construction_wall_ms", "first_forward_backward_wall_ms"):
        value = row.get(key)
        if finite(value, positive=True):
            result[key] = value
    selected = metadata.get("selected_backend")
    if isinstance(selected, str):
        result["selected_backend"] = public_token(selected.split(";")[0])
    return result


def hardware_snapshot(report):
    """Whitelist two hardware fields from the first usable CSV observation."""
    for snapshot in report.get("gpu_observations", []):
        fields = snapshot.get("query_fields", [])
        if snapshot.get("exit_status") != 0 or not all(name in fields for name in ("name", "driver_version", "power.limit")):
            continue
        for line in snapshot.get("rows", []):
            values = next(csv.reader([line]))
            if len(values) != len(fields):
                continue
            record = {name: value.strip() for name, value in zip(fields, values)}
            if record["name"] != report["device"]["name"]:
                continue
            driver = record["driver_version"]
            driver = driver if re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", driver) else None
            try:
                power = float(record["power.limit"])
            except ValueError:
                power = None
            if not finite(power, positive=True):
                power = None
            return dict(driver_version=driver, power_limit_watts=power)
    return dict(driver_version=None, power_limit_watts=None)


def summarize_report(label, report, report_sha256):
    public_token(label)
    protocol = report["protocol"]
    if protocol.get("canonical_layout") != LAYOUT:
        raise ValueError("Canonical API layout contract differs.")
    count, batch = protocol["samples"], protocol["capture_batch"]
    if type(count) is not int or count < 3 or type(batch) is not int or batch < 1:
        raise ValueError("Invalid timing protocol.")
    configuration = source_metadata(report)
    configuration["timing"] = dict(
        canonical_layout=LAYOUT,
        samples=count,
        capture_batch=batch,
        target_ms=protocol["target_ms"],
        fp16_limit=protocol["fp16_limit"],
        bf16_limit=protocol["bf16_limit"],
    )
    configuration_id = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    device = report["device"]
    runs = dict(
        label=label,
        raw_report_sha256=report_sha256,
        raw_report_status=public_token(report["status"]),
        configuration_sha256=configuration_id,
        device=dict(
            name=public_token(device["name"]),
            capability=device["capability"],
            sm_count=device["sm_count"],
            memory_bytes=device["memory_bytes"],
            **hardware_snapshot(report),
        ),
        cases=[],
    )
    grouped = collections.defaultdict(dict)
    for row in report["rows"]:
        if row["backend"] not in BACKENDS:
            raise ValueError("Unknown backend in comparison report.")
        workload = workload_metadata(row)
        key = workload["sha256"], workload["seed"]
        if row["backend"] in grouped[key]:
            raise ValueError("Ambiguous duplicate workload/backend rows.")
        grouped[key][row["backend"]] = row
    for rows in grouped.values():
        workload = workload_metadata(next(iter(rows.values())))
        case = dict(workload=workload, backends={}, comparisons={})
        for name in BACKENDS:
            row = rows.get(name)
            if row is None:
                case["backends"][name] = dict(status="unavailable", accuracy=dict(passed=False), timings={})
                continue
            timing = {
                phase + "/" + mode: timing_summary(row, phase + "/" + mode, count, batch)
                for phase in PHASES
                for mode in MODES
            }
            case["backends"][name] = dict(
                status=public_token(row["status"]),
                accuracy=accuracy_summary(row),
                preparation=prepare_summary(row),
                timings=timing,
            )
            for key in ("failed_operands",):
                name_only = artifact_name(row.get(key))
                if name_only:
                    case["backends"][name][key] = name_only
        for phase in PHASES:
            for mode in MODES:
                key = phase + "/" + mode
                ours = case["backends"]["kohakufa"]["timings"].get(key, {})
                comparisons = {}
                for other in ("flex", "natten"):
                    baseline = case["backends"][other]["timings"].get(key, {})
                    eligible = ours.get("eligible_for_ratio") is True and baseline.get("eligible_for_ratio") is True
                    result = dict(eligible=eligible)
                    if eligible:
                        if ours["rounds"] != baseline["rounds"]:
                            raise ValueError("Cannot pair different sample rounds.")
                        paired = [a / b for a, b in zip(baseline["device_us"]["samples"], ours["device_us"]["samples"])]
                        result.update(
                            baseline_over_kohakufa=baseline["device_us"]["median"] / ours["device_us"]["median"],
                            paired_round_ratios=dict(**distribution(paired), samples=paired),
                        )
                    comparisons[other] = result
                case["comparisons"][key] = comparisons
        runs["cases"].append(case)
    return configuration_id, configuration, runs


def summarize_inputs(inputs):
    configurations, runs = {}, []
    seen_labels = set()
    for label, data in inputs:
        if label in seen_labels:
            raise ValueError("Input labels must be unique.")
        seen_labels.add(label)
        digest = hashlib.sha256(data).hexdigest()
        identifier, config, run = summarize_report(label, json.loads(data), digest)
        configurations[identifier] = config
        runs.append(run)
    return dict(
        schema_version=1,
        scope="Complete contiguous BHSD D64 neighborhood API; resident GPU inputs, geometry preparation reported separately. No complete-model or equal-internal-precision claim.",
        timing_boundary="Device-event interval includes GPU work and launch gaps; graph measurements amortize each replay over capture_batch complete calls. Synchronized host wall intervals are retained separately.",
        ratio_direction="baseline_over_kohakufa > 1 means lower candidate latency; < 1 means lower baseline latency. No aggregation across workloads or GPUs.",
        ratio_requirements="Same report/process, source/runtime, workload hash/seed, phase/mode, full paired rounds, independent accuracy pass, changed-input graph pass and actual kernel traces.",
        configurations=configurations,
        runs=runs,
    )


def markdown(report):
    lines = [
        "# Neighborhood attention performance",
        "",
        report["scope"],
        "",
        report["timing_boundary"],
        "",
        "Latencies are median device intervals in microseconds, followed by sample range. Flex / ours and NATTEN / ours greater than one mean lower candidate latency. Failures and missing verification are excluded from ratios. Cold setup and preparation are retained separately in JSON.",
        "",
    ]
    for phase in PHASES:
        for mode in MODES:
            key = phase + "/" + mode
            lines += [
                "## " + phase.replace("_", " ") + " / " + mode,
                "",
                "<details>",
                "<summary>Complete timing rows</summary>",
                "",
                "| GPU | Workload | Ours, µs | Flex, µs | NATTEN, µs | Flex / ours | NATTEN / ours |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
            for run in report["runs"]:
                for case in run["cases"]:
                    workload = case["workload"]
                    caption = f"{workload['spatial'][0]}×{workload['spatial'][1]} w{workload['window'][0]}×{workload['window'][1]} B{workload['batch']}H{workload['heads']} {workload['dtype']}"
                    cells = []
                    for name in BACKENDS:
                        backend = case["backends"][name]
                        timing = backend["timings"].get(key, {})
                        values = timing.get("device_us")
                        if timing.get("status") == "pass" and values:
                            suffix = " †" if not timing["eligible_for_ratio"] else ""
                            cells.append(f"{values['median']:.2f} [{values['minimum']:.2f}–{values['maximum']:.2f}]{suffix}")
                        else:
                            cells.append(timing.get("status", backend["status"]))
                    ratios = [
                        (
                            f"{case['comparisons'][key][name]['baseline_over_kohakufa']:.3f}×"
                            if case["comparisons"][key][name]["eligible"]
                            else "—"
                        )
                        for name in ("flex", "natten")
                    ]
                    lines.append("| " + " | ".join([run["label"], caption, *cells, *ratios]) + " |")
            lines += ["", "</details>", ""]
    lines += [
        "† A retained timing lacks required qualification, graph validation, paired rounds, or a kernel trace; inspect its exclusion list in JSON. No ratio is reported for it.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    inputs = []
    for item in args.input:
        if "=" not in item:
            parser.error("Each input must be LABEL=PATH.")
        label, name = item.split("=", 1)
        inputs.append((label, Path(name).read_bytes()))
    result = summarize_inputs(inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(result))


if __name__ == "__main__":
    main()
