"""Export cross-architecture qualification evidence without numerical imports.

Only public, selected metadata is copied. Raw exceptions, commands, machine
paths, process identifiers and complete kernel symbols stay in retained captures.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from benchmarks.summarize_neighborhood_key_precision import (
    COMPONENTS,
    PRIOR,
    TARGET,
    directed_rows,
    metadata,
    sharp_rows,
    timing_rows,
    worst,
)
from benchmarks.summarize_neighborhood_precision import digest

SANITIZERS = ("memcheck", "racecheck", "initcheck", "synccheck")
OWNER_FIELDS = (
    "mode",
    "columns",
    "rows",
    "width",
    "warps",
    "height",
    "image_width",
    "window_h",
    "window_w",
    "reload_center",
    "kv_column_statistics",
    "stage_statistics",
    "baseline_precision",
    "experiment_fp32_probability",
    "experiment_weighted_dq",
    "experiment_normalized_mean",
    "experiment_reconstruct_center",
)


def read_json(path):
    if not path.is_file():
        return None, None
    data = path.read_bytes()
    return json.loads(data), data


def status(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9_-]+", value):
        raise ValueError("Status metadata must contain a simple status identifier.")
    return value


def receipt(path):
    raw, data = read_json(path)
    if raw is None:
        return dict(status="missing")
    result = dict(status=status(raw.get("status", "unavailable")), report_sha256=hashlib.sha256(data).hexdigest())
    for name in ("exit_status", "controller_exit_status", "elapsed_seconds"):
        if name in raw:
            result[name] = raw[name]
    return result


def unittest_summary(directory, stage):
    result = receipt(directory / f"{stage}.receipt.json")
    path = directory / f"{stage}.log"
    if path.is_file():
        text = path.read_text(errors="replace")
        counts, skipped = re.findall(r"Ran (\d+) tests?", text), re.findall(r"skipped=(\d+)", text)
        result["tests"] = int(counts[-1]) if counts else None
        result["skipped"] = int(skipped[-1]) if skipped else 0
        result["log_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def owners(value):
    return {
        label: {name: owner[name] for name in OWNER_FIELDS if name in owner}
        for label, owner in value.items()
        if label in ("program", "query", "keyvalue") and owner is not None
    }


def attestation(path):
    raw, data = read_json(path)
    if raw is None:
        return dict(status="missing", selected_modules=[])
    result = dict(
        status=status(raw["status"]),
        report_sha256=hashlib.sha256(data).hexdigest(),
        target_prefix=raw["target_prefix"],
        selected_modules=[],
        scope="Compiler flags and retained target artifacts; graph execution and numerical checks are separate evidence.",
    )
    for name in ("compiled_records_sha256", "script_sha256"):
        if name in raw:
            result[name] = digest(raw[name])
    for module in raw.get("selected_modules", []):
        result["selected_modules"].append(
            dict(
                record_index=int(module["record_index"]),
                owners=owners(module["owners"]),
                target=module["target"],
                ptx_sha256=digest(module["ptx_sha256"]),
                cubin_sha256=digest(module["cubin_sha256"]),
                ptx_evidence=[
                    {
                        name: entry[name]
                        for name in ("mma_opcodes", "mma_sites", "cp_async_4byte_sites", "cp_async_16byte_sites")
                    }
                    for entry in module.get("ptx_evidence", [])
                ],
            )
        )
    return result


def bootstrap_summary(path):
    if path is None:
        return dict(status="not-provided")
    if path.is_dir():
        path = path / "runtime-preflight.json"
    raw, data = read_json(path)
    if raw is None:
        return dict(status="missing")
    result = dict(status=status(raw["status"]), report_sha256=hashlib.sha256(data).hexdigest())
    for name in ("device", "capability", "sm_count", "memory_bytes", "cuda", "packages"):
        if name in raw:
            result[name] = raw[name]
    if "python" in raw:
        result["python_version"] = raw["python"].split()[0]
    result["preflight_commands"] = {
        name: dict(exit_status=command["exit_status"]) for name, command in raw.get("commands", {}).items()
    }
    return result


def phase_summary(directory):
    raw, data = read_json(directory / "phase-multigpu-qualification.json")
    if raw is None:
        return dict(status="missing")
    result = dict(status=status(raw["status"]), report_sha256=hashlib.sha256(data).hexdigest())
    for name in ("qualification_status", "timing_status", "evidence_status", "control_status"):
        if name in raw:
            result[name] = status(raw[name])
    for name in ("source_manifest_sha256", "source_revision"):
        if name in raw:
            result[name] = digest(raw[name]) if name.endswith("sha256") else raw[name]
    for name in (
        "device",
        "target_prefix",
        "selected_directed",
        "sanitizers",
        "unittest_scope",
        "unclassified_control_failures",
    ):
        if name in raw:
            result[name] = raw[name]
    result["stages"] = {
        name: {
            key: value[key] for key in ("status", "exit_status", "controller_exit_status", "receipt_sha256") if key in value
        }
        for name, value in raw.get("stages", {}).items()
    }
    return result


def expected_hashes(previous):
    records = {}
    for row in previous["directed"]:
        index, sha = int(row["fixture_index"]), digest(row["operand_sha256"])
        if index in records and records[index] != sha:
            raise ValueError("Historical input identities disagree.")
        records[index] = sha
    if set(records) != set(range(29)):
        raise ValueError("All 29 historical input identities are required.")
    return records


def qualification(rows, hashes):
    selected = [row for row in rows if row["implementation"] == TARGET]
    identity_checks = [
        dict(
            fixture_index=index,
            observed_rows=len([row for row in rows if row["fixture_index"] == index]),
            expected_sha256=sha,
            matched=bool(any(row["fixture_index"] == index for row in rows))
            and all(
                row["operand_sha256"] == sha and row["historical_operand_verified"]
                for row in rows
                if row["fixture_index"] == index
            ),
        )
        for index, sha in hashes.items()
    ]
    required_ok = all(
        row.get("math_gate", {}).get("required_components")
        == (["out", "dk", "dv"] if row["fixture"] in ("zero-dp-sharp", "constant-dp-sharp") else list(COMPONENTS))
        for row in selected
    )
    formula = sum(row.get("formula", {}).get("passed", False) for row in selected)
    math_required = sum(row.get("math_gate", {}).get("passed", False) for row in selected)
    invariants = sum(row.get("paired_invariants", {}).get("passed", False) for row in selected)
    complete = len(selected) == 29 and {row["fixture_index"] for row in selected} == set(range(29))
    zero = [row for row in selected if row["fixture"] in ("zero-dp-sharp", "constant-dp-sharp")]
    return dict(
        status=(
            "pass"
            if complete
            and all(item["matched"] for item in identity_checks)
            and required_ok
            and formula == math_required == invariants == 29
            and all(row["status"] == "pass" for row in selected)
            else "failed"
        ),
        expected_cases=29,
        observed_selected_cases=len(selected),
        complete=complete,
        input_identity_checks=identity_checks,
        all_29_inputs_match_previous_h100=all(item["matched"] for item in identity_checks),
        required_component_contract_preserved=required_ok,
        formula_passed=formula,
        required_math_passed=math_required,
        paired_invariants_passed=invariants,
        observed_all_four_math_components_passed=sum(
            bool(row.get("math_gate", {}).get("errors")) and all(record["passed"] for record in row["math_gate"]["errors"])
            for row in selected
        ),
        zero_constant_controls=len(zero),
        exact_zero_dk_controls=sum(
            bool(row.get("per_head_fp64"))
            and all(
                item["reference_norm"] == 0 and item["max_absolute_error"] == 0
                for item in row["per_head_fp64"]
                if item["component"] == "dk"
            )
            for row in zero
        ),
    )


def error_summary(rows):
    selected = [row for row in rows if row["implementation"] == TARGET]
    return {
        dtype: {
            family: worst(
                [
                    record
                    for row in selected
                    if row["dtype"] == dtype and (family == "all_observed" or row["fixture_index"] < 21)
                    for record in row.get("per_head_fp64", [])
                ]
            )
            for family in ("historical_21", "all_observed")
        }
        for dtype in sorted({row["dtype"] for row in selected})
    }


def summarize_node(directory, bootstrap, hashes):
    result = dict(phase=phase_summary(directory), bootstrap=bootstrap_summary(bootstrap))
    raw, data = read_json(directory / "directed/report.json")
    if raw is None:
        result.update(directed_metadata=dict(status="missing"), directed=[], qualification=dict(status="missing"))
    else:
        rows = directed_rows(raw)
        result.update(
            directed_metadata=metadata(raw, data, completed=False),
            directed=rows,
            qualification=qualification(rows, hashes),
            worst_selected_math_errors=error_summary(rows),
        )
        complete_groups = {
            row["fixture_index"]
            for row in rows
            if row["implementation"] == TARGET
            and "per_head_fp64" in row
            and any(
                other["fixture_index"] == row["fixture_index"]
                and other["implementation"] == PRIOR
                and "per_head_fp64" in other
                for other in rows
            )
        }
        result["sharp_matched_heads"] = sharp_rows(
            [row for row in rows if row["fixture_index"] in complete_groups and "per_head_fp64" in row]
        )
    raw, data = read_json(directory / "timing/report.json")
    if raw is None:
        result.update(timing_metadata=dict(status="missing"), timing=[])
    else:
        result.update(timing_metadata=metadata(raw, data, completed=False), timing=timing_rows(raw))
        result["numerical_and_timing_source_identical"] = result.get("directed_metadata", {}).get("source") == result[
            "timing_metadata"
        ].get("source")
    result["checks"] = {
        name: unittest_summary(directory, name) for name in ("guard-tests", "cpu-tests", "portable-gpu-tests", *SANITIZERS)
    }
    result["compiled_selection"] = {
        name: attestation(directory / f"attestation-{name}.json")
        for name in ("directed", "portable-gpu-tests", *SANITIZERS, "timing")
    }
    result["row_status_counts"] = dict(Counter(row["status"] for row in result["directed"]))
    return result


def percentage(value):
    return "unavailable" if value is None else f"{value * 100:.5f}%"


def hardware_scope(name, node):
    timing = node.get("timing_metadata", {})
    device = timing.get("device", {})
    bootstrap = node.get("bootstrap", {})
    capability = device.get("capability", bootstrap.get("capability"))
    sm_count = device.get("sm_count", bootstrap.get("sm_count"))
    memory = device.get("memory_bytes", bootstrap.get("memory_bytes"))
    observations = [
        record
        for metadata_name in ("timing_metadata", "directed_metadata")
        for record in node.get(metadata_name, {}).get("gpu_observations", [])
    ]
    drivers = sorted({record["driver_version"] for record in observations if "driver_version" in record})
    limits = sorted({record["power.limit"] for record in observations if "power.limit" in record})
    power = "unavailable" if not limits else f"{limits[0]:g}" if len(limits) == 1 else f"{limits[0]:g}–{limits[-1]:g}"
    return [
        device.get("name", bootstrap.get("device", name)),
        ".".join(str(value) for value in capability) if capability else "unavailable",
        str(sm_count) if sm_count is not None else "unavailable",
        f"{memory / 1024**3:.1f}" if memory is not None else "unavailable",
        ", ".join(drivers) if drivers else "unavailable",
        power,
    ]


def markdown(report):
    lines = [
        "# Neighborhood precision across GPUs",
        "",
        "The explicit dQ+dK experiment uses the same arithmetic and retained rounded inputs on every architecture. Default dispatch is unchanged.",
        "",
        "The [H100 dQ+dK study](neighborhood-key-precision.md) specifies the experiment and its explicit selection controls.",
        "",
        "The mathematical reference uses native FP64 on identical rounded Q/K/V/dO. The independent rounded-arithmetic reference uses the actual rounded forward output; it does not emulate tensor-core reduction order.",
        "",
        "The 21 historical fixtures gate output, dQ, dK and dV independently. The eight zero/constant-dP controls retain their declared output/dK/dV mathematical gates; dQ formula agreement and observed FP64 error remain reported. Limits are 0.5% for FP16 and 1% for BF16 per head and component.",
        "",
        "Measurements apply to the actual device configurations and recorded power limits below.",
        "",
        "| GPU | Compute capability | SMs | VRAM (GiB) | Driver | Power limit (W) |",
        "| --- | --- | ---: | ---: | --- | ---: |",
    ]
    for name, node in report["nodes"].items():
        lines.append("| " + " | ".join(hardware_scope(name, node)) + " |")
    lines += [
        "",
        "| GPU | Saved-input qualification | GPU methods | Sanitizers | Complete qualification |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, node in report["nodes"].items():
        gpu = node["checks"]["portable-gpu-tests"]
        san = sum(
            node["checks"][tool]["status"] == "pass"
            and node["checks"][tool].get("tests") == 2
            and node["checks"][tool].get("skipped") == 0
            and node["compiled_selection"][tool]["status"] == "pass"
            for tool in SANITIZERS
        )
        gpu_scope = gpu["status"] == "pass" and gpu.get("tests") == 15 and gpu.get("skipped") == 0
        presentation_qualified = (
            node["phase"].get("qualification_status") == "pass"
            and node["qualification"]["status"] == "pass"
            and gpu_scope
            and san == 4
        )
        complete_status = "pass" if presentation_qualified else node["phase"].get("qualification_status", "unavailable")
        if complete_status == "pass" and not presentation_qualified:
            complete_status = "scope-not-verified"
        lines.append(
            f"| {name} | {node['qualification']['status']} | {gpu.get('tests', 'unavailable')} ({gpu['status']}; skipped={gpu.get('skipped', 'unavailable')}) | {san}/4 | {complete_status} |"
        )
    lines += [
        "",
        "Worst per-head relative norm errors on the 21 historical fixtures:",
        "",
        "| GPU | Dtype | Output | dQ | dK | dV |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, node in report["nodes"].items():
        for dtype, errors in node.get("worst_selected_math_errors", {}).items():
            values = [percentage(errors["historical_21"][component]) for component in COMPONENTS]
            lines.append("| " + " | ".join([name, dtype.removeprefix("torch."), *values]) + " |")
    lines += [
        "",
        "Secondary resident forward-plus-backward CUDA graph timings follow. They include the complete advertised call, including required backend layout copies. Compilation and immutable geometry preparation are outside this boundary. Missing or failed rows do not establish a speedup.",
        "",
        "Baseline, prior dQ, combined dQ+dK, FlexAttention and NATTEN use different precision policies. These are complete-implementation comparisons, rather than isolated scheduling comparisons under identical arithmetic.",
        "",
        "| GPU | Workload | Implementation | Status | Median (µs) | Change vs prior dQ |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for name, node in report["nodes"].items():
        for row in node["timing"]:
            measured = (
                f"{row['device_ms']['median'] * 1000:.3f}"
                if row["status"] == "pass" and "device_ms" in row
                else "unavailable"
            )
            comparison = row.get("comparisons", {}).get(PRIOR, {}).get("device_ms", {}).get("incremental_percent")
            changed = f"{comparison:+.3f}%" if comparison is not None else "unavailable"
            lines.append(
                f"| {name} | {row['workload']['name']} | {row['implementation']} | {row['status']} | {measured} | {changed} |"
            )
    lines += [
        "",
        "All per-head errors, declared gates, observed control failures, source and input hashes, raw timing samples and compiled-path flags are retained in the companion JSON. Compiler artifacts establish target and selected flags; graph replay and numerical tests supply separate execution evidence. Results qualify these operator workloads and do not establish model-level quality or performance on other shapes.",
        "",
    ]
    return "\n".join(
        lines[:2]
        + [
            "Historical method labels below describe the recorded campaign. The combined dQ+dK method with column-staged statistics is now the public neighborhood default; see [the current contract](../precision.md#neighborhood-default).",
            "",
        ]
        + lines[2:]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", nargs=2, action="append", required=True, metavar=("NAME", "RESULTS"))
    parser.add_argument("--bootstrap", nargs=2, action="append", default=[], metavar=("NAME", "PREFLIGHT"))
    parser.add_argument("--previous-report", type=Path, default=Path("docs/experimental/neighborhood-key-precision.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    names = [name for name, _ in args.node]
    if len(names) != len(set(names)) or any(not re.fullmatch(r"[a-z0-9-]+", name) for name in names):
        parser.error("Node names must be unique simple identifiers.")
    bootstrap = {name: Path(path) for name, path in args.bootstrap}
    if len(bootstrap) != len(args.bootstrap) or set(bootstrap) - set(names):
        parser.error("Bootstrap records must refer to unique selected nodes.")
    previous, data = read_json(args.previous_report)
    if previous is None:
        parser.error("The historical H100 report is required.")
    hashes = expected_hashes(previous)
    result = dict(
        schema="neighborhood-multigpu-precision-v1",
        reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        default_changed=False,
        selected_method=TARGET,
        selected_statistics="column_staged",
        previous_report_sha256=hashlib.sha256(data).hexdigest(),
        references=dict(
            mathematical="Native FP64, identical retained rounded inputs; no emulated FP64.",
            formula="Independent rounded equations using actual rounded O; not an instruction-order emulator.",
        ),
        acceptance=dict(
            fp16=0.005,
            bf16=0.01,
            historical_fixtures=list(COMPONENTS),
            zero_constant_controls=["out", "dk", "dv"],
            dQ_on_zero_constant_controls="Formula gate and observed mathematical metrics retained.",
        ),
        nodes={name: summarize_node(Path(path), bootstrap.get(name), hashes) for name, path in args.node},
    )
    for path, content in (
        (args.output, json.dumps(result, indent=2, allow_nan=False) + "\n"),
        (args.markdown, markdown(result)),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


if __name__ == "__main__":
    main()
