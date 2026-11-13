"""Summarize retained neighborhood errors without changing their references.

Input reports contain already measured baseline results. This utility performs
no new attention calculation and makes no inference about model-level quality.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

COMPONENTS = ("out", "dq", "dk", "dv")
BACKENDS = ("kohakufa", "flex", "natten")
LABELS = {"kohakufa": "Candidate", "flex": "FlexAttention", "natten": "NATTEN"}


def measured_error(backend, component, reference="fp64"):
    value = backend["accuracy"]["components"][component]["worst_head_relative_error"][reference]
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("Expected a finite measured relative error.")
    return value


def summarize(performance, qualification):
    maxima = {}
    paired = {name: {component: [] for component in COMPONENTS} for name in ("flex", "natten")}
    representative = None
    cases = 0
    for run in performance["runs"]:
        for case in run["cases"]:
            cases += 1
            workload = case["workload"]
            dtype = workload["dtype"]
            for name in BACKENDS:
                backend = case["backends"][name]
                if backend["status"] != "pass" or not backend["accuracy"]["passed"]:
                    raise ValueError("Failed or unqualified results cannot establish an accuracy comparison.")
                entry = maxima.setdefault(dtype, {}).setdefault(name, {})
                for component in COMPONENTS:
                    error = measured_error(backend, component)
                    if component not in entry or error > entry[component]["relative_l2_error"]:
                        entry[component] = dict(relative_l2_error=error, device=run["label"], workload=workload["name"])
            for name in ("flex", "natten"):
                for component in COMPONENTS:
                    ours = measured_error(case["backends"]["kohakufa"], component)
                    baseline = measured_error(case["backends"][name], component)
                    if baseline == 0:
                        raise ValueError("A zero-error baseline needs a separate absolute-error comparison.")
                    paired[name][component].append(ours / baseline)
            if run["label"] == "h100" and workload["name"] == "64sq-w7-bf16":
                representative = dict(
                    device="h100",
                    workload=workload,
                    errors={
                        name: {
                            reference: {
                                component: measured_error(case["backends"][name], component, reference)
                                for component in COMPONENTS
                            }
                            for reference in ("fp64", "rounded")
                        }
                        for name in BACKENDS
                    },
                )
    if representative is None or not cases:
        raise ValueError("The specified representative case and a nonempty matrix are required.")
    ranges = {
        name: {
            component: dict(
                cases=len(values),
                minimum=min(values),
                maximum=max(values),
                candidate_lower_error_cases=sum(value < 1 for value in values),
                candidate_higher_error_cases=sum(value > 1 for value in values),
                equal_error_cases=sum(value == 1 for value in values),
            )
            for component, values in components.items()
        }
        for name, components in paired.items()
    }
    directed = []
    groups = qualification["grouped_worst_head_errors"]
    for dtype in ("torch.float16", "torch.bfloat16"):
        selected = {
            group["implementation"]: group
            for group in groups
            if group["case"] == "common-offset" and group["scale"] == 0.125 and group["dtype"] == dtype
        }
        ours, baseline = [selected[name]["worst_head_relative_error"] for name in ("kohakufa", "natten")]
        values = {
            reference: {
                name: record[reference]["dq"]["value"] for name, record in (("kohakufa", ours), ("natten", baseline))
            }
            for reference in ("fp64", "rounded")
        }
        directed.append(
            dict(
                dtype=dtype,
                component="dq",
                errors=values,
                mathematical_relative_error_reduction=(values["fp64"]["natten"] - values["fp64"]["kohakufa"])
                / values["fp64"]["natten"],
            )
        )
    return dict(
        scope="Retained baseline neighborhood operator accuracy; mathematical FP64 and rounded arithmetic are separate references.",
        metric="Per-batch/head relative L2 norm over sampled token positions and all 64 channels; worst head retained per case. Initial directed fixtures evaluate complete small tensors.",
        reference="Native torch.float64 arithmetic on identical already-rounded FP16/BF16 operands. Sparse matrix references ran on CPU; initial small H100 qualification also checked CUDA FP64 reference gates.",
        limitations=[
            "FP64 is finite-precision arithmetic, not exact real arithmetic or arbitrary precision.",
            "Converting rounded operands to FP64 does not recover their original pre-quantization values.",
            "Worst-head ratios match device/workload, but the worst head can differ between backends.",
            "Random-matrix maxima are descriptive; their maxima can come from different cases.",
            "No task-level accuracy or universal pointwise/bitwise accuracy claim.",
        ],
        performance_configuration_sha256=sorted(performance["configurations"]),
        routine=dict(
            devices=len(performance["runs"]),
            cases=cases,
            observed_maxima_vs_fp64=maxima,
            candidate_over_comparator_worst_head_error_ratios=ranges,
            representative=representative,
        ),
        directed=dict(
            scope="Earlier H100 PCIe common-offset fixture, NATTEN cutlass-fna. This is not the subsequent Hopper/Blackwell automatic-backend comparison.",
            flex_attention_measured=False,
            spatial=[5, 9],
            window=[3, 3],
            scale=0.125,
            fixtures=directed,
        ),
    )


def summarize_probe(probe, *, label, input_sha256):
    """Retain all diagnostic heads, including comparator numerical failures."""
    if probe["status"] == "running":
        raise ValueError("An unfinished probe cannot establish an accuracy comparison.")
    cases = {}
    for row in probe["rows"]:
        key = str(row["case_index"])
        case = cases.setdefault(
            key,
            {
                name: row[name]
                for name in (
                    "case_index",
                    "case",
                    "dtype",
                    "shape",
                    "spatial",
                    "window",
                    "scale",
                    "seed",
                    "directed",
                    "compatibility_probe",
                    "operand_sha256",
                    "gate_reference",
                )
            }
            | {"backends": {}},
        )
        if row["operand_sha256"] != case["operand_sha256"]:
            raise ValueError("A paired probe must use identical rounded operands.")
        backend = {"status": row["status"], "per_head": row.get("errors", [])}
        if "errors" in row:
            backend["worst_head_relative_error"] = {
                reference: {
                    component: max(
                        (
                            record[reference]["relative_error"]
                            for record in row["errors"]
                            if record["component"] == component and record[reference]["relative_error"] is not None
                        ),
                        default=None,
                    )
                    for component in COMPONENTS
                }
                for reference in ("fp64", "rounded")
            }
        for name in ("underflow_observation", "error_type", "device_reference_errors"):
            if name in row:
                backend[name] = row[name]
        if row.get("status") == "execution-error" and "kernel_size" in row.get("error_message", ""):
            backend["failure_reason"] = "NATTEN requires every window dimension to be at least 2; this fixture uses (1, 3)."
        metadata = row.get("backend_metadata", {})
        backend["selected_backend"] = metadata.get("selected_backend", metadata.get("natten_backend"))
        backend["launched_cuda_kernels"] = row.get("launched_cuda_kernels", [])
        case["backends"][row["implementation"]] = backend
    return {
        "label": label,
        "status": probe["status"],
        "input_sha256": input_sha256,
        "source_revision": probe["source_revision"],
        "source_sha256": probe["source_sha256"],
        "versions": probe["versions"],
        "device": probe["device"],
        "tuning_variant": probe.get(
            "tuning_variant",
            {
                "name": "no_stat_fills",
                "columns": [32, 32, 32],
                "materialize_output_grads": False,
                "provenance": "Standalone probe at default source dispatch; no runtime variant_context override.",
            },
        ),
        "reference": probe["reference"],
        "acceptance": probe["acceptance"],
        "candidate_failures": probe.get("candidate_failures"),
        "execution_errors": probe.get("execution_errors"),
        "comparator_numerical_differences": probe.get("comparator_numerical_differences"),
        "cases": list(cases.values()),
    }


def probe_markdown(probe):
    candidate_rows = [case["backends"]["kohakufa"] for case in probe["cases"] if "kohakufa" in case["backends"]]
    candidate_passes = sum(row["status"] == "pass" for row in candidate_rows)
    lines = [
        "",
        f"## Current native-backend probe: {probe['label']}",
        "",
        "This separate full-tensor diagnostic uses native CPU FP64 equations on",
        "identical already-rounded operands. It measures no latency or model quality.",
        f"Device: {probe['device']['name']}, {probe['device']['sm_count']} SMs. "
        f"Qualified stack: PyTorch {probe['versions']['torch']}, NATTEN {probe['versions']['natten']}.",
        "The retained PyTorch 2.13 stack matches the newest NATTEN prebuilt pairing",
        "and prior baseline; these results do not evaluate PyTorch 2.15 FlexAttention.",
        "See the [NATTEN installation matrix](https://natten.org/install/) and",
        "[current FlexAttention documentation](https://docs.pytorch.org/docs/2.15/nn.attention.flex_attention.html).",
        f"Candidate variant: `{probe['tuning_variant']['name']}`, "
        f"columns `{probe['tuning_variant']['columns'] if probe['tuning_variant']['columns'] is not None else 'automatic per shape'}`, "
        f"materialize output gradients `{probe['tuning_variant']['materialize_output_grads']}`.",
        f"Exact contiguous traversal flags: `{probe['tuning_variant'].get('exact_contiguous', False) if probe['tuning_variant'].get('exact_contiguous', False) is not None else 'automatic per shape'}`.",
        *(
            [
                "These tiny probe tensors have at most 45 tokens and exercise the",
                "production C32 fallback. The bounded H100 KV64 preset requires at",
                "least 1024 tokens and is qualified separately on the timing matrix",
                "and held-out workloads; the production label does not imply KV64",
                "executes for these small directed probes.",
            ]
            if probe["tuning_variant"]["columns"] is None
            else []
        ),
        f"Probe status: `{probe['status']}`; candidate failed rows: {probe['candidate_failures']}; "
        f"execution errors: {probe['execution_errors']}; comparator rounded-policy differences: "
        f"{probe['comparator_numerical_differences']}.",
        *([f"Retained stage exit status: `{probe['stage_receipt']['exit_status']}`."] if "stage_receipt" in probe else []),
        "",
        f"{candidate_passes}/{len(candidate_rows)} candidate rows passed their declared gates. Three NATTEN rows",
        "could not execute because its public API rejects a (1, 3) window; these",
        "two uniform-boundary cases and one underflow case remain explicit errors.",
        "Negative-scale rows are compatibility probes; NATTEN produced nonfinite",
        "results there. Exact-zero references have undefined relative norm error",
        "and are labeled explicitly. The JSON retains absolute error and finite flags.",
        "The overall probe exit status therefore indicates external execution errors,",
        "not a candidate numerical failure. Other comparator numerical differences",
        "measure compatibility with the candidate rounded policy, not universal defects.",
        "",
        "Full-tensor worst-head mathematical FP64 relative L2 error in percent:",
        "",
        "| fixture | dtype | scale | implementation | output | dQ | dK | dV | status |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for case in probe["cases"]:
        for name in BACKENDS:
            backend = case["backends"].get(name)
            if not backend:
                continue
            values = backend.get("worst_head_relative_error", {}).get("fp64", {})
            displayed = []
            for component in COMPONENTS:
                records = [r["fp64"] for r in backend["per_head"] if r["component"] == component]
                if any(not record["finite"] for record in records):
                    displayed.append("nonfinite")
                elif values.get(component) is not None:
                    displayed.append(f"{values[component] * 100:.4f}%")
                elif records and all(record["max_absolute_error"] == 0 for record in records):
                    displayed.append("exact zero")
                else:
                    displayed.append("unavailable")
            errors = " | ".join(displayed)
            lines.append(
                f"| {case['case']} | {case['dtype'].removeprefix('torch.')} | {case['scale']} | "
                f"{LABELS[name]} | {errors} | {backend['status']} |"
            )
    lines += [
        "",
        "The large-offset fixture sets Q[0] = K[0] = 1024, Q[1] = 1 and",
        "K[1] = token index modulo 7. Its raw scores are exactly representable",
        "as FP32 (2^20 plus integer gaps 0 through 6). The shared offset changes",
        "normalization magnitude while preserving the mathematical probabilities.",
        "This is a defined reconstruction diagnostic, not a typical-workload claim.",
        "",
        "Directed rows are gated against the declared rounded equations; random",
        "rows are gated against mathematical FP64. Passing the directed rounded",
        "gate does not imply a small mathematical error. In particular the large",
        "K component can amplify the intrinsic P/dS rounding floor in dQ.",
        "",
        "On this large-offset fixture, candidate FP16 dK/dV errors are approximately",
        "12–18 times smaller than FlexAttention/native NATTEN, and output error is",
        "also lower. BF16 output/dK/dV errors are lower by smaller factors. This",
        "does not extend to dQ: candidate BF16 dQ error is about 194%, compared",
        "with FlexAttention 163% and native NATTEN 213%. Those large errors make",
        "a general mathematical-gradient accuracy claim inappropriate.",
        "",
        "Backend selection matters. On the positive small-offset BF16 fixture,",
        "native Hopper NATTEN dQ error is 10.07%, below the candidate 12.16%.",
        "The earlier 4.1% relative dQ improvement above compared the candidate",
        "with CUTLASS NATTEN, and does not describe this native Hopper result.",
        "The JSON retains both references, every head/component, operand hashes,",
        "actual launched kernels and source hashes, including failed diagnostics.",
    ]
    return lines


def markdown(report):
    representative = report["routine"]["representative"]
    ranges = report["routine"]["candidate_over_comparator_worst_head_error_ratios"]["flex"]
    gradient_errors_larger = all(
        ranges[component]["candidate_higher_error_cases"] == report["routine"]["cases"] for component in ("dq", "dk")
    )
    lines = [
        "# Neighborhood accuracy against FP64",
        "",
        "The retained ordinary-workload results do **not** demonstrate a general",
        f"mathematical-accuracy gain over FlexAttention or NATTEN. On the {report['routine']['cases']} matched",
        "device/workload cases, the candidate's worst-head sampled dQ error was",
        f"{ranges['dq']['minimum']:.3f}–{ranges['dq']['maximum']:.3f} times FlexAttention's, and dK error was "
        f"{ranges['dk']['minimum']:.3f}–{ranges['dk']['maximum']:.3f} times",
        (
            "FlexAttention's. Both gradient errors were larger in every matched case."
            if gradient_errors_larger
            else "FlexAttention's. Gradient error ordering varied across matched cases."
        ),
        "Output and dV errors were close. NATTEN comparisons varied by case.",
        "",
        "These compare each implementation's worst head within the same workload;",
        "the identity of that head can differ. They are descriptive norm-error",
        "comparisons, without statistical or model-quality conclusions.",
        "",
        *(
            [
                "The newer native-backend probes below show a defined large-offset benefit",
                "for output/dK/dV, while dQ remains worse than FlexAttention and can have",
                "very large mathematical error. There is **no overall accuracy winner**.",
                "",
            ]
            if report.get("probes")
            else []
        ),
        "## What the reference means",
        "",
        "`benchmarks/reference.py` and the sparse equations in",
        "`benchmarks/benchmark_neighborhood.py` use real `torch.float64` arithmetic.",
        "The performance matrix used CPU FP64 references. The initial small H100",
        "qualification also required the reference gates with CUDA FP64. No",
        "double-from-two-floats emulation was used. FP64 remains finite precision",
        "and is not exact real arithmetic or an arbitrary-precision proof.",
        "",
        "The operands are the identical **already-rounded FP16/BF16** Q/K/V/dO",
        "values used by each implementation on that GPU. Their conversion to FP64",
        "does not recover pre-quantization values, so these measurements isolate",
        "operator arithmetic from initial input-quantization error.",
        "",
        "The metric is `norm(actual - reference) / norm(reference)` per batch/head,",
        "over sampled boundary/interior token positions and all 64 channels. dK/dV",
        "references include every query contributing to each sampled key. This is",
        "not a universal per-coordinate error bound or an exhaustive large-image",
        "comparison. Separate small directed fixtures check full tensors.",
        "",
        "`rounded_reference` answers a different question: agreement with the",
        "declared P/dS operand conversions, FP32 QK/dP, and delta from actual rounded",
        "O. Its final products use FP64 to isolate operand-rounding effects. Close",
        "agreement with that reference does not imply the closest mathematical",
        "FP64 result. The candidate retains the [declared precision contract](../precision.md).",
        "",
        "## Representative ordinary workload",
        "",
        "H100 SXM, B1/H8/D64, 64 x 64 image, 7 x 7 window, BF16, scale 0.125.",
        "Values are worst-head sampled relative L2 errors **in percent against",
        "mathematical FP64**:",
        "",
        "| implementation | output | dQ | dK | dV |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in BACKENDS:
        values = representative["errors"][name]["fp64"]
        lines.append("| " + LABELS[name] + " | " + " | ".join(f"{values[c] * 100:.4f}%" for c in COMPONENTS) + " |")
    lines += [
        "",
        "NATTEN used its native Hopper automatic/default preset in this matrix.",
        "These complete implementations retain their own arithmetic policies.",
        "Different operand-rounding policies are consistent with these differences;",
        "the results alone do not identify every contributing error source.",
        "",
        "## Maximum observed random-workload errors",
        "",
        f"Maximum worst-head sampled relative L2 error, in percent, across {report['routine']['devices']} GPUs",
        "and the retained geometries per dtype. A different case may produce each",
        "maximum; these maxima do not establish a paired accuracy winner.",
        "",
        "| dtype | implementation | output | dQ | dK | dV |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for dtype in ("fp16", "bf16"):
        for name in BACKENDS:
            values = report["routine"]["observed_maxima_vs_fp64"][dtype][name]
            lines.append(
                "| "
                + dtype
                + " | "
                + LABELS[name]
                + " | "
                + " | ".join(f"{values[c]['relative_l2_error'] * 100:.4f}%" for c in COMPONENTS)
                + " |"
            )
    lines += [
        "",
        "## Directed common-offset fixture",
        "",
        "The earlier H100 PCIe test used a 5 x 9 image, 3 x 3 window, scale 0.125,",
        "and a large shared score offset. Its NATTEN comparator was explicitly",
        "`cutlass-fna`. FlexAttention was not measured on this fixture, nor were",
        "the later native Hopper/Blackwell NATTEN presets. Full-tensor worst-head",
        "dQ relative errors were:",
        "",
        "| dtype | candidate / FP64 | NATTEN / FP64 | relative error reduction | candidate / rounded | NATTEN / rounded |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fixture in report["directed"]["fixtures"]:
        exact, rounded = [fixture["errors"][kind] for kind in ("fp64", "rounded")]
        lines.append(
            f"| {fixture['dtype'].removeprefix('torch.')} | {exact['kohakufa'] * 100:.4f}% | {exact['natten'] * 100:.4f}% | "
            f"{fixture['mathematical_relative_error_reduction'] * 100:.2f}% | {rounded['kohakufa'] * 100:.4f}% | {rounded['natten'] * 100:.4f}% |"
        )
    lines += [
        "",
        "The mathematical FP64 improvement on this fixture is about 10.5% lower",
        "dQ error in FP16 and 4.1% lower in BF16. The much larger improvement",
        "against the rounded reference must not be described as an equivalent",
        "FP64-accuracy gain. Its BF16 mathematical dQ error is still about 12%,",
        "showing a substantial rounding floor on this cancellation-sensitive case.",
        "",
        "The demonstrated benefit is robust probability reconstruction and closer",
        "agreement with the declared rounded computation on defined difficult",
        "inputs. There is no demonstrated universal accuracy gain over FlexAttention,",
        "and the initial qualification did not measure FlexAttention on this fixture.",
        "The separate current native-backend probe below fills that comparison.",
        "",
        "The [machine-readable audit](neighborhood-accuracy.json) retains source-report",
        "hashes, all component maxima and their locations, matched error-ratio ranges,",
        "and both reference types. The original",
        "[performance summary](neighborhood-performance.json) and",
        "[initial H100 qualification](neighborhood-h100.json) retain the source measurements.",
        "",
        "Regenerate from the repository root:",
        "Set `ACCURACY_REPORT_n` and `ACCURACY_RECEIPT_n` to each retained",
        "off-tree raw report and terminal receipt first.",
        "Raw reports and operands remain outside the repository.",
        "",
        "```sh",
        "python3 -m benchmarks.summarize_neighborhood_accuracy \\",
        "    --performance docs/experimental/neighborhood-performance.json \\",
        "    --qualification docs/experimental/neighborhood-h100.json \\",
        *(
            f'    --probe {probe["label"]} "${{ACCURACY_REPORT_{index}}}" \\'
            for index, probe in enumerate(report.get("probes", []), 1)
        ),
        *(
            f'    --probe-receipt {probe["label"]} "${{ACCURACY_RECEIPT_{index}}}" \\'
            for index, probe in enumerate(report.get("probes", []), 1)
            if "stage_receipt" in probe
        ),
        "    --output docs/experimental/neighborhood-accuracy.json \\",
        "    --markdown docs/experimental/neighborhood-accuracy.md",
        "```",
    ]
    for probe in report.get("probes", []):
        lines.extend(probe_markdown(probe))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--performance", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--probe", nargs=2, action="append", default=[], metavar=("LABEL", "PATH"))
    parser.add_argument("--probe-receipt", nargs=2, action="append", default=[], metavar=("LABEL", "PATH"))
    args = parser.parse_args()
    performance, qualification = [json.loads(path.read_text()) for path in (args.performance, args.qualification)]
    report = summarize(performance, qualification)
    report["input_sha256"] = {
        role: hashlib.sha256(path.read_bytes()).hexdigest()
        for role, path in (("performance", args.performance), ("initial_h100_qualification", args.qualification))
    }
    report["probes"] = [
        summarize_probe(
            json.loads(Path(path).read_text()),
            label=label,
            input_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        )
        for label, path in args.probe
    ]
    for label, path in args.probe_receipt:
        receipt_path = Path(path)
        receipt = json.loads(receipt_path.read_text())
        matching = [probe for probe in report["probes"] if probe["label"] == label]
        if len(matching) != 1:
            raise ValueError("A receipt needs one matching probe label.")
        matching[0]["stage_receipt"] = {
            name: receipt[name]
            for name in ("stage", "job_id", "status", "exit_status", "started_utc", "finished_utc", "elapsed_seconds")
            if name in receipt
        } | {"input_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.markdown:
        args.markdown.write_text(markdown(report))


if __name__ == "__main__":
    main()
