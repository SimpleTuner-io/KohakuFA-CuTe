"""Export compact precision evidence using measured JSON and the standard library.

Only explicitly selected public fields are copied. Raw machine captures,
paths, exception messages and kernel addresses remain in the private archive.
This script performs no tensor calculation or new performance measurement.
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path, PurePosixPath

COMPONENTS = ("out", "dq", "dk", "dv")
IMPLEMENTATIONS = ("baseline", "fp32_p", "weighted_dq", "fp32_p_weighted_dq", "legacy_weighted", "flex", "natten")
METRICS = (
    "finite",
    "reference_norm",
    "relative_error",
    "log10_relative_error",
    "norm_ratio",
    "cosine",
    "max_absolute_error",
)
STATISTICS = ("median", "minimum", "maximum", "mean", "stdev")


def scalar(value):
    if value is not None and not isinstance(value, (bool, int, float)):
        raise ValueError("Expected a measured numeric scalar.")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite measurements must be explicitly represented as null.")
    return value


def digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Expected a SHA256 digest.")
    return value


def metric(value):
    return {name: scalar(value[name]) for name in METRICS if name in value}


def per_head(values):
    result = []
    for value in values:
        if value["component"] not in COMPONENTS:
            raise ValueError("Unknown output component.")
        result.append(
            dict(component=value["component"], batch=int(value["batch"]), head=int(value["head"]), **metric(value))
        )
    return result


def worst(values):
    return {
        component: max(
            (
                value["relative_error"]
                for value in values
                if value["component"] == component and value.get("relative_error") is not None
            ),
            default=None,
        )
        for component in COMPONENTS
    }


def gate(value):
    return dict(
        passed=bool(value["passed"]),
        limit=scalar(value["limit"]),
        errors=[
            dict(**error, passed=bool(original["passed"]))
            for original, error in zip(value["errors"], per_head(value["errors"]))
        ],
    )


def backend(value, implementation):
    # Precision selection is explicit; the generic adapter's baseline metadata
    # does not describe the stronger kernel arithmetic selected by its context.
    result = dict(
        selected_backend=value.get("selected_backend", "unavailable"),
        precision_policy=(
            "default docs/precision.md"
            if implementation == "baseline"
            else "backend-native" if implementation in ("flex", "natten") else "explicit stronger arithmetic experiment"
        ),
    )
    for key in (
        "block_size",
        "preparation_storage_bytes",
        "last_preparation_device_ms",
        "last_preparation_host_ms",
        "first_preparation_device_ms",
        "first_preparation_host_ms",
    ):
        if key in value:
            result[key] = scalar(value[key])
    result["geometry_preparation_reused"] = bool(value.get("geometry_preparation_reused", False))
    if implementation == "natten":
        result["layout_copies_in_complete_call"] = True
    return result


def source(value):
    hashes = {}
    for name, value_hash in value["source_sha256"].items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError("Source metadata must use repository-relative paths.")
        hashes[name] = digest(value_hash)
    revision = value["source_revision"]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Expected a full source revision.")
    return dict(base_revision=revision, file_sha256=hashes)


def observations(values):
    allowed = (
        "driver_version",
        "power.limit",
        "power.draw",
        "clocks.sm",
        "clocks.mem",
        "temperature.gpu",
        "utilization.gpu",
        "memory.used",
        "memory.total",
    )
    result = []
    for value in values:
        for text in value.get("rows", []):
            fields = dict(zip(value["query_fields"], next(csv.reader([text], skipinitialspace=True))))
            row = {}
            for key in allowed:
                raw = fields.get(key, "")
                if key == "driver_version":
                    if raw and all(c in "0123456789." for c in raw):
                        row[key] = raw
                else:
                    try:
                        row[key] = scalar(float(raw))
                    except ValueError:
                        pass
            result.append(row)
    return result


def report_metadata(value, data):
    if value["status"] == "running" or any(row["status"] in ("running", "prepared") for row in value["rows"]):
        raise ValueError("An unfinished report cannot become a completed public result.")
    device = value["device"]
    return dict(
        status=value["status"],
        report_sha256=hashlib.sha256(data).hexdigest(),
        source=source(value),
        versions=value["versions"],
        cuda=value["cuda"],
        device={key: device[key] for key in ("name", "capability", "sm_count", "memory_bytes")},
        gpu_observations=observations(value.get("gpu_observations", [])),
        status_counts=dict(Counter(row["status"] for row in value["rows"])),
    )


def failure(value):
    # Keep the occurrence and supported-case classification, never the raw
    # exception string, traceback, host name, or filesystem destination.
    result = dict(status=value["status"])
    if value["implementation"] == "natten" and 1 in value.get("window", ()):
        result["classification"] = "NATTEN public kernel_size API requires both window dimensions >= 2"
    elif value["status"] == "execution-error":
        result["classification"] = "execution error; retained in the private raw report"
    else:
        result["classification"] = "numerical qualification failure; unchanged acceptance criteria"
    return result


def summarize_directed(value):
    result = []
    for original in value["rows"]:
        name = original["implementation"]
        if name not in IMPLEMENTATIONS:
            raise ValueError("Unknown implementation.")
        row = {
            key: original[key]
            for key in (
                "fixture",
                "fixture_index",
                "implementation",
                "dtype",
                "spatial",
                "window",
                "scale",
                "directed",
                "seed",
                "status",
            )
        }
        row["operand_sha256"] = digest(original["operand_sha256"])
        if "fp64" in original:
            row["per_head_fp64"] = per_head(original["fp64"])
            row["worst_head_relative_l2"] = worst(row["per_head_fp64"])
            row["backend"] = backend(original["backend_metadata"], name)
            row["nonfinite_components"] = [
                {key: record[key] for key in ("component", "batch", "head")}
                for record in row["per_head_fp64"]
                if not record["finite"]
            ]
            if row["nonfinite_components"]:
                row["diagnostic_failure"] = dict(
                    classification=(
                        "Nonfinite comparator output/gradients on a negative-scale compatibility fixture"
                        if original["scale"] < 0 and name in ("flex", "natten")
                        else "Nonfinite output or gradient"
                    )
                )
        if "formula" in original:
            row["formula_gate"] = gate(original["formula"])
            row["paired_invariants"] = {key: original["paired_invariants"][key] for key in ("passed", "exact_components")}
        if "ordinary_fp64_gate" in original:
            row["ordinary_fp64_gate"] = gate(original["ordinary_fp64_gate"])
        if "counterfactual_dq" in original:
            row["counterfactual_dq"] = {
                policy: [dict(batch=int(record["batch"]), head=int(record["head"]), **metric(record)) for record in records]
                for policy, records in original["counterfactual_dq"].items()
            }
        if original["status"] not in ("pass", "diagnostic-complete"):
            row["failure"] = failure(original)
        result.append(row)
    pairs = {}
    for row in result:
        key = row["fixture_index"]
        if key in pairs and pairs[key] != row["operand_sha256"]:
            raise ValueError("A paired fixture must use identical rounded operands.")
        pairs[key] = row["operand_sha256"]
    return result


def summarize_timing(value):
    result = []
    for original in value["rows"]:
        row = {key: original[key] for key in ("workload", "implementation", "status")}
        row["operand_sha256"] = digest(original["operand_sha256"])
        if "backend_metadata" in original:
            row["backend"] = backend(original["backend_metadata"], original["implementation"])
        if "accuracy" in original:
            row["per_head_fp64"] = [
                dict(
                    component=record["component"],
                    batch=int(record["batch"]),
                    head=int(record["head"]),
                    passed=bool(record["passed"]),
                    **metric(record["fp64"]),
                )
                for record in original["accuracy"]
            ]
            row["per_head_default_policy_diagnostic"] = [
                dict(
                    component=record["component"],
                    batch=int(record["batch"]),
                    head=int(record["head"]),
                    **metric(record["rounded"]),
                )
                for record in original["accuracy"]
            ]
            row["worst_head_relative_l2"] = worst(row["per_head_fp64"])
            row["reference_positions"] = original["reference_positions"]
        if "graph_check" in original:
            row["graph_check"] = dict(
                passed=bool(original["graph_check"]["passed"]),
                changed_output=bool(original["graph_check"]["changed_output"]),
                errors=per_head(original["graph_check"]["errors"]),
            )
        if "profile" in original:
            names = original["profile"].get("launched_cuda_kernels", [])
            # Names can embed compiler object addresses. Public evidence keeps
            # family counts; exact symbols and loaded binaries stay archived.
            row["profile"] = dict(
                complete_calls=original["profile"]["complete_calls"],
                total_cuda_launches=len(names),
                kernel_family_counts=dict(
                    Counter(
                        (
                            "neighborhood"
                            if "Neighborhood" in name
                            else (
                                "delta"
                                if "Delta" in name
                                else (
                                    "triton"
                                    if "triton" in name.lower()
                                    else "libnatten" if "natten" in name.lower() or "fmha" in name.lower() else "other"
                                )
                            )
                        )
                        for name in names
                    )
                ),
            )
        if "device_ms" in original:
            row["device_us"] = {key: scalar(original["device_ms"][key]) * 1000 for key in STATISTICS}
            row["wall_us"] = {key: scalar(original["wall_ms"][key]) * 1000 for key in STATISTICS}
            row["samples"] = [
                {key: scalar(sample[key]) for key in ("round", "device_ms", "wall_ms", "graph_replays", "complete_calls")}
                for sample in original["samples"]
            ]
        if original["status"] != "pass":
            row["failure"] = failure(original)
        result.append(row)
    for row in result:
        matched = next(
            (
                baseline
                for baseline in result
                if baseline["workload"]["name"] == row["workload"]["name"] and baseline["implementation"] == "baseline"
            ),
            None,
        )
        if matched is None or matched["operand_sha256"] != row["operand_sha256"]:
            raise ValueError("Timing comparisons require a matched baseline and identical rounded operands.")
        if row["status"] == matched["status"] == "pass":
            row["candidate_over_baseline_latency"] = row["device_us"]["median"] / matched["device_us"]["median"]
            row["baseline_over_candidate_speed"] = 1 / row["candidate_over_baseline_latency"]
            baseline_rounds = {sample["round"]: sample["device_ms"] for sample in matched["samples"]}
            ratios = [sample["device_ms"] / baseline_rounds[sample["round"]] for sample in row["samples"]]
            row["paired_round_latency_ratios"] = dict(
                samples=ratios, median=statistics.median(ratios), minimum=min(ratios), maximum=max(ratios)
            )
    return result


def summarize(directed_data, timing_data):
    directed, timing = [json.loads(data) for data in (directed_data, timing_data)]
    if (
        directed["source_sha256"] != timing["source_sha256"]
        or directed["versions"] != timing["versions"]
        or directed["device"] != timing["device"]
    ):
        raise ValueError("Directed and timing evidence must use the same source, stack and device.")
    result = dict(
        schema="neighborhood-precision-v1",
        default_changed=False,
        scope="Explicit experimental neighborhood backward arithmetic; operator results, no HDiT model or task-level quality evaluation.",
        references=dict(
            mathematical="Native torch.float64 on identical already-rounded operands, excluding pre-quantization input error.",
            formula="Independent FP32 QK/dP/actual-O delta and prescribed P/dS casts, final FP64 products; not a tensor-core reduction-order emulator.",
            small="Complete small tensors, independently enumerated shifted neighborhoods and all output channels.",
            timing="Boundary/interior query and inverse key owners at recorded positions, all heads and 64 channels; not complete large-tensor norm error.",
            counterfactual_delta="ideal_delta uses the ideal FP64-derived delta rounded to FP32 before dS, with the other declared operand boundaries retained.",
            output_cast_floor="Round mathematical FP64 dQ directly to its output dtype; excludes input quantization and additional kernel arithmetic error.",
        ),
        methods=dict(
            baseline="P_low then dS_low, delta from actual rounded O; docs/precision.md.",
            fp32_p="P remains FP32 in dS; dS rounds once, dV still uses P_low.",
            weighted_dq="dQ=s[(dS_low@K)-sum(dS_low)*(P_low@K)/sum(P_low)]; baseline dK/dV.",
            fp32_p_weighted_dq="Combine FP32 P through dS with the normalized weighted-key dQ correction.",
            legacy_weighted="Explicit optional unnormalized projection control; no historical complete-policy equivalence claimed.",
            comparators="Native arithmetic and backend dispatch; NATTEN BHSD/BHWD copies included in complete calls.",
        ),
        acceptance=dict(
            fp16=0.005,
            bf16=0.01,
            rule="Separate per-head norm gates for each output and gradient. Directed mathematical errors remain diagnostics; prescribed-equation gates are retained unchanged.",
            zero_reference="Relative error is null; absolute error and finiteness are retained.",
        ),
        timing_protocol={key: timing["protocol"][key] for key in ("timing", "samples", "target_ms", "capture_batch")},
        directed_metadata=report_metadata(directed, directed_data),
        timing_metadata=report_metadata(timing, timing_data),
        directed=summarize_directed(directed),
        timing=summarize_timing(timing),
    )
    result["mathematical_limitations"] = sharp_key_limitations(result["directed"])
    return result


def sharp_key_limitations(rows):
    """Expose sharp-attention dK errors despite prescribed-equation agreement.

    Select the combined variant's worst dK head once per signed fixture, then
    retain that identical head for every comparator. A small reference norm
    never converts a large mathematical relative error into an accuracy pass.
    """
    result = []
    for combined in rows:
        if combined["fixture"] != "sharp-signed" or combined["implementation"] != "fp32_p_weighted_dq":
            continue
        selected = max(
            (record for record in combined["per_head_fp64"] if record["component"] == "dk"),
            key=lambda record: record["relative_error"] if record["relative_error"] is not None else math.inf,
        )
        matched = {}
        for original in rows:
            if original["fixture_index"] != combined["fixture_index"] or original["implementation"] not in (
                "baseline",
                "fp32_p_weighted_dq",
                "flex",
                "natten",
            ):
                continue
            record = next(
                record
                for record in original.get("per_head_fp64", [])
                if record["component"] == "dk"
                and record["batch"] == selected["batch"]
                and record["head"] == selected["head"]
            )
            matched[original["implementation"]] = dict(
                **metric(record), formula_gate_passed=original.get("formula_gate", {}).get("passed")
            )
        result.append(
            dict(
                fixture="sharp-signed",
                fixture_index=combined["fixture_index"],
                dtype=combined["dtype"],
                scale=combined["scale"],
                component="dk",
                batch=selected["batch"],
                head=selected["head"],
                reference_norm=selected["reference_norm"],
                matched_implementations=matched,
            )
        )
    return dict(
        scope="Complete small-tensor mathematical FP64 dK error on sharp signed-scale fixtures; heads matched to the combined variant's worst head.",
        interpretation="The dQ correction does not center dK. dK retains actual rounded-O delta and low-precision dS. Agreement with those prescribed equations is separate from mathematical FP64 accuracy; no further causal diagnosis is claimed.",
        sharp_signed_dk=result,
    )


def forensic_summary(data, receipt_data):
    value, receipt = [json.loads(content) for content in (data, receipt_data)]
    result = dict(
        report_sha256=hashlib.sha256(data).hexdigest(),
        receipt_sha256=hashlib.sha256(receipt_data).hexdigest(),
        status=value["status"],
        original_formula_gate_failure_preserved=True,
        receipt={key: receipt[key] for key in ("stage", "job_id", "status", "exit_status", "elapsed_seconds")},
        script_sha256=digest(value["script_sha256"]),
        operand_file_sha256=digest(value["operand_sha256"]),
        actual_file_sha256=digest(value["actual_sha256"]),
        retained_actual_bytewise_reproduced=value["retained_actual_bytewise_reproduced"],
        torch_mm_out_dtype_documented=bool(value["torch_mm_out_dtype_documented"]),
        torch_mm_out_dtype_executed=bool(value["torch_mm_out_dtype_executed"]),
        largest_cpu_formula_discrepancy=value["largest_cpu_formula_discrepancy"],
        target_row_cast_crossings=value["target_row_cast_crossings"],
        fresh_vs_cpu_reference=per_head(value["fresh_vs_cpu_reference"]),
        fresh_vs_cuda_reference=per_head(value["fresh_vs_cuda_reference"]),
        best_reconstruction={
            key: value["best_reconstruction"][key]
            for key in ("probability", "dp", "delta", "changed_ds_entries", "changed_target_row_ds")
        },
        reconstructed_dq_heads=[
            record for record in per_head(value["best_reconstruction_other_heads"]) if record["component"] == "dq"
        ],
        zero_error_reconstruction_models=sum(record["head1_vs_actual"]["relative_error"] == 0 for record in value["models"]),
    )
    for name in ("max_delta_difference_vs_actual", "max_dp_difference_vs_cpu", "max_score_difference_vs_cpu"):
        result[name] = {key: scalar(number) for key, number in value[name].items()}
    return result


def qualification_summary(value):
    result = {}
    for stage, record in value.items():
        if stage not in (
            "cpu",
            "experimental_gpu",
            "default_neighborhood_gpu",
            "memcheck",
            "racecheck",
            "initcheck",
            "synccheck",
        ):
            raise ValueError("Unknown qualification stage.")
        result[stage] = {
            key: record[key]
            for key in (
                "status",
                "exit_status",
                "tests",
                "skipped",
                "methods",
                "variants",
                "dtypes",
                "elapsed_seconds",
                "receipt_sha256",
            )
            if key in record
        }
        if "receipt_sha256" in result[stage]:
            digest(result[stage]["receipt_sha256"])
    return result


def percent(value):
    return "undefined" if value is None else f"{value * 100:.5f}%"


def markdown(data):
    runtime = data["timing_metadata"]
    device = runtime["device"]
    lines = [
        "# Experimental neighborhood precision",
        "",
        "This report qualifies the dQ experiment. The [subsequent dK study](dk-precision.md) combines it with a reconstructed key-gradient correction; the [multi-GPU report](multigpu-precision.md) qualifies that combined method on five additional devices. Start with the [current guide](../README.md) for selection and coverage.",
        "",
        "The default arithmetic is unchanged. These explicit experiments compare stronger backward equations on identical rounded FP16/BF16 operands. They measure the neighborhood operator; no HDiT model or task quality was evaluated.",
        "",
        "The combined method improves measured dQ accuracy, but dK remains a material limitation: the sharp negative-scale fixture has 98.12% BF16 and 9.81% FP16 dK relative norm error against mathematical FP64. Passing the prescribed rounded equations does not establish near-FP64 accuracy for every gradient.",
        "",
        f"Measured on {device['name']} ({device['sm_count']} SMs), PyTorch {runtime['versions']['torch']}, CUDA {runtime['cuda']}, CUTLASS DSL {runtime['versions']['nvidia-cutlass-dsl']}, NATTEN {runtime['versions'].get('natten', 'unavailable')}. The qualified stack is retained for matched comparisons. Exact source/library metadata, all per-head component errors, sample medians/ranges and negative results are in [the compact JSON](../../../results/neighborhood/dq-precision.json).",
        "",
        "At qualification time, [PyTorch stable documentation](https://docs.pytorch.org/docs/stable/) resolves to 2.15. The [official NATTEN installation matrix](https://natten.org/install/) supplies 0.21.7 prebuilt libnatten wheels for PyTorch 2.13 and 2.12, including the retained 2.13/CUDA 13.0 pairing. This campaign keeps that already-qualified matching binary stack for same-process comparisons rather than changing framework and compiler policy during the experiment.",
        "",
        "FP64 means native `torch.float64` on the same already-quantized inputs. Small fixtures measure complete tensors. Large timing cases probe recorded boundary/interior query and inverse-key owners over all heads and channels; those errors are sampled, rather than complete large-tensor norms. NATTEN and FlexAttention are independently assessed implementations.",
        "",
        "## Arithmetic",
        "",
        "`fp32_p` keeps reconstructed P in FP32 through dS formation, then rounds dS once to the input dtype. dV still uses input-rounded P. `weighted_dq` uses the default dS and subtracts `sum(dS_low) * (P_low @ K) / sum(P_low)` from the unscaled FP32 dQ accumulation. `fp32_p_weighted_dq` combines both. The scale is applied after accumulation; delta comes from the actual rounded forward output in every GPU variant. Forward arithmetic is shared.",
        "",
        "The normalized weighted-key correction removes the spurious common-key term caused when rounded dS no longer sums to zero. This is a stronger numerical method, with an extra query-gradient product and reductions; it changes the default rounded equations. It does not recover discarded input bits.",
        "",
        "## Large common-key offset",
        "",
        "Worst-head relative L2 errors against mathematical FP64, in percent. Each 5×9 fixture uses a 3×3 shifted neighborhood and common Q/K component 1024, with exactly representable FP32 score differences. The JSON retains every head, signed-scale fixture and component.",
        "",
        "| dtype | implementation | O | dQ | dK | dV |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    compiler = data.get("loaded_compiler")
    if compiler:
        index = next(index for index, text in enumerate(lines) if text == "## Arithmetic")
        lines[index:index] = [
            f"The retained generated CuTe PTX identifies CUDA compiler {compiler['compiler_build']} and PTX ISA {compiler['ptx_isa']}. This loaded-code evidence is distinct from PyTorch's CUDA {runtime['cuda']} runtime/build tag. The compiler metadata is retained in JSON.",
            "",
        ]
    for row in data["directed"]:
        if row["fixture"] != "large-common-offset" or "worst_head_relative_l2" not in row:
            continue
        errors = row["worst_head_relative_l2"]
        lines.append(
            f"| {row['dtype'].removeprefix('torch.')} | {row['implementation']} | "
            + " | ".join(percent(errors[c]) for c in COMPONENTS)
            + " |"
        )
    lines += [
        "",
        "The correction targets a real cancellation error: multiplying a nonzero rounded dS row sum by a large common K component. Keeping P in FP32 alone leaves dS rounding and rounded-output delta error. The improvement is specific measured evidence; it does not establish universal pointwise accuracy or a task-level winner.",
        "",
        "## Remaining sharp-attention dK error",
        "",
        "The combined dQ correction does not repair every gradient. On the complete small sharp-attention fixtures, dK still differs substantially from mathematical FP64, despite passing the unchanged prescribed-equation gates. dK retains delta from the actual rounded O and input-rounded dS. The measurements identify this remaining limitation; they do not independently establish its complete numerical cause.",
        "",
        "Each row below selects the combined variant's worst dK head and compares that same head in the other implementations. Reference L2 and maximum absolute discrepancy provide scale for cancellation-sensitive errors; a small reference norm does not turn a large relative discrepancy into an accuracy pass. In particular, BF16 negative-scale head 1 has reference L2 0.175785 and maximum absolute error 0.061478, alongside 98.12% relative error. FlexAttention has a comparable error on that fixture, which does not make either implementation a mathematical oracle.",
        "",
        "| dtype | scale | head | FP64 dK L2 | baseline dK | combined dK | Flex dK | NATTEN dK | combined max absolute error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in data["mathematical_limitations"]["sharp_signed_dk"]:
        matched = row["matched_implementations"]
        errors = [
            percent(matched[name]["relative_error"]) if matched[name]["finite"] else "nonfinite"
            for name in ("baseline", "fp32_p_weighted_dq", "flex", "natten")
        ]
        maximum = matched["fp32_p_weighted_dq"]["max_absolute_error"]
        lines.append(
            f"| {row['dtype'].removeprefix('torch.')} | {row['scale']} | {row['head']} | {row['reference_norm']:.6f} | "
            + " | ".join(errors)
            + f" | {maximum:.6f} |"
        )
    lines += [
        "",
        "## Remaining dQ error",
        "",
        "These CPU counterfactuals separate operand boundaries using FP64 final products. Ideal delta is derived in FP64 and then rounded to FP32 before dS. Omitting the dS cast also omits the final low-precision output cast in that diagnostic. The final-cast floor rounds mathematical FP64 dQ directly to the output dtype.",
        "",
        "| dtype | default operands | FP32 P | ideal delta + FP32 P | additionally omit dS cast | ideal final-cast floor |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in data["directed"]:
        if row["fixture"] != "large-common-offset" or row["implementation"] != "baseline":
            continue
        cf = row.get("counterfactual_dq", {})
        keys = ("default", "fp32_p", "fp32_p_ideal_delta", "fp32_p_ideal_delta_unrounded_ds", "output_cast_floor")
        errors = [
            max(
                (record["relative_error"] for record in cf.get(key, []) if record.get("relative_error") is not None),
                default=None,
            )
            for key in keys
        ]
        lines.append(f"| {row['dtype'].removeprefix('torch.')} | " + " | ".join(percent(error) for error in errors) + " |")
    lines += [
        "",
        "The corrected kernel's remaining error must be compared with its dtype's final-cast floor per head. Further improvements can require another rounding policy or residual products and must be evaluated with their complete-call cost. The counterfactuals are diagnostic equations, rather than measured GPU variants.",
        "",
        "## Complete forward plus backward cost",
        "",
        f"Resident CUDA graph device-event medians in microseconds: {data['timing_protocol']['samples']} alternating timing rounds, {data['timing_protocol']['capture_batch']} complete calls per replay. Compilation, allocation and immutable geometry preparation are outside the resident interval; required per-call layout copies and all forward/backward kernels are inside. Preparation costs remain in JSON. Changed-input graph checks precede timing. Profiling is separate from uninstrumented samples.",
        "",
        "| case | baseline | FP32 P | weighted dQ | combined | FlexAttention | NATTEN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    names = list(dict.fromkeys(row["workload"]["name"] for row in data["timing"]))
    for name in names:
        rows = {row["implementation"]: row for row in data["timing"] if row["workload"]["name"] == name}
        values = [
            f"{rows[key]['device_us']['median']:.2f}" if key in rows and rows[key]["status"] == "pass" else "unavailable"
            for key in ("baseline", "fp32_p", "weighted_dq", "fp32_p_weighted_dq", "flex", "natten")
        ]
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    lines += [
        "",
        "Sampled mathematical FP64 errors for the ordinary timing inputs:",
        "",
        "| case | implementation | O | dQ | dK | dV |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in data["timing"]:
        if "worst_head_relative_l2" in row:
            errors = row["worst_head_relative_l2"]
            lines.append(
                f"| {row['workload']['name']} | {row['implementation']} | "
                + " | ".join(percent(errors[c]) for c in COMPONENTS)
                + " |"
            )
    failures = [row for row in data["directed"] + data["timing"] if "failure" in row]
    nonfinite = [row for row in data["directed"] if "diagnostic_failure" in row]
    lines += [
        "",
        "## Qualification and limitations",
        "",
        "The experiment retains independent prescribed-equation gates at 0.5% FP16 and 1% BF16 per head and component, plus ordinary mathematical FP64 gates. These are norm criteria and do not require universal bitwise agreement across FP32 reduction orders. Formula errors, mathematical errors and final-cast floors remain separate. Paired unchanged-component checks and changed-input graph checks are retained in JSON.",
        "",
        f"There are {len(failures)} retained non-success rows across the directed and timing reports, plus {len(nonfinite)} completed comparator diagnostics with nonfinite results. Their classifications are:",
        "",
    ]
    counts = Counter(row["failure"]["classification"] for row in failures)
    counts.update(row["diagnostic_failure"]["classification"] for row in nonfinite)
    lines.extend(f"- {count}: {classification}." for classification, count in counts.items())
    if not counts:
        lines.append("- No failed or unsupported rows in these reports.")
    for row in failures:
        if "formula_gate" not in row:
            continue
        for record in row["formula_gate"]["errors"]:
            if not record["passed"]:
                lines += [
                    "",
                    f"The {row['dtype'].removeprefix('torch.')} `{row['implementation']}` variant fails its unchanged formula gate on `{row['fixture']}` ({record['component']}, head {record['head']}): relative norm error {percent(record.get('relative_error'))}, maximum absolute discrepancy {record.get('max_absolute_error')}. This failure remains recorded; it is not relabelled as a passing gate.",
                ]
    forensic = data.get("forensic")
    if forensic:
        lines += [
            "",
            "A separate immutable retained-operand diagnosis reproduces all four original outputs bytewise. CPU and CUDA IEEE FP32 references agree on the failed dQ gate. Replacing their dP with native FP16-input products with FP32 output reproduces both heads' final dQ bitwise; independently reconstructed probability and actual saved normalization statistics give the same result. This identifies an FP32 reduction-order difference amplified by the declared dS cast, with no acceptance threshold changed.",
            "",
            "At query 7/key 15 in head 1, pre-cast dS is −1.4194334745407104 with the IEEE product and −1.4194337129592896 with the FP16-input product. Those values lie one FP32 ULP on either side of the FP16 midpoint −1.41943359375. Rounding selects −1.4189453125 versus −1.419921875: one FP16 step, which the common key and signed scale turn into the observed dQ discrepancy of exactly 0.125. The compact JSON retains the original failed gate, diagnostic source/operand/report hashes, reconstruction metrics and actual successful diagnostic exit receipt.",
        ]
    qualification = data.get("qualification", {})
    if qualification:
        lines += ["", "Qualification receipts:", "", "| stage | result | scope |", "| --- | --- | --- |"]
        for name, record in qualification.items():
            scope = ", ".join(
                f"{record[key]} {key}" for key in ("tests", "skipped", "methods", "variants", "dtypes") if key in record
            )
            lines.append(f"| {name} | {record['status']} | {scope} |")
    lines += [
        "",
        "Execution errors and unsupported calls have no successful timing. Worst-head comparisons can select different heads; all matched per-head records are retained. FP64 is finite precision, initial input quantization is excluded, and measurements on this H100 do not establish behavior on another GPU. The original default and its acceptance criteria remain unchanged.",
        "",
        "Reproduce the compact artifacts from retained completed reports:",
        "",
        "```sh",
        "python3.13 -m benchmarks.neighborhood.summarize_precision \\",
        "  --directed RESULTS/directed/report.json --timing RESULTS/timing/report.json \\",
        "  --output results/neighborhood/dq-precision.json \\",
        "  --markdown docs/neighborhood/reports/dq-precision.md",
        "```",
        "",
        "Use `--forensic`, `--forensic-receipt`, `--compiler-metadata` and `--qualification` with the retained compact metadata to include the supplementary evidence.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directed", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--forensic", type=Path)
    parser.add_argument("--forensic-receipt", type=Path)
    parser.add_argument("--compiler-metadata", type=Path)
    parser.add_argument("--qualification", type=Path)
    args = parser.parse_args()
    result = summarize(args.directed.read_bytes(), args.timing.read_bytes())
    if bool(args.forensic) != bool(args.forensic_receipt):
        parser.error("Forensic report and actual exit receipt are required together.")
    if args.forensic:
        result["forensic"] = forensic_summary(args.forensic.read_bytes(), args.forensic_receipt.read_bytes())
    if args.compiler_metadata:
        compiler = json.loads(args.compiler_metadata.read_bytes())
        result["loaded_compiler"] = {key: compiler[key] for key in ("compiler_build", "ptx_isa", "scope")}
    if args.qualification:
        result["qualification"] = qualification_summary(json.loads(args.qualification.read_bytes()))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    args.markdown.write_text(markdown(result))


if __name__ == "__main__":
    main()
