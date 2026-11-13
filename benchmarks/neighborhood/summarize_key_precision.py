"""Export measured dK precision evidence using only the Python standard library.

Numeric references and timing are performed elsewhere. This reporter copies
selected public fields, never raw exception text, host paths or kernel symbols.
"""

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path, PurePosixPath

from benchmarks.neighborhood.summarize_precision import digest, metric, observations, per_head, source

COMPONENTS = ("out", "dq", "dk", "dv")
BASELINE = "baseline"
PRIOR = "fp32_p_weighted_dq"
TARGET = "weighted_dq_reconstructed_dk"
CONTROL = "reconstructed_dk"
METHODS = (BASELINE, PRIOR, CONTROL, TARGET, "flex", "natten")
STRATEGIES = ("direct", "column", "staged", "column_staged")


def worst(records):
    return {
        component: max(
            (
                record["relative_error"]
                for record in records
                if record["component"] == component and record.get("relative_error") is not None
            ),
            default=None,
        )
        for component in COMPONENTS
    }


def gate(value):
    records = per_head(value["errors"])
    for record, original in zip(records, value["errors"]):
        record["passed"] = bool(original["passed"])
    result = dict(passed=bool(value["passed"]), limit=value["limit"], errors=records)
    for name in ("required_components", "required_for_qualification"):
        if name in value:
            result[name] = value[name]
    return result


def backend(value):
    return {
        key: value[key]
        for key in (
            "selected_backend",
            "geometry_preparation_reused",
            "preparation_storage_bytes",
            "last_preparation_device_ms",
            "last_preparation_host_ms",
            "first_preparation_device_ms",
            "first_preparation_host_ms",
        )
        if key in value
    }


def metadata(raw, data, *, completed=True):
    if completed and (
        raw.get("status") == "running" or any(row.get("status") in ("running", "prepared") for row in raw.get("rows", []))
    ):
        raise ValueError("Incomplete reports cannot establish a final measured result.")
    result = dict(status=raw.get("status", "unavailable"), report_sha256=hashlib.sha256(data).hexdigest())
    if "source_sha256" in raw:
        result["source"] = source(raw)
    for name in ("script_sha256", "previous_report_sha256"):
        if name in raw:
            result[name] = digest(raw[name])
    if "versions" in raw:
        result["versions"] = raw["versions"]
    if "cuda" in raw:
        result["cuda"] = raw["cuda"]
    if "device" in raw:
        result["device"] = {key: raw["device"][key] for key in ("name", "capability", "sm_count", "memory_bytes")}
    result["row_status_counts"] = dict(Counter(row["status"] for row in raw.get("rows", [])))
    result["gpu_observations"] = observations(raw.get("gpu_observations", []))
    return result


def strategy(raw):
    value = raw.get("key_statistics_experiment", {}).get("new_strong_method", raw.get("strategy", "direct"))
    if value not in STRATEGIES:
        raise ValueError("Unknown explicit statistics scheduling strategy.")
    return value


def directed_rows(raw):
    result = []
    identities = {}
    for value in raw["rows"]:
        if value["implementation"] not in METHODS:
            raise ValueError("Unknown numerical method.")
        row = {
            key: value[key]
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
        row["operand_sha256"] = digest(value["operand_sha256"])
        row["historical_operand_verified"] = bool(value.get("historical_operand_verified", False))
        index = row["fixture_index"]
        if index in identities and identities[index] != row["operand_sha256"]:
            raise ValueError("Paired fixture operands changed.")
        identities[index] = row["operand_sha256"]
        if "fp64" in value:
            row["per_head_fp64"] = per_head(value["fp64"])
            row["worst_head_relative_l2"] = worst(row["per_head_fp64"])
        for name in ("formula", "math_gate"):
            if name in value:
                row[name] = gate(value[name])
        if "paired_invariants" in value:
            row["paired_invariants"] = dict(
                passed=bool(value["paired_invariants"]["passed"]),
                checks=[
                    {key: check[key] for key in ("component", "against", "passed")}
                    for check in value["paired_invariants"]["checks"]
                ],
            )
        if "backend_metadata" in value:
            row["backend"] = backend(value["backend_metadata"])
        if "correction_observations" in value:
            row["correction_observations"] = value["correction_observations"]
        if row["status"] == "execution-error":
            row["failure_classification"] = "execution error; retained raw artifacts are not a successful result"
        result.append(row)
    return result


def profile(value):
    names = value.get("launched_cuda_kernels", [])
    calls = value.get("complete_calls", 1)
    counts = Counter(
        (
            "output_delta"
            if "OutputDelta" in name
            else (
                "spatial_attention_owner"
                if "NeighborhoodAttention" in name
                else (
                    "triton"
                    if "triton" in name.lower()
                    else "libnatten" if "natten" in name.lower() or "fmha" in name.lower() else "other"
                )
            )
        )
        for name in names
    )
    result = dict(
        complete_calls=calls,
        cuda_launches=len(names),
        cuda_launches_per_call=len(names) / calls,
        family_counts=dict(counts),
        family_counts_per_call={name: count / calls for name, count in counts.items()},
    )
    if "cpu_empty_operation_counts" in value:
        result["cpu_empty_operation_counts"] = {
            name: int(count)
            for name, count in value["cpu_empty_operation_counts"].items()
            if name.startswith("aten::") and "empty" in name
        }
    return result


def timing_rows(raw):
    result = []
    for value in raw["rows"]:
        row = {key: value[key] for key in ("workload", "implementation", "status")}
        if row["implementation"] not in METHODS:
            raise ValueError("Unknown timed implementation.")
        row["operand_sha256"] = digest(value["operand_sha256"])
        if "backend_metadata" in value:
            row["backend"] = backend(value["backend_metadata"])
        if "selected" in value:
            row["selected"] = {key: value["selected"][key] for key in ("precision", "statistics", "columns", "traversal")}
        if "accuracy" in value:
            row["per_head_fp64"] = [
                dict(
                    component=record["component"],
                    batch=record["batch"],
                    head=record["head"],
                    passed=bool(record["passed"]),
                    **metric(record["fp64"]),
                )
                for record in value["accuracy"]
            ]
            row["worst_head_relative_l2"] = worst(row["per_head_fp64"])
            row["reference_positions"] = value["reference_positions"]
        if "graph_check" in value and value["graph_check"] is not None:
            row["graph_check"] = dict(
                passed=bool(value["graph_check"]["passed"]),
                changed_output=bool(value["graph_check"]["changed_output"]),
                errors=per_head(value["graph_check"]["errors"]),
            )
        if "profile" in value:
            row["profile"] = profile(value["profile"])
        if "device_ms" in value:
            for name in ("device_ms", "wall_ms"):
                row[name] = {key: value[name][key] for key in ("median", "minimum", "maximum", "mean", "stdev")}
            row["samples"] = [
                {key: sample[key] for key in ("round", "device_ms", "wall_ms", "graph_replays", "complete_calls")}
                for sample in value["samples"]
            ]
        result.append(row)
    for row in result:
        row["comparisons"] = {}
        for control in (BASELINE, PRIOR):
            baseline = next(
                (
                    other
                    for other in result
                    if other["workload"]["name"] == row["workload"]["name"] and other["implementation"] == control
                ),
                None,
            )
            if baseline is None:
                continue
            if baseline["operand_sha256"] != row["operand_sha256"]:
                raise ValueError("Timing comparisons require identical rounded operands.")
            if baseline["status"] != row["status"] or row["status"] != "pass":
                continue
            comparison = {}
            for clock in ("device_ms", "wall_ms"):
                by_round = {sample["round"]: sample[clock] for sample in baseline["samples"]}
                ratios = [sample[clock] / by_round[sample["round"]] for sample in row["samples"]]
                ratio = row[clock]["median"] / baseline[clock]["median"]
                comparison[clock] = dict(
                    candidate_over_control_latency=ratio,
                    control_over_candidate_speed=1 / ratio,
                    incremental_percent=(ratio - 1) * 100,
                    paired_round_latency_ratios=dict(
                        samples=ratios, median=statistics.median(ratios), minimum=min(ratios), maximum=max(ratios)
                    ),
                )
            row["comparisons"][control] = comparison
        if (
            row["implementation"] == TARGET
            and PRIOR in row["comparisons"]
            and raw.get("protocol", {}).get("mode") != "eager"
        ):
            increment = row["comparisons"][PRIOR]["device_ms"]["candidate_over_control_latency"] - 1
            row["incremental_goal"] = dict(
                limit=0.05,
                passed=increment <= 0.05,
                requires_investigation=increment > 0.10,
                scope="Resident full forward plus backward median versus prior combined arithmetic; shape-specific descriptive result.",
            )
    return result


def sharp_rows(rows):
    result = []
    for target in rows:
        if target["implementation"] != TARGET or target["fixture"] != "sharp-signed":
            continue
        paired = [row for row in rows if row["fixture_index"] == target["fixture_index"]]
        reference = next(row for row in paired if row["implementation"] == PRIOR)
        for old in reference["per_head_fp64"]:
            if old["component"] != "dk":
                continue
            records = {
                row["implementation"]: next(
                    record
                    for record in row["per_head_fp64"]
                    if record["component"] == "dk" and record["batch"] == old["batch"] and record["head"] == old["head"]
                )
                for row in paired
            }
            result.append(
                dict(
                    fixture_index=target["fixture_index"],
                    dtype=target["dtype"],
                    scale=target["scale"],
                    batch=old["batch"],
                    head=old["head"],
                    reference_norm=old["reference_norm"],
                    implementations={name: metric(record) for name, record in records.items()},
                )
            )
    return result


def validation_summary(raw, data):
    result = metadata(raw, data, completed=False)
    for name in ("stage", "job_id", "phase", "exit_status", "elapsed_seconds", "source_manifest_sha256"):
        if name in raw:
            result[name] = raw[name]
    if "stages" in raw:
        result["stages"] = {
            name: {
                key: value[key]
                for key in (
                    "status",
                    "controller_exit_status",
                    "exit_status",
                    "tests",
                    "skipped",
                    "methods",
                    "variants",
                    "dtypes",
                    "errors",
                    "hazards",
                    "receipt_sha256",
                )
                if key in value
            }
            for name, value in raw["stages"].items()
        }
    if "rows" in raw and raw["rows"] and "byte_equal" in raw["rows"][0]:
        result["strategy"] = strategy(raw)
        result["rows"] = []
        for original in raw["rows"]:
            row = {
                key: original[key]
                for key in ("case", "dtype", "shape", "spatial", "window", "scale", "status", "operand_sha256")
            }
            row["byte_equal"] = original.get("byte_equal", {})
            row["errors"] = per_head(original.get("errors", []))
            row["selected"] = {
                name: {
                    key: value[key]
                    for key in (
                        "backend",
                        "precision",
                        "statistics",
                        "columns",
                        "traversal",
                        "dout_stride",
                        "dout_storage_offset",
                    )
                    if key in value
                }
                for name, value in original.get("selected", {}).items()
            }
            result["rows"].append(row)
    return result


def execution_path_audit(data):
    result = []
    for record in json.loads(data):
        row = dict(
            stage=record["stage"],
            compiled_sha256=digest(record["compiled_sha256"]),
            newstrong_stats=[
                dict(column=bool(value["column"]), staged=bool(value["staged"]), modules=int(value["modules"]))
                for value in record["newstrong_stats"]
            ],
        )
        for key in ("report_sha256", "receipt_sha256"):
            if key in record:
                row[key] = digest(record[key])
        for key in (
            "stage_exit_status",
            "same_eight_operand_hashes",
            "source_sha256_identical_to_first_heldout",
            "all_24_row_operand_hashes_identical_by_case",
        ):
            if key in record:
                row[key] = record[key]
        if "operand_sha256" in record:
            row["operand_sha256"] = {name: digest(value) for name, value in record["operand_sha256"].items()}
        if "source_sha256" in record:
            for name in record["source_sha256"]:
                path = PurePosixPath(name)
                if path.is_absolute() or ".." in path.parts or "\\" in name:
                    raise ValueError("Audit source paths must be repository-relative.")
            row["source_sha256"] = {name: digest(value) for name, value in record["source_sha256"].items()}
        result.append(row)
    return dict(report_sha256=hashlib.sha256(data).hexdigest(), stages=result)


def summarize(
    directed_data,
    timing_data,
    *,
    eager_data=None,
    heldout_data=None,
    path_audit_data=None,
    validations=None,
    negative_sweeps=None,
):
    directed, timing = [json.loads(data) for data in (directed_data, timing_data)]
    rows = directed_rows(directed)
    selected = [row for row in rows if row["implementation"] == TARGET]
    zero = [row for row in selected if row["fixture"] in ("zero-dp-sharp", "constant-dp-sharp")]
    result = dict(
        schema="neighborhood-key-precision-v1",
        default_changed=False,
        scope="Explicit stronger neighborhood key-gradient arithmetic; operator measurements, no model-level quality claim.",
        arithmetic=dict(
            method=TARGET,
            query="Prior normalized weighted dQ arithmetic retained bytewise.",
            correction="t=sum_FP32(P32*(dP32-delta_actualO))/sum_FP32(P32), reconstructed during the existing query traversal.",
            key="dS_key=round_input(P32*((dP32-delta_actualO)-t)); dK products accumulate FP32, scale after accumulation.",
            storage="A separate FP32 row buffer preserves the actual rounded-output delta; the two subtractions retain the small residual.",
            extra_work="Two query row reductions and one correction buffer; no additional attention pass, GEMM or GPU launch.",
            default="Default docs/precision.md and automatic dispatch are unchanged; scheduling selectors remain explicit experiments.",
        ),
        references=dict(
            mathematical="Native torch.float64 on identical rounded Q/K/V/dO, excluding initial input quantization error.",
            formula="Independent IEEE FP32 products/reductions and prescribed operand casts, FP64 final reference products; not a tensor-core instruction-order emulator.",
            directed="Complete small tensors, independently enumerated shifted neighborhoods.",
            timing="Sparse boundary/interior query and inverse-key owners, all heads/channels at recorded positions; not full large-tensor norms.",
        ),
        acceptance=dict(
            fp16=0.005,
            bf16=0.01,
            mathematical="Separate output/dQ/dK/dV gates for the selected method on historical fixtures and ordinary timings. New zero/constant-dP controls target O/dK/dV while preserving and reporting prior dQ.",
            formula="Prescribed-equation agreement and mathematical FP64 error remain distinct.",
            performance="Predeclared aim <=5% incremental resident full-call median versus prior combined method; investigate cases >10%. No tolerance or performance goal is widened after measurement.",
        ),
        selected_statistics_strategy=strategy(timing),
        directed_metadata=metadata(directed, directed_data),
        timing_metadata=metadata(timing, timing_data),
        directed=rows,
        timing=timing_rows(timing),
        sharp_matched_heads=sharp_rows(rows),
        targeted_qualification=dict(
            cases=len(selected),
            passed=sum(row["status"] == "pass" for row in selected),
            formula_passed=sum(row["formula"]["passed"] for row in selected),
            required_math_passed=sum(row["math_gate"]["passed"] for row in selected),
            all_component_math_passed=sum(
                all(record["passed"] for record in row["math_gate"]["errors"]) for row in selected
            ),
            unchanged_components_passed=sum(row["paired_invariants"]["passed"] for row in selected),
            exact_zero_dk_controls=len(zero),
            exact_zero_dk_passed=sum(
                all(
                    record["reference_norm"] == 0 and record["max_absolute_error"] == 0
                    for record in row["per_head_fp64"]
                    if record["component"] == "dk"
                )
                for row in zero
            ),
        ),
        legacy_failures=[
            dict(
                fixture=row["fixture"],
                fixture_index=row["fixture_index"],
                dtype=row["dtype"],
                implementation=row["implementation"],
                status=row["status"],
                formula=row.get("formula"),
                math_gate=row.get("math_gate"),
            )
            for row in rows
            if row["status"] not in ("pass", "diagnostic-complete")
        ],
    )
    result["numerical_and_timing_source_identical"] = directed.get("source_sha256") == timing.get("source_sha256")
    result["targeted_worst_math_errors"] = {
        dtype: {
            component: max(
                (
                    dict(
                        value=record["relative_error"],
                        fixture=row["fixture"],
                        fixture_index=row["fixture_index"],
                        batch=record["batch"],
                        head=record["head"],
                    )
                    for row in selected
                    if row["dtype"] == dtype
                    for record in row["per_head_fp64"]
                    if record["component"] == component and record["relative_error"] is not None
                ),
                key=lambda record: record["value"],
                default=None,
            )
            for component in COMPONENTS
        }
        for dtype in sorted({row["dtype"] for row in selected})
    }
    result["validation"] = {name: validation_summary(json.loads(data), data) for name, data in (validations or {}).items()}
    result["negative_sweeps"] = {
        name: dict(
            metadata=metadata(json.loads(data), data, completed=False),
            strategy=strategy(json.loads(data)),
            timing=timing_rows(json.loads(data)),
        )
        for name, data in (negative_sweeps or {}).items()
    }
    audited = {}
    if path_audit_data is not None:
        result["execution_path_audit"] = execution_path_audit(path_audit_data)
        audited = {record["stage"]: record for record in result["execution_path_audit"]["stages"]}
    failed_heldout = None
    for label, content in (negative_sweeps or {}).items():
        original = json.loads(content)
        audit = audited.get("heldout-final")
        if (
            "heldout_evaluation" in original
            and audit
            and all(not record["column"] and not record["staged"] for record in audit["newstrong_stats"])
        ):
            negative = result["negative_sweeps"][label]
            negative.update(
                classification="Harness path-selection failure: cached variant alias executed direct statistics despite the recorded column_staged label.",
                declared_strategy=strategy(original),
                verified_actual_strategy="direct",
                excluded_from_selected_evaluation=True,
                compiled_path_audit=audit,
            )
            failed_heldout = original
    if eager_data is not None:
        eager = json.loads(eager_data)
        result["eager"] = dict(
            metadata=metadata(eager, eager_data),
            strategy=strategy(eager),
            protocol=eager["protocol"],
            rows=timing_rows(eager),
        )
    if heldout_data is not None:
        heldout = json.loads(heldout_data)
        if strategy(heldout) != result["selected_statistics_strategy"]:
            raise ValueError("Held-out evaluation must retain the already-selected statistics schedule.")
        wrapper = heldout["key_statistics_experiment"]
        calls = [
            dict(method=value["method"], strategy=value["strategy"], calls=int(value["calls"]))
            for value in wrapper.get("applied_context_calls", [])
        ]
        if not any(
            value["method"] == TARGET and value["strategy"] == result["selected_statistics_strategy"] and value["calls"] > 0
            for value in calls
        ):
            raise ValueError(
                "Held-out evaluation requires positive verified context calls, not a requested-strategy label alone."
            )
        audit = audited.get("heldout-verified")
        flags = {"direct": (False, False), "column": (True, False), "staged": (False, True), "column_staged": (True, True)}[
            strategy(heldout)
        ]
        if (
            audit is None
            or not audit["newstrong_stats"]
            or sum(value["modules"] for value in audit["newstrong_stats"]) != 8
            or any((value["column"], value["staged"]) != flags for value in audit["newstrong_stats"])
        ):
            raise ValueError("Selected held-out evidence requires the actual compiled flags on all eight KV modules.")
        evaluation = heldout["heldout_evaluation"]
        result["heldout"] = dict(
            metadata=metadata(heldout, heldout_data),
            strategy=strategy(heldout),
            seed=int(evaluation["seed"]),
            scope="New shapes and operands evaluated after schedule selection, without selection or retuning on these results.",
            context_verification=dict(
                applied_context_calls=calls,
                wrapper_sha256=digest(wrapper["wrapper_sha256"]),
                preloaded_aliases_rebound=wrapper.get("preloaded_aliases_rebound", []),
            ),
            compiled_path_audit=audit,
            rows=timing_rows(heldout),
        )
        if failed_heldout is not None:
            before = {
                row["workload"]["name"]: row["operand_sha256"]
                for row in failed_heldout["rows"]
                if row["implementation"] == TARGET
            }
            after = {
                row["workload"]["name"]: row["operand_sha256"] for row in heldout["rows"] if row["implementation"] == TARGET
            }
            if before != after:
                raise ValueError("The harness repair rerun must retain exactly the original held-out operands.")
            result["heldout"]["same_operands_as_misselected_run_verified"] = True
    return result


def percentage(value):
    return "undefined" if value is None else f"{value*100:.5f}%"


def kernel_evidence_summary(raw, data):
    result = dict(
        report_sha256=hashlib.sha256(data).hexdigest(),
        source_manifest_sha256=digest(raw["source_manifest_sha256"]),
        scope="Selected retained compiler-returned KV machine-code inspection and separately instrumented kernel intervals; static instruction sites are not runtime percentages. No direct CUPTI CUfunction-to-module binding was established.",
    )
    result["workload"] = {key: raw["workload"][key] for key in ("dtype", "spatial", "window")}
    result["profiled_kv_us"] = {name: float(raw["profiled_kv_us"][name]) for name in ("prior", "direct", "selected")}
    counts = raw["instruction_counts"]
    result["instruction_counts"] = {
        key: int(counts[key])
        for key in (
            "direct_ldg",
            "selected_ldg",
            "selected_lds",
            "selected_stat_cp_async_sites",
            "direct_hmma",
            "selected_hmma",
        )
    }
    return result


def markdown(data):
    device = data["timing_metadata"].get("device", {})
    selected = data["selected_statistics_strategy"]
    qualification = data["targeted_qualification"]
    lines = [
        "# Experimental neighborhood key-gradient precision",
        "",
        "This H100 report defines the combined method. The [multi-GPU report](multigpu-precision.md) replays its frozen inputs on B200, B300, RTX 4090, RTX 5090 and L40S. See the [current guide](../README.md) for the API and explicit-selection example.",
        "",
        "The default arithmetic is unchanged. This explicit follow-up reconstructs the softmax correction during the existing query pass to reduce the sharp-attention dK error. Results below concern defined neighborhood operators and already-rounded operands, rather than model quality or universal pointwise accuracy.",
        "",
        f"Measured device: {device.get('name','unavailable')}, {device.get('sm_count','unavailable')} SMs. Selected KV statistics schedule: `{selected}`. Exact stack, device, source hashes, all per-head component errors, raw timing samples and negative results are retained in [the compact JSON](../../../results/neighborhood/dk-precision.json).",
        "",
        "## Arithmetic and accuracy contract",
        "",
        "`t = sum(P32 * (dP32 - delta_O)) / sum(P32)` is computed with FP32 row accumulation during the existing query traversal. A separate FP32 row buffer keeps `delta_O` from the actual rounded forward output. The KV owner forms `dS_key = round_input(P32 * ((dP32 - delta_O) - t))`; its products accumulate FP32 and the attention scale is applied afterward. No additional GEMM, neighborhood traversal or GPU launch is required.",
        "",
        "The selected `weighted_dq_reconstructed_dk` method retains the prior normalized weighted-query correction. O, dQ and dV are checked bytewise against the previous method; dV still uses input-rounded P. The unprojected `reconstructed_dk` variant is a dK control and does not claim to repair its known dQ stress errors.",
        "",
        "Native FP64 references evaluate identical already-rounded inputs. Mathematical accuracy and independently prescribed rounded equations are tested separately. The predefined per-head component limits remain 0.5% FP16 and 1% BF16. New exact zero/constant-dP controls target dK and preserve the previous dQ policy.",
        "",
        "Use the explicit experiment around the complete forward/backward call, including warmup and graph capture. These process-wide selections are serialized benchmark controls, with cache keys for the chosen arithmetic and scheduling. Q/K/V use the existing BHSD, head-dimension-64 FP16/BF16 neighborhood API.",
        "",
        "```python",
        "import torch",
        "from benchmarks.neighborhood.precision import key_statistics, variant",
        "from kohakufa_cute import neighborhood_attention",
        "",
        f'with variant("{TARGET}"), key_statistics("{selected}"):',
        "    out = neighborhood_attention(q, k, v, spatial_shape=(height, width), window_size=window)",
        "    dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)",
        "```",
        "",
        "## Sharp attention: matched dK heads",
        "",
        "Relative L2 errors against mathematical FP64, in percent. These are full small-tensor comparisons on the exact retained seed-178 operands, with both signed scales. Every head is shown; reference norms and maximum absolute discrepancies keep cancellation-sensitive errors interpretable.",
        "",
        "| dtype | scale | head | FP64 dK L2 | original baseline | prior dQ correction | new dQ+dK correction | new max absolute error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    heldout = data.get("heldout")
    if heldout:
        exceptions = [
            row
            for row in heldout["rows"]
            if row["implementation"] == TARGET and row.get("incremental_goal", {}).get("passed") is False
        ]
        if exceptions:
            prefix = (
                "The main timing matrix meets its predefined incremental-cost goal, but independent held-out testing retains performance exceptions: "
                if all(
                    row.get("incremental_goal", {}).get("passed", False)
                    for row in data["timing"]
                    if row["implementation"] == TARGET
                )
                else "Independent held-out testing retains performance exceptions: "
            )
            text = (
                prefix
                + ", ".join(
                    f"`{row['workload']['name']}` adds {row['comparisons'][PRIOR]['device_ms']['incremental_percent']:.2f}% versus the prior dQ method"
                    for row in exceptions
                )
                + ". These misses of the 5% aim remain explicit; no tuning followed the verified held-out evaluation."
            )
            lines[6:6] = [text, ""]
    for row in data["sharp_matched_heads"]:
        values = row["implementations"]
        new = values[TARGET]
        lines.append(
            f"| {row['dtype'].removeprefix('torch.')} | {row['scale']} | {row['head']} | {row['reference_norm']:.6f} | "
            + " | ".join(percentage(values[name]["relative_error"]) for name in (BASELINE, PRIOR, TARGET))
            + f" | {new['max_absolute_error']:.8f} |"
        )
    lines += [
        "",
        f"The selected method passes {qualification['passed']}/{qualification['cases']} defined fixture qualifications, including {qualification['required_math_passed']} required mathematical gates and {qualification['formula_passed']} independent policy gates. All four observed component gates pass on {qualification['all_component_math_passed']} fixtures, including the new controls whose prior dQ policy was preserved. All {qualification['exact_zero_dk_passed']}/{qualification['exact_zero_dk_controls']} zero/constant-dP controls retain exactly zero dK. Its O/dQ/dV unchanged-component checks pass on {qualification['unchanged_components_passed']} fixtures. These counts describe the measured fixtures and do not promise exact zero gradients for every possible cancellation.",
        "",
        "Worst full-small-tensor per-head mathematical errors across the targeted fixture matrix; each component maximum may come from a different fixture or head:",
        "",
        "| dtype | O | dQ | dK | dV |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for dtype, components in data["targeted_worst_math_errors"].items():
        lines.append(
            f"| {dtype.removeprefix('torch.')} | "
            + " | ".join(percentage(components[name]["value"]) if components[name] else "undefined" for name in COMPONENTS)
            + " |"
        )
    lines += [
        "",
        "The original unprojected-query control retains its prior FP16 negative common-offset dQ failure. It is kept as a failed prescribed-reference gate, rather than hidden or converted into a passing result. The prior [dQ precision report](dq-precision.md) records the retained-product diagnosis: a native FP16-input GEMM with FP32 output changes dP enough to cross one dS rounding midpoint. The new weighted-query method preserves the already-qualified dQ behavior.",
        "",
        "## Complete resident forward plus backward cost",
        "",
        "CUDA graph device-event medians in microseconds, with matched quantized operands and alternating measurement order. Required per-call work is included; compilation, allocation warmup and immutable geometry preparation are outside the resident interval. The predeclared aim is at most 5% incremental median latency versus the prior combined dQ method, with cases above 10% requiring investigation. Shape exceptions remain explicit.",
        "",
        "| case | original baseline | prior dQ correction | new dQ+dK correction | incremental vs prior | incremental vs baseline | <=5% goal |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    names = list(dict.fromkeys(row["workload"]["name"] for row in data["timing"]))
    for name in names:
        rows = {row["implementation"]: row for row in data["timing"] if row["workload"]["name"] == name}
        if TARGET not in rows:
            continue
        target = rows[TARGET]
        values = [
            (
                f"{rows[key]['device_ms']['median']*1000:.2f}"
                if key in rows and rows[key]["status"] == "pass"
                else "unavailable"
            )
            for key in (BASELINE, PRIOR, TARGET)
        ]
        ratios = [
            target.get("comparisons", {}).get(control, {}).get("device_ms", {}).get("incremental_percent")
            for control in (PRIOR, BASELINE)
        ]
        display = ["unavailable" if ratio is None else f"{ratio:+.2f}%" for ratio in ratios]
        goal = target.get("incremental_goal", {}).get("passed")
        lines.append(
            f"| {name} | "
            + " | ".join(values + display + ["pass" if goal is True else "miss" if goal is False else "unavailable"])
            + " |"
        )
    lines += [
        "",
        "All timing cases retain separate sampled mathematical FP64 errors for each component in JSON. Sparse probes cover recorded boundary/interior query owners and every query contributing to sampled key owners. They do not establish full large-tensor norm bounds. Failed or unsupported calls do not receive successful timing ratios.",
        "",
        "Selected-method sampled mathematical FP64 errors, worst head per component:",
        "",
        "| case | O | dQ | dK | dV |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in data["timing"]:
        if row["implementation"] == TARGET and "worst_head_relative_l2" in row:
            lines.append(
                f"| {row['workload']['name']} | "
                + " | ".join(percentage(row["worst_head_relative_l2"][component]) for component in COMPONENTS)
                + " |"
            )
    lines.append("")
    heldout = data.get("heldout")
    if heldout:
        lines += [
            "## Held-out shapes after schedule selection",
            "",
            f"These shapes and operands use seed {heldout['seed']} and the already-selected `{heldout['strategy']}` schedule. They are evaluated after selection, without further tuning. Both dtype cases, accuracy failures and misses of the original 5% aim remain visible; the JSON retains all component errors and paired samples.",
            "",
            "| case | prior us | new us | incremental vs prior | sampled new dK error | <=5% goal |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for name in dict.fromkeys(row["workload"]["name"] for row in heldout["rows"]):
            rows = {row["implementation"]: row for row in heldout["rows"] if row["workload"]["name"] == name}
            if any(key not in rows or rows[key]["status"] != "pass" for key in (PRIOR, TARGET)):
                lines.append(f"| {name} | unavailable | unavailable | unavailable | unavailable | unavailable |")
                continue
            old, new = rows[PRIOR], rows[TARGET]
            increment = new["comparisons"][PRIOR]["device_ms"]["incremental_percent"]
            goal = "pass" if new["incremental_goal"]["passed"] else "miss"
            lines.append(
                f"| {name} | {old['device_ms']['median']*1000:.2f} | {new['device_ms']['median']*1000:.2f} | {increment:+.2f}% | {percentage(new['worst_head_relative_l2']['dk'])} | {goal} |"
            )
        lines.append("")
        if heldout.get("same_operands_as_misselected_run_verified"):
            lines += [
                "The first attempt at this held-out evaluation had a harness selection failure: a cached Python alias kept the direct statistics path while its report label said `column_staged`. Inspection found both scheduling flags false on all eight compiled KV modules. Its timings remain as a direct-path diagnostic in `negative_sweeps` and are excluded from the selected schedule table. The repaired harness reruns the identical eight shapes and seed-10816 operand hashes, with positive context counters and both actual compiled flags verified on all eight selected KV modules. Kernel source and the selected policy are unchanged; this is a harness repair, not retuning on held-out results.",
                "",
            ]
    eager = data.get("eager")
    if eager:
        lines += [
            "## Complete eager API cost",
            "",
            "These calls include output/statistics/scratch allocation operations, dispatch, autograd and the complete forward/backward API. The caching allocator and compilation are warmed; no CUDA graph hides per-call host work. Host wall medians and CUDA stream elapsed medians are both reported. CUDA elapsed time includes eager launch gaps, rather than measuring summed kernel execution time. Profiling runs are separate from timing.",
            "",
            "| case | prior wall us | new wall us | wall increment | prior device us | new device us |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name in dict.fromkeys(row["workload"]["name"] for row in eager["rows"]):
            rows = {row["implementation"]: row for row in eager["rows"] if row["workload"]["name"] == name}
            if any(key not in rows or rows[key]["status"] != "pass" for key in (PRIOR, TARGET)):
                continue
            old, new = rows[PRIOR], rows[TARGET]
            increment = new["comparisons"][PRIOR]["wall_ms"]["incremental_percent"]
            lines.append(
                f"| {name} | {old['wall_ms']['median']*1000:.2f} | {new['wall_ms']['median']*1000:.2f} | {increment:+.2f}% | {old['device_ms']['median']*1000:.2f} | {new['device_ms']['median']*1000:.2f} |"
            )
        lines.append("")
        paired = [
            value
            for row in eager["rows"]
            if row["implementation"] == TARGET
            for value in row.get("comparisons", {})
            .get(PRIOR, {})
            .get("wall_ms", {})
            .get("paired_round_latency_ratios", {})
            .get("samples", [])
        ]
        if paired:
            lines += [
                f"Paired eager wall-clock ratios vary from {(min(paired)-1)*100:+.2f}% to {(max(paired)-1)*100:+.2f}% across the retained rounds. The table's ratios of medians are descriptive and do not establish a precise small host-overhead bound; resident graph timing and eager host timing are separate measurements.",
                "",
            ]
    target_profiles = [
        row["profile"]["cuda_launches_per_call"]
        for row in data["timing"]
        if row["implementation"] == TARGET and "profile" in row
    ]
    if target_profiles:
        lines += [
            "## Executed path and qualification",
            "",
            "Untimed profiles retain actual launched-kernel family counts. The selected method has "
            + ", ".join(f"{value:g}" for value in sorted(set(target_profiles)))
            + " CUDA launches per complete call on the profiled cases. Profiled symbols and retained compiler-returned binaries remain in the raw archive; public metadata omits compiler object addresses. Exported names and resource metadata were matched to the profiles; direct CUPTI CUfunction-to-module binding was not established. Eager profiles also retain CPU empty/allocation operation counts when measured.",
            "",
        ]
    evidence = data.get("kernel_evidence")
    if evidence:
        intervals = evidence["profiled_kv_us"]
        counts = evidence["instruction_counts"]
        lines += [
            f"Separate instrumented KV intervals on the retained {evidence['workload']['dtype']} workload are {intervals['prior']:.2f} us for the prior method, {intervals['direct']:.2f} us for the direct-statistics new method, and {intervals['selected']:.2f} us for the selected schedule. These are kernel intervals from profiling, rather than complete-call medians.",
            "",
            f"Inspection of the matching retained KV binary finds {counts['direct_ldg']} repeated LDG sites in the direct path versus {counts['selected_ldg']} in the selected path, with {counts['selected_lds']} LDS sites and {counts['selected_stat_cp_async_sites']} asynchronous statistics-copy sites. HMMA counts remain {counts['direct_hmma']} and {counts['selected_hmma']} for the compared work. Static instruction counts do not establish runtime percentages or bytes transferred.",
            "",
        ]
    if not data["numerical_and_timing_source_identical"]:
        lines += [
            "Numerical and timing evidence use different retained source snapshots. Their hashes are kept separately; final scheduling qualification and direct-versus-selected byte checks must accompany any claim that the chosen statistics schedule preserves the measured arithmetic.",
            "",
        ]
    if data["validation"]:
        lines += [
            "Supplementary qualification receipts and direct-versus-selected byte checks:",
            "",
            "| evidence | stage | result | scope |",
            "| --- | --- | --- | --- |",
        ]
        for label, value in data["validation"].items():
            if value.get("stages"):
                for stage, record in value["stages"].items():
                    status = record.get(
                        "status",
                        "pass" if record.get("exit_status", record.get("controller_exit_status")) == 0 else "unavailable",
                    )
                    scope = ", ".join(
                        f"{record[key]} {key}"
                        for key in ("tests", "skipped", "methods", "errors", "hazards")
                        if key in record
                    )
                    lines.append(f"| {label} | {stage} | {status} | {scope} |")
            else:
                lines.append(f"| {label} | {value.get('stage','complete report')} | {value['status']} | |")
        lines.append("")
    if data["negative_sweeps"]:
        lines += [
            "Earlier direct-load and alternative scheduling measurements remain in JSON under `negative_sweeps`; their failed goals, failed qualification and shape regressions are retained separately from the selected final timing matrix.",
            "",
        ]
    lines += [
        "The method is explicit experimental arithmetic. Automatic production dispatch and the default precision contract are unchanged. Measurements qualify this hardware, stack and workload set; they do not establish a universal numerical bound or a speedup on another GPU.",
        "",
        "Regenerate from retained completed reports:",
        "",
        "```sh",
        "python3.13 -m benchmarks.neighborhood.summarize_key_precision \\",
        "  --directed RESULTS/directed/report.json --timing RESULTS/timing/report.json \\",
        "  --output results/neighborhood/dk-precision.json \\",
        "  --markdown docs/neighborhood/reports/dk-precision.md",
        "```",
        "",
        "Optional `--eager`, `--heldout`, `--path-audit`, `--kernel-evidence`, repeated `--validation LABEL=PATH` and `--negative-sweep LABEL=PATH` retain supplementary evidence. Selected held-out reports require context counters and the compiled-path audit.",
        "",
    ]
    return "\n".join(lines)


def labeled(arguments):
    result = {}
    for index, argument in enumerate(arguments or []):
        label, path = argument.split("=", 1) if "=" in argument else (f"evidence-{index+1}", argument)
        if not label or any(
            char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in label
        ):
            raise ValueError("Use a plain public evidence label.")
        if label in result:
            raise ValueError("Evidence labels must be unique.")
        result[label] = Path(path).read_bytes()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directed", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--eager", type=Path)
    parser.add_argument("--heldout", type=Path)
    parser.add_argument("--path-audit", type=Path)
    parser.add_argument("--kernel-evidence", type=Path)
    parser.add_argument("--validation", action="append")
    parser.add_argument("--negative-sweep", action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.directed.read_bytes(),
        args.timing.read_bytes(),
        eager_data=args.eager.read_bytes() if args.eager else None,
        heldout_data=args.heldout.read_bytes() if args.heldout else None,
        path_audit_data=args.path_audit.read_bytes() if args.path_audit else None,
        validations=labeled(args.validation),
        negative_sweeps=labeled(args.negative_sweep),
    )
    if args.kernel_evidence:
        evidence = args.kernel_evidence.read_bytes()
        result["kernel_evidence"] = kernel_evidence_summary(json.loads(evidence), evidence)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    args.markdown.write_text(markdown(result))


if __name__ == "__main__":
    main()
