"""Matched complete-call neighborhood experiments under the default arithmetic.

Raw outputs, operands, and traces belong outside the repository. The screen
changes one scheduling factor at a time before the combined candidate. It
does not enable an architecture-specific preset or change operand rounding.
"""

import argparse
import datetime
import hashlib
import importlib
import importlib.metadata
import json
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from benchmarks.neighborhood.backends import make_backend
from benchmarks.neighborhood.benchmark import (
    Execution,
    assess,
    atomic_save,
    error_record,
    execute,
    paired_order,
    profile_execution,
    snapshot_gpu,
    sparse_references,
    summarize,
    workloads,
)
from benchmarks.neighborhood.tuning import describe_columns, neighborhood_columns
from kohakufa_cute.neighborhood_geometry import normalize_exact_contiguous


@dataclass(frozen=True)
class Variant:
    name: str
    columns: tuple[int, int, int] | None = (32, 32, 32)
    materialize_output_grads: bool = True
    implementation: str = "kohakufa"
    exact_contiguous: bool | tuple[bool, bool, bool] | None = False


VARIANTS = {
    variant.name: variant
    for variant in (
        Variant("baseline"),
        Variant("no_stat_fills", materialize_output_grads=False),
        Variant("forward64", (64, 32, 32)),
        Variant("query64", (32, 64, 32)),
        Variant("keyvalue64", (32, 32, 64)),
        Variant("columns64", (64, 64, 64)),
        Variant("columns64_no_stat_fills", (64, 64, 64), False),
        Variant("flex", implementation="flex"),
        Variant("forward128", (128, 32, 32)),
        Variant("query128", (32, 128, 32)),
        Variant("keyvalue128", (32, 32, 128)),
        Variant("columns128", (128, 128, 128)),
        Variant("columns128_no_stat_fills", (128, 128, 128), False),
        Variant("keyvalue64_no_stat_fills", (32, 32, 64), False),
        Variant("keyvalue128_no_stat_fills", (32, 32, 128), False),
        Variant("exact32_no_stat_fills", materialize_output_grads=False, exact_contiguous=True),
        Variant("exactmixed_no_stat_fills", (32, 32, 64), False, exact_contiguous=True),
        Variant("exactmixed128_no_stat_fills", (32, 32, 128), False, exact_contiguous=True),
        Variant("exactmixed_noq_no_stat_fills", (32, 32, 64), False, exact_contiguous=(True, False, True)),
        Variant("production", columns=None, materialize_output_grads=False, exact_contiguous=None),
    )
}
SCREEN_VARIANTS = tuple(VARIANTS)[:8]
FULL_VARIANTS = ("baseline", "columns64_no_stat_fills", "flex")
SCREEN_CASES = {"64sq-w7-bf16", "64sq-w13-bf16", "tail-33x49-w7-bf16"}


def selected_workloads(suite, names=None):
    cases = workloads()
    if suite == "screen":
        cases = [case for case in cases if case.name in SCREEN_CASES]
    elif suite != "full":
        raise ValueError("Unknown tuning suite.")
    if names is not None:
        cases = [case for case in cases if case.name in names]
        if set(names) - {case.name for case in cases}:
            raise ValueError("Unknown or excluded workload requested.")
    if not cases:
        raise ValueError("No workloads selected.")
    return cases


@contextmanager
def variant_context(variant):
    """Keep one selection active through forward, autograd and graph checks.

    The same host selection wrapper is used for every variant, including Flex.
    Its eager overhead remains in reported intervals. Captured graph replay
    does not consult these Python flags.
    """
    ops = importlib.import_module("kohakufa_cute.neighborhood_ops")
    previous = ops._MATERIALIZE_OUTPUT_GRADS
    with neighborhood_columns(variant.columns, exact_contiguous=variant.exact_contiguous):
        ops._MATERIALIZE_OUTPUT_GRADS = variant.materialize_output_grads
        try:
            yield
        finally:
            ops._MATERIALIZE_OUTPUT_GRADS = previous


class VariantExecution(Execution):
    def __init__(self, variant, *args, **kwargs):
        self.variant = variant
        super().__init__(*args, **kwargs)

    def eager(self):
        with variant_context(self.variant):
            return super().eager()

    def changed_inputs(self):
        with variant_context(self.variant):
            return super().changed_inputs()


def baseline_equivalence(actual, baseline, *, required, baseline_selected=False):
    """Compare full tensors, including signed-zero bits, outside timed calls."""
    if baseline is None:
        return dict(
            status="unavailable_baseline_execution_failed" if baseline_selected else "unavailable_baseline_not_selected",
            required_for_variant=required,
            gate_applied=False,
        )
    components = {}
    for name, got, reference in zip(("out", "dq", "dk", "dv"), actual, baseline):
        got = got.detach().cpu()
        same_type = got.dtype == reference.dtype and got.shape == reference.shape
        components[name] = dict(
            torch_equal=bool(torch.equal(got, reference)),
            dtype_and_shape_equal=same_type,
            bitwise_equal=same_type
            and bool(torch.equal(got.contiguous().view(torch.uint8), reference.contiguous().view(torch.uint8))),
        )
    return dict(
        status="checked",
        required_for_variant=required,
        gate_applied=required,
        components=components,
        all_bitwise_equal=all(component["bitwise_equal"] for component in components.values()),
    )


SCHEDULE_REFERENCES = {
    "exact32_no_stat_fills": "baseline",
    "exactmixed_no_stat_fills": "keyvalue64_no_stat_fills",
    "exactmixed128_no_stat_fills": "keyvalue128_no_stat_fills",
    "exactmixed_noq_no_stat_fills": "keyvalue64_no_stat_fills",
}


def qualification_order(names):
    """Qualify selected old-traversal references before their candidates."""

    def candidate(name):
        value = VARIANTS[name].exact_contiguous
        return value is None or any(normalize_exact_contiguous(value))

    return sorted(names, key=lambda name: (name != "baseline", candidate(name)))


def schedule_equivalence(actual, references, name, selected, effective=None):
    """Require exact traversal to preserve its same-column result bits."""
    reference_name = SCHEDULE_REFERENCES.get(name)
    if name == "production":
        if effective is None:
            return dict(status="unavailable_effective_schedule", required_for_variant=True, gate_applied=False)
        reference_name = "baseline" if tuple(effective["columns"]) == (32, 32, 32) else "keyvalue64_no_stat_fills"
    if reference_name is None:
        return dict(status="not_required", gate_applied=False)
    result = baseline_equivalence(
        actual, references.get(reference_name), required=True, baseline_selected=reference_name in selected
    )
    result["reference_variant"] = reference_name
    result["comparison_scope"] = "Identical column widths and arithmetic; only inactive traversal slots differ."
    if result["status"].startswith("unavailable_baseline"):
        result["status"] = result["status"].replace("baseline", "reference")
    return result


def retain_failure(directory, name, case, inputs, dout, actual, baseline=None, schedule_reference=None):
    path = directory / (name + "-failure.pt")
    torch.save(
        dict(
            inputs=[tensor.detach().cpu() for tensor in inputs],
            dout=dout.detach().cpu(),
            actual=None if actual is None else [tensor.detach().cpu() for tensor in actual],
            baseline=baseline,
            schedule_reference=schedule_reference,
            workload=asdict(case),
        ),
        path,
    )
    return path.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=["screen", "full"], default="screen")
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS))
    parser.add_argument("--samples", type=int)
    parser.add_argument("--target-ms", type=float)
    parser.add_argument("--capture-batch", type=int, default=8)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--case", action="append")
    args = parser.parse_args()
    args.samples = args.samples if args.samples is not None else (5 if args.suite == "screen" else 7)
    args.target_ms = args.target_ms if args.target_ms is not None else (50.0 if args.suite == "screen" else 100.0)
    if args.samples < 3 or args.target_ms <= 0 or args.capture_batch < 1:
        parser.error("Need at least three samples, positive target time and capture batch.")
    root = Path(__file__).resolve().parents[2]
    if args.output.resolve().is_relative_to(root):
        parser.error("Write raw outputs outside the repository.")
    revision = subprocess_revision(root)
    if revision != args.source_revision:
        parser.error("Source base revision does not match.")
    try:
        cases = selected_workloads(args.suite, args.case)
    except ValueError as error:
        parser.error(str(error))
    names = args.variants or (SCREEN_VARIANTS if args.suite == "screen" else FULL_VARIANTS)
    if len(names) != len(set(names)):
        parser.error("Each variant must be named exactly once.")
    names = qualification_order(names)
    if not torch.cuda.is_available():
        parser.error("CUDA is required; skipped work is not a timing result.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch._dynamo.config.recompile_limit = 128
    torch._dynamo.config.accumulated_recompile_limit = 512
    properties = torch.cuda.get_device_properties(0)
    source_names = [
        "benchmarks/neighborhood/tune.py",
        "benchmarks/neighborhood/tuning.py",
        "benchmarks/neighborhood/benchmark.py",
        "benchmarks/neighborhood/backends.py",
        "benchmarks/neighborhood/reference.py",
        "benchmarks/reference.py",
        "docs/precision.md",
    ]
    sources = sorted((root / "src/kohakufa_cute").rglob("*.py")) + [root / name for name in source_names]
    packages = ["torch", "nvidia-cutlass-dsl", "apache-tvm-ffi", "natten", "triton", "cuda-bindings"]
    report = dict(
        schema="neighborhood-tuning-v1",
        status="running",
        suite=args.suite,
        source_revision=revision,
        source_sha256={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        python=sys.version,
        versions={package: importlib.metadata.version(package) for package in packages},
        device=dict(
            name=properties.name,
            capability=list(torch.cuda.get_device_capability()),
            sm_count=properties.multi_processor_count,
            memory_bytes=properties.total_memory,
        ),
        variants={name: {**asdict(VARIANTS[name]), **describe_columns(VARIANTS[name].columns)} for name in names},
        protocol=dict(
            scope="Complete contiguous BHSD D64 forward and forward-plus-backward calls; geometry resident.",
            arithmetic="docs/precision.md unchanged; compare mathematical FP64 and rounded-reference diagnostics separately.",
            baseline="C32 all owners, materialize_output_grads=True reproduces the prior two unused statistic-gradient fills.",
            hypothesis="C64 reduces repeated traversal/load/barrier work; per-owner tests isolate changed tiles, while no_stat_fills removes unused autograd operations.",
            modes=["graph"] if args.suite == "screen" else ["eager", "graph"],
            phases=["forward", "forward_backward"],
            samples=args.samples,
            target_ms=args.target_ms,
            capture_batch=args.capture_batch,
            sample_order="Rotating and alternating variant order for paired rounds.",
            host_scope="Eager intervals include explicit experiment selection controls for all variants; graph replay uses fixed captured kernels.",
            graph_validation="Changed Q/K/V/dO against fresh eager under the identical variant context; operands restored.",
            accuracy="Per-head sampled mathematical FP64 norm error: FP16 0.5%, BF16 1%; rounded diagnostics use actual rounded O.",
            exact_equivalence="When baseline is selected it qualifies first; same-C32 no_stat_fills must match every O/dQ/dK/dV byte, including signed zero. Exact traversal candidates must additionally match a selected old-traversal reference at identical column widths; schedule_equivalence identifies that reference. Other variants record full-tensor equality without a bitwise gate. Omitted references are explicit missing validation.",
            resource_scope="Compiler cubin/PTX, register/shared-memory/spill data must be retained by the campaign wrapper; staging estimates are not measured resources.",
        ),
        gpu_observations=[snapshot_gpu()],
        rows=[],
    )
    atomic_save(args.output, report)
    any_failure = False
    for case in cases:
        seed = 809 + workloads().index(case)
        torch.manual_seed(seed)
        dtype = torch.float16 if case.dtype == "fp16" else torch.bfloat16
        inputs = [torch.randn(case.shape, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)]
        dout = torch.randn_like(inputs[0])
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        active, rows = {}, {}
        baseline_cpu = None
        schedule_references = {}
        for name in names:
            variant, actual = VARIANTS[name], None
            row = dict(
                workload=asdict(case),
                seed=seed,
                workload_sha256=hashlib.sha256(json.dumps(asdict(case), sort_keys=True).encode()).hexdigest(),
                variant=name,
                backend=variant.implementation,
                configuration=asdict(variant),
                status="preparing",
                timings={},
                graph_checks={},
                profiles={},
            )
            report["rows"].append(row)
            rows[name] = row
            atomic_save(args.output, report)
            try:
                torch.cuda.synchronize()
                started = time.perf_counter()
                backend = make_backend(variant.implementation, case.spatial, case.window, case.scale, "cuda")
                row["construction_wall_ms"] = (time.perf_counter() - started) * 1000
                started = time.perf_counter()
                with torch.cuda.stream(stream), variant_context(variant):
                    actual = execute(backend.call, inputs, dout, True)
                    if variant.implementation == "kohakufa":
                        from kohakufa_cute.neighborhood_kernels import effective_schedule

                        columns, traversal = effective_schedule(inputs, case.spatial, case.window)
                        row["effective_schedule"] = dict(
                            columns=columns,
                            exact_contiguous_flags=traversal,
                            exact_contiguous_applied_by_owner={
                                owner: flag and 64 % case.spatial[1] == 0
                                for owner, flag in zip(("forward", "query", "keyvalue"), traversal)
                            },
                            metadata_source="Runtime launch resolver under the identical variant context.",
                        )
                    else:
                        row["effective_schedule"] = dict(status="other_implementation")
                stream.synchronize()
                row["first_forward_backward_wall_ms"] = (time.perf_counter() - started) * 1000
                if name == "baseline" or name in SCHEDULE_REFERENCES.values():
                    schedule_references[name] = tuple(tensor.detach().cpu().clone() for tensor in actual)
                if name == "baseline":
                    baseline_cpu = schedule_references[name]
                equivalence = baseline_equivalence(
                    actual, baseline_cpu, required=name == "no_stat_fills", baseline_selected="baseline" in names
                )
                row["baseline_equivalence"] = equivalence
                if equivalence["gate_applied"] and not equivalence["all_bitwise_equal"]:
                    raise ArithmeticError("Removing unused statistic-gradient fills changed numerical result bits.")
                exact_equivalence = schedule_equivalence(
                    actual, schedule_references, name, names, row.get("effective_schedule")
                )
                row["schedule_equivalence"] = exact_equivalence
                if exact_equivalence["gate_applied"] and not exact_equivalence["all_bitwise_equal"]:
                    raise ArithmeticError("Exact traversal changed numerical result bits at identical column widths.")
                if not all(bool(torch.isfinite(tensor).all()) for tensor in actual):
                    raise ArithmeticError("Nonfinite output or gradient.")
                references = sparse_references(inputs, dout, actual[0], case.spatial, case.window, scale=case.scale)
                errors = assess(actual, references, dtype)
                row["accuracy"] = dict(
                    passed=all(error["passed"] for error in errors), positions=references["positions"], errors=errors
                )
                row["backend_metadata"] = dict(backend.metadata)
                if not row["accuracy"]["passed"]:
                    raise ArithmeticError("Independent sampled FP64 gate failed.")
                active[name] = backend
                row["status"] = "ready"
            except Exception as error:
                row.update(status="qualification-error", error=error_record(error))
                try:
                    row["failed_operands"] = retain_failure(
                        args.output.parent,
                        case.name + "-" + name,
                        case,
                        inputs,
                        dout,
                        actual,
                        baseline_cpu,
                        schedule_references.get(row.get("schedule_equivalence", {}).get("reference_variant")),
                    )
                except Exception as retention_error:
                    row["operand_retention_error"] = error_record(retention_error)
                any_failure = True
            atomic_save(args.output, report)
        for phase in ("forward", "forward_backward"):
            for mode in report["protocol"]["modes"]:
                executions = {}
                key = phase + "/" + mode
                for name, backend in active.items():
                    row = rows[name]
                    timing = dict(status="preparing", samples=[])
                    row["timings"][key] = timing
                    atomic_save(args.output, report)
                    try:
                        started = time.perf_counter()
                        execution = VariantExecution(
                            VARIANTS[name], backend, inputs, dout, phase, mode, stream, args.capture_batch
                        )
                        timing["warm_and_capture_wall_ms"] = (time.perf_counter() - started) * 1000
                        if mode == "graph":
                            check = execution.changed_inputs()
                            row["graph_checks"][phase] = check
                            if not check["passed"]:
                                filename = case.name + "-" + name + "-" + phase + "-graph-failure.pt"
                                torch.save(execution.failed_probe, args.output.parent / filename)
                                check["failed_operands"] = filename
                                raise ArithmeticError("Graph changed-input validation failed.")
                        count = execution.calibrate(args.target_ms)
                        for _ in range(3):
                            execution.sample(count)
                        executions[name] = execution, count
                        timing.update(status="sampling", calls_per_graph=execution.batch, repeats_per_sample=count)
                    except Exception as error:
                        timing.update(status="execution-error", error=error_record(error))
                        any_failure = True
                    atomic_save(args.output, report)
                for repeat in range(args.samples):
                    for name in paired_order(executions, repeat):
                        execution, count = executions[name]
                        timing = rows[name]["timings"][key]
                        if timing["status"] == "execution-error":
                            continue
                        try:
                            timing["samples"].append(dict(round=repeat, **execution.sample(count)))
                        except Exception as error:
                            timing.update(status="execution-error", error=error_record(error))
                            any_failure = True
                        atomic_save(args.output, report)
                for name, (execution, count) in executions.items():
                    timing = rows[name]["timings"][key]
                    if timing["status"] != "execution-error":
                        timing.update(
                            status="pass",
                            device_ms=summarize([sample["device_ms"] for sample in timing["samples"]]),
                            wall_ms=summarize([sample["wall_ms"] for sample in timing["samples"]]),
                        )
                    if args.profile:
                        try:
                            path = args.output.parent / f"{case.name}-{name}-{phase}-{mode}.trace.json"
                            rows[name]["profiles"][key] = profile_execution(execution, path)
                        except Exception as error:
                            rows[name]["profiles"][key] = dict(status="profile-error", error=error_record(error))
                            any_failure = True
                    atomic_save(args.output, report)
                del executions
                execution = None
                torch.cuda.synchronize()
        for row in rows.values():
            if row["status"] == "ready":
                row["status"] = (
                    "pass" if all(timing["status"] == "pass" for timing in row["timings"].values()) else "partial-failure"
                )
        report["gpu_observations"].append(snapshot_gpu())
        atomic_save(args.output, report)
        del active, inputs, dout
        actual = backend = references = baseline_cpu = None
        schedule_references.clear()
        torch.cuda.empty_cache()
    report.update(
        status="completed-with-failures" if any_failure else "pass",
        finished_utc=datetime.datetime.now(datetime.UTC).isoformat(),
    )
    atomic_save(args.output, report)
    raise SystemExit(1 if any_failure else 0)


def subprocess_revision(root):
    import subprocess

    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


if __name__ == "__main__":
    main()
