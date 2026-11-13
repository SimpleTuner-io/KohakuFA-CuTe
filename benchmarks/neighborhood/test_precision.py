"""Measure explicit stronger backward equations and complete-call cost.

Run on the rented CUDA host. Raw operands, reports, traces and samples must
remain outside the repository. Mathematical FP64 diagnostics and prescribed
rounded-equation qualification are separate; neither comparator is an oracle.
"""

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import torch

from benchmarks.neighborhood.backends import make_backend
from benchmarks.neighborhood.benchmark import (
    Execution,
    assess,
    atomic_save,
    paired_order,
    profile_execution,
    snapshot_gpu,
    sparse_references,
    summarize,
    workloads,
)
from benchmarks.neighborhood.precision import VARIANTS, variant
from benchmarks.neighborhood.precision_reference import (
    component_metrics,
    counterfactual_dq,
    mathematical_reference,
    precision_reference,
    reference_gate,
)
from benchmarks.neighborhood.qualify import execute
from benchmarks.neighborhood.qualify_accuracy import cases, tensor_hash
from benchmarks.reference import metrics

# Kept independently of production configuration so the reference is a check
# of the prescribed experiment, rather than a reflection of kernel flags.
POLICIES = {
    "baseline": {},
    "fp32_p": dict(fp32_p=True),
    "weighted_dq": dict(correction="normalized"),
    "fp32_p_weighted_dq": dict(fp32_p=True, correction="normalized"),
    "legacy_weighted": dict(fp32_p=True, correction="weighted"),
}


def context(name):
    return variant(name) if name in VARIANTS else contextlib.nullcontext()


def fixtures():
    yield from cases()
    for dtype in (torch.float16, torch.bfloat16):
        for sign in (1, -1):
            torch.manual_seed(177)
            inputs = [torch.randn(1, 2, 153, 64, dtype=dtype, device="cuda") for _ in range(3)]
            yield "partial-even-window", inputs, torch.randn_like(inputs[0]), (9, 17), (3, 4), sign * 0.125, False, 177
            torch.manual_seed(178)
            inputs = [torch.randn(1, 2, 45, 64, dtype=dtype, device="cuda") for _ in range(3)]
            inputs[0].mul_(16)
            inputs[1].mul_(16)
            yield "sharp-signed", inputs, torch.randn_like(inputs[0]), (5, 9), (3, 3), sign * 0.125, True, 178
        torch.manual_seed(179)
        inputs = [torch.zeros(1, 2, 45, 64, dtype=dtype, device="cuda") for _ in range(3)]
        inputs[0][..., 0], inputs[0][..., 1] = 1024, 1
        inputs[1][..., 0] = -1024
        inputs[1][..., 1] = torch.arange(45, device="cuda") % 7
        inputs[2].copy_(torch.randn_like(inputs[2]))
        yield "large-common-offset-negative", inputs, torch.randn_like(inputs[0]), (5, 9), (3, 3), -0.125, True, 179


def paired_invariants(actual, baseline, name):
    components = [0]
    if name in ("weighted_dq",):
        components += [2, 3]
    elif name in ("fp32_p", "fp32_p_weighted_dq", "legacy_weighted"):
        components += [3]
    return dict(
        passed=all(
            torch.equal(actual[index].contiguous().view(torch.uint8), baseline[index].contiguous().view(torch.uint8))
            for index in components
        ),
        exact_components=[("out", "dq", "dk", "dv")[index] for index in components],
    )


def directed(args, report):
    failures = 0
    for index, fixture in enumerate(fixtures()):
        label, inputs, dout, spatial, window, scale, stress, seed = fixture
        if args.fixture and label not in args.fixture:
            continue
        cpu = [x.detach().cpu() for x in inputs]
        gradient = dout.cpu()
        artifact = args.output.parent / f"{args.output.stem}-fixture-{index}.pt"
        torch.save(dict(inputs=cpu, dout=gradient, spatial=spatial, window=window, scale=scale, seed=seed), artifact)
        exact = mathematical_reference(cpu, gradient, spatial, window, scale)
        baseline = None
        names = list(dict.fromkeys(["baseline", *args.variants, *args.backends]))
        for name in names:
            row = dict(
                stage="directed",
                fixture=label,
                fixture_index=index,
                implementation=name,
                dtype=str(inputs[0].dtype),
                spatial=spatial,
                window=window,
                scale=scale,
                directed=stress,
                seed=seed,
                operand_artifact=artifact.name,
                operand_sha256=tensor_hash([*cpu, gradient]),
                status="running",
            )
            report["rows"].append(row)
            atomic_save(args.output, report)
            try:
                backend = make_backend("kohakufa" if name in VARIANTS else name, spatial, window, scale, device="cuda")
                with context(name):
                    actual = execute(backend.call, inputs, dout)
                torch.cuda.synchronize()
                actual = tuple(x.detach().cpu() for x in actual)
                actual_path = args.output.parent / f"{args.output.stem}-fixture-{index}-{name}.pt"
                torch.save(dict(actual=actual, operand_artifact=artifact.name), actual_path)
                row["actual_artifact"] = actual_path.name
                row["backend_metadata"] = backend.metadata
                row["fp64"] = component_metrics(actual, exact)
                if name == "baseline":
                    baseline = actual
                    counterfactuals = counterfactual_dq(cpu, gradient, spatial, window, scale, actual[0])
                    row["counterfactual_dq"] = {
                        policy: [
                            dict(batch=b, head=h, **metrics(value[b, h], exact[1][b, h]))
                            for b in range(value.shape[0])
                            for h in range(value.shape[1])
                        ]
                        for policy, value in counterfactuals.items()
                    }
                if name in VARIANTS:
                    expected = precision_reference(cpu, gradient, spatial, window, scale, actual[0], **POLICIES[name])
                    row["formula"] = reference_gate(actual, expected, inputs[0].dtype)
                    row["paired_invariants"] = paired_invariants(actual, baseline, name)
                    passed = row["formula"]["passed"] and row["paired_invariants"]["passed"]
                    if not stress:
                        row["ordinary_fp64_gate"] = reference_gate(actual, exact, inputs[0].dtype)
                        passed &= row["ordinary_fp64_gate"]["passed"]
                    row["status"] = "pass" if passed else "qualification-failure"
                    failures += not passed
                else:
                    row["status"] = "diagnostic-complete"
            except Exception as error:
                row.update(status="execution-error", error_type=type(error).__name__, error=str(error))
                failures += name in VARIANTS
            atomic_save(args.output, report)
    return failures


def timing(args, report):
    selected = [case for case in workloads("full") if not args.case or case.name in args.case]
    if args.case and set(args.case) - {case.name for case in selected}:
        raise ValueError("Unknown timing case.")
    failures = 0
    for case in selected:
        torch.manual_seed(809)
        dtype = torch.float16 if case.dtype == "fp16" else torch.bfloat16
        inputs = [torch.randn(case.shape, device="cuda", dtype=dtype).requires_grad_() for _ in range(3)]
        dout = torch.randn_like(inputs[0])
        artifact = args.output.parent / f"{args.output.stem}-{case.name}-inputs.pt"
        torch.save(dict(inputs=[x.detach().cpu() for x in inputs], dout=dout.cpu()), artifact)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        executions, rows = {}, {}
        for name in list(dict.fromkeys(["baseline", *args.variants, *args.backends])):
            row = dict(
                stage="timing",
                workload=case.__dict__,
                implementation=name,
                status="running",
                samples=[],
                operand_artifact=artifact.name,
                operand_sha256=tensor_hash([*inputs, dout]),
            )
            rows[name] = row
            report["rows"].append(row)
            atomic_save(args.output, report)
            try:
                backend = make_backend(
                    "kohakufa" if name in VARIANTS else name, case.spatial, case.window, case.scale, device="cuda"
                )
                with context(name):
                    execution = Execution(backend, inputs, dout, "forward_backward", "graph", stream, args.capture_batch)
                    row["graph_check"] = execution.changed_inputs()
                    row["repeats"] = execution.calibrate(args.target_ms)
                    if args.profile:
                        trace = args.output.parent / f"{args.output.stem}-{case.name}-{name}.trace.json"
                        row["profile"] = profile_execution(execution, trace)
                row["backend_metadata"] = backend.metadata
                references = sparse_references(
                    inputs, dout, execution.output[0], case.spatial, case.window, scale=case.scale
                )
                row["accuracy"] = assess(execution.output, references, dtype)
                row["reference_positions"] = references["positions"]
                row["status"] = (
                    "prepared"
                    if row["graph_check"]["passed"] and all(x["passed"] for x in row["accuracy"])
                    else "qualification-failure"
                )
                if row["status"] == "prepared":
                    executions[name] = execution
                elif name in VARIANTS:
                    failures += 1
                if hasattr(execution, "failed_probe"):
                    failpath = args.output.parent / f"{args.output.stem}-{case.name}-{name}-graph-failure.pt"
                    torch.save(execution.failed_probe, failpath)
                    row["graph_failure_artifact"] = failpath.name
            except Exception as error:
                row.update(status="execution-error", error_type=type(error).__name__, error=str(error))
                failures += name in VARIANTS
            atomic_save(args.output, report)
        for repeat in range(args.samples):
            for name in paired_order(executions, repeat):
                with context(name):
                    sample = executions[name].sample(rows[name]["repeats"])
                rows[name]["samples"].append(dict(round=repeat, **sample))
                atomic_save(args.output, report)
        for name in executions:
            row = rows[name]
            row["device_ms"] = summarize([sample["device_ms"] for sample in row["samples"]])
            row["wall_ms"] = summarize([sample["wall_ms"] for sample in row["samples"]])
            row["status"] = "pass"
        report["gpu_observations"].append(snapshot_gpu())
        atomic_save(args.output, report)
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("directed", "timing", "all"), default="all")
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    parser.add_argument("--backends", choices=("flex", "natten"), nargs="*", default=[])
    parser.add_argument(
        "--fixture",
        action="append",
        choices=(
            "random",
            "common-offset",
            "large-common-offset",
            "uniform-inverse-boundary",
            "probability-underflow",
            "partial-even-window",
            "sharp-signed",
            "large-common-offset-negative",
        ),
    )
    parser.add_argument("--case", action="append")
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--target-ms", type=float, default=100)
    parser.add_argument("--capture-batch", type=int, default=8)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.output.resolve().is_relative_to(root):
        parser.error("Write raw artifacts outside the repository.")
    if args.samples < 3 or args.target_ms <= 0 or args.capture_batch < 1:
        parser.error("Need three samples, positive target and capture batch.")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if revision != args.source_revision:
        parser.error("Source base revision does not match frozen snapshot.")
    if not torch.cuda.is_available():
        parser.error("CUDA required; skipped work is not qualification.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch._dynamo.config.recompile_limit = 128
    torch._dynamo.config.accumulated_recompile_limit = 512
    sources = (
        sorted((root / "src/kohakufa_cute").rglob("*.py"))
        + list((root / "benchmarks/neighborhood").glob("*.py"))
        + [root / "benchmarks/reference.py", root / "docs/precision.md", root / "tests/test_neighborhood_precision.py"]
    )
    packages = ("torch", "nvidia-cutlass-dsl", "apache-tvm-ffi", "natten", "triton", "cuda-bindings")
    versions = {}
    for name in packages:
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            versions[name] = importlib.metadata.version(name)
    properties = torch.cuda.get_device_properties(0)
    report = dict(
        status="running",
        source_revision=revision,
        versions=versions,
        cuda=torch.version.cuda,
        source_sha256={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        device=dict(
            name=properties.name,
            capability=torch.cuda.get_device_capability(),
            sm_count=properties.multi_processor_count,
            memory_bytes=properties.total_memory,
        ),
        protocol=dict(
            reference="Native FP64 on identical rounded operands; no emulated FP64.",
            formula="Independent FP32 QK/dP/delta and prescribed P/dS casts, final FP64 products; reduction order not replayed.",
            directed_fp64="Diagnostic; does not redefine declared rounded equations.",
            ordinary_fp16_limit=0.005,
            ordinary_bf16_limit=0.01,
            timing="Warmed complete forward plus backward; resident CUDA graph device events. Backend layout copies inside call; immutable geometry preparation separately recorded.",
            samples=args.samples,
            target_ms=args.target_ms,
            capture_batch=args.capture_batch,
        ),
        gpu_observations=[snapshot_gpu()],
        rows=[],
    )
    atomic_save(args.output, report)
    failures = 0
    try:
        if args.stage in ("directed", "all"):
            failures += directed(args, report)
        if args.stage in ("timing", "all"):
            failures += timing(args, report)
        report["status"] = "pass" if failures == 0 else "qualification-failure"
        report["candidate_failures"] = failures
    except BaseException as error:
        report.update(status="execution-error", error_type=type(error).__name__, error=str(error))
        atomic_save(args.output, report)
        raise
    atomic_save(args.output, report)
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
