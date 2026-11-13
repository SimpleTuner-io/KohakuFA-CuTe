"""Probe native neighborhood implementations against mathematical FP64.

This compact diagnostic measures no latency. Keep operands, outputs and traces
outside the repository. Comparator differences from the candidate's rounded
contract remain visible and do not redefine that contract.
"""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import torch

from benchmarks.neighborhood.backends import make_backend
from benchmarks.neighborhood.qualify import assess, execute
from benchmarks.neighborhood.reference import neighborhood_mask
from benchmarks.reference import fp64_reference


def cases(device="cuda"):
    """Original seed-792 offset construction plus qualified small controls."""
    for dtype in (torch.float16, torch.bfloat16):
        torch.manual_seed(791)
        inputs = [torch.randn(1, 2, 45, 64, dtype=dtype, device=device) for _ in range(3)]
        yield "random", inputs, torch.randn_like(inputs[0]), (5, 9), (3, 3), 0.125, False, 791
        for sign in (1, -1):
            torch.manual_seed(792)
            inputs = [torch.zeros(1, 2, 45, 64, dtype=dtype, device=device) for _ in range(3)]
            q, k, v = inputs
            q[..., 0], q[..., 1] = 64, 1
            k[..., 0] = sign * 64
            k[..., 1] = torch.arange(45, device=device) % 7
            v.copy_(torch.randn_like(v))
            yield "common-offset", inputs, torch.randn_like(q), (5, 9), (3, 3), sign * 0.125, True, 792
        torch.manual_seed(792)
        inputs = [torch.zeros(1, 2, 45, 64, dtype=dtype, device=device) for _ in range(3)]
        q, k, v = inputs
        q[..., 0], q[..., 1] = 1024, 1
        k[..., 0] = 1024
        k[..., 1] = torch.arange(45, device=device) % 7
        v.copy_(torch.randn_like(v))
        yield "large-common-offset", inputs, torch.randn_like(q), (5, 9), (3, 3), 0.125, True, 792
        inputs = [torch.zeros(1, 1, 5, 64, dtype=dtype, device=device) for _ in range(3)]
        yield "uniform-inverse-boundary", inputs, torch.ones_like(inputs[0]), (1, 5), (1, 3), 0.125, True, None
    inputs = [torch.zeros(1, 1, 5, 64, dtype=torch.float16, device=device) for _ in range(3)]
    inputs[0][..., 0] = 1
    inputs[1][..., 0, 0] = -20
    yield "probability-underflow", inputs, torch.full_like(inputs[0], 8192), (1, 5), (1, 3), 1.0, True, None


def tensor_hash(values):
    digest = hashlib.sha256()
    for value in values:
        data = value.detach().cpu().contiguous()
        digest.update(str((str(data.dtype), tuple(data.shape))).encode())
        digest.update(data.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def save(path, report):
    serializable = json.loads(json.dumps(report), parse_constant=lambda value: None)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(serializable, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backends", nargs="+", choices=["kohakufa", "flex", "natten"], default=["kohakufa", "flex", "natten"]
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--reference-agreement", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.output.resolve().is_relative_to(root):
        parser.error("Keep raw accuracy artifacts outside the repository.")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if revision != args.source_revision:
        parser.error("Snapshot base revision does not match.")
    if not torch.cuda.is_available():
        parser.error("CUDA is required; a skipped run is not qualification.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch._dynamo.config.recompile_limit = 128
    torch._dynamo.config.accumulated_recompile_limit = 512
    sources = sorted((root / "src/kohakufa_cute").rglob("*.py")) + [
        Path(__file__),
        root / "benchmarks/neighborhood/backends.py",
        root / "benchmarks/neighborhood/reference.py",
        root / "benchmarks/neighborhood/qualify.py",
        root / "benchmarks/reference.py",
        root / "docs/precision.md",
    ]
    properties = torch.cuda.get_device_properties(0)
    report = dict(
        status="running",
        source_revision=revision,
        source_sha256={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        versions={
            name: importlib.metadata.version(name)
            for name in ("torch", "nvidia-cutlass-dsl", "apache-tvm-ffi", "natten", "triton", "cuda-bindings")
        },
        device=dict(
            name=properties.name,
            capability=list(torch.cuda.get_device_capability()),
            sm_count=properties.multi_processor_count,
            memory_bytes=properties.total_memory,
        ),
        reference=dict(
            mathematical="Native torch.float64 equations on CPU, identical already-rounded input operands; no emulated-double algorithm.",
            rounded="Existing single-rounded P/dS contract with actual rounded O; final products FP64, QK/dP FP32.",
            cuda_reference_gates_also_required=args.reference_agreement,
            limitations="FP64 is finite precision, not exact real arithmetic; pre-quantization input error is excluded.",
        ),
        acceptance=dict(
            fp16=0.005,
            bf16=0.01,
            random="Separate full-tensor per-head mathematical FP64 relative norm limits for output and each gradient.",
            directed="Separate per-head rounded-reference limits. Mathematical FP64 errors are diagnostics, not replaced by the rounded errors.",
            comparator="Same reported gates are mathematical accuracy or candidate-policy compatibility diagnostics. Comparator numerical differences do not fail the candidate campaign.",
        ),
        rows=[],
    )
    save(args.output, report)
    execution_errors = 0
    candidate_failures = 0
    try:
        # Freeze every tiny case before a possible device error prevents copies.
        frozen = []
        for index, fixture in enumerate(cases()):
            label, inputs, dout, spatial, window, scale, directed, seed = fixture
            cpu = [value.detach().cpu() for value in inputs]
            filename = f"{args.output.stem}-case-{index}-inputs.pt"
            torch.save(
                dict(inputs=cpu, dout=dout.cpu(), spatial=spatial, window=window, scale=scale, seed=seed, fixture=label),
                args.output.parent / filename,
            )
            frozen.append((fixture, filename, tensor_hash([*cpu, dout.cpu()])))
        for index, (fixture, filename, operand_hash) in enumerate(frozen):
            label, inputs, dout, spatial, window, scale, directed, seed = fixture
            for name in args.backends:
                row = dict(
                    case_index=index,
                    case=label,
                    implementation=name,
                    dtype=str(inputs[0].dtype),
                    shape=list(inputs[0].shape),
                    spatial=spatial,
                    window=window,
                    scale=scale,
                    seed=seed,
                    directed=directed,
                    compatibility_probe=scale < 0,
                    operand_artifact=filename,
                    operand_sha256=operand_hash,
                    gate_reference="rounded" if directed else "fp64",
                    status="running",
                )
                report["rows"].append(row)
                save(args.output, report)
                try:
                    backend = make_backend(name, spatial, window, scale, device="cuda")
                    actual = execute(backend.call, inputs, dout)
                    torch.cuda.synchronize()
                    row["backend_metadata"] = backend.metadata
                    row["errors"] = assess(actual, inputs, dout, spatial, window, scale, directed=directed)
                    passed = all(record["pass"] for record in row["errors"])
                    if args.reference_agreement:
                        row["device_reference_errors"] = assess(
                            actual, inputs, dout, spatial, window, scale, directed=directed, device="cuda"
                        )
                        passed &= all(record["pass"] for record in row["device_reference_errors"])
                    row["status"] = "pass" if passed else "numerical-difference"
                    if name == "kohakufa" and not passed:
                        candidate_failures += 1
                    actual_file = f"{args.output.stem}-case-{index}-{name}-actual.pt"
                    torch.save(
                        dict(actual=[tensor.detach().cpu() for tensor in actual], inputs_artifact=filename),
                        args.output.parent / actual_file,
                    )
                    row["actual_artifact"] = actual_file
                    if label == "probability-underflow":
                        exact = fp64_reference(
                            [tensor.detach().cpu() for tensor in inputs],
                            dout.cpu(),
                            neighborhood_mask(spatial, window),
                            scale=scale,
                        )
                        row["underflow_observation"] = dict(
                            mathematical_dv_first_key_first_channel=float(exact[3][0, 0, 0, 0]),
                            actual_dv_first_key_first_channel=float(actual[3][0, 0, 0, 0]),
                            actual_nonzeros_first_key=int(actual[3][0, 0, 0].count_nonzero()),
                        )
                    if args.profile:
                        with torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
                        ) as trace:
                            execute(backend.call, inputs, dout)
                            torch.cuda.synchronize()
                        trace_path = args.output.with_name(f"{args.output.stem}-case-{index}-{name}.trace.json")
                        trace.export_chrome_trace(str(trace_path))
                        row["trace_file"] = trace_path.name
                        row["launched_cuda_kernels"] = [
                            event.name for event in trace.events() if event.device_type == torch.autograd.DeviceType.CUDA
                        ]
                except Exception as error:
                    execution_errors += 1
                    row.update(status="execution-error", error_type=type(error).__name__, error_message=str(error))
                save(args.output, report)
    except BaseException as error:
        report.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "execution-error",
            error_type=type(error).__name__,
        )
        save(args.output, report)
        raise
    report.update(
        status="execution-error" if execution_errors else ("candidate-numerical-failure" if candidate_failures else "pass"),
        candidate_failures=candidate_failures,
        execution_errors=execution_errors,
        comparator_numerical_differences=sum(
            row["implementation"] != "kohakufa" and row["status"] == "numerical-difference" for row in report["rows"]
        ),
    )
    save(args.output, report)
    raise SystemExit(1 if execution_errors or candidate_failures else 0)


if __name__ == "__main__":
    main()
