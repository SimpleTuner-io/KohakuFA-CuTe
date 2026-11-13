"""Retain independent neighborhood accuracy and optional NATTEN comparison.

This runner measures no latency. Keep its JSON, operands, and profiler captures
outside the tracked tree; review publication artifacts with check_public_text.
"""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import torch

from benchmarks.neighborhood.key_precision_reference import key_precision_reference
from benchmarks.neighborhood.reference import neighborhood_mask
from benchmarks.reference import fp64_reference, metrics, rounded_reference
from kohakufa_cute import neighborhood_attention


def assess(actual, inputs, dout, spatial, window, scale, *, directed, device="cpu", policy="weighted_dq_reconstructed_dk"):
    operands = [t.detach().to(device) for t in inputs]
    outputs = [t.detach().to(device) for t in actual]
    gradient = dout.detach().to(device)
    mask = neighborhood_mask(spatial, window, device=device)
    exact = fp64_reference(operands, gradient, mask, scale=scale)
    rounded = rounded_reference(operands, gradient, mask, scale=scale, output=outputs[0])
    prescribed = key_precision_reference(
        operands,
        gradient,
        spatial,
        window,
        inputs[0].shape[-1] ** -0.5 if scale is None else scale,
        outputs[0],
        method=policy,
    )
    limit = 0.01 if inputs[0].dtype == torch.bfloat16 else 0.005
    records = []
    for name, got, mathematical, contract, selected in zip(("out", "dq", "dk", "dv"), outputs, exact, rounded, prescribed):
        for batch in range(got.shape[0]):
            for head in range(got.shape[1]):
                errors = {
                    "fp64": metrics(got[batch, head], mathematical[batch, head]),
                    "rounded": metrics(got[batch, head], contract[batch, head]),
                    "policy": metrics(got[batch, head], selected[batch, head]),
                }
                required = errors["policy" if directed else "fp64"]
                passed = required["finite"] and (
                    required["relative_error"] < limit if required["reference_norm"] else required["max_absolute_error"] == 0
                )
                records.append({"component": name, "batch": batch, "head": head, "pass": passed, **errors})
    return records


def execute(function, inputs, dout):
    operands = [t.detach().requires_grad_() for t in inputs]
    out = function(*operands)
    return out, *torch.autograd.grad(out, operands, dout)


def cases(device="cuda"):
    for dtype in (torch.float16, torch.bfloat16):
        for spatial, window in (((3, 5), (3, 3)), ((9, 17), (3, 4)), ((5, 7), (5, 7))):
            for scale in (0.125, -0.125, 0.0):
                torch.manual_seed(791)
                shape = (1, 2, spatial[0] * spatial[1], 64)
                inputs = [torch.randn(shape, device=device, dtype=dtype) for _ in range(3)]
                yield "random", inputs, torch.randn_like(inputs[0]), spatial, window, scale, False
        for sign in (1, -1):
            torch.manual_seed(792)
            inputs = [torch.zeros(1, 2, 45, 64, device=device, dtype=dtype) for _ in range(3)]
            q, k, v = inputs
            q[..., 0], q[..., 1] = 64, 1
            k[..., 0] = sign * 64
            k[..., 1] = torch.arange(45, device=device) % 7
            v.copy_(torch.randn_like(v))
            yield "common-offset", inputs, torch.randn_like(q), (5, 9), (3, 3), sign * 0.125, True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-revision", required=True, help="Expected snapshot base commit; file hashes also retain patches."
    )
    parser.add_argument("--compare-natten", action="store_true")
    parser.add_argument(
        "--natten-backend", help="Explicit NATTEN backend, such as cutlass-fna; availability is device-specific."
    )
    parser.add_argument(
        "--reference-agreement", action="store_true", help="Also require the same gates with device references."
    )
    parser.add_argument(
        "--profile", action="store_true", help="Record launched CUDA kernel names in separate instrumented calls."
    )
    args = parser.parse_args()
    if args.compare_natten and not args.natten_backend:
        parser.error("--compare-natten requires --natten-backend")
    root = Path(__file__).resolve().parents[2]
    if args.output.resolve().is_relative_to(root):
        parser.error("Raw qualification artifacts must be outside the repository.")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if revision != args.source_revision:
        parser.error("Source revision does not match the snapshot base.")
    if not torch.cuda.is_available():
        parser.error("CUDA is required; a skipped run is not qualification.")
    torch.set_num_threads(4)
    properties = torch.cuda.get_device_properties(0)
    sources = sorted((root / "src/kohakufa_cute").rglob("*.py")) + [
        Path(__file__),
        root / "benchmarks/neighborhood/reference.py",
        root / "benchmarks/neighborhood/key_precision_reference.py",
        root / "benchmarks/neighborhood/precision_reference.py",
        root / "benchmarks/reference.py",
        root / "tests/test_neighborhood.py",
        root / "docs/precision.md",
    ]
    packages = ["torch", "nvidia-cutlass-dsl", "apache-tvm-ffi"] + (["natten"] if args.compare_natten else [])
    report = {
        "status": "running",
        "source_revision": revision,
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "versions": {name: importlib.metadata.version(name) for name in packages},
        "cuda": torch.version.cuda,
        "device": properties.name,
        "capability": list(torch.cuda.get_device_capability()),
        "sm_count": properties.multi_processor_count,
        "total_memory": properties.total_memory,
        "scope": "Synthetic attention forward and backward correctness; no HDiT model, transforms, or latency measured.",
        "neighborhood_policy": "weighted_dq_reconstructed_dk",
        "key_statistics": "column_staged",
        "acceptance": {
            "random": "Per-head mathematical FP64 norm error",
            "directed": "Per-head prescribed neighborhood-policy norm error; historical rounded diagnostics retained",
            "bf16": 0.01,
            "fp16": 0.005,
        },
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        # Preserve the finite flags while representing undefined diagnostics as
        # JSON null, including failures returned by another implementation.
        serializable = json.loads(json.dumps(report), parse_constant=lambda value: None)
        args.output.write_text(json.dumps(serializable, indent=2, allow_nan=False) + "\n")

    save()
    retained = None
    try:
        for index, (label, inputs, dout, spatial, window, scale, directed) in enumerate(cases()):
            # Preserve operands before a device error can make transfers fail.
            retained = {"inputs": [t.cpu() for t in inputs], "dout": dout.cpu()}
            functions = {
                "kohakufa": lambda q, k, v: neighborhood_attention(
                    q, k, v, spatial_shape=spatial, window_size=window, scale=scale
                )
            }
            if args.compare_natten:
                from natten import na2d

                def natten(q, k, v):
                    tensors = [t.reshape(t.shape[0], t.shape[1], *spatial, 64).permute(0, 2, 3, 1, 4) for t in (q, k, v)]
                    out = na2d(
                        *tensors,
                        kernel_size=window,
                        stride=1,
                        dilation=1,
                        is_causal=False,
                        scale=scale,
                        backend=args.natten_backend,
                    )
                    return out.permute(0, 3, 1, 2, 4).flatten(2, 3)

                functions["natten"] = natten
            for implementation, function in functions.items():
                row = {
                    "case": label,
                    "implementation": implementation,
                    "dtype": str(inputs[0].dtype),
                    "shape": list(inputs[0].shape),
                    "spatial": spatial,
                    "window": window,
                    "scale": scale,
                    "directed": directed,
                    "status": "running",
                }
                if implementation == "natten":
                    row["requested_backend"] = args.natten_backend
                report["rows"].append(row)
                save()
                actual = execute(function, inputs, dout)
                errors = assess(actual, inputs, dout, spatial, window, scale, directed=directed)
                row["errors"] = errors
                passed = all(record["pass"] for record in errors)
                if args.reference_agreement:
                    row["device_reference_errors"] = assess(
                        actual, inputs, dout, spatial, window, scale, directed=directed, device="cuda"
                    )
                    passed &= all(record["pass"] for record in row["device_reference_errors"])
                row["status"] = "pass" if passed else "numerical-failure"
                if not passed:
                    artifact = args.output.with_name(f"{args.output.stem}-case-{index}-{implementation}.pt")
                    torch.save(
                        {
                            "inputs": [t.cpu() for t in inputs],
                            "dout": dout.cpu(),
                            "actual": [t.detach().cpu() for t in actual],
                        },
                        artifact,
                    )
                    row["operands"] = artifact.name
                if args.profile:
                    from torch.profiler import ProfilerActivity, profile

                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as capture:
                        execute(function, inputs, dout)
                        torch.cuda.synchronize()
                    trace = args.output.with_name(f"{args.output.stem}-case-{index}-{implementation}.trace.json")
                    capture.export_chrome_trace(str(trace))
                    row["trace_file"] = trace.name
                    row["launched_cuda_kernels"] = [
                        event.name for event in capture.events() if event.device_type == torch.autograd.DeviceType.CUDA
                    ]
                save()
        # NATTEN is an independently assessed comparator; its failures remain
        # in the report without changing the candidate's qualification target.
        report["status"] = (
            "pass"
            if all(r["status"] == "pass" for r in report["rows"] if r["implementation"] == "kohakufa")
            else "numerical-failure"
        )
    except BaseException as error:
        report["status"] = "execution-error"
        report["error_type"] = type(error).__name__
        if report["rows"] and retained is not None:
            report["rows"][-1]["status"] = "execution-error"
            artifact = args.output.with_name(f"{args.output.stem}-execution-failure.pt")
            torch.save(retained, artifact)
            report["rows"][-1]["operands"] = artifact.name
        save()
        raise
    save()
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
