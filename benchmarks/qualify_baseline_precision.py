"""Check rounded operands and time complete attention calls after precision removal."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import torch

from benchmarks.reference import fp64_reference, metrics, rounded_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, nargs="+", default=[16, 64, 128, 256, 512])
    parser.add_argument("--seq", type=int, nargs="+", default=[128, 1024, 4096])
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dtype", nargs="+", choices=["fp16", "bf16"], default=["bf16", "fp16"])
    parser.add_argument("--modes", nargs="+", choices=["dense", "causal"], default=["dense", "causal"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--window-ms", type=float, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="candidate")
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--compare-previous", action="store_true")
    parser.add_argument("--export-code", type=Path)
    args = parser.parse_args()
    if args.export_code is not None:
        os.environ["CUTE_DSL_KEEP"] = "ptx,cubin"
        os.environ.setdefault("CUTE_DSL_DUMP_DIR", str(args.export_code.resolve()))
        Path(os.environ["CUTE_DSL_DUMP_DIR"]).mkdir(parents=True, exist_ok=True)
    from benchmarks.compare_ada import capture, measure
    from benchmarks.compare_l40s import board_state, graph_kernels, loaded_libraries
    from benchmarks.experimental_precision_control import previous_precision
    from kohakufa_cute import kernels, scaled_dot_product_attention

    prop = torch.cuda.get_device_properties(0)
    report = {
        "label": args.label,
        "schema_version": 4,
        "date": datetime.now(timezone.utc).date().isoformat(),
        "base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "device": {
            "name": prop.name,
            "multiprocessors": prop.multi_processor_count,
            "capability": [prop.major, prop.minor],
            "memory_bytes": prop.total_memory,
            "board_before": board_state(),
        },
        "stack": {"python": platform.python_version(), "torch": str(torch.__version__), "cuda": torch.version.cuda},
        "packages": {name: importlib.metadata.version(name) for name in ("nvidia-cutlass-dsl", "apache-tvm-ffi", "triton")},
        "source_sha256": {
            "src/kohakufa_cute/" + p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(kernels.__file__).parent.iterdir())
            if p.suffix in (".py", ".cpp")
        },
        "protocol": "Independent out/dQ/dK/dV versus FP64 mathematical and single-rounded-P/dS equations. Backward rounded reference uses actual O for delta. CUDA graph changed-input checks; rotate CuTe and default SDPA timing order. Device-event graph replay; compilation excluded.",
        "measurements": [],
        "kernels": {},
        "benchmark_sha256": {
            "benchmarks/" + name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in (
                "qualify_baseline_precision.py",
                "experimental_precision_control.py",
                "reference.py",
                "compare_ada.py",
                "compare_l40s.py",
            )
        },
        "precision_contract": "Single rounded FP16/BF16 P; dS formed from rounded P and actual O/dO FP32 delta, rounded once; FP32 accumulation followed by gradient scaling.",
        "controls": {
            "cute": "Current default arithmetic and schedules.",
            "previous": "Pre-change arithmetic reconstructed on this compiler; separately verified against the preceding checkout.",
            "sdpa": "Default PyTorch SDPA; actual graph symbols identify the selected kernels.",
        },
        "repeats": args.repeats,
        "target_window_ms": args.window_ms,
    }
    compiled = []
    original_compile = kernels.compile_kernel

    def retain(*a, **kw):
        function = original_compile(*a, **kw)
        compiled.append(function)
        if args.export_code is not None:
            args.export_code.mkdir(parents=True, exist_ok=True)
            for extension in ("ptx", "cubin"):
                data = getattr(function, "__" + extension + "__")
                if isinstance(data, str):
                    data = data.encode() if "\n" in data else Path(data).read_bytes()
                elif isinstance(data, bytes) and data.startswith(b"/") and b"\n" not in data:
                    data = Path(data.decode()).read_bytes()
                assert data and (b".entry" in data if extension == "ptx" else data.startswith(b"\x7fELF"))
                name = hashlib.sha256(data).hexdigest() + "." + extension
                (args.export_code / name).write_bytes(data)
        return function

    kernels.compile_kernel = retain
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for dtype_name in args.dtype:
        for sequence in args.seq:
            for dimension in args.dim:
                for mode in args.modes:
                    torch.manual_seed(2031)
                    dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
                    inputs = [torch.randn(1, args.heads, sequence, dimension, device="cuda", dtype=dtype) for _ in range(4)]
                    q, k, v, dout = inputs
                    operands = [t.detach().requires_grad_(True) for t in (q, k, v)]
                    causal = mode == "causal"
                    output = scaled_dot_product_attention(*operands, is_causal=causal)
                    actual = (output, *torch.autograd.grad(output, operands, dout))
                    errors = {"fp64": [], "rounded": []}
                    for head in range(args.heads):
                        part = tuple(t[:, head : head + 1] for t in (q, k, v))
                        grad = dout[:, head : head + 1]
                        for name, reference in (
                            ("fp64", fp64_reference(part, grad, causal=causal)),
                            ("rounded", rounded_reference(part, grad, causal=causal, output=output[:, head : head + 1])),
                        ):
                            errors[name].append([metrics(t[:, head : head + 1], r) for t, r in zip(actual, reference)])
                    # Ordinary random inputs retain the existing norm gate. The
                    # directed cancellation tests separately check rounded math.
                    for part in errors["fp64"]:
                        for error in part:
                            assert error["finite"] and error["relative_error"] < (
                                0.01 if dtype == torch.bfloat16 else 0.005
                            ), error
                    accuracy = {
                        name: {
                            component: {
                                "finite": all(part[i]["finite"] for part in heads),
                                "max_relative_error": max(
                                    part[i]["relative_error"] for part in heads if part[i]["relative_error"] is not None
                                ),
                                "max_absolute_error": max(part[i]["max_absolute_error"] for part in heads),
                            }
                            for i, component in enumerate(("out", "dq", "dk", "dv"))
                        }
                        for name, heads in errors.items()
                    }
                    row = {"shape": list(q.shape), "dtype": dtype_name, "mode": mode, "accuracy": accuracy}
                    if not args.skip_timing:
                        graphs, owners, resources = {}, {}, {}
                        backends = {
                            "cute": scaled_dot_product_attention,
                            "sdpa": torch.nn.functional.scaled_dot_product_attention,
                        }
                        if args.compare_previous:
                            backends["previous"] = scaled_dot_product_attention
                        for backend, function in backends.items():
                            context = previous_precision() if backend == "previous" else nullcontext()
                            with context:
                                for phase in ("forward", "backward", "joint"):
                                    graphs[backend, phase], owners[backend, phase] = capture(
                                        function, (q, k, v), dout, causal, phase, keep_graph=True
                                    )
                                    resources[backend + "/" + phase] = []
                                    for node in graph_kernels(graphs[backend, phase]):
                                        static = {k: v for k, v in node.items() if k != "grid"}
                                        identifier = hashlib.sha256(json.dumps(static, sort_keys=True).encode()).hexdigest()[
                                            :16
                                        ]
                                        report["kernels"].setdefault(identifier, static)
                                        resources[backend + "/" + phase].append({"kernel": identifier, "grid": node["grid"]})
                        samples = {
                            backend: {phase: [] for phase in ("forward", "backward", "joint")} for backend in backends
                        }
                        order = list(backends)
                        for repeat in range(args.repeats):
                            offset = repeat % len(order)
                            for backend in order[offset:] + order[:offset]:
                                for phase in ("forward", "backward", "joint"):
                                    graph = graphs[backend, phase]
                                    pilot = measure(graph, 16)
                                    count = max(16, min(8192, int(args.window_ms / pilot)))
                                    samples[backend][phase].append(measure(graph, count))
                        row["graph_resources"] = resources
                        row["samples_ms"] = samples
                        row["medians_ms"] = {
                            b: {p: statistics.median(v) for p, v in phases.items()} for b, phases in samples.items()
                        }
                        del graphs, owners
                    report["measurements"].append(row)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(json.dumps({k: row[k] for k in ("shape", "dtype", "mode", "medians_ms") if k in row}), flush=True)
                    del inputs, operands, output, actual
                    torch.cuda.empty_cache()
    report["device"]["board_after"] = board_state()
    report["loaded_libraries"] = loaded_libraries()
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
