"""Compare explicit attention variants in one process, including preparation."""

import argparse
import hashlib
import importlib
import itertools
import json
import math
import re
import statistics
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operations", type=json.loads, required=True, help="JSON list of labelled factories or backends.")
    parser.add_argument("--seq", nargs="+", type=int, default=[128, 1024, 4096])
    parser.add_argument("--dim", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=371)
    parser.add_argument("--dtype", nargs="+", choices=("bf16", "fp16"), default=["bf16", "fp16"])
    parser.add_argument("--phases", nargs="+", choices=("forward", "backward", "joint"), default=["joint"])
    parser.add_argument("--warmup-ms", type=float, default=1000)
    parser.add_argument("--window-ms", type=float, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--source-revision", help="Full revision label supplied by the frozen-snapshot installer.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.source_revision and not re.fullmatch(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})", args.source_revision):
        parser.error("The source revision must be a full hexadecimal Git object ID.")
    if (
        min(*args.seq, *args.dim, args.heads, args.window_ms, args.repeats) <= 0
        or args.warmup_ms < 0
        or not all(math.isfinite(value) for value in (args.window_ms, args.warmup_ms))
    ):
        parser.error("Shapes, windows and repeats must be positive; warmup must be nonnegative.")

    import torch

    import kohakufa_cute
    from benchmarks.compare_ada import capture
    from benchmarks.compare_l40s import graph_kernels
    from benchmarks.flashattention_backends import configuration_metadata, load_backend
    from benchmarks.validate_sm80_static import assess

    operations = {}
    metadata = {}
    for spec in args.operations:
        label = spec["label"]
        if label in operations:
            parser.error("Operation labels must be unique.")
        if "factory" in spec:
            module, factory = spec["factory"].split(":")
            operations[label] = getattr(importlib.import_module(module), factory)(**spec.get("kwargs", {}))
            metadata[label] = configuration_metadata(spec)
        else:
            operations[label] = load_backend(spec["backend"])
            metadata[label] = operations[label].metadata()
    if len(operations) < 2:
        parser.error("Compare at least two explicit operations.")
    report = {
        "scope": "Same-process alternating graph replay and complete eager calls; forward retains gradients.",
        "event_scope": "The eager CUDA-event interval includes launch gaps; graph replay is a separate device-path control.",
        "device": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_revision": args.source_revision,
        "source_revision_origin": "argument" if args.source_revision else "unavailable",
        "source_sha256": {
            "src/kohakufa_cute/"
            + path.relative_to(Path(kohakufa_cute.__file__).parent).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(kohakufa_cute.__file__).parent.rglob("*"))
            if path.is_file() and path.suffix in (".py", ".cpp")
        },
        "operations": metadata,
        "protocol": {key: getattr(args, key) for key in ("warmup_ms", "window_ms", "repeats", "seed")},
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for sequence, dimension, dtype_name, causal in itertools.product(args.seq, args.dim, args.dtype, (False, True)):
        torch.manual_seed(args.seed)
        dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
        inputs = [
            torch.randn(1, args.heads, sequence, dimension, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)
        ]
        dout = torch.randn_like(inputs[0])
        # Final comparison uses a different input seed from schedule selection.
        qualification = {}
        for spec in args.operations:
            label = spec["label"]
            output = operations[label](*inputs, is_causal=causal)
            actual = output, *torch.autograd.grad(output, inputs, dout)
            qualification[label] = assess(actual, inputs, dout, causal=causal, scale=None, directed=False, pointwise=False)
        for phase in args.phases:
            executions, graphs, resources = {}, {}, {}
            for label, operation in operations.items():
                saved = operation(*inputs, is_causal=causal) if phase == "backward" else None

                def execute(op=operation, output=saved, selected_phase=phase):
                    if selected_phase == "backward":
                        return torch.autograd.grad(output, inputs, dout, retain_graph=True)
                    result = op(*inputs, is_causal=causal)
                    if selected_phase == "joint":
                        return result, *torch.autograd.grad(result, inputs, dout)
                    return (result,)

                executions[label] = execute
                for _ in range(5):
                    execute()
                graphs[label] = capture(operation, [t.detach() for t in inputs], dout, causal, phase, keep_graph=True)
                resources[label] = graph_kernels(graphs[label][0])
            for mode in ("graph", "eager"):
                calls = {label: graph[0].replay for label, graph in graphs.items()} if mode == "graph" else executions
                torch.cuda.synchronize()
                begin = time.perf_counter()
                while (time.perf_counter() - begin) * 1000 < args.warmup_ms:
                    for execute in calls.values():
                        execute()
                    torch.cuda.synchronize()
                counts = {}
                for label, execute in calls.items():
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    for _ in range(10):
                        execute()
                    torch.cuda.synchronize()
                    latency = (time.perf_counter() - begin) * 1000 / 10
                    counts[label] = max(10, math.ceil(args.window_ms / max(latency, 0.0001)))
                samples = {label: [] for label in calls}
                order = list(calls)
                start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                start.record()
                end.record()
                end.synchronize()
                for repeat in range(args.repeats):
                    offset = repeat % len(order)
                    for label in order[offset:] + order[:offset]:
                        for _ in range(3):
                            calls[label]()
                        torch.cuda.synchronize()
                        begin = time.perf_counter()
                        start.record()
                        for _ in range(counts[label]):
                            result = calls[label]()
                        end.record()
                        end.synchronize()
                        samples[label].append(
                            {
                                "cuda_event_ms": start.elapsed_time(end) / counts[label],
                                "wall_ms": (time.perf_counter() - begin) * 1000 / counts[label],
                            }
                        )
                row = {
                    "shape": list(inputs[0].shape),
                    "dtype": dtype_name,
                    "causal": causal,
                    "phase": phase,
                    "mode": mode,
                    "calls_per_window": counts,
                    "samples": samples,
                    "median_ms": {
                        label: {
                            key: statistics.median(value[key] for value in values) for key in ("cuda_event_ms", "wall_ms")
                        }
                        for label, values in samples.items()
                    },
                    "graph_resources": resources,
                    "qualification": qualification,
                }
                report["rows"].append(row)
                temporary = args.output.with_suffix(".partial")
                temporary.write_text(json.dumps(report, indent=2) + "\n")
                temporary.replace(args.output)
                print({key: row[key] for key in ("shape", "dtype", "causal", "phase", "mode", "median_ms")}, flush=True)


if __name__ == "__main__":
    main()
