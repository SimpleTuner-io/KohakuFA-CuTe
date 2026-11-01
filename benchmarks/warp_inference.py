"""Compare explicit warp inference with the unchanged training entry under NoGrad."""

import argparse
import hashlib
import itertools
import json
import math
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import torch

from benchmarks.reference import fp64_reference, metrics


@torch.no_grad()
def references(inputs, causal=False, scale=None):
    expected = []
    for head in range(inputs[0].shape[1]):
        operands = [tensor[:, head : head + 1] for tensor in inputs]
        expected.append(fp64_reference(operands, torch.zeros_like(operands[0]), causal=causal, scale=scale)[0])
    return expected


def assess_output(output, expected):
    limit = 0.01 if output.dtype == torch.bfloat16 else 0.005
    records = []
    for head, reference in enumerate(expected):
        error = metrics(output[:, head : head + 1], reference)
        assert error["finite"], (head, error)
        if error["reference_norm"]:
            assert error["relative_error"] < limit, (head, error)
        else:
            assert error["max_absolute_error"] == 0, (head, error)
        records.append(error)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", nargs="+", type=int, default=[128, 1024, 4096])
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dim", nargs="+", type=int, choices=(64, 128), default=[64, 128])
    parser.add_argument("--dtype", nargs="+", choices=("bf16", "fp16"), default=["bf16", "fp16"])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--window-ms", type=float, default=50)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(*args.seq, args.heads, args.rounds, args.window_ms) <= 0:
        parser.error("Shapes, rounds and windows must be positive.")

    from benchmarks.compare_l40s import graph_kernels
    from benchmarks.flashattention_backends import load_backend
    from kohakufa_cute import scaled_dot_product_attention
    from kohakufa_cute.fa4_sm80._schedules import a100_schedule
    from kohakufa_cute.native import extension

    fa2 = load_backend("fa2")
    report = {
        "scope": "Warmed NoGrad device-event graph time" if args.graph else "Complete warmed NoGrad eager wall time",
        "device": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "fa2": fa2.metadata(),
        "rounds": args.rounds,
        "window_ms": args.window_ms,
        "graph": args.graph,
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": {
            str(p.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__).resolve().parents[1] / "src/kohakufa_cute/native.cpp",
                Path(__file__).resolve().parents[1] / "src/kohakufa_cute/native.py",
                *sorted((Path(__file__).resolve().parents[1] / "src/kohakufa_cute/fa4_sm80").glob("*.py")),
            ]
        },
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for sequence, dimension, dtype_name, causal in itertools.product(args.seq, args.dim, args.dtype, (False, True)):
        torch.manual_seed(1009)
        dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
        inputs = [torch.randn(1, args.heads, sequence, dimension, device="cuda", dtype=dtype) for _ in range(3)]
        schedule = a100_schedule(dimension, sequence, causal)
        runner = extension().AttentionRunner("warp", asdict(schedule))
        functions = {
            "old_runner": lambda: runner(*inputs, causal=causal),
            "inference": lambda: runner.inference(*inputs, causal=causal),
            "public": lambda: scaled_dot_product_attention(*inputs, is_causal=causal),
            "fa2": lambda: fa2(*inputs, is_causal=causal),
            "sdpa": lambda: torch.nn.functional.scaled_dot_product_attention(*inputs, is_causal=causal),
        }
        expected = references(inputs, causal)
        actual_graphs, owners, execution, errors = {}, [], {}, {}
        with torch.no_grad():
            for name, function in functions.items():
                for _ in range(3):
                    output = function()
                errors[name] = assess_output(output, expected)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(graph):
                    output = function()
                actual_graphs[name] = graph_kernels(graph)
                owners.append((graph, output))
                execution[name] = graph.replay if args.graph else function
            torch.cuda.synchronize()
            started = time.perf_counter()
            while time.perf_counter() - started < 1:
                for _ in range(4):
                    execution["inference"]()
                torch.cuda.synchronize()
            counts = {}
            for name, function in execution.items():
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(20):
                    function()
                torch.cuda.synchronize()
                counts[name] = max(20, math.ceil(args.window_ms / ((time.perf_counter() - start) * 1000 / 20)))
            order, samples = list(execution), {name: [] for name in execution}
            for repeat in range(args.rounds):
                offset = repeat % len(order)
                for name in order[offset:] + order[:offset]:
                    torch.cuda.synchronize()
                    if args.graph:
                        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(counts[name]):
                            execution[name]()
                        stop.record()
                        stop.synchronize()
                        elapsed = start.elapsed_time(stop)
                    else:
                        start = time.perf_counter()
                        for _ in range(counts[name]):
                            execution[name]()
                        torch.cuda.synchronize()
                        elapsed = (time.perf_counter() - start) * 1000
                    samples[name].append(elapsed / counts[name])
            paths = {}
            for name, function in functions.items():
                with torch.profiler.profile(
                    activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA),
                    profile_memory=True,
                ) as profiler:
                    function()
                torch.cuda.synchronize()
                paths[name] = {
                    "cuda_kernels": sorted(
                        {event.name for event in profiler.events() if "cuda" in str(event.device_type).lower()}
                    ),
                    "allocation_operators": {
                        event.key: event.count for event in profiler.key_averages() if "empty" in event.key
                    },
                }
        row = {
            "shape": list(inputs[0].shape),
            "dtype": dtype_name,
            "causal": causal,
            "schedule": asdict(schedule),
            "errors": errors,
            "kernels": actual_graphs,
            "eager_paths": paths,
            "calls_per_window": counts,
            "samples_ms": samples,
            "median_ms": {name: statistics.median(values) for name, values in samples.items()},
        }
        report["rows"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print({key: row[key] for key in ("shape", "dtype", "causal", "median_ms")}, flush=True)


if __name__ == "__main__":
    main()
