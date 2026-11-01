"""Compare complete eager SM80 calls with Python and native preparation."""

import argparse
import hashlib
import itertools
import json
import math
import statistics
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", nargs="+", type=int, default=[128, 1024, 4096])
    parser.add_argument("--dim", nargs="+", type=int, choices=(64, 128), default=[64, 128])
    parser.add_argument("--dtype", nargs="+", choices=("bf16", "fp16"), default=["bf16", "fp16"])
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--warmup-ms", type=float, default=1000)
    parser.add_argument("--window-ms", type=float, default=100)
    parser.add_argument("--configuration", type=json.loads, help="Explicit warp schedule as a JSON object.")
    parser.add_argument("--fa2", action="store_true", help="Include the separately loaded standalone FA2 backend.")
    parser.add_argument("--graph", action="store_true", help="Measure matched graph replay as a device-path control.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 1 or args.calls < 1 or args.warmup_ms < 0 or args.window_ms <= 0:
        parser.error("Use positive rounds, calls and measurement windows, and a nonnegative warmup.")

    import torch

    from benchmarks.sm80_native_host import attention as native_host
    from kohakufa_cute.fa4_sm80._attention import Schedule
    from kohakufa_cute.fa4_sm80._attention import attention as python_host
    from kohakufa_cute.native import extension

    module = extension()
    schedule = Schedule(**(args.configuration or {}))
    runner = module.AttentionRunner("warp", args.configuration) if args.configuration is not None else None
    fa2 = None
    if args.fa2:
        from benchmarks.flashattention_backends import load_backend

        fa2 = load_backend("fa2")
    root = Path(__file__).resolve().parents[1]
    paths = [root / "src/kohakufa_cute/native.cpp", root / "src/kohakufa_cute/native.py"]
    paths.extend(sorted((root / "src/kohakufa_cute/fa4_sm80").glob("*.py")))
    report = {
        "scope": "Complete warmed eager calls, alternating backend order; synchronization at window boundaries.",
        "rounds": args.rounds,
        "calls_per_window": args.calls,
        "minimum_window_ms": args.window_ms,
        "warmup_ms": args.warmup_ms,
        "configuration": args.configuration,
        "graph_replay": args.graph,
        "fa2": fa2.metadata() if fa2 is not None else None,
        "device": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for sequence, dimension, dtype_name, causal in itertools.product(args.seq, args.dim, args.dtype, (False, True)):
        dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
        torch.manual_seed(859)
        inputs = [
            torch.randn(1, 16, sequence, dimension, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)
        ]
        dout = torch.randn_like(inputs[0])
        backends = {
            "generic": lambda: module.attention(*inputs, None, None, causal),
            "python_host": lambda: python_host(*inputs, causal=causal, schedule=schedule),
            "native_host": (
                (lambda: runner(*inputs, causal=causal))
                if runner is not None
                else (lambda: native_host(*inputs, is_causal=causal))
            ),
            "sdpa": lambda: torch.nn.functional.scaled_dot_product_attention(*inputs, is_causal=causal),
        }
        if fa2 is not None:
            backends["fa2"] = lambda: fa2(*inputs, is_causal=causal)
        for phase in ("forward", "joint"):

            def run(function):
                if phase == "forward":
                    with torch.no_grad():
                        return function()
                out = function()
                return out, *torch.autograd.grad(out, inputs, dout)

            for function in backends.values():
                for _ in range(5):
                    run(function)
            torch.cuda.synchronize()
            executions = {name: (lambda function=function: run(function)) for name, function in backends.items()}
            graph_storage = []
            if args.graph:
                for name, execute in executions.items():
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        actual = execute()
                    graph_storage.append((graph, actual))
                    executions[name] = graph.replay
            warm_start = time.perf_counter()
            while (time.perf_counter() - warm_start) * 1000 < args.warmup_ms:
                for _ in range(4):
                    executions["native_host"]()
                torch.cuda.synchronize()
            iterations = {}
            for name, execute in executions.items():
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(args.calls):
                    execute()
                torch.cuda.synchronize()
                latency = (time.perf_counter() - start) * 1000 / args.calls
                iterations[name] = max(args.calls, math.ceil(args.window_ms / max(latency, 0.0001)))
            samples = {name: [] for name in backends}
            order = list(backends)
            for repeat in range(args.rounds):
                offset = repeat % len(order)
                for name in order[offset:] + order[:offset]:
                    for _ in range(3):
                        executions[name]()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(iterations[name]):
                        result = executions[name]()
                    torch.cuda.synchronize()
                    samples[name].append((time.perf_counter() - start) * 1000 / iterations[name])
            row = {
                "shape": list(inputs[0].shape),
                "dtype": dtype_name,
                "causal": causal,
                "phase": phase,
                "calls_per_window": iterations,
                "samples_ms": samples,
                "median_ms": {name: statistics.median(values) for name, values in samples.items()},
            }
            report["rows"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print({k: v for k, v in row.items() if k != "samples_ms"}, flush=True)


if __name__ == "__main__":
    main()
