"""Matched complete eager wall time for native, static schedules, and SDPA."""

import argparse
import hashlib
import itertools
import json
import math
import statistics
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch

from kohakufa_cute import api, fa4_sm80
from kohakufa_cute.fa4_sm80._attention import attention as previous_attention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", type=int, nargs="+", default=(128, 1024, 4096))
    parser.add_argument("--dim", type=int, nargs="+", default=(64, 128))
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--calls", type=int, default=10)
    parser.add_argument("--warmup-ms", type=float, default=1000)
    parser.add_argument("--window-ms", type=float, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    backends = ("native", "previous", "candidate", "sdpa", "public")
    report = {
        "scope": "Complete warmed eager wall time; synchronization at window boundaries; backend profiling outside timing.",
        "device": torch.cuda.get_device_name(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": str(torch.__version__),
        "repeats": args.repeats,
        "minimum_calls_per_window": args.calls,
        "warmup_ms": args.warmup_ms,
        "minimum_window_ms": args.window_ms,
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__).resolve().parents[1] / "src/kohakufa_cute/fa4_sm80").glob("*.py")
        },
        "backend_boundaries": {
            "native": "Existing native CuTe; automatic static selection disabled.",
            "previous": "Explicit immutable Schedule() through the static Python entry.",
            "candidate": "Explicit selected A100 schedule; bypasses automatic performance bounds.",
            "public": "Full public API with actual automatic selection.",
            "sdpa": "PyTorch SDPA; actual selected kernels captured separately.",
        },
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for sequence, dimension, dtype, causal in itertools.product(
        args.seq, args.dim, (torch.bfloat16, torch.float16), (False, True)
    ):
        torch.manual_seed(877)
        inputs = [
            torch.randn(1, args.heads, sequence, dimension, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)
        ]
        dout = torch.randn_like(inputs[0])
        selected = fa4_sm80.compatible(*inputs, None, 0, causal, None, False)
        for phase in ("forward", "joint"):

            def execute(backend):
                def forward():
                    if backend == "previous":
                        return previous_attention(*inputs, causal=causal)
                    if backend == "candidate":
                        return fa4_sm80.attention(*inputs, causal=causal)
                    if backend == "sdpa":
                        return torch.nn.functional.scaled_dot_product_attention(*inputs, is_causal=causal)
                    return api.scaled_dot_product_attention(*inputs, is_causal=causal)

                if phase == "forward":
                    with torch.no_grad():
                        return forward()
                out = forward()
                return out, *torch.autograd.grad(out, inputs, dout)

            # Compile and allocate separately, then keep the declared complete
            # workload active long enough to settle its sustained device state.
            for backend in backends:
                context = patch.object(fa4_sm80, "compatible", return_value=False) if backend == "native" else nullcontext()
                with context:
                    for _ in range(3):
                        execute(backend)
            torch.cuda.synchronize()
            warm_start = time.perf_counter()
            while (time.perf_counter() - warm_start) * 1000 < args.warmup_ms:
                for _ in range(4):
                    execute("candidate")
                torch.cuda.synchronize()
            warm_elapsed_ms = (time.perf_counter() - warm_start) * 1000
            iterations = {}
            for backend in backends:
                context = patch.object(fa4_sm80, "compatible", return_value=False) if backend == "native" else nullcontext()
                with context:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(args.calls):
                        execute(backend)
                    torch.cuda.synchronize()
                    per_call_ms = (time.perf_counter() - start) * 1000 / args.calls
                iterations[backend] = max(args.calls, math.ceil(args.window_ms / max(per_call_ms, 0.0001)))
            samples = {backend: [] for backend in backends}
            for repeat in range(args.repeats):
                offset = repeat % len(backends)
                for backend in backends[offset:] + backends[:offset]:
                    context = (
                        patch.object(fa4_sm80, "compatible", return_value=False) if backend == "native" else nullcontext()
                    )
                    with context:
                        for _ in range(3):
                            result = execute(backend)
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        for _ in range(iterations[backend]):
                            result = execute(backend)
                        torch.cuda.synchronize()
                        samples[backend].append((time.perf_counter() - start) * 1000 / iterations[backend])
            eager_paths = {}
            for backend in ("candidate", "public", "sdpa"):
                with torch.profiler.profile(
                    activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA)
                ) as profiler:
                    result = execute(backend)
                torch.cuda.synchronize()
                output = result[0] if phase == "joint" else result
                grad_fn = output.grad_fn
                eager_paths[backend] = {
                    "operators": sorted({event.name for event in profiler.events() if "scaled_dot_product" in event.name}),
                    "cuda_kernels": sorted(
                        {event.name for event in profiler.events() if "cuda" in str(event.device_type).lower()}
                    ),
                    "grad_fn": type(grad_fn).__name__ if grad_fn is not None else None,
                    "schedule": asdict(grad_fn.schedule) if hasattr(grad_fn, "schedule") else None,
                }
            row = {
                "shape": list(inputs[0].shape),
                "dtype": str(dtype),
                "causal": causal,
                "phase": phase,
                "public_selected_static": selected,
                "warm_elapsed_ms": warm_elapsed_ms,
                "calls_per_window": iterations,
                "samples_ms": samples,
                "median_ms": {backend: statistics.median(values) for backend, values in samples.items()},
                "eager_paths": eager_paths,
                "sdpa_operators": eager_paths["sdpa"]["operators"],
                "sdpa_cuda_kernels": eager_paths["sdpa"]["cuda_kernels"],
                "sdpa_grad_fn": eager_paths["sdpa"]["grad_fn"],
            }
            report["rows"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                {key: value for key, value in row.items() if key not in ("samples_ms", "sdpa_cuda_kernels", "eager_paths")},
                flush=True,
            )


if __name__ == "__main__":
    main()
