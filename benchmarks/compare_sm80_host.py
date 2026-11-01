"""Compare complete eager SM80 calls with Python and native preparation."""

import argparse
import hashlib
import itertools
import json
import statistics
import time
from pathlib import Path


def main():
    import torch

    from benchmarks.sm80_native_host import attention as native_host
    from kohakufa_cute.fa4_sm80._attention import attention as python_host
    from kohakufa_cute.native import extension

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", nargs="+", type=int, default=[128, 1024, 4096])
    parser.add_argument("--dtype", nargs="+", choices=("bf16", "fp16"), default=["bf16", "fp16"])
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    module = extension()
    root = Path(__file__).resolve().parents[1]
    paths = [root / "src/kohakufa_cute/native.cpp", root / "src/kohakufa_cute/native.py"]
    paths.extend(sorted((root / "src/kohakufa_cute/fa4_sm80").glob("*.py")))
    report = {
        "scope": "Complete warmed eager calls, alternating backend order; synchronization at window boundaries.",
        "rounds": args.rounds,
        "calls_per_window": args.calls,
        "device": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for sequence, dimension, dtype_name, causal in itertools.product(args.seq, (64, 128), args.dtype, (False, True)):
        dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
        torch.manual_seed(859)
        inputs = [
            torch.randn(1, 16, sequence, dimension, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)
        ]
        dout = torch.randn_like(inputs[0])
        backends = {
            "generic": lambda: module.attention(*inputs, None, None, causal),
            "python_host": lambda: python_host(*inputs, causal=causal),
            "native_host": lambda: native_host(*inputs, is_causal=causal),
            "sdpa": lambda: torch.nn.functional.scaled_dot_product_attention(*inputs, is_causal=causal),
        }
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
            samples = {name: [] for name in backends}
            order = list(backends)
            for repeat in range(args.rounds):
                offset = repeat % len(order)
                for name in order[offset:] + order[:offset]:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(args.calls):
                        result = run(backends[name])
                    torch.cuda.synchronize()
                    samples[name].append((time.perf_counter() - start) * 1000 / args.calls)
            row = {
                "shape": list(inputs[0].shape),
                "dtype": dtype_name,
                "causal": causal,
                "phase": phase,
                "samples_ms": samples,
                "median_ms": {name: statistics.median(values) for name, values in samples.items()},
            }
            report["rows"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print({k: v for k, v in row.items() if k != "samples_ms"}, flush=True)


if __name__ == "__main__":
    main()
