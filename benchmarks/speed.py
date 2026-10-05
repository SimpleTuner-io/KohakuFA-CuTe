"""Measure attention forward and forward/backward using rotating CUDA-graph replays."""

import argparse
import json
import statistics
from contextlib import nullcontext
from pathlib import Path

import torch

from benchmarks.precision import evaluate, timing
from kohakufa_cute import automatic_scaled_dot_product_attention, scaled_dot_product_attention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--dim", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["sdpa", "cudnn", "cute", "cute-auto", "kohakufa"],
        default=["sdpa", "cudnn", "cute"],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("Timing requires CUDA.")
    if min(*args.seq, *args.dim, args.heads, args.iterations, args.repeats) < 1:
        parser.error("Shapes, iterations and repeats must be positive.")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    functions = {
        "sdpa": torch.nn.functional.scaled_dot_product_attention,
        "cudnn": torch.nn.functional.scaled_dot_product_attention,
        "cute": scaled_dot_product_attention,
        "cute-auto": automatic_scaled_dot_product_attention,
    }
    if "kohakufa" in args.backends:
        from kohakufa import attention

        def upstream(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
            return attention(q, k, v, mask=attn_mask, causal=is_causal, scale=scale)

        functions["kohakufa"] = upstream
    rows = []
    for sequence in args.seq:
        for dimension in args.dim:
            torch.manual_seed(21)
            q, k, v, dout = [torch.randn(1, args.heads, sequence, dimension, device="cuda", dtype=dtype) for _ in range(4)]
            reference = evaluate(functions["sdpa"], (q, k, v), dout, None, args.causal)
            samples = {backend: [] for backend in args.backends}
            unsupported = {}
            for repeat in range(args.repeats):
                offset = repeat % len(args.backends)
                for backend in args.backends[offset:] + args.backends[:offset]:
                    if backend in unsupported:
                        continue
                    context = (
                        torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION)
                        if backend == "cudnn"
                        else nullcontext()
                    )
                    try:
                        with context:
                            actual = evaluate(functions[backend], (q, k, v), dout, None, args.causal)
                            for got, want in zip(actual, reference):
                                torch.testing.assert_close(
                                    got,
                                    want,
                                    atol=0.03 if dtype == torch.bfloat16 else 0.005,
                                    rtol=0.03 if dtype == torch.bfloat16 else 0.005,
                                )
                            sample = timing(functions[backend], (q, k, v), dout, None, args.causal, args.iterations)
                    except RuntimeError as error:
                        if backend != "cudnn" or "No available kernel" not in str(error):
                            raise
                        unsupported[backend] = "No available cuDNN kernel."
                        continue
                    samples[backend].append(sample)
            medians = {
                backend: {
                    key: statistics.median(sample[key] for sample in values) for key in ("forward_ms", "forward_backward_ms")
                }
                for backend, values in samples.items()
                if values
            }
            row = {"shape": list(q.shape), "samples": samples, "medians": medians, "unsupported": unsupported}
            rows.append(row)
            print(json.dumps(row), flush=True)
    report = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "dtype": args.dtype,
        "causal": args.causal,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "ordering": "rotate backend order by one each repetition",
        "timing": "CUDA event milliseconds per graph replay; forward_backward includes forward",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
