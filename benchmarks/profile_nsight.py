"""Capture warmed attention with Nsight's CUDA profiler API ranges.

Forward-only runs retain gradients, matching benchmarks.speed's training
forward path. Keep profiler reports outside the checkout.
"""

import argparse
import json

import torch

from kohakufa_cute import scaled_dot_product_attention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("sdpa", "cute"), required=True)
    parser.add_argument("--phase", choices=("forward", "backward", "joint"), default="joint")
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--eager", action="store_true", help="Profile ordinary launches instead of CUDA graph replay")
    args = parser.parse_args()
    if min(args.seq, args.dim, args.heads, args.iterations, args.warmup) < 1:
        parser.error("Shapes and iteration counts must be positive")
    torch.manual_seed(21)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    inputs = [
        torch.randn(1, args.heads, args.seq, args.dim, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)
    ]
    dout = torch.randn_like(inputs[0])
    operation = scaled_dot_product_attention if args.backend == "cute" else torch.nn.functional.scaled_dot_product_attention
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        saved = operation(*inputs, is_causal=args.causal) if args.phase == "backward" else None

    def run():
        if args.phase == "backward":
            return torch.autograd.grad(saved, inputs, dout, retain_graph=True)
        output = operation(*inputs, is_causal=args.causal)
        if args.phase == "joint":
            return output, *torch.autograd.grad(output, inputs, dout)
        return (output,)

    with torch.cuda.stream(stream):
        for _ in range(args.warmup):
            values = run()
    torch.cuda.synchronize()
    graph = None
    if not args.eager:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            values = run()
        graph.replay()
        torch.cuda.synchronize()
    assert all(torch.isfinite(t).all() for t in values)
    label = f"{args.backend}:{args.phase}:S{args.seq}:D{args.dim}:{'causal' if args.causal else 'dense'}"
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(label)
    try:
        with torch.cuda.stream(stream):
            for _ in range(args.iterations):
                if graph is None:
                    values = run()
                else:
                    graph.replay()
            torch.cuda.synchronize()
    finally:
        torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()
    assert all(torch.isfinite(t).all() for t in values)
    print(json.dumps(dict(vars(args), shape=list(inputs[0].shape), finite=True)))


if __name__ == "__main__":
    main()
