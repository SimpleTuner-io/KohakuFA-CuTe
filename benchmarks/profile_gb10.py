"""Profile the complete public GB10 or standalone FA2 forward/backward call."""

import argparse
import json
from pathlib import Path

import torch

from benchmarks.compare_ada import capture
from benchmarks.compare_l40s import graph_kernels
from benchmarks.experimental_gb10_shared import make_public_attention
from benchmarks.flashattention_backends import load_backend


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("public", "cute", "fa2"), required=True)
    p.add_argument("--eager", action="store_true")
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    function = make_public_attention() if args.backend == "public" else load_backend(args.backend)
    torch.manual_seed(2031)
    inputs = tuple(torch.randn(1, 4, 1024, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
    dout = torch.randn_like(inputs[0])

    def eager():
        out = function(*inputs)
        return out, *torch.autograd.grad(out, inputs, dout)

    graph, owners = capture(function, [tensor.detach() for tensor in inputs], dout, False, "joint", keep_graph=True)
    execute = eager if args.eager else graph.replay
    for _ in range(20):
        actual = execute()
    torch.cuda.synchronize()
    assert torch.cuda.cudart().cudaProfilerStart() == 0
    for iteration in range(args.iterations):
        with torch.cuda.nvtx.range(f"{args.backend}/{'eager' if args.eager else 'graph'}/{iteration}"):
            actual = execute()
    torch.cuda.synchronize()
    assert torch.cuda.cudart().cudaProfilerStop() == 0
    args.output.write_text(
        json.dumps(
            {
                "backend": args.backend,
                "execution": "eager" if args.eager else "graph",
                "iterations": args.iterations,
                "graph_kernels": graph_kernels(graph),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
