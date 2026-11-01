"""Qualify both wide ownership kernels, tails, and changed-input graph replay."""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    import torch

    from benchmarks.compare_ada import capture
    from benchmarks.compare_l40s import graph_kernels
    from benchmarks.experimental_b200_wide import wide_backward
    from benchmarks.reference import fp64_reference, metrics
    from kohakufa_cute import scaled_dot_product_attention as attention

    torch.manual_seed(3829)
    records = []
    with wide_backward(512) if args.candidate else nullcontext():
        for length, heads, causal in ((512, 16, False), (513, 16, True), (1024, 8, False)):
            inputs = [torch.randn(1, heads, length, 512, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
            dout = torch.randn_like(inputs[0])
            operands = [t.detach().requires_grad_() for t in inputs]
            output = attention(*operands, is_causal=causal)
            actual = output, *torch.autograd.grad(output, operands, dout)
            reference = fp64_reference(inputs, dout, causal=causal)
            errors = [
                dict(metrics(got[:, head : head + 1], want[:, head : head + 1]), component=component, head=head)
                for component, got, want in zip(("out", "dq", "dk", "dv"), actual, reference)
                for head in range(heads)
            ]
            passed = all(e["finite"] and e["relative_error"] < 0.01 for e in errors)
            graph, retained = capture(attention, inputs, dout, causal, "joint", keep_graph=True)
            kernels = graph_kernels(graph)
            selected = [k for k in kernels if "RegisterAux" in k["symbol"]]
            if not passed or len(selected) != 2:
                raise AssertionError("The tested call must execute both wide owners and preserve each component norm gate.")
            records.append(dict(shape=list(output.shape), causal=causal, errors=errors, kernels=kernels, passed=passed))
            print(list(output.shape), causal, "both wide owners and changed graph pass", flush=True)
    args.output.write_text(json.dumps(dict(records=records), indent=2) + "\n")


if __name__ == "__main__":
    main()
