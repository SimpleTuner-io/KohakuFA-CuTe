"""Exercise repeated statistic handoffs without launching reference attention kernels."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory", required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-graphs", action="store_true", help="Bound an initialization diagnostic to eager execution")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch

    from benchmarks.compare_ada import capture
    from benchmarks.reference import fp64_reference, metrics

    name, factory = args.factory.split(":")
    module = importlib.import_module(name)
    attention = getattr(module, factory)()
    cases = [(129, 3, 64, True)] if args.quick else [(129, 3, 64, True), (257, 5, 128, False), (513, 3, 64, False)]
    dtypes = (torch.bfloat16,) if args.quick else (torch.bfloat16, torch.float16)
    records = []
    torch.manual_seed(2207)
    for dtype in dtypes:
        for length, heads, dimension, causal in cases:
            inputs = [torch.randn(1, heads, length, dimension, dtype=dtype, device="cuda") for _ in range(3)]
            dout = torch.randn_like(inputs[0])
            operands = [t.detach().requires_grad_() for t in inputs]
            out = attention(*operands, is_causal=causal)
            actual = out, *torch.autograd.grad(out, operands, dout)
            reference = fp64_reference(inputs, dout, causal=causal)
            errors = [
                dict(metrics(got[:, h : h + 1], want[:, h : h + 1]), head=h, component=key)
                for h in range(heads)
                for key, got, want in zip(("out", "dq", "dk", "dv"), actual, reference)
            ]
            limit = 0.01 if dtype == torch.bfloat16 else 0.005
            passed = all(
                e["finite"] and (e["relative_error"] < limit if e["reference_norm"] else e["max_absolute_error"] == 0)
                for e in errors
            )
            if not passed:
                raise AssertionError("The selected implementation violates the unchanged per-head component norm gate.")
            if not args.no_graphs:
                capture(attention, tuple(inputs), dout, causal, "joint")
            records.append(
                dict(
                    shape=list(out.shape),
                    dtype=str(dtype),
                    causal=causal,
                    passed=passed,
                    norm_limit=limit,
                    errors=errors,
                    changed_input_graph=not args.no_graphs,
                )
            )
            print(list(out.shape), dtype, "pass", flush=True)
    import kohakufa_cute.fa4_blackwell

    folder = Path(kohakufa_cute.fa4_blackwell.__file__).parent
    static_sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob("*.py"))}
    args.output.write_text(
        json.dumps(
            dict(
                factory=args.factory,
                static_sources=static_sources,
                checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                records=records,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
