"""Directed qualification for the retained experimental static FA4 schedule."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    forward_schedule = "packed"

    import torch

    from benchmarks.blackwell_fa4_experiments import make_static_attention
    from benchmarks.compare_ada import capture
    from benchmarks.reference import fp64_reference, metrics, rounded_reference
    from tests.test_key_precision import key_cancellation
    from tests.test_value_precision import square_value_cancellation, value_residual_underflow

    attention = make_static_attention()
    records = []

    def check(label, inputs, dout, *, causal=False, scale=None, directed=False, graph=False):
        operands = [t.detach().requires_grad_() for t in inputs]
        out = attention(*operands, is_causal=causal, scale=scale)
        actual = out, *torch.autograd.grad(out, operands, dout)
        rounded = rounded_reference(inputs, dout, causal=causal, scale=scale, output=out)
        mathematical = fp64_reference(inputs, dout, causal=causal, scale=scale)
        errors = {
            reference_name: [
                dict(metrics(got[:, head : head + 1], want[:, head : head + 1]), head=head, component=name)
                for head in range(out.shape[1])
                for name, got, want in zip(("out", "dq", "dk", "dv"), actual, expected)
            ]
            for reference_name, expected in (("rounded", rounded), ("mathematical", mathematical))
        }
        limit = 0.01 if out.dtype == torch.bfloat16 else 0.005
        accepted = errors["rounded" if directed else "mathematical"]
        passed = all(
            e["finite"] and (e["relative_error"] < limit if e["reference_norm"] else e["max_absolute_error"] == 0)
            for e in accepted
        )
        if graph:
            capture(attention, inputs, dout, causal, "joint")
        record = dict(
            label=label,
            shape=list(out.shape),
            dtype=str(out.dtype),
            causal=causal,
            scale=scale,
            input_strides=[list(t.stride()) for t in inputs],
            directed=directed,
            graph_changed_inputs=graph,
            acceptance="rounded" if directed else "mathematical",
            norm_limit=limit,
            passed=passed,
            errors=errors,
        )
        records.append(record)
        args.output.write_text(json.dumps(dict(records=records), indent=2) + "\n")
        print(label, "pass" if passed else "FAIL", [max((e["relative_error"] or 0) for e in accepted)], flush=True)
        if not passed:
            raise AssertionError("The candidate violates the unchanged component norm limits.")

    torch.manual_seed(2033)
    cases = [(129, 64, 3, True)] if args.smoke else [(129, 64, 3, True), (257, 128, 5, False), (1024, 128, 24, True)]
    if not args.smoke:
        cases.extend([(2049, 64, 4, False), (4097, 128, 3, True)])
    for dtype in (torch.bfloat16, torch.float16):
        for length, dimension, heads, causal in cases:
            tensors = [torch.randn(1, heads, length, dimension, dtype=dtype, device="cuda") for _ in range(4)]
            check("random-tail", tuple(tensors[:3]), tensors[3], causal=causal, graph=True)
            if length == 257:
                strided = [
                    torch.randn(1, length, heads, dimension, dtype=dtype, device="cuda").transpose(1, 2)
                    for _ in range(4)
                ]
                check("bshd-storage-view", tuple(strided[:3]), strided[3], causal=causal, graph=True)

        if args.smoke:
            continue
        for dimension in (64, 128):
            inputs, dout = key_cancellation(dimension, dtype, "cuda", heads=1, square=True)
            check("negative-maximum-key-cancellation", inputs, dout, directed=True)
            inputs, dout, scale, _ = square_value_cancellation(dimension, dtype, "cuda", heads=1)
            check("value-cancellation", inputs, dout, scale=scale, directed=True)
    if not args.smoke:
        inputs, dout, scale, _ = value_residual_underflow(64, "cuda", heads=1)
        check("expected-fp16-probability-underflow", inputs, dout, scale=scale, directed=True)
    source = Path(__file__).with_name("blackwell_fa4_experiments.py")
    import kohakufa_cute.fa4_blackwell

    folder = Path(kohakufa_cute.fa4_blackwell.__file__).parent
    static_sources = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(folder.glob("*.py"))}
    report = dict(
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        static_sources=static_sources,
        maximum="producer-shared",
        factory="benchmarks.blackwell_fa4_experiments:make_static_attention",
        forward_schedule=forward_schedule,
        records=records,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
