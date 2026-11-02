"""Exercise new GB10 forward-only dispatch and its live fallback boundaries."""

import argparse
import itertools
import json
from pathlib import Path
from unittest.mock import patch

import torch

from benchmarks.gb10_inference import assess_output, references
from benchmarks.reference import fp64_reference, metrics


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    from kohakufa_cute.fa4_sm80 import gb10

    torch.set_num_threads(4)
    torch.manual_seed(262147)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    outdir = args.output.parent / (args.output.stem + "-operands")
    outdir.mkdir(exist_ok=True)
    report = dict(decisions=[], training=[], forward_ad=[], compiled=[])

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def check(xs, expected, **kw):
        got = gb10.inference_compatible(*xs, **kw)
        report["decisions"].append(
            dict(shape=list(xs[0].shape), options={k: str(v) for k, v in kw.items()}, expected=expected, actual=got)
        )
        save()
        assert got == expected

    for dim, dtype, causal in itertools.product((16, 64, 256), (torch.bfloat16, torch.float16), (False, True)):
        tag = f'd{dim}-{str(dtype).split(".")[-1]}-c{int(causal)}'
        xs = [torch.randn(1, 16, 128, dim, device="cuda", dtype=dtype) for _ in range(3)]
        check(xs, True, causal=causal)
        for pos in range(3):
            for case in ("negative", "strided", "unaligned", "grad"):
                values = [t.detach() for t in xs]
                if case == "negative":
                    values[pos] = values[pos]._neg_view()
                elif case == "strided":
                    values[pos] = torch.randn(1, 16, 128, dim * 2, device="cuda", dtype=dtype)[..., ::2]
                elif case == "unaligned":
                    values[pos] = torch.randn(values[pos].numel() + 1, device="cuda", dtype=dtype)[1:].view_as(values[pos])
                else:
                    values[pos].requires_grad_(True)
                if case == "grad":
                    with torch.enable_grad():
                        check(values, False, causal=causal)
                    check(values, True, causal=causal)
                else:
                    check(values, False, causal=causal)
        for kw in ({"mask": object()}, {"dropout": 0.1}, {"gqa": True}, {"scale": 0.0}, {"scale": -0.125}):
            check(xs, False, **kw)
        for shape in ((2, 16, 128, dim), (1, 8, 128, dim), (1, 16, 129, dim), (1, 16, 512, dim)):
            check([torch.randn(*shape, device="cuda", dtype=dtype) for _ in range(3)], False)
        for owner, flag in ((torch.compiler, "is_compiling"), (torch, "are_deterministic_algorithms_enabled")):
            with patch.object(owner, flag, return_value=True):
                check(xs, False)
        with torch.enable_grad():
            leaves = [t.detach().requires_grad_(True) for t in xs]
            dout = torch.randn_like(leaves[0])
            torch.save(
                dict(inputs=[t.detach().cpu() for t in leaves], dout=dout.cpu(), causal=causal), outdir / (tag + ".pt")
            )
            out = gb10.scaled_dot_product_attention(*leaves, is_causal=causal)
            got = (out, *torch.autograd.grad(out, leaves, dout))
        errors = []
        for h in range(16):
            ref = fp64_reference(
                tuple(t.detach().cpu()[:, h : h + 1] for t in leaves), dout.cpu()[:, h : h + 1], causal=causal
            )
            row = [metrics(t.detach().cpu()[:, h : h + 1], r) for t, r in zip(got, ref)]
            assert all(
                e["finite"]
                and (
                    e["relative_error"] < (0.01 if dtype == torch.bfloat16 else 0.005)
                    if e["reference_norm"]
                    else e["max_absolute_error"] == 0
                )
                for e in row
            ), row
            errors.append(row)
        report["training"].append(dict(tag=tag, errors=errors))
        save()
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(xs[0], torch.ones_like(xs[0]))
            try:
                gb10.scaled_dot_product_attention(dual, xs[1], xs[2], is_causal=causal)
            except (ValueError, RuntimeError) as error:
                assert "forward-AD" in str(error)
            else:
                raise AssertionError("ForwardAD tangent discarded")
        report["forward_ad"].append(tag)
        # Each shape/dtype/visibility case is independent; avoid exhausting Dynamo's per-code cache.
        torch.compiler.reset()
        compiled = torch.compile(gb10.scaled_dot_product_attention, backend="eager", fullgraph=True)
        output = compiled(*xs, is_causal=causal)
        report["compiled"].append(
            dict(tag=tag, errors=assess_output(output.cpu(), references(tuple(t.cpu() for t in xs), causal)))
        )
        save()
    report["status"] = "pass"
    save()


if __name__ == "__main__":
    main()
