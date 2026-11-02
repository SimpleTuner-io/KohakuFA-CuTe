"""Qualify actual GB10 D64 inference and preserve live training/fallback semantics."""

import argparse
import itertools
import json
from pathlib import Path
from unittest.mock import patch

import torch

from benchmarks.gb10_long_d64 import assess_output, references, retain_tensor
from benchmarks.gb10_width_controls import parent as load_parent


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--parent-source", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1179649)
    a = p.parse_args()
    from benchmarks.compare_l40s import graph_kernels
    from kohakufa_cute.fa4_sm80 import gb10

    assert gb10._qualified_device(torch.cuda.current_device())
    torch.set_num_threads(4)
    parent = load_parent(str(a.parent_source))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(seed=a.seed, cases=[], fallback=[], status="started")

    def save():
        a.output.write_text(json.dumps(report, indent=2) + "\n")

    for seq, dtype, causal in itertools.product((4096,), (torch.bfloat16, torch.float16), (False, True)):
        tag = f's{seq}-{str(dtype).split(".")[-1]}-c{int(causal)}'
        torch.manual_seed(a.seed + seq + (100003 if dtype == torch.float16 else 0))
        xs = [torch.randn(1, 16, seq, 64, device="cuda", dtype=dtype) for _ in range(3)]
        xs[0].add_(3)
        xs[1].sub_(3)
        row = dict(tag=tag, status="started", inputs=[[retain_tensor(t, a.output.parent) for t in xs]], checks=[])
        report["cases"].append(row)
        save()
        saved = []
        assert gb10.inference_compatible(*xs, causal=causal)
        with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
            out = gb10.scaled_dot_product_attention(*xs, is_causal=causal)
        assert not saved and not out.requires_grad
        row["checks"].append(dict(kind="negative_offset_eager", errors=assess_output(out.cpu(), references(xs, causal))))
        if seq == 4096:

            class IdentityTensor(torch.Tensor):
                pass

            for values in ([torch.nn.Parameter(t) for t in xs], [t.as_subclass(IdentityTensor) for t in xs]):
                assert gb10.inference_compatible(*values, causal=causal)
                assess_output(gb10.scaled_dot_product_attention(*values, is_causal=causal).cpu(), references(xs, causal))
            row["parameter_and_identity_subclass"] = "pass"
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            out = gb10.scaled_dot_product_attention(*xs, is_causal=causal)
        row["graph"] = graph_kernels(graph)
        for change in range(3):
            for t in xs:
                t.copy_(torch.randn_like(t))
            row["inputs"].append([retain_tensor(t, a.output.parent) for t in xs])
            save()
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            expected = references(xs, causal)
            row["checks"].append(
                dict(kind="changed_poisoned_graph", change=change, errors=assess_output(out.cpu(), expected))
            )
        train = []
        dout = torch.randn_like(xs[0])
        row["dout"] = retain_tensor(dout, a.output.parent)
        with torch.enable_grad():
            for operation in (parent.scaled_dot_product_attention, gb10.scaled_dot_product_attention):
                leaves = [t.detach().requires_grad_(True) for t in xs]
                assert not gb10.inference_compatible(*leaves, causal=causal)
                o = operation(*leaves, is_causal=causal)
                assert o.requires_grad
                train.append(tuple(t.detach().cpu() for t in (o, *torch.autograd.grad(o, leaves, dout))))
        assert all(torch.equal(before, after) for before, after in zip(*train))
        row["training_parent_bitwise_equal"] = True
        with torch.enable_grad():
            for position in range(3):
                leaves = [t.detach() for t in xs]
                leaves[position].requires_grad_(True)
                assert not gb10.inference_compatible(*leaves, causal=causal)
                o = gb10.scaled_dot_product_attention(*leaves, is_causal=causal)
                assert o.requires_grad
                g = torch.autograd.grad(o, leaves[position], dout)[0]
                assert torch.isfinite(g).all()
        for context in (torch.no_grad, torch.inference_mode):
            with context():
                assert gb10.inference_compatible(*leaves, causal=causal)
                o = gb10.scaled_dot_product_attention(*leaves, is_causal=causal)
                assert not o.requires_grad
                assess_output(o.cpu(), expected)
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(xs[0], torch.ones_like(xs[0]))
            try:
                gb10.scaled_dot_product_attention(dual, xs[1], xs[2], is_causal=causal)
            except (ValueError, RuntimeError) as error:
                assert "forward-AD" in str(error)
            else:
                raise AssertionError("Forward AD tangent discarded")
        torch.compiler.reset()
        compiled = torch.compile(gb10.scaled_dot_product_attention, backend="eager", fullgraph=True)
        row["compiled_fp64"] = assess_output(compiled(*xs, is_causal=causal).cpu(), expected)
        row["status"] = "pass"
        save()
        print(tag, "pass", flush=True)
    for dtype in (torch.bfloat16, torch.float16):
        for case in (
            "tail",
            "negative_view",
            "negative_scale",
            "zero_scale",
            "mask",
            "hidden",
            "strided",
            "unaligned",
            "heads",
            "dimension",
            "gqa",
        ):
            seq = 4095 if case == "tail" else 4096
            heads = 4 if case == "heads" else 16
            dim = 512 if case == "dimension" else 64
            xs = [torch.randn(1, heads, seq, dim, device="cuda", dtype=dtype) for _ in range(3)]
            kw = {}
            if case == "negative_view":
                xs[0] = xs[0]._neg_view()
            if case == "strided":
                xs[0] = torch.randn(1, heads, seq, dim * 2, device="cuda", dtype=dtype)[..., ::2]
            if case == "unaligned":
                xs[0] = torch.randn(xs[0].numel() + 1, device="cuda", dtype=dtype)[1:].view_as(xs[0])
            if case.endswith("scale"):
                kw["scale"] = -0.125 if case == "negative_scale" else 0.0
            if case in ("mask", "hidden"):
                kw["attn_mask"] = (
                    torch.ones(seq, seq, device="cuda", dtype=torch.bool)
                    if case == "mask"
                    else torch.zeros(seq, seq, device="cuda", dtype=torch.bool)
                )
            if case == "gqa":
                xs[1:] = [t[:, :8].contiguous() for t in xs[1:]]
                kw["enable_gqa"] = True
            assert not gb10.inference_compatible(
                *xs, mask=kw.get("attn_mask"), scale=kw.get("scale"), gqa=kw.get("enable_gqa", False)
            )
            item = dict(
                case=case,
                dtype=str(dtype),
                status="started",
                inputs=[retain_tensor(t, a.output.parent) for t in xs],
                views=[
                    dict(
                        shape=list(t.shape),
                        stride=list(t.stride()),
                        storage_offset=t.storage_offset(),
                        negative=t.is_neg(),
                        base=retain_tensor(t._base, a.output.parent) if t._base is not None else None,
                    )
                    for t in xs
                ],
                options={
                    k: (retain_tensor(v, a.output.parent) if isinstance(v, torch.Tensor) else v) for k, v in kw.items()
                },
            )
            report["fallback"].append(item)
            save()
            before = parent.scaled_dot_product_attention(*xs, **kw)
            after = gb10.scaled_dot_product_attention(*xs, **kw)
            assert torch.equal(before, after)
            if case == "hidden":
                assert not torch.count_nonzero(after)
            item.update(status="pass", parent_bitwise_equal=True)
            save()
    # Verify selected initialization failures propagate even with populated runner caches.
    xs = [torch.randn(1, 16, 4096, 64, device="cuda", dtype=torch.float16) for _ in range(3)]
    with patch.object(gb10, "_long_d64_runner", side_effect=RuntimeError("Selected adapter failed")):
        try:
            gb10.scaled_dot_product_attention(*xs)
        except RuntimeError as error:
            assert "Selected adapter failed" in str(error)
        else:
            raise AssertionError("Selected adapter failure fell through")
    report["selected_failure"] = "propagated"
    report["status"] = "pass"
    save()


if __name__ == "__main__":
    main()
