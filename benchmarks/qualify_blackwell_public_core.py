"""Verify bounded public Blackwell dispatch and changing-input graph ownership."""

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    from benchmarks.compare_ada import capture
    from benchmarks.compare_l40s import graph_kernels
    from benchmarks.reference import fp64_reference, metrics
    from kohakufa_cute import api, fa4_blackwell

    if not fa4_blackwell.available():
        raise AssertionError("The qualified optional runtime was not available.")
    torch.manual_seed(2231)
    records = []
    for name in ("attention", "scaled_dot_product_attention", "automatic_scaled_dot_product_attention"):
        original = getattr(api, name)

        def attention(q, k, v, *, is_causal=False, **kwargs):
            return original(
                q, k, v, **({"causal": is_causal} if name == "attention" else {"is_causal": is_causal}), **kwargs
            )

        inputs = tuple(torch.randn(1, 32, 2049, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        dout = torch.randn_like(inputs[0])
        operands = [t.detach().requires_grad_() for t in inputs]
        with patch.object(fa4_blackwell, "run_attention", wraps=fa4_blackwell.run_attention) as selected:
            with torch.cuda.nvtx.range("public/eager/" + name):
                out = attention(*operands, is_causal=True)
                actual = out, *torch.autograd.grad(out, operands, dout)
            graph, owner = capture(attention, inputs, dout, True, "joint", keep_graph=True)
            if selected.call_count == 0:
                raise AssertionError("The public API did not select its qualified schedule.")
        reference = fp64_reference(inputs, dout, causal=True)
        errors = [
            dict(metrics(got[:, h : h + 1], want[:, h : h + 1]), component=component, head=h)
            for h in range(32)
            for component, got, want in zip(("out", "dq", "dk", "dv"), actual, reference)
        ]
        if not all(
            e["finite"] and (e["relative_error"] < 0.01 if e["reference_norm"] else e["max_absolute_error"] == 0)
            for e in errors
        ):
            raise AssertionError("The public path violates the unchanged component norm limits.")
        kernels = graph_kernels(graph)
        if not any("fa4_blackwell" in k["symbol"] for k in kernels):
            raise AssertionError("The public graph did not launch the selected static schedule.")
        records.append(dict(api=name, shape=list(out.shape), graph_changed_inputs=True, errors=errors, kernels=kernels))
        print(name, "selected and passed", flush=True)
    with patch.object(
        fa4_blackwell, "run_attention", side_effect=AssertionError("Unsupported metadata selected the optional schedule.")
    ):
        cases = [("short", 1, 3, 129, 64, 1, None, None)]
        for label, batch, heads, length, dimension, group, scale, masking in cases:
            q = torch.randn(batch, heads, length, dimension, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            k, v = [
                torch.randn(
                    batch, heads // group, length, dimension, device="cuda", dtype=torch.bfloat16, requires_grad=True
                )
                for _ in range(2)
            ]
            mask = torch.ones(1, 1, 1, length, device="cuda", dtype=torch.bool) if masking else None
            out = api.scaled_dot_product_attention(q, k, v, scale=scale, attn_mask=mask, enable_gqa=group > 1)
            gradients = torch.autograd.grad(out, (q, k, v), torch.ones_like(out))
            if not all(torch.isfinite(t).all() for t in (out, *gradients)):
                raise AssertionError("A native fallback produced nonfinite values.")
            records.append(dict(fallback=label, finite=True, shape=list(q.shape)))
            print(label, "native fallback passed", flush=True)
    compiled = torch.compile(api.scaled_dot_product_attention, fullgraph=True)
    inputs = tuple(torch.randn(1, 3, 2048, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    dout = torch.randn_like(inputs[0])
    graph, owner = capture(compiled, inputs, dout, True, "joint", keep_graph=True)
    compiled_kernels = graph_kernels(graph)
    if any("fa4_blackwell" in k["symbol"] for k in compiled_kernels):
        raise AssertionError("Compiled execution selected an unqualified optional schedule.")
    if not any("kohakufa" in k["symbol"] for k in compiled_kernels):
        raise AssertionError("Compiled execution did not launch the native CuTe fallback.")
    records.append(
        dict(
            fallback="compiled-fullgraph",
            shape=list(inputs[0].shape),
            graph_changed_inputs=True,
            kernels=compiled_kernels,
        )
    )
    print("compiled-fullgraph native fallback passed", flush=True)
    folder = Path(fa4_blackwell.__file__).parent
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob("*.py"))}
    report = dict(
        available=True,
        static_sources=sources,
        api_sha256=hashlib.sha256(Path(api.__file__).read_bytes()).hexdigest(),
        records=records,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
