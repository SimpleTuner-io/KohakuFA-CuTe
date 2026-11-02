"""Qualify the guarded S1024 path through actual public APIs and live policies."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(argv)
    import json

    import torch

    from benchmarks.reference import fp64_reference, metrics
    from benchmarks.rtx4090_automatic_controls import parent_api
    from benchmarks.validate_sm80_static import assess
    from kohakufa_cute import api, fa4_sm80
    from kohakufa_cute.fa4_sm80 import rtx4090

    root = options.output
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    checks = 0
    for dtype in (torch.bfloat16, torch.float16):
        for dimension in (64, 128):
            for causal in (False, True):
                torch.manual_seed(947209)
                inputs = [
                    torch.nn.Parameter(torch.randn(1, 16, 1024, dimension, device="cuda", dtype=dtype)) for _ in range(3)
                ]
                dout = torch.randn_like(inputs[0])
                torch.save(
                    {"inputs": inputs, "dout": dout, "causal": causal},
                    root / f'actual-{str(dtype).split(".")[-1]}-{dimension}-{causal}.pt',
                )
                for namespace in ("strict", "sdpa", "automatic"):
                    operation = {
                        "strict": api.attention,
                        "sdpa": api.scaled_dot_product_attention,
                        "automatic": api.automatic_scaled_dot_product_attention,
                    }[namespace]
                    kwargs = {"causal": causal} if namespace == "strict" else {"is_causal": causal, "enable_gqa": True}
                    out = operation(*inputs, **kwargs)
                    actual = out, *torch.autograd.grad(out, inputs, dout)
                    errors = assess(actual, inputs, dout, causal=causal, scale=None, directed=False, pointwise=False)
                    rows.append(
                        {
                            "dtype": str(dtype),
                            "dimension": dimension,
                            "causal": causal,
                            "namespace": namespace,
                            "max_fp64_norm": max(e["fp64"]["relative_error"] for e in errors),
                        }
                    )
                q, k, v = inputs
                for bad in (
                    q[:, :8],
                    q[:, :, :513],
                    q.float(),
                    torch._neg_view(q),
                    torch.empty(1, 16, 1024, dimension + 1, device="cuda", dtype=dtype, requires_grad=True)[..., :dimension],
                ):
                    assert api._try_rtx4090_s1024(bad, k, v, None, 0, causal, None, False) is None
                    checks += 1
                flat = torch.empty(q.numel() + 16, device="cuda", dtype=dtype)
                mutable = flat[: q.numel()].view_as(q).requires_grad_(True)
                for offset in (0, 1, 0):
                    with torch.no_grad():
                        mutable.set_(flat.untyped_storage(), offset, q.shape, q.stride())
                    selected = api._try_rtx4090_s1024(mutable, k, v, None, 0, causal, None, False)
                    assert (selected is None) == bool(offset)
                    checks += 1
                for scale in (0.0, -0.125, float("nan")):
                    assert api._try_rtx4090_s1024(q, k, v, None, 0, causal, scale, False) is None
                    checks += 1
                for manager in (torch.no_grad(), torch.inference_mode()):
                    with manager:
                        assert api._try_rtx4090_s1024(q, k, v, None, 0, causal, None, False) is None
                        torch.testing.assert_close(
                            api.scaled_dot_product_attention(q, k, v, is_causal=causal),
                            parent_api().scaled_dot_product_attention(q, k, v, is_causal=causal),
                            atol=0,
                            rtol=0,
                        )
                        checks += 1
                for owner, name in ((torch.compiler, "is_compiling"), (torch, "are_deterministic_algorithms_enabled")):
                    with patch.object(owner, name, return_value=True):
                        assert api._try_rtx4090_s1024(q, k, v, None, 0, causal, None, False) is None
                    checks += 1
                with patch.object(fa4_sm80, "available", return_value=False):
                    assert api._try_rtx4090_s1024(q, k, v, None, 0, causal, None, False) is None
                checks += 1
                if dimension == 64 and not causal:
                    references = [
                        fp64_reference(tuple(t[:, h : h + 1] for t in inputs), dout[:, h : h + 1], causal=causal)
                        for h in range(16)
                    ]
                    for flags in range(1, 8):
                        values = [t.detach().requires_grad_(bool(flags & (1 << i))) for i, t in enumerate(inputs)]
                        requested = [i for i, t in enumerate(values) if t.requires_grad]
                        out = api.scaled_dot_product_attention(*values, is_causal=causal)
                        grads = torch.autograd.grad(out, [values[i] for i in requested], dout)
                        for grad, i in zip(grads, requested):
                            for h in range(16):
                                error = metrics(grad[:, h : h + 1], references[h][i + 1])
                                assert error["finite"] and error["relative_error"] < (
                                    0.01 if dtype == torch.bfloat16 else 0.005
                                )
                        checks += 1
    # Actual boolean mask/GQA/rectangular fallbacks and compiled bypass.
    for queries, keys, kh, masking in ((1024, 1024, 16, True), (1024, 1025, 16, False), (1024, 1024, 8, False)):
        q = torch.randn(1, 16, queries, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k, v = [torch.randn(1, kh, keys, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(2)]
        mask = (torch.arange(keys, device="cuda") % 3 != 0).reshape(1, 1, 1, keys) if masking else None
        out = api.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        expected = parent_api().scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
        checks += 1
    q = torch.randn(1, 16, 1024, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    operation = torch.compile(api.scaled_dot_product_attention, backend="eager", fullgraph=True)
    with patch.object(rtx4090, "_s1024_guarded_call", side_effect=AssertionError("Compiled call entered eager helper")):
        out = operation(q, q, q)
    torch.testing.assert_close(out, parent_api().scaled_dot_product_attention(q, q, q), atol=0, rtol=0)
    (root / "report.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "public_component_rows": rows,
                "live_policy_cache_and_fallback_checks": checks,
                "actual_compiled_bypass": True,
            },
            indent=2,
        )
        + "\n"
    )
    print("Actual public component/policy/cache/compiled qualification passed", flush=True)


if __name__ == "__main__":
    main()
