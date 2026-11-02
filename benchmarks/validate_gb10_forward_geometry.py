"""Qualify dense GB10 forward geometry and its declared backward rounding."""

import argparse
import json
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True, help="Report outside the checkout; operands are saved beside it.")
    p.add_argument("--forward-only", action="store_true")
    a = p.parse_args()

    from benchmarks.compare_ada import capture
    from benchmarks.reference import fp64_reference, metrics
    from benchmarks.validate_a40_shared import assess
    from kohakufa_cute.fa4_sm80 import gb10
    from kohakufa_cute.fa4_sm80._host import native_runner

    if not torch.cuda.is_available() or not gb10._qualified_device(torch.cuda.current_device()):
        p.error("This qualification requires the full 48-SM GB10 optional runtime.")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    operands_dir = a.output.parent / (a.output.stem + "-operands")
    operands_dir.mkdir(exist_ok=False)
    r = {
        "scope": "Changed forward geometry; full independent per-head O/dQ/dK/dV checks outside sanitizers, forward ownership checks inside tools",
        "forward_only": a.forward_only,
        "reference_threads": 4,
        "criterion": "Random timing-family cases use independent per-head FP64 norm gates; directed negative-offset cases use identical norm limits on declared once-rounded equations and retain mathematical diagnostics. Original old/new offset failure and rounding-floor comparison retained separately.",
        "rows": [],
    }

    def save():
        a.output.write_text(json.dumps(r, indent=2) + "\n")

    torch.set_num_threads(4)
    for dtype in (torch.bfloat16, torch.float16):
        lengths = (129, 257, 1024) if a.forward_only else (1024, 4096)
        for length in lengths:
            for offset in (False, True):
                torch.manual_seed(82193 + length)
                q, k, v = [torch.randn(1, 16, length, 128, device="cuda", dtype=dtype) for _ in range(3)]
                if offset:
                    q.add_(3)
                    k.sub_(3)
                dout = torch.randn_like(q)
                name = f'{str(dtype).split(".")[-1]}-s{length}-offset{int(offset)}'
                torch.save(
                    {"inputs": [t.cpu() for t in (q, k, v)], "dout": dout.cpu(), "offset": offset},
                    operands_dir / (name + ".pt"),
                )
                if a.forward_only:
                    schedule = gb10.schedule(4096, False, 16, dtype)
                    operation = native_runner(schedule)
                    out = operation(q, k, v, causal=False)
                    expected = fp64_reference(tuple(t.cpu() for t in (q, k, v)), dout.cpu())[0]
                    errors = []
                    for head in range(16):
                        e = metrics(out[:, head : head + 1].cpu(), expected[:, head : head + 1])
                        assert e["finite"] and e["relative_error"] < (0.01 if dtype == torch.bfloat16 else 0.005), e
                        errors.append(e)
                    graph, storage = capture(
                        lambda q, k, v, is_causal=False: operation(q, k, v, causal=False),
                        (q, k, v),
                        dout,
                        False,
                        "forward",
                        keep_graph=True,
                    )
                else:
                    q, k, v = [t.detach().requires_grad_(True) for t in (q, k, v)]
                    out = gb10.scaled_dot_product_attention(q, k, v)
                    actual = (out, *torch.autograd.grad(out, (q, k, v), dout))
                    torch.save({"actual": [t.detach().cpu() for t in actual]}, operands_dir / (name + "-actual.pt"))
                    errors = assess(actual, (q, k, v), dout, causal=False, scale=None, directed=offset, pointwise=False)
                    graph, storage = capture(
                        gb10.scaled_dot_product_attention, (q, k, v), dout, False, "joint", keep_graph=True
                    )
                inputs, gradient, _, actual = storage
                with torch.no_grad():
                    for t in inputs:
                        t.copy_(torch.randn_like(t))
                    gradient.copy_(torch.randn_like(gradient))
                    for t in actual:
                        t.fill_(float("nan"))
                torch.save(
                    {"inputs": [t.detach().cpu() for t in inputs], "dout": gradient.cpu()},
                    operands_dir / (name + "-graph.pt"),
                )
                graph.replay()
                torch.cuda.synchronize()
                if a.forward_only:
                    expected = fp64_reference(tuple(t.detach().cpu() for t in inputs), gradient.cpu())[0]
                    changed = []
                    for head in range(16):
                        e = metrics(actual[0][:, head : head + 1].cpu(), expected[:, head : head + 1])
                        assert e["finite"] and e["relative_error"] < (0.01 if dtype == torch.bfloat16 else 0.005), e
                        changed.append(e)
                else:
                    changed = assess(actual, inputs, gradient, causal=False, scale=None, directed=False, pointwise=False)
                r["rows"].append({"case": name, "eager": errors, "changed_poisoned_graph": changed, "status": "pass"})
                save()
    r["status"] = "pass"
    save()
    print("geometry qualification pass", len(r["rows"]), flush=True)


if __name__ == "__main__":
    main()
