"""Actual public D256 selection, literal statistics, fallbacks and live graphs."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    from benchmarks.compare_ada import capture
    from benchmarks.validate_sm80_static import assess, statistics
    from kohakufa_cute import api
    from kohakufa_cute.fa4_sm80 import a40

    args.output.parent.mkdir(parents=True, exist_ok=True)
    operands_dir = args.output.with_suffix(".operands")
    operands_dir.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    report = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": str(torch.__version__),
        "cases": [],
        "scope": "Actual public API; same rounded equations/FP64 per-head norms, statistics and changed-poisoned graphs. CPU references.",
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def run(label, shape, dtype, causal, *, selected=True, scale=None, negative=False, graph=False):
        torch.manual_seed(59213 + len(report["cases"]))
        q, k, v, do = [torch.randn(shape, device="cuda", dtype=dtype) for _ in range(4)]
        if negative:
            q = torch._neg_view(q)
        inputs = tuple(t.detach().requires_grad_(True) for t in (q, k, v))
        cpu = tuple(t.detach().cpu() for t in inputs)
        dout = do.cpu()
        torch.save({"inputs": cpu, "dout": dout, "causal": causal, "scale": scale}, operands_dir / (label + ".pt"))
        assert a40.compatible(*inputs, causal=causal, scale=scale) == selected, (label, "dispatch")
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
            if selected:
                with patch.object(
                    api, "scaled_dot_product_attention", side_effect=AssertionError("unexpected fallback")
                ):
                    out = a40.scaled_dot_product_attention(*inputs, is_causal=causal, scale=scale)
            else:
                out = a40.scaled_dot_product_attention(*inputs, is_causal=causal, scale=scale)
        actual = out, *torch.autograd.grad(out, inputs, do)
        errors = assess(
            tuple(t.detach().cpu() for t in actual),
            cpu,
            dout,
            causal=causal,
            scale=scale,
            directed=False,
            pointwise=False,
        )
        stats = []
        if selected:
            assert len(saved) == 6
            maximum, lognorm = (t.detach().cpu() for t in saved[-2:])
            amount = shape[-1] ** -0.5 if scale is None else scale
            for h in range(shape[1]):
                stats.append(
                    statistics(
                        tuple(t[:, h : h + 1] for t in cpu),
                        maximum[:, h : h + 1],
                        lognorm[:, h : h + 1],
                        causal,
                        amount,
                    )
                )
        row = {
            "case": label,
            "shape": shape,
            "dtype": str(dtype),
            "causal": causal,
            "scale": scale,
            "selected": selected,
            "errors": errors,
            "statistics": stats,
        }
        if graph:
            g, owners = capture(a40.scaled_dot_product_attention, inputs, do, causal, "joint")
            live, _, _, outputs = owners
            torch.manual_seed(59891)
            with torch.no_grad():
                for tensor in (*live, do):
                    tensor.copy_(torch.randn_like(tensor))
                for tensor in outputs:
                    tensor.fill_(float("nan"))
            torch.save(
                {"inputs": tuple(t.detach().cpu() for t in live), "dout": do.cpu(), "causal": causal},
                operands_dir / (label + "-changed.pt"),
            )
            g.replay()
            torch.cuda.synchronize()
            row["changed_poisoned_graph"] = assess(
                tuple(t.detach().cpu() for t in outputs),
                tuple(t.detach().cpu() for t in live),
                do.cpu(),
                causal=causal,
                scale=None,
                directed=False,
                pointwise=False,
            )
        report["cases"].append(row)
        save()

    if args.smoke:
        for dtype, causal in ((torch.bfloat16, False), (torch.float16, True)):
            run("smoke-" + str(dtype), (1, 16, 1024, 256), dtype, causal, graph=True)
    else:
        for dtype in (torch.bfloat16, torch.float16):
            for s in (1024, 4096):
                for causal in (False, True):
                    run(f"selected-{dtype}-{s}-{causal}", (1, 16, s, 256), dtype, causal, graph=s == 1024)
            for scale in (0.125, 0.03125):
                run(f"scale-{dtype}-{scale}", (1, 16, 1024, 256), dtype, True, scale=scale)
            run("negative-" + str(dtype), (1, 16, 1024, 256), dtype, False, negative=True)
            for dim in (16, 64, 128, 256, 512):
                run(f"fallback-{dtype}-{dim}", (1, 2, 129, dim), dtype, True, selected=False)
    report["status"] = "pass"
    save()


if __name__ == "__main__":
    main()
