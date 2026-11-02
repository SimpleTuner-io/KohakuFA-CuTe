"""Actual public short D256 selection, literal statistics, fallbacks and live graphs."""

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
    from benchmarks.reference import fp64_reference, metrics
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

    def run(label, shape, dtype, causal, *, selected=True, scale=None, negative=False, graph=False, strided_do=False):
        torch.manual_seed(59213 + len(report["cases"]))
        q, k, v, do = [torch.randn(shape, device="cuda", dtype=dtype) for _ in range(4)]
        if strided_do:
            do = torch.randn((*shape[:-1], shape[-1] * 2), device="cuda", dtype=dtype)[..., ::2]
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
        if selected:
            with torch.no_grad():
                forward = a40.scaled_dot_product_attention(*inputs, is_causal=causal, scale=scale)
            assert not forward.requires_grad
            expected = fp64_reference(cpu, dout, causal=causal, scale=scale)[0]
            row["no_grad_forward"] = []
            for head in range(shape[1]):
                error = metrics(forward[:, head : head + 1].cpu(), expected[:, head : head + 1])
                assert error["finite"] and error["relative_error"] < (0.01 if dtype == torch.bfloat16 else 0.005)
                row["no_grad_forward"].append(error)
        report["cases"].append(row)
        save()

    for dtype in (torch.bfloat16, torch.float16):
        for causal in (False, True):
            if args.smoke and (dtype == torch.bfloat16) == causal:
                continue
            prefix = f"short-{dtype}-{causal}"
            run(prefix, (1, 16, 128, 256), dtype, causal, graph=True, strided_do=True)
            if args.smoke:
                continue
            run(prefix + "-scale", (1, 16, 128, 256), dtype, causal, scale=0.125)
            run(prefix + "-negative", (1, 16, 128, 256), dtype, causal, negative=True)
            for shape in (
                (1, 16, 127, 256),
                (1, 16, 129, 256),
                (1, 16, 512, 256),
                (1, 8, 128, 256),
                (2, 3, 128, 256),
                (1, 16, 128, 16),
                (1, 16, 128, 64),
                (1, 16, 128, 128),
                (1, 16, 128, 512),
            ):
                run(prefix + "-fallback-" + "x".join(map(str, shape)), shape, dtype, causal, selected=False)
    report["status"] = "pass"
    save()


if __name__ == "__main__":
    main()
