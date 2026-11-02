"""Held-out actual public shared-V long calls and changed graph operands."""

import argparse
import itertools
import json
from pathlib import Path
from unittest.mock import patch

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    from benchmarks.compare_ada import capture
    from benchmarks.compare_l40s import graph_kernels
    from benchmarks.validate_sm80_static import assess, statistics
    from kohakufa_cute import api
    from kohakufa_cute.fa4_sm80 import _host

    report = {
        "rows": [],
        "reference": "Original per-head device FP64 mathematical and rounded diagnostics; unchanged gates.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    for dtype, heads, causal in itertools.product((torch.bfloat16, torch.float16), (8, 16), (False, True)):
        torch.manual_seed(4457)
        q, k, v, dout = [torch.randn(1, heads, 4096, 256, device="cuda", dtype=dtype) for _ in range(4)]
        inputs = tuple(t.requires_grad_(True) for t in (q, k, v))
        row = {"dtype": str(dtype), "heads": heads, "causal": causal}
        report["current"] = row
        save()
        torch.save(
            {"inputs": tuple(t.detach().cpu() for t in inputs), "dout": dout.cpu(), "case": row},
            args.output.with_name(args.output.stem + f".input-{len(report['rows'])}.pt"),
        )
        actual, saved = (), []
        try:
            with (
                patch.object(_host, "native_runner", wraps=_host.native_runner) as called,
                torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t),
            ):
                output = api.scaled_dot_product_attention(*inputs, is_causal=causal)
            assert called.call_count == 1
            selected = called.call_args.args[0]
            assert selected.backward_n == 64 and not selected.v_in_regs
            assert selected.causal_head_group == (4 if causal else 1)
            assert len(saved) == 6
            row["statistics"] = statistics(inputs, saved[-2], saved[-1], causal, 256**-0.5)
            actual = output, *torch.autograd.grad(output, inputs, dout)
            row["errors"] = assess(actual, inputs, dout, causal=causal, scale=None, directed=False, pointwise=False)
            graph, storage = capture(
                api.scaled_dot_product_attention, [t.detach() for t in inputs], dout, causal, "joint", keep_graph=True
            )
            leaves, _, _, actual = storage
            row["kernels"] = graph_kernels(graph)
            assert len(row["kernels"]) == 4 and all(r["BINARY_VERSION"] == 80 for r in row["kernels"])
            row["replays"] = []
            for seed in (4463, 4481):
                torch.manual_seed(seed)
                with torch.no_grad():
                    for t in (*leaves, dout):
                        t.copy_(torch.randn_like(t))
                    for t in actual:
                        t.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                row["replays"].append(
                    assess(actual, leaves, dout, causal=causal, scale=None, directed=False, pointwise=False)
                )
        except Exception:
            torch.save(
                {
                    "inputs": tuple(t.detach().cpu() for t in inputs),
                    "dout": dout.cpu(),
                    "actual": tuple(t.detach().cpu() for t in actual),
                    "case": row,
                },
                args.output.with_suffix(".failed.pt"),
            )
            raise
        report["rows"].append(row)
        report.pop("current")
        save()
    report["status"] = "pass"
    save()


if __name__ == "__main__":
    main()
