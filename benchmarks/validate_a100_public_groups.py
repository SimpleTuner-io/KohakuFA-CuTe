"""Check actual long public selection, literal statistics and changing graphs."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agreement", action="store_true")
    args = parser.parse_args()
    from benchmarks.compare_ada import capture
    from benchmarks.compare_l40s import graph_kernels
    from benchmarks.validate_sm80_static import assess, statistics
    from kohakufa_cute import api, fa4_sm80
    from kohakufa_cute.fa4_sm80 import _host

    torch.set_num_threads(4)
    report = {
        "rows": [],
        "reference": "Unchanged original gates on bit-preserving CPU copies, including actual O.",
        "agreement": args.agreement,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def copied(tensors):
        return tuple(t.detach().cpu() for t in tensors)

    def check(actual, inputs, dout):
        criteria = dict(causal=True, scale=None, directed=False, pointwise=False)
        cuda = assess(actual, inputs, dout, **criteria) if args.agreement else None
        cpu = assess(copied(actual), copied(inputs), dout.detach().cpu(), **criteria)
        return {"cpu": cpu, "cuda": cuda}

    cases = ((torch.bfloat16, 8, 64), (torch.float16, 16, 128))
    for dtype, heads, dimension in cases:
        torch.manual_seed(4099)
        inputs = [torch.randn(1, heads, 4096, dimension, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)]
        dout = torch.randn_like(inputs[0])
        row = {"shape": list(inputs[0].shape), "dtype": str(dtype)}
        report["current"] = row
        save()
        saved = []
        actual = ()
        try:
            with (
                patch.object(_host, "native_runner", wraps=_host.native_runner) as called,
                torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t),
            ):
                output = api.scaled_dot_product_attention(*inputs, is_causal=True)
            assert called.call_count == 1 and called.call_args.args[0].causal_head_group == (4 if heads == 8 else 16)
            assert len(saved) == 6
            row["selected_group"] = called.call_args.args[0].causal_head_group
            row["statistics"] = []
            for head in range(heads):
                part = tuple(t[:, head : head + 1] for t in inputs)
                maximum, logsum = (t[:, head : head + 1] for t in saved[-2:])
                if args.agreement:
                    statistics(part, maximum, logsum, True, dimension**-0.5)
                row["statistics"].append(
                    statistics(copied(part), maximum.detach().cpu(), logsum.detach().cpu(), True, dimension**-0.5)
                )
            actual = output, *torch.autograd.grad(output, inputs, dout)
            row["errors"] = check(actual, inputs, dout)
            graph, storage = capture(
                api.scaled_dot_product_attention, [t.detach() for t in inputs], dout, True, "joint", keep_graph=True
            )
            operands, _, _, actual = storage
            row["kernels"] = graph_kernels(graph)
            assert len(row["kernels"]) == 4 and all(r["BINARY_VERSION"] == 80 for r in row["kernels"])
            row["replays"] = []
            for seed in (4103, 4111):
                torch.manual_seed(seed)
                with torch.no_grad():
                    for tensor in (*operands, dout):
                        tensor.copy_(torch.randn_like(tensor))
                    for tensor in actual:
                        tensor.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                row["replays"].append(check(actual, operands, dout))
        except Exception:
            torch.save(
                {"inputs": copied(inputs), "dout": dout.detach().cpu(), "actual": copied(actual), "case": row},
                args.output.with_suffix(".failed.pt"),
            )
            raise
        report["rows"].append(row)
        del report["current"]
        save()
    report["status"] = "pass"
    save()


if __name__ == "__main__":
    main()
