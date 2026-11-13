"""Check medium causal head-order controls with fixed rounded equations."""

import argparse
import json
from pathlib import Path

import torch

from benchmarks.experiments.ada.compare_ada import capture
from benchmarks.experiments.ada.compare_l40s import graph_kernels
from benchmarks.experiments.ampere.a100_medium_group_controls import make_attention, make_direct
from benchmarks.experiments.ampere.validate_sm80_static import assess, statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", type=int, choices=(1, 4, 8), required=True)
    parser.add_argument("--sanitizer-smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = {"group": args.group, "rows": [], "reference": "CPU copies, unchanged per-head mathematical/rounded gates."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    public, direct = make_attention(group=args.group), make_direct(group=args.group)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    cases = (
        ((torch.bfloat16, 2048, False), (torch.float16, 257, True))
        if args.sanitizer_smoke
        else tuple(
            (dtype, length, negative)
            for dtype in (torch.bfloat16, torch.float16)
            for length, negative in ((2048, False), (2048, True), (129, True), (257, False))
        )
    )
    for dtype, length, negative in cases:
        torch.manual_seed(839747)
        inputs = [torch.randn(1, 8, length, 128, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)]
        dout = torch.randn_like(inputs[0])
        if negative:
            with torch.no_grad():
                inputs[0].mul_(0.3)
                inputs[1].mul_(0.3)
                inputs[0][..., 0], inputs[1][..., 0] = 20, -20
        row = {"group": args.group, "dtype": str(dtype), "sequence": length, "negative": negative, "public": length == 2048}
        operands_path = args.output.with_name(args.output.stem + f"-{dtype}-{length}-{negative}.pt")
        report["current"] = row
        save()
        captured = {"inputs": tuple(t.detach().cpu() for t in inputs), "dout": dout.detach().cpu(), "case": row.copy()}
        torch.save(captured, operands_path)
        function = public if length == 2048 else direct
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
            output = function(*inputs, is_causal=True)
        assert len(saved) == 6
        actual = output, *torch.autograd.grad(output, inputs, dout)
        copied = tuple(t.detach().cpu() for t in actual)
        captured["actual"] = copied
        captured["statistics"] = tuple(t.detach().cpu() for t in saved[-2:])
        torch.save(captured, operands_path)
        row["errors"] = assess(
            copied, captured["inputs"], captured["dout"], causal=True, scale=None, directed=negative, pointwise=False
        )
        row["statistics"] = [
            statistics(
                tuple(t[:, h : h + 1] for t in captured["inputs"]),
                captured["statistics"][0][:, h : h + 1],
                captured["statistics"][1][:, h : h + 1],
                True,
                128**-0.5,
            )
            for h in range(8)
        ]
        if length == 2048 and not negative:
            graph, storage = capture(function, [t.detach() for t in inputs], dout, True, "joint", keep_graph=True)
            operands, captured_dout, _, actual = storage
            row["resources"] = graph_kernels(graph)
            assert len(row["resources"]) == 4
            assert all(node["BINARY_VERSION"] == 80 for node in row["resources"])
            with torch.no_grad():
                for tensor in (*operands, captured_dout):
                    tensor.add_(0.03125)
                for tensor in actual:
                    tensor.fill_(float("nan"))
            operands_path = operands_path.with_name(operands_path.stem + ".replay.pt")
            captured = {
                "inputs": tuple(t.detach().cpu() for t in operands),
                "dout": captured_dout.detach().cpu(),
                "case": {**row, "replay": True},
            }
            torch.save(captured, operands_path)
            graph.replay()
            torch.cuda.synchronize()
            copied = tuple(t.detach().cpu() for t in actual)
            captured["actual"] = copied
            torch.save(captured, operands_path)
            row["replay"] = assess(
                copied, captured["inputs"], captured["dout"], causal=True, scale=None, directed=False, pointwise=False
            )
        report["rows"].append(row)
        report.pop("current")
        save()
    report["status"] = "pass"
    save()


if __name__ == "__main__":
    main()
