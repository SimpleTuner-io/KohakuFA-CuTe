"""Public medium-causal arithmetic, changed graphs, and live routing checks."""

import argparse
import itertools
import json
from pathlib import Path

import torch

from benchmarks.experiments.ada.validate_l40s_static import assess
from benchmarks.experiments.ampere.validate_sm80_static import statistics
from kohakufa_cute import api, fa4_sm80
from kohakufa_cute.fa4_sm80 import _l40s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    def retain(label, inputs, dout, actual, saved, directed=False):
        cpu = tuple(t.detach().cpu() for t in inputs)
        assert len(saved) == 6
        errors = assess(
            tuple(t.detach().cpu() for t in actual),
            cpu,
            dout.cpu(),
            causal=True,
            scale=None,
            directed=directed,
            pointwise=False,
        )
        stats = [
            statistics(
                tuple(t[:, h : h + 1] for t in cpu),
                saved[-2].cpu()[:, h : h + 1],
                saved[-1].cpu()[:, h : h + 1],
                True,
                0.125,
            )
            for h in range(16)
        ]
        rows.append({"case": label, "errors": errors, "statistics": stats})
        args.output.write_text(json.dumps({"rows": rows}, indent=2) + "\n")

    for dtype, case in itertools.product(
        (torch.bfloat16, torch.float16), ("random",) if args.smoke else ("random", "negative", "sharp")
    ):
        torch.manual_seed(601773)
        inputs = [torch.randn(1, 16, 1024, 64, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)]
        dout = torch.randn_like(inputs[0])
        if case == "negative":
            with torch.no_grad():
                inputs[0].mul_(0.2)
                inputs[1].mul_(0.2)
                inputs[0][..., 0] = 20
                inputs[1][..., 0] = -20
        if case == "sharp":
            with torch.no_grad():
                inputs[0].mul_(8)
                inputs[1].mul_(8)
        label = f"{dtype}-{case}"
        folder = args.output.parent / "operands"
        folder.mkdir(exist_ok=True)
        torch.save({"inputs": [t.detach().cpu() for t in inputs], "dout": dout.cpu()}, folder / (label + ".pt"))
        assert fa4_sm80.compatible(*inputs, None, 0, True, None, False)
        saved = []
        try:
            with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
                out = api.scaled_dot_product_attention(*inputs, is_causal=True)
            actual = (out, *torch.autograd.grad(out, inputs, dout))
            retain(label, inputs, dout, actual, saved, case != "random")
        except Exception:
            torch.save(
                {"inputs": [t.detach().cpu() for t in inputs], "dout": dout.cpu()},
                args.output.with_suffix(".failed.pt"),
            )
            raise
        if case != "random":
            continue
        # Replays must read changing inputs and ignore poisoned output/gradient storage.
        from benchmarks.experiments.ada.compare_ada import capture

        graph, (graph_inputs, _, _, storage) = capture(api.scaled_dot_product_attention, inputs, dout, True, "joint")
        for index, amplitude in enumerate((0.5, 1.25)):
            with torch.no_grad():
                for t in graph_inputs:
                    t.copy_(torch.randn_like(t) * amplitude)
                for t in storage:
                    t.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            # Independent eager call validates the changed operands using the same public path.
            saved = []
            with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
                out = api.scaled_dot_product_attention(*graph_inputs, is_causal=True)
            torch.autograd.grad(out, graph_inputs, dout)
            retain(label + f"-graph-{index}", graph_inputs, dout, storage, saved)
    result = json.loads(args.output.read_text())
    result["status"] = "pass"
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
