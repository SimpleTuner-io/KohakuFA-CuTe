"""Qualify experimental full-GB10 S128 forward-only widths on saved inputs."""

import argparse
import itertools
import json
from pathlib import Path

import torch

from benchmarks.gb10_inference import assess_output, references


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--public", action="store_true")
    p.add_argument("--seed", type=int, default=196607)
    args = p.parse_args()
    from benchmarks.compare_l40s import graph_kernels
    from kohakufa_cute.fa4_sm80 import gb10
    from kohakufa_cute.fa4_sm80._host import native_runner

    assert gb10._qualified_device(torch.cuda.current_device())
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    inputs_dir = args.output.parent / (args.output.stem + "-operands")
    inputs_dir.mkdir(exist_ok=True)
    report = dict(seed=args.seed, torch=str(torch.__version__), device=torch.cuda.get_device_name(), cases=[], rejection=[])

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    for dim, dtype, causal in itertools.product((16, 64, 256), (torch.bfloat16, torch.float16), (False, True)):
        tag = f'd{dim}-{str(dtype).split(".")[-1]}-c{int(causal)}'
        values = [torch.randn(1, 16, 128, dim, device="cuda", dtype=dtype) for _ in range(3)]
        torch.save(dict(inputs=[t.cpu() for t in values], causal=causal), inputs_dir / (tag + ".pt"))
        row = dict(tag=tag, status="started", checks=[])
        report["cases"].append(row)
        save()
        runner = native_runner(gb10.inference_schedule(128, causal))
        expected = references(tuple(t.cpu() for t in values), causal)
        operation = (
            (lambda *xs, causal=causal: gb10.scaled_dot_product_attention(*xs, is_causal=causal))
            if args.public
            else (lambda *xs, causal=causal: runner.inference(*xs, causal=causal))
        )
        if args.public:
            assert gb10.inference_compatible(*values, causal=causal)
        out = operation(*values)
        row["checks"].append(assess_output(out.cpu(), expected))
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            out = operation(*values)
        graph.replay()
        torch.cuda.synchronize()
        row["graph_resources"] = graph_kernels(graph)
        row["checks"].append(assess_output(out.cpu(), expected))
        for change in range(3):
            for t in values:
                t.copy_(torch.randn_like(t))
            torch.save(dict(inputs=[t.cpu() for t in values], causal=causal), inputs_dir / (tag + f"-changed{change}.pt"))
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            row["checks"].append(assess_output(out.cpu(), references(tuple(t.cpu() for t in values), causal)))
        for kind in ("requested_grad", "negative_view", "negative_scale", "zero_scale", "normal_mode"):
            xs = [t.detach() for t in values]
            opts = dict(causal=causal)
            if kind == "normal_mode" and dim == 64:
                continue
            if kind == "requested_grad":
                xs[0].requires_grad_(True)
            if kind == "negative_view":
                xs[0] = xs[0]._neg_view()
            if kind.endswith("scale"):
                opts["scale"] = -0.125 if kind == "negative_scale" else 0.0
            try:
                with torch.enable_grad() if kind == "requested_grad" else torch.no_grad():
                    (runner if kind == "normal_mode" else runner.inference)(*xs, **opts)
            except (ValueError, RuntimeError):
                report["rejection"].append(dict(tag=tag, case=kind, status="pass"))
            else:
                raise AssertionError((tag, kind, "unsupported call accepted"))
        row["status"] = "pass"
        save()
        print(tag, "pass", flush=True)
    report["status"] = "pass"
    save()


if __name__ == "__main__":
    main()
