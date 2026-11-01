"""Check causal CTA remapping across residual head groups and batches."""

import argparse
import itertools
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", type=int, choices=(1, 2, 4, 16), required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--variant", choices=("q2-register", "q1-shared", "q2-shared"), default="q2-register")
    parser.add_argument("--reference-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--compare-reference-devices",
        action="store_true",
        help="Run both unchanged reference gates outside sanitizers.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from benchmarks.gb10_native_head_groups import make_attention
    from benchmarks.validate_sm80_static import assess, statistics

    operation = make_attention(head_group=args.group, direct=True, variant=args.variant)
    layouts = ((1, 3), (2, 3), (1, 18)) if args.smoke else ((1, 1), (1, 3), (2, 3), (1, 18), (2, 7), (1, 24))
    sequences = (129,) if args.smoke else (129, 257)
    if args.reference_device == "cpu":
        torch.set_num_threads(4)
    report = {
        "group": args.group,
        "variant": args.variant,
        "smoke": args.smoke,
        "reference_device": args.reference_device,
        "compare_reference_devices": args.compare_reference_devices,
        "cpu_threads": torch.get_num_threads() if args.reference_device == "cpu" else None,
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def copied(tensors, device):
        # Preserve the input/output dtype and its already rounded values.
        return tuple(t.detach().to(device) for t in tensors)

    def check(actual, inputs, dout, device):
        return assess(
            copied(actual, device),
            copied(inputs, device),
            dout.detach().to(device),
            causal=True,
            scale=None,
            directed=False,
            pointwise=False,
        )

    for dtype, dimension, (batch, heads), sequence in itertools.product(
        (torch.bfloat16, torch.float16), (128,), layouts, sequences
    ):
        torch.manual_seed(4121)
        inputs = [
            torch.randn(batch, heads, sequence, dimension, device="cuda", dtype=dtype, requires_grad=True)
            for _ in range(3)
        ]
        dout = torch.randn_like(inputs[0])
        row = {"shape": list(inputs[0].shape), "dtype": str(dtype), "causal": True}
        report["current"] = row
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        actual, operands = (), inputs
        output = maximum = logsum = None
        try:
            saved = []
            with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
                output = operation(*inputs, is_causal=True)
            assert len(saved) == 6, "Native configured entry did not save its expected six tensors."
            row["autograd_node"] = output.grad_fn.name()
            maximum, logsum = saved[-2:]
            row["statistics"] = statistics(
                copied(inputs, args.reference_device),
                maximum.detach().to(args.reference_device),
                logsum.detach().to(args.reference_device),
                True,
                dimension**-0.5,
            )
            actual = output, *torch.autograd.grad(output, inputs, dout)
            row["errors"] = check(actual, inputs, dout, args.reference_device)
            if args.compare_reference_devices:
                from benchmarks.reference import fp64_reference, metrics, rounded_reference

                other = "cuda" if args.reference_device == "cpu" else "cpu"
                row["alternate_reference"] = {
                    "device": other,
                    "statistics": statistics(
                        copied(inputs, other), maximum.to(other), logsum.to(other), True, dimension**-0.5
                    ),
                    "errors": check(actual, inputs, dout, other),
                }
                cpu_inputs, cpu_dout = copied(inputs, "cpu"), dout.detach().cpu()
                cpu_math = fp64_reference(cpu_inputs, cpu_dout, causal=True)
                gpu_math = fp64_reference(inputs, dout, causal=True)
                cpu_rounded = rounded_reference(cpu_inputs, cpu_dout, causal=True, output=actual[0].detach().cpu())
                gpu_rounded = rounded_reference(inputs, dout, causal=True, output=actual[0])
                row["reference_agreement"] = {}
                for component, cpu, gpu, cpu_low, gpu_low in zip(
                    ("out", "dq", "dk", "dv"), cpu_math, gpu_math, cpu_rounded, gpu_rounded
                ):
                    # FP32 products/reductions can cross an operand-rounding
                    # boundary. Retain that diagnostic; both devices above
                    # must independently pass the original acceptance gates.
                    torch.testing.assert_close(gpu.cpu(), cpu, rtol=1e-10, atol=1e-10)
                    row["reference_agreement"][component] = {
                        "fp64": metrics(gpu.cpu(), cpu),
                        "rounded": metrics(gpu_low.cpu(), cpu_low),
                    }
            if sequence == 129:
                operands = [t.detach().clone().requires_grad_(True) for t in inputs]
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())

                def execute():
                    captured = []
                    with torch.autograd.graph.saved_tensors_hooks(lambda t: captured.append(t) or t, lambda t: t):
                        out = operation(*operands, is_causal=True)
                    result = out, *torch.autograd.grad(out, operands, dout)
                    return result, *captured[-2:]

                with torch.cuda.stream(stream):
                    for _ in range(3):
                        execute()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        actual, graph_maximum, graph_logsum = execute()
                torch.cuda.current_stream().wait_stream(stream)
                replay_errors = []
                replay_statistics = []
                for _ in range(2):
                    with torch.no_grad():
                        for tensor in (*operands, dout):
                            tensor.copy_(torch.randn_like(tensor))
                        for tensor in (*actual, graph_maximum, graph_logsum):
                            tensor.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize()
                    replay_errors.append(check(actual, operands, dout, args.reference_device))
                    replay_statistics.append(
                        statistics(
                            copied(operands, args.reference_device),
                            graph_maximum.to(args.reference_device),
                            graph_logsum.to(args.reference_device),
                            True,
                            dimension**-0.5,
                        )
                    )
                row["changed_poisoned_statistics"] = replay_statistics
                row["changed_poisoned_replays"] = replay_errors
        except Exception:
            torch.save(
                {
                    "inputs": copied(inputs, "cpu"),
                    "replay_inputs": copied(operands, "cpu"),
                    "actual": copied(actual, "cpu"),
                    "forward_output": None if output is None else output.detach().cpu(),
                    "maximum": None if maximum is None else maximum.detach().cpu(),
                    "logsum": None if logsum is None else logsum.detach().cpu(),
                    "dout": dout.detach().cpu(),
                    "case": row,
                },
                args.output.with_suffix(".failed.pt"),
            )
            raise
        report["rows"].append(row)
        del report["current"]
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = "pass"
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
