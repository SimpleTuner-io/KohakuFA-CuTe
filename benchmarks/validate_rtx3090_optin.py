"""Qualify the bounded ordinary RTX 3090 public opt-in entry."""

import argparse
import itertools
import json
from pathlib import Path

import torch


def make_attention():
    from kohakufa_cute import rtx3090_attention

    return rtx3090_attention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--reference-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--compare-reference-devices", action="store_true", help="Run both unchanged reference gates outside sanitizers."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from benchmarks.compare_ada import capture
    from benchmarks.validate_sm80_static import assess, statistics
    from kohakufa_cute import rtx3090_attention
    from kohakufa_cute.rtx3090 import _device_matches, _eligible

    assert _device_matches(torch.cuda.current_device())
    operation = rtx3090_attention
    layouts = ((1, 16),)
    sequences = (1024, 1025) if args.smoke else (1024, 1025, 4096)
    if args.reference_device == "cpu":
        torch.set_num_threads(4)
    report = {
        "entry": "rtx3090_attention",
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
            causal=causal,
            scale=None,
            directed=False,
            pointwise=False,
        )

    for dtype, dimension, (batch, heads), sequence, causal in itertools.product(
        (torch.bfloat16, torch.float16), (64, 128), layouts, sequences, (False, True)
    ):
        torch.manual_seed(4121)
        inputs = [
            torch.randn(batch, heads, sequence, dimension, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)
        ]
        dout = torch.randn_like(inputs[0])
        row = {"shape": list(inputs[0].shape), "dtype": str(dtype), "causal": causal}
        report["current"] = row
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        actual, operands = (), inputs
        output = maximum = logsum = None
        try:
            saved = []
            with torch.autograd.graph.saved_tensors_hooks(
                lambda tensor: saved.append(tensor) or tensor, lambda tensor: tensor
            ):
                output = operation(*inputs, is_causal=causal)
            assert _eligible(*inputs, None, 0, causal, None, False)
            assert len(saved) == 6
            maximum, logsum = saved[-2:]
            row["statistics"] = statistics(
                copied(inputs, args.reference_device),
                maximum.detach().to(args.reference_device),
                logsum.detach().to(args.reference_device),
                causal,
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
                        copied(inputs, other), maximum.to(other), logsum.to(other), causal, dimension**-0.5
                    ),
                    "errors": check(actual, inputs, dout, other),
                }
                row["reference_agreement"] = []
                for head in range(heads):
                    part = tuple(t[:, head : head + 1] for t in inputs)
                    grad = dout[:, head : head + 1]
                    cpu_inputs, cpu_dout = copied(part, "cpu"), grad.detach().cpu()
                    cpu_math = fp64_reference(cpu_inputs, cpu_dout, causal=causal)
                    gpu_math = fp64_reference(part, grad, causal=causal)
                    actual_out = actual[0][:, head : head + 1].detach()
                    cpu_rounded = rounded_reference(cpu_inputs, cpu_dout, causal=causal, output=actual_out.cpu())
                    gpu_rounded = rounded_reference(part, grad, causal=causal, output=actual_out)
                    for component, cpu, gpu, cpu_low, gpu_low in zip(
                        ("out", "dq", "dk", "dv"), cpu_math, gpu_math, cpu_rounded, gpu_rounded
                    ):
                        torch.testing.assert_close(gpu.cpu(), cpu, rtol=1e-10, atol=1e-10)
                        row["reference_agreement"].append(
                            {
                                "head": head,
                                "component": component,
                                "fp64": metrics(gpu.cpu(), cpu),
                                "rounded": metrics(gpu_low.cpu(), cpu_low),
                            }
                        )
            if sequence == 1025:
                graph, storage = capture(operation, [t.detach() for t in inputs], dout, causal, "joint")
                operands, _, _, actual = storage
                replay_errors = []
                for _ in range(2):
                    with torch.no_grad():
                        for tensor in (*operands, dout):
                            tensor.copy_(torch.randn_like(tensor))
                        for tensor in actual:
                            tensor.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize()
                    replay_errors.append(check(actual, operands, dout, args.reference_device))
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
