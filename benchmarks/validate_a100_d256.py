"""Broader D256 evaluation envelope; final automatic selection is checked separately."""

import argparse
import itertools
import json
from pathlib import Path
from unittest.mock import patch

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agreement", action="store_true")
    parser.add_argument("--tool-smoke", action="store_true")
    args = parser.parse_args()
    from benchmarks.a100_d256_evaluation_controls import make_attention
    from benchmarks.compare_ada import capture
    from benchmarks.compare_l40s import graph_kernels
    from benchmarks.device_rounded_reference import assess_device_intermediates
    from benchmarks.validate_sm80_static import statistics
    from kohakufa_cute.fa4_sm80 import _host
    from tests.test_value_precision import square_value_cancellation, value_residual_underflow

    function = make_attention()
    torch.set_num_threads(4)
    report = {"rows": [], "agreement": args.agreement, "tool_smoke": args.tool_smoke}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def copied(tensors):
        return tuple(t.detach().cpu() for t in tensors)

    def compare(actual, inputs, dout, causal, scale, directed):
        return assess_device_intermediates(
            actual, inputs, dout, causal=causal, scale=scale, directed=directed, agreement=args.agreement
        )

    def run(label, inputs, dout, *, causal=False, scale=None, selected=True, directed=False, graph=False):
        row = {
            "label": label,
            "shape": list(inputs[0].shape),
            "dtype": str(inputs[0].dtype),
            "causal": causal,
            "scale": scale,
        }
        report["current"] = row
        save()
        number = len(report["rows"])
        torch.save(
            {"inputs": copied(inputs), "dout": dout.detach().cpu(), "case": row},
            args.output.with_name(args.output.stem + f".input-{number}.pt"),
        )
        operands = [t.detach().requires_grad_(True) for t in inputs]
        actual, saved = (), []
        try:
            with (
                patch.object(_host, "native_runner", wraps=_host.native_runner) as called,
                torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t),
            ):
                output = function(*operands, is_causal=causal, scale=scale)
            assert called.called == selected, (label, selected, called.call_count)
            row["selected"] = called.called
            if selected:
                chosen = called.call_args.args[0]
                assert chosen.backward_n == 32 and chosen.v_in_regs and chosen.causal_head_group == 1
                assert len(saved) == 6
                effective_scale = inputs[0].shape[-1] ** -0.5 if scale is None else scale
                if args.agreement:
                    statistics(operands, saved[-2], saved[-1], causal, effective_scale)
                row["statistics"] = statistics(copied(operands), *copied(saved[-2:]), causal, effective_scale)
            actual = output, *torch.autograd.grad(output, operands, dout)
            row["errors"] = compare(actual, operands, dout, causal, scale, directed)
            if graph:
                captured, storage = capture(function, [t.detach() for t in inputs], dout, causal, "joint", keep_graph=True)
                row["kernels"] = graph_kernels(captured)
                if selected:
                    assert len(row["kernels"]) == 4 and all(k["BINARY_VERSION"] == 80 for k in row["kernels"])
                leaves, _, _, actual = storage
                row["replays"] = []
                for seed in (4337, 4349):
                    torch.manual_seed(seed)
                    with torch.no_grad():
                        for t in (*leaves, dout):
                            t.copy_(torch.randn_like(t))
                        for t in actual:
                            t.fill_(float("nan"))
                    captured.replay()
                    torch.cuda.synchronize()
                    row["replays"].append(compare(actual, leaves, dout, causal, scale, False))
        except Exception:
            torch.save(
                {
                    "inputs": copied(inputs),
                    "dout": dout.detach().cpu(),
                    "actual": copied(actual),
                    "saved": copied(saved),
                    "case": row,
                },
                args.output.with_suffix(".failed.pt"),
            )
            raise
        report["rows"].append(row)
        report.pop("current")
        save()

    sequences = (128,) if args.tool_smoke else (128, 1024, 4096)
    for dtype, heads, sequence, causal in itertools.product(
        (torch.bfloat16, torch.float16), (8, 16), sequences, (False, True)
    ):
        torch.manual_seed(4327 + len(report["rows"]))
        q, k, v, dout = [torch.randn(1, heads, sequence, 256, device="cuda", dtype=dtype) for _ in range(4)]
        run("selected-random", (q, k, v), dout, causal=causal, graph=sequence == 128)
    for dtype in (torch.bfloat16, torch.float16):
        inputs, dout, scale, _ = square_value_cancellation(256, dtype, "cuda", length=128, heads=8)
        run("rounded-cancellation", inputs, dout, scale=scale, directed=True)
    inputs, dout, scale, _ = value_residual_underflow(256, "cuda", length=128, heads=8)
    run("rounded-underflow", inputs, dout, scale=scale, directed=True)
    for causal in (False, True):
        q, k, v, dout = [torch.randn(1, 8, 128, 256, device="cuda", dtype=torch.bfloat16) * 0.1 for _ in range(4)]
        q[..., 0], k[..., 0] = 20, -20
        run("negative-common-offset", (q, k, v), dout, causal=causal, directed=True)
    for sequence, heads in ((129, 8), (257, 16), (128, 9)):
        q, k, v, dout = [torch.randn(1, heads, sequence, 256, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
        run("excluded-shape", (q, k, v), dout, selected=False)
    for scale in (-0.125, 0.0):
        q, k, v, dout = [torch.randn(1, 8, 128, 256, device="cuda", dtype=torch.float16) for _ in range(4)]
        run("signed-scale-fallback", (q, k, v), dout, scale=scale, selected=False)
    from kohakufa_cute.fa4_sm80._schedules import a100_schedule

    runner = _host.native_runner(a100_schedule(256, 128, False, 8))
    with torch.no_grad():
        try:
            runner.inference(q, k, v)
        except ValueError:
            report["explicit_inference_d256"] = "rejected; existing width boundary preserved"
        else:
            raise AssertionError("This training change must not widen explicit inference.")
    report["status"] = "pass"
    save()


if __name__ == "__main__":
    main()
