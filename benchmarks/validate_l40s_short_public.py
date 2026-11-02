"""Qualify the static S128 D16 public entry under the maintained arithmetic."""

import argparse
import importlib.util
import itertools
import json
from pathlib import Path

import torch

from benchmarks.reference import fp64_reference, metrics, rounded_reference
from benchmarks.validate_sm80_static import assess, statistics
from kohakufa_cute.api import scaled_dot_product_attention as rtx3090_attention
from kohakufa_cute.fa4_sm80._l40s import compatible as _eligible
from tests.test_value_precision import square_value_cancellation, value_residual_underflow


def make_attention():
    return rtx3090_attention


def make_parent_attention(source):
    specification = importlib.util.spec_from_file_location("kohakufa_cute._rtx3090_parent_short", source)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.rtx3090_attention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--compare-reference-devices", action="store_true")
    args = parser.parse_args()
    from benchmarks.compare_ada import capture

    torch.set_num_threads(4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"entry": "public_full_L40S_short", "reference_device": "cpu", "rows": []}

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    for dtype, causal in itertools.product((torch.bfloat16, torch.float16), (False, True)):
        cases = (
            ("random", "negative", "sharp")
            if args.smoke
            else ("random", "negative", "sharp", "cancellation", "underflow", "underflow-scaled")
        )
        for name in cases:
            torch.manual_seed(296647)
            inputs = [torch.randn(1, 16, 128, 16, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)]
            dout = torch.randn_like(inputs[0])
            scale = None
            if name == "negative":
                with torch.no_grad():
                    inputs[0].mul_(0.3)
                    inputs[1].mul_(0.3)
                    inputs[0][..., 0], inputs[1][..., 0] = 20, -20
            elif name == "sharp":
                with torch.no_grad():
                    for t in inputs:
                        t.zero_()
                    inputs[0][..., 0] = 40
                    inputs[1][..., 0] = 5
                    inputs[1][:, :, 0, 0] = 10
                    inputs[2][..., 0] = -1
                    inputs[2][:, :, 0, 0] = 1
                    dout.zero_()
                    dout[..., 0] = 1
            elif name in ("cancellation", "underflow", "underflow-scaled"):
                factory = square_value_cancellation if name == "cancellation" else value_residual_underflow
                if name == "cancellation":
                    inputs, dout, scale, _ = factory(16, dtype, "cuda", length=128, heads=16)
                elif dtype == torch.float16:
                    if name == "underflow-scaled" and not causal:
                        continue
                    inputs, dout, scale, _ = factory(16, "cuda", length=128, heads=16)
                    if name == "underflow-scaled":
                        dout.div_(8)
                else:
                    continue
                inputs = [t.detach().requires_grad_(True) for t in inputs]
            row = {"dtype": str(dtype), "causal": causal, "case": name}
            operand_dir = args.output.parent / (args.output.stem + "-operands")
            operand_dir.mkdir(exist_ok=True)
            torch.save(
                {"inputs": [t.detach().cpu() for t in inputs], "dout": dout.cpu(), "causal": causal, "scale": scale},
                operand_dir / (str(dtype).split(".")[-1] + f"-c{int(causal)}-" + name + ".pt"),
            )
            report["current"] = row
            save()
            actual = ()
            try:
                saved = []
                with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
                    output = rtx3090_attention(*inputs, is_causal=causal, scale=scale)
                assert _eligible(*inputs, None, 0, causal, scale, False) and len(saved) == 6
                actual = output, *torch.autograd.grad(output, inputs, dout)
                cpu_inputs = tuple(t.detach().cpu() for t in inputs)
                cpu_dout = dout.cpu()
                row["statistics"] = statistics(cpu_inputs, saved[-2].cpu(), saved[-1].cpu(), causal, scale or 0.25)
                if name == "underflow" and causal:
                    # The dense fixture overflows final FP16 dV when made causal.
                    # Preserve it as a cast-overflow diagnostic, outside finite norm cases.
                    expected = rounded_reference(cpu_inputs, cpu_dout, causal=True, scale=scale, output=actual[0].cpu())
                    row["diagnostic"] = "expected final FP16 overflow"
                    row["components"] = []
                    for component, got, reference in zip(("out", "dq", "dk", "dv"), actual, expected):
                        got = got.cpu()
                        cast = reference.to(dtype)
                        assert not got.isnan().any()
                        assert torch.equal(got.isinf(), cast.isinf())
                        overflow = cast.isinf()
                        assert torch.equal(got[overflow].signbit(), cast[overflow].signbit())
                        for head in range(16):
                            finite = torch.isfinite(cast[:, head : head + 1])
                            error = metrics(got[:, head : head + 1][finite], reference[:, head : head + 1][finite])
                            assert error["finite"]
                            assert (
                                error["relative_error"] < 0.005
                                if error["reference_norm"]
                                else error["max_absolute_error"] == 0
                            )
                            row["components"].append(
                                {
                                    "head": head,
                                    "component": component,
                                    "expected_infinities": int(overflow[:, head : head + 1].sum()),
                                    "finite_subset_error": error,
                                }
                            )
                    report.setdefault("diagnostics", []).append(row)
                    del report["current"]
                    save()
                    continue
                row["errors"] = assess(
                    tuple(t.cpu() for t in actual),
                    cpu_inputs,
                    cpu_dout,
                    causal=causal,
                    scale=scale,
                    directed=name != "random",
                    pointwise=False,
                )
                if args.compare_reference_devices and name == "random":
                    for h in range(16):
                        part = tuple(t[:, h : h + 1] for t in inputs)
                        cpu = fp64_reference(tuple(t.cpu() for t in part), dout[:, h : h + 1].cpu(), causal=causal)
                        gpu = fp64_reference(part, dout[:, h : h + 1], causal=causal)
                        for a, b in zip(cpu, gpu):
                            torch.testing.assert_close(a, b.cpu(), rtol=1e-10, atol=1e-10)
                    row["reference_agreement"] = True
                if name == "random":
                    graph, storage = capture(
                        rtx3090_attention, [t.detach() for t in inputs], dout, causal, "joint", keep_graph=True
                    )
                    leaves, _, _, values = storage
                    with torch.no_grad():
                        for t in (*leaves, dout):
                            t.copy_(torch.randn_like(t))
                        for t in values:
                            t.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize()
                    row["changed_poisoned_graph"] = assess(
                        tuple(t.cpu() for t in values),
                        tuple(t.cpu() for t in leaves),
                        dout.cpu(),
                        causal=causal,
                        scale=None,
                        directed=False,
                        pointwise=False,
                    )
                    bases = [t.detach().clone().requires_grad_(True) for t in inputs]
                    views = [torch.ops.aten._neg_view.default(t) for t in bases]
                    grad = torch.ops.aten._neg_view.default(torch.randn_like(dout))
                    out = rtx3090_attention(*views, is_causal=causal)
                    got = out, *[-g for g in torch.autograd.grad(out, bases, grad)]
                    row["negative_views"] = assess(
                        tuple(t.cpu() for t in got),
                        tuple(t.cpu() for t in views),
                        grad.cpu(),
                        causal=causal,
                        scale=None,
                        directed=False,
                        pointwise=False,
                    )
            except Exception:
                torch.save(
                    {
                        "inputs": [t.detach().cpu() for t in inputs],
                        "dout": dout.cpu(),
                        "actual": [t.cpu() for t in actual],
                        "case": row,
                    },
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
