"""Qualify the actual explicit GB10 forward-only API and its mode boundaries."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Small actual-API cases for sanitizer ownership checks.")
    args = parser.parse_args()
    from benchmarks.compare_l40s import graph_kernels
    from benchmarks.warp_inference import assess_output, references
    from kohakufa_cute import api
    from kohakufa_cute.fa4_sm80 import gb10

    assert gb10._qualified_device(torch.cuda.current_device())
    torch.set_num_threads(4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    operands = args.output.parent / (args.output.stem + "-operands")
    operands.mkdir(exist_ok=False)
    report = {
        "scope": "Actual explicit GB10 API; CPU FP64 output gates 1% BF16 / 0.5% FP16, changed poisoned graphs and live dispatch transitions; unchanged native inference arithmetic",
        "rows": [],
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    for dtype in (torch.bfloat16, torch.float16):
        for length in (128, 1024) if args.smoke else (128, 512, 1024):
            for causal in (False, True):
                torch.manual_seed(39251 + length)
                inputs = [torch.randn(1, 16, length, 128, device="cuda", dtype=dtype) for _ in range(3)]
                inputs[0].add_(3)
                inputs[1].sub_(3)
                name = f"{dtype}-s{length}-c{int(causal)}"
                torch.save({"inputs": [t.cpu() for t in inputs], "causal": causal}, operands / (name + ".pt"))
                with torch.no_grad():
                    assert gb10.inference_compatible(*inputs, causal=causal) == (
                        not (length == 512 and not causal and dtype == torch.bfloat16)
                    )
                    output = gb10.scaled_dot_product_attention(*inputs, is_causal=causal)
                    eager = assess_output(output.cpu(), references(tuple(t.cpu() for t in inputs), causal))
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            gb10.scaled_dot_product_attention(*inputs, is_causal=causal)
                        graph = torch.cuda.CUDAGraph(keep_graph=True)
                        with torch.cuda.graph(graph, stream=stream):
                            output = gb10.scaled_dot_product_attention(*inputs, is_causal=causal)
                    torch.cuda.current_stream().wait_stream(stream)
                    for t in inputs:
                        t.copy_(torch.randn_like(t))
                    output.fill_(float("nan"))
                    torch.save({"inputs": [t.cpu() for t in inputs]}, operands / (name + "-graph.pt"))
                    graph.replay()
                    torch.cuda.synchronize()
                    changed = assess_output(output.cpu(), references(tuple(t.cpu() for t in inputs), causal))
                report["rows"].append(
                    {"case": name, "eager": eager, "changed_poisoned_graph": changed, "graph": graph_kernels(graph)}
                )
                save()

    # Reuse populated mode/shape caches while changing autograd state and flags.
    transitions = []
    for length in (128, 512):
        inputs = [torch.randn(1, 16, length, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
        for mode in ("training", "no_grad", "inference_mode", "training_again"):
            context = (
                torch.no_grad()
                if mode == "no_grad"
                else torch.inference_mode() if mode == "inference_mode" else torch.enable_grad()
            )
            with context:
                output = gb10.scaled_dot_product_attention(*inputs)
                if "training" in mode:
                    assert not gb10.inference_compatible(*inputs)
                    assert output.requires_grad
                    gradient = torch.autograd.grad(output, inputs, torch.ones_like(output))
                    assert all(torch.isfinite(g).all() for g in gradient)
                else:
                    assert gb10.inference_compatible(*inputs) == (length != 512)
                    assert not output.requires_grad
                assess_output(output.detach().cpu(), references(tuple(t.detach().cpu() for t in inputs)))
            transitions.append({"sequence": length, "mode": mode, "status": "pass"})
        for position in range(3):
            values = [t.detach() for t in inputs]
            values[position].requires_grad_(True)
            assert not gb10.inference_compatible(*values)
            output = gb10.scaled_dot_product_attention(*values)
            assert output.requires_grad
            assert torch.isfinite(torch.autograd.grad(output, values[position], torch.ones_like(output))[0]).all()
    report["transitions"] = transitions
    save()

    fallback = []
    for case in ("tail", "negative_view", "negative", "zero", "mask", "hidden", "strided", "gqa", "heads", "dimension"):
        length = 129 if case == "tail" else 128
        shape = (1, 4 if case == "heads" else 16, length, 512 if case == "dimension" else 128)
        inputs = [torch.randn(*shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        if case == "negative_view":
            inputs[0] = inputs[0]._neg_view()
        if case == "strided":
            inputs[0] = torch.randn(*shape[:-1], shape[-1] * 2, device="cuda", dtype=inputs[0].dtype)[..., ::2]
        if case == "gqa":
            inputs[1:] = [t[:, :8].contiguous() for t in inputs[1:]]
        mask = torch.ones(length, length, device="cuda", dtype=torch.bool) if case in ("mask", "hidden") else None
        if mask is not None:
            mask[-3:] = False
            if case == "hidden":
                mask.zero_()
        scale = {"negative": -0.125, "zero": 0.0}.get(case)
        options = {"attn_mask": mask, "scale": scale, "enable_gqa": case == "gqa"}
        with (
            torch.no_grad(),
            patch.object(gb10, "native_runner", side_effect=AssertionError("Unexpected selected runner")),
        ):
            output = gb10.scaled_dot_product_attention(*inputs, **options)
        from benchmarks.reference import fp64_reference, metrics

        expected = fp64_reference(
            tuple(t.cpu() for t in inputs),
            torch.zeros_like(inputs[0]).cpu(),
            mask=None if mask is None else mask.cpu(),
            scale=scale,
        )[0]
        errors = assess_output(output.cpu(), [expected[:, h : h + 1] for h in range(shape[1])])
        fallback.append({"case": case, "errors": errors, "status": "pass"})
    report["fallback"] = fallback

    # Compiled execution must bypass the explicit selector before reading tensor metadata.
    compiled_rows = []
    compiled = torch.compile(gb10.scaled_dot_product_attention, backend="eager", fullgraph=True)
    for dtype in (torch.bfloat16, torch.float16):
        values = [torch.randn(1, 16, 128, 128, device="cuda", dtype=dtype) for _ in range(3)]
        with (
            torch.no_grad(),
            patch.object(gb10, "native_runner", side_effect=AssertionError("Compiled selected runner")),
        ):
            output = compiled(*values)
        error = assess_output(output.cpu(), references(tuple(t.cpu() for t in values)))
        compiled_rows.append({"dtype": str(dtype), "output": error, "status": "pass"})
    report["compiled_fallback"] = compiled_rows

    # Pending negation uses the previously qualified canonicalizing training runner.
    with torch.no_grad():
        values = [torch.randn(1, 16, 512, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        values[0] = values[0]._neg_view()
        assert not gb10.inference_compatible(*values)
        assert gb10.compatible(*values)
        output = gb10.scaled_dot_product_attention(*values)
        report["negative_view_training_family"] = assess_output(output.cpu(), references(tuple(t.cpu() for t in values)))

    inputs = [torch.randn(1, 16, 128, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(inputs[0], torch.ones_like(inputs[0]))
        try:
            gb10.scaled_dot_product_attention(dual, *inputs[1:])
        except (RuntimeError, ValueError) as error:
            assert "forward-AD" in str(error), str(error)
            report["forward_ad"] = "explicitly_rejected"
        else:
            raise AssertionError("The forward-only API silently discarded a forward-AD tangent")
    report["status"] = "pass"
    save()
    print("GB10 public inference qualification pass", len(report["rows"]), flush=True)


if __name__ == "__main__":
    main()
