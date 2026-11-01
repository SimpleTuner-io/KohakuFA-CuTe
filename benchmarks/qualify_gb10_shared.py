"""Qualify explicit GB10 shared schedules against the existing arithmetic gates."""

import argparse
import itertools
import json
from pathlib import Path

import torch

from benchmarks.experimental_gb10_shared import make_attention
from benchmarks.validate_sm80_static import assess, statistics
from tests.test_key_precision import key_cancellation
from tests.test_value_precision import square_value_cancellation, value_residual_underflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=("native", "python"), default="native")
    parser.add_argument("--forward-m", type=int, default=64)
    parser.add_argument("--dout-stages", type=int, default=1)
    parser.add_argument("--query-stages", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dense-m32", action="store_true")
    parser.add_argument("--stage1-shared-v", action="store_true")
    parser.add_argument("--public-selected", action="store_true")
    parser.add_argument("--long-causal-only", action="store_true", help="Exercise only the new H4/S4096 FP16 public route.")
    parser.add_argument("--heads", type=int, choices=(4, 16), default=4)
    parser.add_argument(
        "--cpu-reference",
        action="store_true",
        help="Run unchanged FP64 reference gates on CPU copies with four threads.",
    )
    parser.add_argument(
        "--reference-agreement",
        action="store_true",
        help="Outside sanitizers, check both CUDA and CPU reference gates.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.long_causal_only and (not args.public_selected or args.heads != 4):
        parser.error("--long-causal-only requires --public-selected --heads 4")
    if args.reference_agreement and not args.cpu_reference:
        parser.error("--reference-agreement requires --cpu-reference")
    if args.cpu_reference:
        torch.set_num_threads(4)
    operation = make_attention(
        host=args.host,
        forward_m=args.forward_m,
        dout_stages=args.dout_stages,
        query_stages=args.query_stages,
        dense_m32=args.dense_m32,
        stage1_shared_v=args.stage1_shared_v,
    )
    if args.public_selected:
        from kohakufa_cute.fa4_sm80.gb10 import scaled_dot_product_attention

        operation = scaled_dot_product_attention
    dimensions = (128,) if args.public_selected else (64, 128)
    heads = args.heads if args.public_selected else 2
    report = {"configuration": vars(args) | {"output": args.output.name}, "rows": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def check(actual, operands, dout, *, label, causal, scale, directed, pointwise, maximum=None, logsum=None):
        try:

            def evaluate(values, inputs, gradient, row_maximum, row_logsum):
                errors = assess(values, inputs, gradient, causal=causal, scale=scale, directed=directed, pointwise=pointwise)
                stats = (
                    None
                    if row_maximum is None
                    else statistics(
                        inputs, row_maximum, row_logsum, causal, inputs[0].shape[-1] ** -0.5 if scale is None else scale
                    )
                )
                return errors, stats

            cuda_result = evaluate(actual, operands, dout, maximum, logsum) if args.reference_agreement else None
            if args.cpu_reference:
                values = tuple(t.detach().cpu() for t in actual)
                inputs = tuple(t.detach().cpu() for t in operands)
                gradient = dout.detach().cpu()
                row_maximum = None if maximum is None else maximum.detach().cpu()
                row_logsum = None if logsum is None else logsum.detach().cpu()
            else:
                values, inputs, gradient, row_maximum, row_logsum = actual, operands, dout, maximum, logsum
            errors, stats = evaluate(values, inputs, gradient, row_maximum, row_logsum)
            if cuda_result is not None:
                differences = [
                    abs((cpu[policy][metric] or 0) - (gpu[policy][metric] or 0))
                    for cpu, gpu in zip(errors, cuda_result[0])
                    for policy in ("fp64", "rounded")
                    for metric in ("relative_error", "max_absolute_error")
                ]
                report.setdefault("reference_agreement", []).append(
                    {
                        "case": label,
                        "both_gates_pass": True,
                        "components": len(errors),
                        "maximum_error_metric_difference": max(differences),
                        "cpu_statistics": stats,
                        "cuda_statistics": cuda_result[1],
                    }
                )
            return errors, stats
        except Exception as error:
            failure = args.output.with_suffix(".failure.pt")
            torch.save(
                {
                    "case": label,
                    "inputs": operands,
                    "dout": dout,
                    "actual": actual,
                    "maximum": maximum,
                    "logsum": logsum,
                    "causal": causal,
                    "scale": scale,
                    "configuration": report["configuration"],
                },
                failure,
            )
            report.update(status="failed", failure={"case": label, "error": str(error), "operands": failure.name})
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            raise

    def retain(label, inputs, dout, *, causal=False, scale=None, graph=False):
        operands = tuple(t.detach().requires_grad_(True) for t in inputs)

        def execute():
            if args.public_selected:
                from kohakufa_cute.fa4_sm80.gb10 import compatible

                assert compatible(*operands, causal=causal, scale=scale), "Public test did not select the qualified family."
            saved = []

            def pack(tensor):
                saved.append(tensor)
                return tensor

            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                out = operation(*operands, is_causal=causal, scale=scale)
            assert len(saved) == 6, "Expected Q/K/V/O/raw maximum/log normalizer."
            maximum, logsum = saved[-2:]
            actual = out, *torch.autograd.grad(out, operands, dout)
            return actual, maximum, logsum

        if graph:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    execute()
                captured = torch.cuda.CUDAGraph()
                with torch.cuda.graph(captured, stream=stream):
                    actual, maximum, logsum = execute()
            torch.cuda.current_stream().wait_stream(stream)
            for seed in (249, 251):
                torch.manual_seed(seed)
                with torch.no_grad():
                    for tensor in (*operands, dout):
                        tensor.copy_(torch.randn_like(tensor))
                    for tensor in (*actual, maximum, logsum):
                        tensor.fill_(float("nan"))
                captured.replay()
                torch.cuda.synchronize()
                errors, stats = check(
                    actual,
                    operands,
                    dout,
                    label=label,
                    causal=causal,
                    scale=scale,
                    directed=False,
                    pointwise=False,
                    maximum=maximum,
                    logsum=logsum,
                )
                report["rows"].append({"case": label, "seed": seed, "errors": errors, "statistics": stats})
        else:
            actual, maximum, logsum = execute()
            directed = label.startswith("rounded") or label in ("negative-common-offset", "retained-common-offset")
            errors, stats = check(
                actual,
                operands,
                dout,
                label=label,
                causal=causal,
                scale=scale,
                directed=directed,
                pointwise=label.startswith("rounded"),
                maximum=maximum,
                logsum=logsum,
            )
            report["rows"].append(
                {
                    "case": label,
                    "shape": list(operands[0].shape),
                    "dtype": str(operands[0].dtype),
                    "causal": causal,
                    "scale": scale,
                    "errors": errors,
                    "statistics": stats,
                }
            )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(label, tuple(operands[0].shape), str(operands[0].dtype), causal, "pass", flush=True)

    if args.long_causal_only:
        torch.manual_seed(1709)
        inputs = tuple(torch.randn(1, 4, 4096, 128, device="cuda", dtype=torch.float16) for _ in range(3))
        dout = torch.randn_like(inputs[0])
        retain("random-public", inputs, dout, causal=True)
        inputs[0].mul_(0.3)
        inputs[1].mul_(0.3)
        inputs[0][..., 0], inputs[1][..., 0] = 20, -20
        retain("negative-common-offset", inputs, dout, causal=True)
        inputs, dout, scale, _ = square_value_cancellation(128, torch.float16, "cuda", length=4096, heads=4)
        retain("rounded-value-cancellation", inputs, dout, causal=True, scale=scale)
        inputs, dout = key_cancellation(128, torch.float16, "cuda", heads=4, square=True)
        retain("rounded-key-cancellation", inputs, dout, causal=True)
        inputs, dout, scale, _ = value_residual_underflow(128, "cuda", length=4096, heads=4)
        # Causal prefixes amplify the fixture's large dO into dV > 400,000.
        # An exact power-of-two rescaling keeps the same rounded probabilities
        # and underflow while making the final FP16 gradient representable.
        dout.mul_(0.125)
        report["underflow_fixture"] = {"dout_scale": 0.125, "purpose": "Representable causal FP16 output gradient"}
        retain("rounded-fp16-underflow", inputs, dout, causal=True, scale=scale)
        inputs = tuple(torch.randn(1, 4, 4096, 128, device="cuda", dtype=torch.float16) for _ in range(3))
        retain("changed-poisoned-graph", inputs, torch.randn_like(inputs[0]), causal=True, graph=True)
        report["status"] = "pass"
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return

    matrix = itertools.product(
        (torch.bfloat16, torch.float16),
        dimensions,
        (
            ((512,) if args.smoke else (512, 1024, 4096))
            if args.public_selected
            else ((65,) if args.smoke else (65, 129, 257))
        ),
        (False, True),
    )
    for dtype, dimension, sequence, causal in matrix:
        if args.public_selected and causal and sequence == 1024:
            continue
        torch.manual_seed(221)
        inputs = tuple(torch.randn(1, heads, sequence, dimension, device="cuda", dtype=dtype) for _ in range(3))
        dout = torch.randn_like(inputs[0])
        retain("random-public" if args.public_selected else "random-tail", inputs, dout, causal=causal)
        if not args.smoke or args.reference_agreement:
            inputs[0].mul_(0.3)
            inputs[1].mul_(0.3)
            inputs[0][..., 0], inputs[1][..., 0] = 20, -20
            retain("negative-common-offset", inputs, dout, causal=causal)
    if not args.smoke and not args.public_selected:
        # Replay the prior GB10 diagnostic exactly: the old script applied both
        # mathematical and rounded gates and reported 1.647329843% FP64 error.
        torch.manual_seed(1212041)
        inputs = tuple(torch.randn(1, 4, 129, 64, device="cuda", dtype=torch.bfloat16) * 0.5 for _ in range(3))
        dout = torch.randn_like(inputs[0])
        for tensor in inputs:
            tensor.copy_(torch.randn_like(tensor) * 0.5)
        inputs[0].fill_(1)
        inputs[1].add_(-3)
        torch.save(
            {
                "inputs": inputs,
                "dout": dout,
                "scale": 0.125,
                "fixture_seed": 1212041,
                "prior_fp64_relative_error": 0.016473298431499483,
            },
            args.output.with_suffix(".retained-offset.pt"),
        )
        retain("retained-common-offset", inputs, dout, scale=0.125)
    for dtype, dimension in itertools.product((torch.bfloat16, torch.float16), dimensions):
        inputs, dout, scale, _ = square_value_cancellation(
            dimension, dtype, "cuda", length=128 if args.smoke and not args.public_selected else 1024, heads=heads
        )
        retain("rounded-value-cancellation", inputs, dout, scale=scale)
        if not args.smoke or args.reference_agreement:
            inputs, dout = key_cancellation(dimension, dtype, "cuda", heads=heads, square=True)
            retain("rounded-key-cancellation", inputs, dout)
    for dimension in dimensions:
        inputs, dout, scale, _ = value_residual_underflow(
            dimension, "cuda", length=128 if args.smoke and not args.public_selected else 1024, heads=heads
        )
        retain("rounded-fp16-underflow", inputs, dout, scale=scale)
    for dtype, causal in ((torch.bfloat16, True), (torch.float16, False)):
        length = (512 if causal else 1024) if args.public_selected else 129
        inputs = tuple(torch.randn(1, heads, length, 128, device="cuda", dtype=dtype) for _ in range(3))
        retain("changed-poisoned-graph", inputs, torch.randn_like(inputs[0]), causal=causal, graph=True)
    report["status"] = "pass"
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
