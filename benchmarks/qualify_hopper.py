"""Qualify a Hopper attention factory on directed inputs and changed graphs."""

import argparse
import hashlib
import importlib
import itertools
import json
from pathlib import Path

import torch

from benchmarks.reference import fp64_reference, metrics, rounded_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", type=int, default=2, help="Use enough work tiles to exercise persistent reuse.")
    parser.add_argument("--check-heads", nargs="+", type=int, help="Explicit independent-reference head subset.")
    parser.add_argument("--modes", nargs="+", choices=("dense", "causal"), help="Restrict a focused ownership probe.")
    parser.add_argument("--candidate", required=True, help="Explicit module:factory with the SDPA calling convention.")
    parser.add_argument("--candidate-kwargs", default="{}")
    parser.add_argument("--lengths", nargs="+", type=int)
    parser.add_argument("--scales", nargs="+", choices=("default", "negative", "zero"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--quick-positive",
        action="store_true",
        help="Check both dtypes and visibility modes with the positive-scale forward enabled.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.heads < 1:
        parser.error("The head count must be positive.")
    checked_heads = list(range(args.heads)) if args.check_heads is None else args.check_heads
    if any(head < 0 or head >= args.heads for head in checked_heads):
        parser.error("Every checked head must be present in the input.")
    from benchmarks.compare_l40s import graph_kernels
    from kohakufa_cute import attention

    candidate_hash = None
    if args.candidate:
        module_name, factory = args.candidate.split(":")
        module = importlib.import_module(module_name)
        candidate_hash = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        function = getattr(module, factory)(**json.loads(args.candidate_kwargs))

        def attention(q, k, v, *, causal, scale):
            return function(q, k, v, is_causal=causal, scale=scale)

    rows = []
    dtypes = (torch.bfloat16,) if args.quick and not args.quick_positive else (torch.bfloat16, torch.float16)
    sequences = (257,) if args.quick or args.quick_positive else (129, 1025)
    if args.lengths:
        if min(args.lengths) < 1:
            parser.error("Lengths must be positive.")
        sequences = tuple(args.lengths)
    visibility = (True,) if args.quick and not args.quick_positive else (False, True)
    if args.modes:
        visibility = tuple(mode == "causal" for mode in args.modes)
    scales = (None,) if args.quick_positive else (-0.25,) if args.quick else (None, -0.25, 0.0)
    if args.scales:
        scale_values = {"default": None, "negative": -0.25, "zero": 0.0}
        scales = tuple(scale_values[name] for name in args.scales)
    for dtype, dimension, sequence, causal, scale in itertools.product(dtypes, (64, 128), sequences, visibility, scales):
        torch.manual_seed(2049)
        inputs = [
            torch.randn(1, args.heads, sequence, dimension, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)
        ]
        dout = torch.randn_like(inputs[0])

        def run():
            out = attention(*inputs, causal=causal, scale=scale)
            return out, *torch.autograd.grad(out, inputs, dout)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, stream=stream):
                actual = run()
        torch.cuda.current_stream().wait_stream(stream)
        errors = []
        for generation in (0, 1):
            if generation:
                with torch.no_grad():
                    for tensor in inputs:
                        tensor.copy_(torch.randn_like(tensor) * 0.3)
                    # A shared negative component tests unscaled statistics.
                    inputs[0][..., 0], inputs[1][..., 0] = 20, -20
                    dout.copy_(torch.randn_like(dout))
            with torch.no_grad():
                for tensor in actual:
                    tensor.fill_(torch.nan)
            graph.replay()
            torch.cuda.synchronize()
            eager = run()
            for got, want in zip(actual, eager):
                if not torch.isfinite(got).all():
                    raise AssertionError("A full output or gradient contains nonfinite values.")
                if args.candidate:
                    # Joint dQ owners can reduce partials in a different
                    # FP32 order across graph/eager scheduling. Independent
                    # per-head rounded-reference checks still follow below.
                    tolerance = 0.02 if dtype == torch.bfloat16 else 0.003
                    torch.testing.assert_close(got, want, atol=tolerance, rtol=tolerance)
                elif not torch.equal(got, want):
                    raise AssertionError("Changed-input graph does not reproduce the same eager kernels.")
            for head in checked_heads:
                operands = tuple(t[:, head : head + 1] for t in inputs)
                grad = dout[:, head : head + 1]
                rounded = rounded_reference(operands, grad, causal=causal, scale=scale, output=actual[0][:, head : head + 1])
                mathematical = fp64_reference(operands, grad, causal=causal, scale=scale)
                error = {
                    "generation": generation,
                    "head": head,
                    "rounded": [metrics(got[:, head : head + 1], want) for got, want in zip(actual, rounded)],
                    "fp64": [metrics(got[:, head : head + 1], want) for got, want in zip(actual, mathematical)],
                }
                for item in error["rounded"]:
                    if not item["finite"] or (
                        item["relative_error"] >= (0.01 if dtype == torch.bfloat16 else 0.005)
                        if item["reference_norm"]
                        else item["max_absolute_error"] != 0
                    ):
                        raise AssertionError(error)
                errors.append(error)
        row = {
            "shape": list(inputs[0].shape),
            "dtype": str(dtype),
            "causal": causal,
            "scale": scale,
            "changed_poisoned_graph": "pass",
            "all_heads_finite": True,
            "independent_reference_heads": checked_heads,
            "graph_kernels": graph_kernels(graph),
            "errors": errors,
        }
        rows.append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "rows": rows,
                    "candidate_kwargs": json.loads(args.candidate_kwargs),
                    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "candidate": args.candidate,
                    "candidate_source_sha256": candidate_hash,
                },
                indent=2,
            )
            + "\n"
        )
        print({key: row[key] for key in ("shape", "dtype", "causal", "scale", "changed_poisoned_graph")}, flush=True)


if __name__ == "__main__":
    main()
