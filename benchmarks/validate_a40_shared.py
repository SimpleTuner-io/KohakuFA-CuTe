"""Directed arithmetic, tails, and changed/poisoned graphs for the shared static A40 schedule."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import torch

from benchmarks.a40_shared_candidates import make_attention
from benchmarks.reference import fp64_reference, metrics, rounded_reference
from benchmarks.validate_sm80_static import statistics
from tests.test_key_precision import key_cancellation
from tests.test_value_precision import square_value_cancellation, value_residual_underflow


def assess(actual, inputs, dout, *, causal, scale, directed, pointwise):
    records = []
    dtype = inputs[0].dtype
    limit = 0.01 if dtype == torch.bfloat16 else 0.005
    coordinate = 0.02 if dtype == torch.bfloat16 else 0.003
    for head in range(inputs[0].shape[1]):
        part = tuple(t[:, head : head + 1] for t in inputs)
        grad = dout[:, head : head + 1]
        mathematical = fp64_reference(part, grad, causal=causal, scale=scale)
        rounded = rounded_reference(part, grad, causal=causal, scale=scale, output=actual[0][:, head : head + 1])
        for component, got, exact, reference in zip(("out", "dq", "dk", "dv"), actual, mathematical, rounded):
            got = got[:, head : head + 1]
            math_error, contract_error = metrics(got, exact), metrics(got, reference)
            required = contract_error if directed else math_error
            assert required["finite"], (head, component, required)
            if required["reference_norm"]:
                assert required["relative_error"] < limit, (head, component, required)
            else:
                assert required["max_absolute_error"] == 0, (head, component, required)
            if pointwise:
                torch.testing.assert_close(got.double(), reference, atol=coordinate, rtol=coordinate)
            records.append({"head": head, "component": component, "fp64": math_error, "rounded": contract_error})
    return records


STATISTICS = []


def run(function, inputs, dout, *, causal=False, scale=None, reference_device=None):
    operands = [t.detach().requires_grad_(True) for t in inputs]
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
        out = function(*operands, is_causal=causal, scale=scale)
    actual_scale = inputs[0].shape[-1] ** -0.5 if scale is None else scale
    if actual_scale > 0:
        assert len(saved) == 6, "The explicit native call did not save its six expected tensors."
        maximum, logsum = saved[-2:]
        device = inputs[0].device if reference_device is None else reference_device
        STATISTICS.append(
            statistics(
                tuple(t.detach().to(device) for t in inputs),
                maximum.detach().to(device),
                logsum.detach().to(device),
                causal,
                actual_scale,
            )
        )
    return out, *torch.autograd.grad(out, operands, dout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--dependencies")
    parser.add_argument("--dimensions", nargs="+", type=int, choices=(64, 128, 256), default=[128])
    parser.add_argument("--head-group", type=int, choices=(1, 4, 16))
    parser.add_argument("--reference-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--smoke", action="store_true", help="Small tail and changed-graph probes for sanitizers.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    function = make_attention(args.source, args.dependencies, direct=True, head_group=args.head_group)
    reference_device = torch.device(args.reference_device)
    if reference_device.type == "cpu":
        torch.set_num_threads(4)

    def run_case(inputs, dout, **kwargs):
        try:
            return run(function, inputs, dout, reference_device=reference_device, **kwargs)
        except Exception:
            torch.save(
                {"inputs": tuple(t.detach().cpu() for t in inputs), "dout": dout.detach().cpu(), "kwargs": kwargs},
                args.output.with_suffix(".failed.pt"),
            )
            raise

    report = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": str(torch.__version__),
        "source_sha256": {
            str(path.name): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__).resolve().parents[1] / "src/kohakufa_cute/fa4_sm80").glob("*.py")
        },
        "candidate_sha256": hashlib.sha256(
            Path(__file__).with_name("a40_shared_candidates.py").read_bytes()
        ).hexdigest(),
        "smoke": args.smoke,
        "dimensions": args.dimensions,
        "head_group_control": args.head_group,
        "reference_device": args.reference_device,
        "host": "native C++ configured runner; full shared joint schedule",
        "statistics_checks": STATISTICS,
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def assess_case(label, inputs, dout, actual, *, causal, scale, directed):
        try:
            return assess(
                tuple(t.detach().to(reference_device) for t in actual),
                tuple(t.detach().to(reference_device) for t in inputs),
                dout.detach().to(reference_device),
                causal=causal,
                scale=scale,
                directed=directed,
                pointwise=directed
                and label in ("rounded-cancellation", "rounded-key-cancellation", "rounded-fp16-underflow"),
            )
        except Exception:
            torch.save(
                {
                    "inputs": tuple(t.detach().cpu() for t in inputs),
                    "dout": dout.detach().cpu(),
                    "actual": tuple(t.detach().cpu() for t in actual),
                    "case": label,
                    "causal": causal,
                    "scale": scale,
                },
                args.output.with_suffix(".failed.pt"),
            )
            raise

    def retain(label, inputs, dout, actual, *, causal, scale=None, directed=False):
        row = {
            "case": label,
            "shape": list(inputs[0].shape),
            "dtype": str(inputs[0].dtype),
            "causal": causal,
            "scale": scale,
            "directed_rounded_gate": directed,
            "errors": assess_case(
                label,
                inputs,
                dout,
                actual,
                causal=causal,
                scale=scale,
                directed=directed,
            ),
        }
        report["rows"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print({key: row[key] for key in ("case", "shape", "dtype", "causal", "scale")}, flush=True)

    matrix = (
        ((torch.bfloat16, args.dimensions[0], 129, True), (torch.float16, args.dimensions[-1], 129, False))
        if args.smoke
        else itertools.product((torch.bfloat16, torch.float16), args.dimensions, (129, 257), (False, True))
    )
    for dtype, dimension, sequence, causal in matrix:
        torch.manual_seed(221)
        q, k, v, dout = [torch.randn(1, 2, sequence, dimension, device="cuda", dtype=dtype) for _ in range(4)]
        inputs = q, k, v
        actual = run_case(inputs, dout, causal=causal)
        retain("random-tail", inputs, dout, actual, causal=causal)
        if not args.smoke:
            q.mul_(0.3)
            k.mul_(0.3)
            q[..., 0], k[..., 0] = 20, -20
            for scale in (dimension**-0.5, -0.125, 0.0):
                actual = run_case(inputs, dout, causal=causal, scale=scale)
                retain("negative-common-offset", inputs, dout, actual, causal=causal, scale=scale, directed=True)
    if not args.smoke:
        for dtype, dimension in itertools.product((torch.bfloat16, torch.float16), args.dimensions):
            inputs, dout, scale, _ = square_value_cancellation(dimension, dtype, "cuda", length=1024, heads=2)
            actual = run_case(inputs, dout, scale=scale)
            retain("rounded-cancellation", inputs, dout, actual, causal=False, scale=scale, directed=True)
        for dtype, dimension in itertools.product((torch.bfloat16, torch.float16), args.dimensions):
            inputs, dout = key_cancellation(dimension, dtype, "cuda", heads=2, square=True)
            actual = run_case(inputs, dout)
            retain("rounded-key-cancellation", inputs, dout, actual, causal=False, directed=True)
        for dtype, dimension in itertools.product((torch.bfloat16, torch.float16), args.dimensions):
            q, k, v, dout = [torch.zeros(1, 2, 129, dimension, device="cuda", dtype=dtype) for _ in range(4)]
            q[..., 0], k[..., 0], k[:, :, 0, 0] = 10, 9 if dtype == torch.float16 else 5, 10
            v[..., 0], v[:, :, 0, 0], dout[..., 0] = -1, 1, 1
            actual = run_case((q, k, v), dout, scale=1.0)
            retain("sharp-saturated", (q, k, v), dout, actual, causal=False, scale=1.0, directed=True)
        for dtype, dimension, heads in itertools.product(
            (torch.bfloat16, torch.float16), args.dimensions, (18, 24, 32)
        ):
            torch.manual_seed(223)
            q, k, v, dout = [torch.randn(1, heads, 257, dimension, device="cuda", dtype=dtype) for _ in range(4)]
            actual = run_case((q, k, v), dout, causal=True)
            retain("model-head-count-tail", (q, k, v), dout, actual, causal=True)
        for dtype, dimension, sequence, causal in itertools.product(
            (torch.bfloat16, torch.float16), args.dimensions, (1025,), (False, True)
        ):
            torch.manual_seed(227)
            q, k, v, dout = [torch.randn(1, 2, sequence, dimension, device="cuda", dtype=dtype) for _ in range(4)]
            actual = run_case((q, k, v), dout, causal=causal)
            retain("long-random-tail", (q, k, v), dout, actual, causal=causal)
        for dimension in args.dimensions:
            inputs, dout, scale, _ = value_residual_underflow(dimension, "cuda", length=1024, heads=2)
            actual = run_case(inputs, dout, scale=scale)
            retain("rounded-fp16-underflow", inputs, dout, actual, causal=False, scale=scale, directed=True)

    from benchmarks.compare_ada import capture

    for dimension, (dtype, causal) in itertools.product(
        args.dimensions, ((torch.bfloat16, True), (torch.float16, False))
    ):
        q, k, v, dout = [torch.randn(1, 2, 129, dimension, device="cuda", dtype=dtype) for _ in range(4)]
        graph, storage = capture(function, (q, k, v), dout, causal, "joint")
        operands, _, _, actual = storage
        for seed in (249, 251):
            torch.manual_seed(seed)
            with torch.no_grad():
                for tensor in (*operands, dout):
                    tensor.copy_(torch.randn_like(tensor))
                for tensor in actual:
                    tensor.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            retain("changed-poisoned-graph", operands, dout, actual, causal=causal)
    report["status"] = "pass"
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
