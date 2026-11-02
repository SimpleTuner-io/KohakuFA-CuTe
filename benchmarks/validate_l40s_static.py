"""Directed arithmetic, tails, and changed/poisoned graphs for the shared static L40S schedule."""

import argparse
import hashlib
import inspect
import itertools
import json
from pathlib import Path

import torch

from benchmarks.l40s_schedules import make_attention
from benchmarks.reference import fp64_reference, metrics, rounded_reference
from benchmarks.validate_sm80_static import statistics
from tests.test_key_precision import key_cancellation
from tests.test_value_precision import square_value_cancellation, value_residual_underflow

CPU_REFERENCE = False


def assess(actual, inputs, dout, *, causal, scale, directed, pointwise):
    if CPU_REFERENCE:
        actual = tuple(t.detach().cpu() for t in actual)
        inputs = tuple(t.detach().cpu() for t in inputs)
        dout = dout.detach().cpu()
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


def run(function, inputs, dout, *, causal=False, scale=None):
    operands = [t.detach().requires_grad_(True) for t in inputs]
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda tensor: saved.append(tensor) or tensor, lambda t: t):
        out = function(*operands, is_causal=causal, scale=scale)
    actual_scale = inputs[0].shape[-1] ** -0.5 if scale is None else scale
    if actual_scale > 0:
        assert len(saved) == 6, "The explicit call did not preserve the six shared attention buffers."
        maximum, logsum = saved[-2:]
        STATISTICS.append(
            statistics(
                tuple(t.detach().cpu() for t in inputs) if CPU_REFERENCE else inputs,
                maximum.detach().cpu() if CPU_REFERENCE else maximum,
                logsum.detach().cpu() if CPU_REFERENCE else logsum,
                causal,
                actual_scale,
            )
        )
    return out, *torch.autograd.grad(out, operands, dout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu-reference",
        action="store_true",
        help="Run identical reference equations on CPU while instrumenting every CUDA operation.",
    )
    parser.add_argument("--candidate-kwargs", type=json.loads, default={"native": True})
    parser.add_argument("--smoke", action="store_true", help="Small tail and changed-graph probes for sanitizers.")
    parser.add_argument("--causal-smoke", action="store_true", help="Instrument both causal widths and dtypes.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    global CPU_REFERENCE
    CPU_REFERENCE = args.cpu_reference
    function = make_attention(**args.candidate_kwargs)
    report = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": str(torch.__version__),
        "source_sha256": {
            str(path.name): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__).resolve().parents[1] / "src/kohakufa_cute/fa4_sm80").glob("*.py")
        },
        "schedule_sha256": hashlib.sha256(Path(__file__).with_name("l40s_schedules.py").read_bytes()).hexdigest(),
        "smoke": args.smoke,
        "reference_device": "cpu" if CPU_REFERENCE else "cuda",
        "candidate_kwargs": args.candidate_kwargs,
        "candidate_factory": make_attention.__module__ + ":" + make_attention.__name__,
        "candidate_factory_sha256": hashlib.sha256(Path(inspect.getfile(make_attention)).read_bytes()).hexdigest(),
        "reference_cpu_threads": torch.get_num_threads() if CPU_REFERENCE else None,
        "causal_smoke": args.causal_smoke,
        "statistics_checks": STATISTICS,
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def retain(label, inputs, dout, actual, *, causal, scale=None, directed=False):
        row = {
            "case": label,
            "shape": list(inputs[0].shape),
            "dtype": str(inputs[0].dtype),
            "causal": causal,
            "scale": scale,
            "directed_rounded_gate": directed,
            "errors": assess(
                actual,
                inputs,
                dout,
                causal=causal,
                scale=scale,
                directed=directed,
                pointwise=directed
                and label in ("rounded-cancellation", "rounded-key-cancellation", "rounded-fp16-underflow"),
            ),
        }
        report["rows"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print({key: row[key] for key in ("case", "shape", "dtype", "causal", "scale")}, flush=True)

    matrix = (
        itertools.product((torch.bfloat16, torch.float16), (64, 128), (129,), (True,))
        if args.smoke and args.causal_smoke
        else (
            ((torch.bfloat16, 64, 129, True), (torch.float16, 128, 129, False))
            if args.smoke
            else itertools.product((torch.bfloat16, torch.float16), (64, 128), (129, 257), (False, True))
        )
    )
    for dtype, dimension, sequence, causal in matrix:
        torch.manual_seed(221)
        q, k, v, dout = [torch.randn(1, 2, sequence, dimension, device="cuda", dtype=dtype) for _ in range(4)]
        inputs = q, k, v
        actual = run(function, inputs, dout, causal=causal)
        retain("random-tail", inputs, dout, actual, causal=causal)
        if not args.smoke:
            q.mul_(0.3)
            k.mul_(0.3)
            q[..., 0], k[..., 0] = 20, -20
            for scale in (dimension**-0.5, -0.125, 0.0):
                actual = run(function, inputs, dout, causal=causal, scale=scale)
                retain("negative-common-offset", inputs, dout, actual, causal=causal, scale=scale, directed=True)
    if not args.smoke:
        for dtype, dimension in itertools.product((torch.bfloat16, torch.float16), (64, 128)):
            inputs, dout, scale, _ = square_value_cancellation(dimension, dtype, "cuda", length=1024, heads=2)
            actual = run(function, inputs, dout, scale=scale)
            retain("rounded-cancellation", inputs, dout, actual, causal=False, scale=scale, directed=True)
        for dtype, dimension in itertools.product((torch.bfloat16, torch.float16), (64, 128)):
            inputs, dout = key_cancellation(dimension, dtype, "cuda", heads=2, square=True)
            actual = run(function, inputs, dout)
            retain("rounded-key-cancellation", inputs, dout, actual, causal=False, directed=True)
        for dtype, dimension in itertools.product((torch.bfloat16, torch.float16), (64, 128)):
            q, k, v, dout = [torch.zeros(1, 2, 129, dimension, device="cuda", dtype=dtype) for _ in range(4)]
            q[..., 0], k[..., 0], k[:, :, 0, 0] = 10, 9 if dtype == torch.float16 else 5, 10
            v[..., 0], v[:, :, 0, 0], dout[..., 0] = -1, 1, 1
            actual = run(function, (q, k, v), dout, scale=1.0)
            retain("sharp-saturated", (q, k, v), dout, actual, causal=False, scale=1.0, directed=True)
        for dtype, (heads, dimension) in itertools.product(
            (torch.bfloat16, torch.float16), ((18, 64), (24, 128), (32, 128))
        ):
            torch.manual_seed(223)
            q, k, v, dout = [torch.randn(1, heads, 257, dimension, device="cuda", dtype=dtype) for _ in range(4)]
            actual = run(function, (q, k, v), dout, causal=True)
            retain("model-head-count-tail", (q, k, v), dout, actual, causal=True)
        for dtype, dimension, sequence, causal in itertools.product(
            (torch.bfloat16, torch.float16), (64, 128), (1025,), (False, True)
        ):
            torch.manual_seed(227)
            q, k, v, dout = [torch.randn(1, 2, sequence, dimension, device="cuda", dtype=dtype) for _ in range(4)]
            actual = run(function, (q, k, v), dout, causal=causal)
            retain("long-random-tail", (q, k, v), dout, actual, causal=causal)
        for dimension in (64, 128):
            inputs, dout, scale, _ = value_residual_underflow(dimension, "cuda", length=1024, heads=2)
            actual = run(function, inputs, dout, scale=scale)
            retain("rounded-fp16-underflow", inputs, dout, actual, causal=False, scale=scale, directed=True)

    from benchmarks.compare_ada import capture

    graph_cases = ((torch.bfloat16, True), (torch.float16, args.causal_smoke))
    for dtype, causal in graph_cases:
        q, k, v, dout = [torch.randn(1, 2, 129, 128, device="cuda", dtype=dtype) for _ in range(4)]
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
