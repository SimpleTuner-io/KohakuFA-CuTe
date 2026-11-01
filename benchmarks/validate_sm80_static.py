"""Qualify the explicit shifted SM80 forward and its joint backward schedule."""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import torch

from benchmarks.reference import fp64_reference, metrics, rounded_reference
from kohakufa_cute import api, fa4_sm80
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


def statistics(inputs, maximum, logsum, causal, scale):
    """Check that saved buffers contain literal max and shifted log2 norm.

    These are diagnostic FP32-statistic checks, separate from the established
    output/gradient norm gates. The absolute allowance covers FP32 dot-product
    reduction differences, including the directed common-offset inputs.
    """
    q, k, _ = inputs
    scores = q.double() @ k.double().transpose(-2, -1)
    if causal:
        visible = torch.ones(scores.shape[-2:], device=q.device, dtype=torch.bool).tril()
        scores.masked_fill_(~visible, -torch.inf)
    expected_maximum = scores.amax(-1)
    expected_logsum = torch.logsumexp((scores - expected_maximum[..., None]) * scale, -1) / math.log(2)
    torch.testing.assert_close(maximum.double(), expected_maximum, atol=2e-4, rtol=1e-6)
    torch.testing.assert_close(logsum.double(), expected_logsum, atol=2e-4, rtol=1e-6)
    return {
        "maximum_absolute_error": (maximum.double() - expected_maximum).abs().max().item(),
        "logsum_absolute_error": (logsum.double() - expected_logsum).abs().max().item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--direct", action="store_true", help="Exercise the static kernel outside automatic performance bounds."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not fa4_sm80.available():
        raise RuntimeError("The optional static SM80 dependency is unavailable.")
    function = api.scaled_dot_product_attention
    if args.direct:

        def function(q, k, v, *, is_causal=False, scale=None):
            if scale is not None and scale <= 0:
                return api.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale)
            return fa4_sm80.attention(q, k, v, causal=is_causal, scale=scale)

    report = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": str(torch.__version__),
        "source_sha256": {
            str(path.name): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__).resolve().parents[1] / "src/kohakufa_cute/fa4_sm80").glob("*.py")
        },
        "smoke": args.smoke,
        "direct_static": args.direct,
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def retain(label, inputs, dout, *, causal=False, scale=None):
        operands = [t.detach().requires_grad_(True) for t in inputs]
        out = function(*operands, is_causal=causal, scale=scale)
        actual_scale = inputs[0].shape[-1] ** -0.5 if scale is None else scale
        selected = hasattr(out.grad_fn, "schedule")
        if args.direct and actual_scale > 0:
            assert hasattr(out.grad_fn, "schedule"), "The selected public call did not execute the static schedule."
        if selected:
            maximum, logsum = out.grad_fn.saved_tensors[-2:]
        actual = out, *torch.autograd.grad(out, operands, dout)
        directed = label.startswith("rounded") or label == "negative-common-offset"
        row = {
            "case": label,
            "shape": list(inputs[0].shape),
            "dtype": str(inputs[0].dtype),
            "causal": causal,
            "scale": scale,
            "selected_static": selected,
            "errors": assess(
                actual,
                inputs,
                dout,
                causal=causal,
                scale=scale,
                directed=directed,
                pointwise=label in ("rounded-cancellation", "rounded-fp16-underflow"),
            ),
        }
        actual_scale = inputs[0].shape[-1] ** -0.5 if scale is None else scale
        if selected:
            row["statistics"] = statistics(inputs, maximum, logsum, causal, actual_scale)
        else:
            row["statistics"] = "native CuTe fallback; arithmetic checked"
        report["rows"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print({key: row[key] for key in ("case", "shape", "dtype", "causal", "scale", "statistics")}, flush=True)

    matrix = (
        ((torch.bfloat16, 64, 129, True), (torch.float16, 128, 129, False))
        if args.smoke
        else itertools.product((torch.bfloat16, torch.float16), (64, 128), (129, 257), (False, True))
    )
    for dtype, dimension, sequence, causal in matrix:
        torch.manual_seed(221)
        q, k, v, dout = [torch.randn(1, 2, sequence, dimension, device="cuda", dtype=dtype) for _ in range(4)]
        inputs = q, k, v
        retain("random-tail", inputs, dout, causal=causal)
        if not args.smoke:
            q.mul_(0.3)
            k.mul_(0.3)
            q[..., 0], k[..., 0] = 20, -20
            for scale in (dimension**-0.5, -0.125, 0.0):
                retain("negative-common-offset", inputs, dout, causal=causal, scale=scale)
    if not args.smoke:
        for dtype, dimension in itertools.product((torch.bfloat16, torch.float16), (64, 128)):
            inputs, dout, scale, _ = square_value_cancellation(dimension, dtype, "cuda", length=1024, heads=2)
            retain("rounded-cancellation", inputs, dout, scale=scale)
        for dimension in (64, 128):
            inputs, dout, scale, _ = value_residual_underflow(dimension, "cuda", length=1024, heads=2)
            retain("rounded-fp16-underflow", inputs, dout, scale=scale)

    from benchmarks.compare_ada import capture

    for dtype, causal in ((torch.bfloat16, True), (torch.float16, False)):
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
            report["rows"].append(
                {
                    "case": "changed-poisoned-graph",
                    "dtype": str(dtype),
                    "causal": causal,
                    "errors": assess(actual, operands, dout, causal=causal, scale=None, directed=False, pointwise=False),
                }
            )
    report["status"] = "pass"
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
