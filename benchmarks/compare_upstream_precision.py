"""Compare directed rounded-operand cases with the original Gluon implementation."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch

from benchmarks.adversarial import cases, make_case
from benchmarks.reference import fp64_reference, metrics, rounded_reference
from tests.test_key_precision import key_cancellation
from tests.test_value_precision import square_value_cancellation, value_residual_underflow


def fixture(case):
    if case.startswith("tracker-"):
        _, dtype_name, name = case.split("-", 2)
        source = next(x for x in cases(extended=False) if x.name == name)
        dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
        inputs, dout, mask = make_case(source, device="cuda", dtype=dtype)
        assert mask is None
        # The original TMA implementation needs at least D16. Zero embedding
        # preserves the reported D8 computation with its original scale.
        if source.dim < 16:
            inputs = tuple(torch.nn.functional.pad(x, (0, 16 - source.dim)) for x in inputs)
            dout = torch.nn.functional.pad(dout, (0, 16 - source.dim))
        return inputs, dout, source.scale if source.scale is not None else source.dim**-0.5, source.causal, 0
    if case.endswith("negative-max"):
        dtype = torch.bfloat16 if case.startswith("bf16") else torch.float16
        sequence, dimension, seed, head = (128, 128, 1207, 9) if dtype == torch.bfloat16 else (1025, 16, 1193, 6)
        torch.manual_seed(seed)
        q, k, v, dout = [torch.randn(1, 16, sequence, dimension, device="cuda", dtype=dtype) for _ in range(4)]
        q.mul_(0.3)
        k.mul_(0.3)
        q[..., 0], k[..., 0] = 20, -20
        return (q, k, v), dout, dimension**-0.5, True, head
    if case == "bf16-key":
        inputs, dout = key_cancellation(256, torch.bfloat16, "cuda", square=True)
        return inputs, dout, None, False, 0
    if case == "fp16-underflow":
        inputs, dout, scale, _ = value_residual_underflow(128, "cuda", length=1024)
    else:
        dtype = torch.bfloat16 if case == "bf16-value" else torch.float16
        inputs, dout, scale, _ = square_value_cancellation(128, dtype, "cuda", length=1024)
    return inputs, dout, scale, False, 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-checkout", type=Path, required=True)
    parser.add_argument("--expected-revision", default="0c5ede6859307bedc0c6089523c16b1ca51b7da9")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import kohakufa
    import triton

    from kohakufa_cute import attention

    checkout = args.upstream_checkout.resolve()
    if not Path(kohakufa.__file__).resolve().is_relative_to(checkout):
        raise RuntimeError("The imported upstream package is not from the selected checkout.")
    revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if revision != args.expected_revision:
        raise RuntimeError("The upstream revision does not match the requested control.")
    report = {
        "upstream_revision": revision,
        "upstream_backward_sha256": hashlib.sha256((checkout / "src/kohakufa/sm100/bwd.py").read_bytes()).hexdigest(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "triton": triton.__version__,
        "scope": "Directed arithmetic and selected-kernel evidence; profiling durations are not performance measurements.",
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected = ["bf16-key", "bf16-value", "fp16-value", "fp16-underflow", "bf16-negative-max", "fp16-negative-max"]
    selected += [f"tracker-{dtype}-{case.name}" for dtype in ("bf16", "fp16") for case in cases(extended=False)]
    for case in selected:
        inputs, dout, scale, causal, head = fixture(case)
        row = {
            "case": case,
            "shape": list(inputs[0].shape),
            "causal": causal,
            "evaluated_head": head,
            "logical_dimension": 8 if case.endswith("issue160") else inputs[0].shape[-1],
            "input_strides": [list(x.stride()) for x in inputs],
            "backends": {},
        }
        for name, function in (("cute", attention), ("upstream", kohakufa.attention)):
            operands = [x.detach().requires_grad_() for x in inputs]
            out = function(*operands, scale=scale, causal=causal)
            actual = (out, *torch.autograd.grad(out, operands, dout))
            one = tuple(x[:, head : head + 1] for x in inputs)
            grad = dout[:, head : head + 1]
            mathematical = fp64_reference(one, grad, scale=scale, causal=causal)
            declared = rounded_reference(one, grad, scale=scale, causal=causal, output=out[:, head : head + 1])
            result = {
                "fp64": [metrics(x[:, head : head + 1], y) for x, y in zip(actual, mathematical)],
                "rounded": [metrics(x[:, head : head + 1], y) for x, y in zip(actual, declared)],
            }
            for component, index in (("dk", 2), ("dv", 3)):
                result[component + "_coordinate"] = actual[index][0, head, 0, 0].item()
                result["mathematical_" + component] = mathematical[index][0, 0, 0, 0].item()
                result["rounded_" + component] = declared[index][0, 0, 0, 0].item()
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            ) as profile:
                observed = function(*operands, scale=scale, causal=causal)
                torch.autograd.grad(observed, operands, dout)
                torch.cuda.synchronize()
            result["launched_kernels"] = [
                event.name for event in profile.events() if event.device_type == torch.autograd.DeviceType.CUDA
            ]
            if not result["launched_kernels"]:
                raise RuntimeError("The profiler did not observe the selected implementation.")
            row["backends"][name] = result
        report["rows"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print({"case": case, "evaluated_head": head}, flush=True)


if __name__ == "__main__":
    main()
