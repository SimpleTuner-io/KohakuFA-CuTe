"""Isolate static Hopper gradient-output tracking with optional NaN allocation.

Forward output is always initialized before its TMA overwrite, separating its
known tracking limitation from dK/dV reads. Initialization is a diagnostic only.
"""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
from unittest.mock import patch

os.environ["CUTE_DSL_KEEP"] = "ptx,cubin"

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, choices=(64, 128), required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), required=True)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--read", choices=("dk", "dv", "all"), required=True)
    parser.add_argument("--initialize-gradients", action="store_true")
    parser.add_argument("--candidate", help="Explicit module:factory for a matched initialization diagnostic.")
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["CUTE_DSL_DUMP_DIR"] = str(args.output / "automatic")
    from benchmarks.compiled_artifacts import retain_code
    from kohakufa_cute import fa4_hopper, kernels

    compiled, allocations, checks = [], {}, {}
    original_compile = kernels.cute.compile

    def compile_keep(program, *arguments, **options):
        function = original_compile(program, *arguments, **options)
        compiled.append(
            {
                "program": type(program).__module__ + "." + type(program).__qualname__,
                "artifacts": retain_code(function, args.output / "code"),
            }
        )
        return function

    def describe(name, tensor):
        allocations[name] = {
            "ptr": tensor.data_ptr(),
            "end": tensor.data_ptr() + tensor.numel() * tensor.element_size(),
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
        }
        print("ALLOCATION", name, json.dumps(allocations[name]), flush=True)

    empty_like = torch.empty_like

    def initialize(*arguments, **options):
        result = empty_like(*arguments, **options)
        result.fill_(torch.nan)
        return result

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.manual_seed(847)
    inputs = [
        torch.randn(1, args.heads, args.length, args.dim, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)
    ]
    dout = torch.randn_like(inputs[0])

    def operation(q, k, v, *, is_causal=False):
        return fa4_hopper.attention(q, k, v, causal=is_causal)

    if args.candidate:
        module, factory = args.candidate.split(":")
        operation = getattr(importlib.import_module(module), factory)()
    else:
        assert fa4_hopper.compatible(*inputs, None, 0.0, args.causal, None, False)
    for name, tensor in zip(("q", "k", "v", "dout"), (*inputs, dout)):
        describe(name, tensor)
    with patch.object(kernels.cute, "compile", compile_keep):
        with patch.object(torch, "empty_like", initialize):
            output = operation(*inputs, is_causal=args.causal)
        describe("out", output)
        assert torch.isfinite(output).all(), "Forward left an initialized NaN unwritten"
        with patch.object(torch, "empty_like", initialize if args.initialize_gradients else empty_like):
            gradients = torch.autograd.grad(output, inputs, dout)
    for name, tensor in zip(("dq", "dk", "dv"), gradients):
        describe(name, tensor)
    for name in ("dq", "dk", "dv") if args.read == "all" else (args.read,):
        tensor = gradients[("dq", "dk", "dv").index(name)]
        checks[name] = {"finite": bool(torch.isfinite(tensor).all()), "sum": tensor.double().sum().item()}
        assert checks[name]["finite"], "Gradient left an initialized NaN unwritten"
        print("READ", name, json.dumps(checks[name]), flush=True)
    report = {
        "scope": "Paired initialization diagnostic; output initialized in every control, gradient initialization only when requested.",
        "configuration": {
            "dimension": args.dim,
            "dtype": args.dtype,
            "causal": args.causal,
            "read": args.read,
            "initialize_gradients": args.initialize_gradients,
            "candidate": args.candidate,
        },
        "allocations": allocations,
        "checks": checks,
        "compiled": compiled,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
