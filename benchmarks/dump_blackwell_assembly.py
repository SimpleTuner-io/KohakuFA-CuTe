"""Capture validated CuTe/Gluon PTX, cubins and SASS for one production shape."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch


def register_reconfiguration(ptx, sass):
    """Compare requested register redistribution with the executed instruction set.

    PTX requests can disappear during assembly (for example, ptxas C7508).
    Counts need not match after optimization; a missing SASS instruction calls
    for inspecting compiler diagnostics rather than assuming a budget took effect.
    """
    requested = len(re.findall(r"\bsetmaxnreg\.(?:inc|dec)\.sync\.aligned\.u32\b", ptx))
    emitted = len(re.findall(r"\bUSETMAXREG(?:\.|\s)", sass))
    return {
        "ptx_request_sites": requested,
        "sass_instruction_sites": emitted,
        "requests_without_sass_instruction": bool(requested and not emitted),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["cute", "gluon"], required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--kohakufa-tuning", type=Path, nargs="+")
    parser.add_argument("--cuobjdump", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.backend == "gluon" and (args.baseline_source or not args.kohakufa_tuning):
        parser.error("Gluon requires tuning files and does not accept a CuTe baseline source.")
    args.output.mkdir(parents=True, exist_ok=True)
    dump = args.output / "cute-dumps"
    # The compiler can return artifact paths without writing them if the dump
    # directory does not exist when the first module is compiled.
    dump.mkdir(exist_ok=True)
    os.environ["CUTE_DSL_KEEP"] = "ptx,cubin"
    os.environ["CUTE_DSL_DUMP_DIR"] = str(dump.resolve())
    os.environ["CUTE_DSL_DISABLE_FILE_CACHING"] = "1"
    os.environ["TRITON_CACHE_DIR"] = str((args.output / "triton-cache").resolve())

    import torch
    import triton

    from benchmarks.compare_blackwell_codegen import baseline_factory, load_source
    from benchmarks.precision import evaluate, load_upstream_tuning, upstream_configurations, upstream_provenance
    from kohakufa_cute import blackwell_pipeline, kernels, scaled_dot_product_attention

    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        parser.error("Assembly comparison requires a datacentre Blackwell GPU.")
    cuobjdump = args.cuobjdump or Path(triton.__file__).parent / "backends/nvidia/bin/cuobjdump"
    compiled, gluon_kernels = [], {}
    report = {
        "backend": args.backend,
        "shape": [1, args.heads, args.seq, args.dim],
        "dtype": args.dtype,
        "causal": args.causal,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "multiprocessors": torch.cuda.get_device_properties(0).multi_processor_count,
        "torch": str(torch.__version__),
        "triton": triton.__version__,
        "cute_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
        "tuning": load_upstream_tuning(args.kohakufa_tuning),
        "candidate_source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(kernels.__file__).parent.iterdir())
            if path.suffix in (".py", ".cpp")
        },
        "artifacts": [],
        "register_reconfiguration": [],
    }
    with ExitStack() as stack:
        if args.backend == "gluon":
            from kohakufa import attention
            from triton.runtime.jit import JITFunction

            original_run = JITFunction.run

            def run(self, *a, **kwargs):
                result = original_run(self, *a, **kwargs)
                if result is not None:
                    gluon_kernels[result.hash] = result
                return result

            stack.enter_context(patch.object(JITFunction, "run", run))

            def function(q, k, v, attn_mask=None, is_causal=False, enable_gqa=False):
                return attention(q, k, v, mask=attn_mask, causal=is_causal)

        else:
            function = scaled_dot_product_attention
            if args.baseline_source:
                baseline = load_source(args.baseline_source)
                report["baseline_sha256"] = hashlib.sha256(args.baseline_source.read_bytes()).hexdigest()
                stack.enter_context(
                    patch.multiple(
                        blackwell_pipeline,
                        **{name: baseline_factory(baseline, name) for name in ("Forward", "Query", "KeyValue")},
                    )
                )
            original_compile = kernels.compile_kernel

            def compile_and_record(*a, **kwargs):
                result = original_compile(*a, **kwargs)
                compiled.append((a[1], result))
                return result

            stack.enter_context(patch.object(kernels, "compile_kernel", compile_and_record))
        torch.manual_seed(21)
        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
        inputs = [torch.randn(*report["shape"], device="cuda", dtype=dtype) for _ in range(4)]
        reference = evaluate(torch.nn.functional.scaled_dot_product_attention, inputs[:3], inputs[3], None, args.causal)
        actual = evaluate(function, inputs[:3], inputs[3], None, args.causal)
        tolerance = 0.03 if dtype == torch.bfloat16 else 0.005
        for got, want in zip(actual, reference):
            torch.testing.assert_close(got, want, atol=tolerance, rtol=tolerance)
        torch.cuda.synchronize()
        report["validation"] = {
            "passed": True,
            "atol": tolerance,
            "rtol": tolerance,
            "max_abs_out_dq_dk_dv": [
                float((got.float() - want.float()).abs().max()) for got, want in zip(actual, reference)
            ],
        }
        if args.backend == "gluon":
            report["configurations"] = upstream_configurations(inputs[0], inputs[1], args.causal)
            report["upstream_source"] = upstream_provenance()
            for index, kernel in enumerate(gluon_kernels.values()):
                directory = args.output / f"{index:02d}-{kernel.name}"
                directory.mkdir(exist_ok=True)
                (directory / "kernel.ptx").write_text(kernel.asm["ptx"])
                (directory / "kernel.cubin").write_bytes(kernel.asm["cubin"])
                (directory / "metadata.json").write_text(
                    json.dumps(
                        {"metadata": kernel.metadata._asdict(), "n_regs": kernel.n_regs, "n_spills": kernel.n_spills},
                        indent=2,
                        default=str,
                    )
                    + "\n"
                )
        else:
            for mode, function in compiled:
                directory = args.output / mode
                directory.mkdir(exist_ok=True)
                for extension in ("ptx", "cubin"):
                    value = getattr(function, f"__{extension}__")
                    if isinstance(value, str) and "\n" not in value and Path(value).is_file():
                        value = Path(value).read_bytes()
                    payload = value.encode() if isinstance(value, str) else value
                    valid = payload and (b".entry" in payload if extension == "ptx" else payload.startswith(b"\x7fELF"))
                    if not valid:
                        raise RuntimeError(f"Missing or invalid {mode} {extension} artifact; inspect {dump}.")
                    (directory / f"kernel.{extension}").write_bytes(payload)
    for cubin in sorted(args.output.glob("*/kernel.cubin")):
        for option, extension in (("--dump-sass", ".sass"), ("--dump-resource-usage", ".resources.txt")):
            result = subprocess.run([str(cuobjdump), option, str(cubin)], text=True, capture_output=True, check=True)
            cubin.with_suffix(extension).write_text(result.stdout)
        report["register_reconfiguration"].append(
            {
                "cubin": str(cubin.relative_to(args.output)),
                **register_reconfiguration(cubin.with_suffix(".ptx").read_text(), cubin.with_suffix(".sass").read_text()),
            }
        )
        for path in sorted(cubin.parent.glob("kernel.*")):
            report["artifacts"].append(
                {"path": str(path.relative_to(args.output)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            )
    report["cuobjdump_version"] = subprocess.check_output([str(cuobjdump), "--version"], text=True)
    (args.output / "capture.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"backend": args.backend, "shape": report["shape"], "validated": True, "artifacts": len(report["artifacts"])}
        )
    )


if __name__ == "__main__":
    main()
