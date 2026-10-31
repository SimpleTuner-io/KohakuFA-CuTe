"""Compare actual standalone FlashAttention calls with qualified CuTe attention."""

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import torch

from benchmarks.compiled_artifacts import retain_code
from benchmarks.flashattention_backends import Backend, UnsupportedCase, configuration_metadata, load_backend
from benchmarks.reference import fp64_reference, metrics, rounded_reference


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--backends", nargs="+", choices=("cute", "sdpa", "fa2", "fa3", "fa4"), required=True)
    result.add_argument("--seq", nargs="+", type=int, default=[128, 1024, 4096])
    result.add_argument("--dim", nargs="+", type=int, default=[16, 64, 128, 256, 512])
    result.add_argument("--heads", type=int, default=16)
    result.add_argument("--dtype", nargs="+", choices=("bf16", "fp16"), default=["bf16", "fp16"])
    result.add_argument("--modes", nargs="+", choices=("dense", "causal"), default=["dense", "causal"])
    result.add_argument(
        "--phases", nargs="+", choices=("forward", "backward", "joint"), default=["forward", "backward", "joint"]
    )
    result.add_argument("--repeats", type=int, default=5)
    result.add_argument("--window-ms", type=float, default=100)
    result.add_argument("--candidate", help="Explicit module:factory returning an SDPA-shaped callable.")
    result.add_argument("--candidate-kwargs", default="{}")
    result.add_argument("--context", help="Explicit module:context_factory[:string_argument] for candidate CuTe dispatch.")
    result.add_argument(
        "--flash-revision", help="Revision verified by the installer; loaded modules are additionally hashed."
    )
    result.add_argument("--export-code", type=Path)
    result.add_argument("--output", type=Path, required=True)
    return result


def main():
    arguments = parser()
    args = arguments.parse_args()
    if min(*args.seq, *args.dim, args.heads, args.repeats, args.window_ms) <= 0:
        arguments.error("Shapes, repeat counts and timing windows must be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("This comparison requires a CUDA GPU.")
    if args.candidate and args.context:
        raise ValueError("Select a candidate callable or a dispatch context.")
    if args.export_code:
        args.export_code.mkdir(parents=True, exist_ok=True)
        os.environ.update(CUTE_DSL_KEEP="ptx,cubin", CUTE_DSL_DUMP_DIR=str(args.export_code.resolve()))
    from benchmarks.compare_ada import capture, measure
    from benchmarks.compare_l40s import board_state, driver_api_version, graph_kernels, loaded_libraries
    from kohakufa_cute import kernels

    compiled, compiler_artifacts, seen_artifacts = [], [], set()
    active_scope = {"stage": "initialization"}
    original = kernels.compile_kernel
    original_cute_compile = kernels.cute.compile

    def observe(function, origin, program=None):
        compiled.append(function)
        key = id(function), origin, json.dumps(active_scope, sort_keys=True)
        if args.export_code and key not in seen_artifacts:
            seen_artifacts.add(key)
            compiler_artifacts.append(
                {
                    "scope": dict(active_scope),
                    "origin": origin,
                    "program": program,
                    "artifacts": retain_code(function, args.export_code / "retained"),
                }
            )

    def retain(*a, **kw):
        function = original(*a, **kw)
        observe(function, "kohakufa_compile_kernel", a[1] if len(a) > 1 else kw.get("mode"))
        return function

    def retain_cute(program, *a, **kw):
        function = original_cute_compile(program, *a, **kw)
        identifier = (
            getattr(program, "__module__", type(program).__module__)
            + "."
            + getattr(program, "__qualname__", type(program).__qualname__)
        )
        observe(function, "cute_compile", identifier)
        return function

    kernels.compile_kernel = retain
    kernels.cute.compile = retain_cute
    properties = torch.cuda.get_device_properties(0)
    report = {
        "schema_version": 1,
        "date": datetime.now(timezone.utc).isoformat(),
        "base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            "src/kohakufa_cute/"
            + p.relative_to(Path(kernels.__file__).parent).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(kernels.__file__).parent.rglob("*"))
            if p.suffix in (".py", ".cpp")
        },
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "adapter_sha256": hashlib.sha256(Path(__file__).with_name("flashattention_backends.py").read_bytes()).hexdigest(),
        "device": {
            "name": properties.name,
            "capability": [properties.major, properties.minor],
            "sm": properties.multi_processor_count,
            "memory_bytes": properties.total_memory,
            "board_before": board_state(),
        },
        "stack": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "driver_api": driver_api_version(),
        },
        "packages": {p: importlib.metadata.version(p) for p in ("nvidia-cutlass-dsl", "apache-tvm-ffi", "triton")},
        "protocol": {
            "layout": "BHSD API boundary; FA receives BSHD views and returns a BHSD view, including any backend preparation",
            "causal": "square, top-left equals bottom-right",
            "backward": "retained forward; all dQ/dK/dV",
            "joint": "forward and all gradients",
            "timing": "warmed CUDA-graph device events; rotate backend order; no profiling during timing",
            "repeats": args.repeats,
            "target_window_ms": args.window_ms,
            "precision": "CuTe single-rounded P/dS, FP32 accumulation and actual-O delta; standalone FA retains its own arithmetic policy",
        },
        "flash_revision": args.flash_revision,
        "candidate": {
            "factory": args.candidate,
            "kwargs": configuration_metadata(json.loads(args.candidate_kwargs)),
            "context": args.context.split(":", 2)[:2] if args.context else None,
            "context_arguments": configuration_metadata(args.context.split(":", 2)[2:]) if args.context else [],
        },
        "backends": {},
        "measurements": [],
        "compiler_artifacts": compiler_artifacts,
        "compiler_artifact_binding": (
            "Compiler-returned code is retained by content hash with its observed case/backend scope. "
            "These records alone do not uniquely bind an actual loaded CUDA graph function; "
            "exact instruction comparisons require independent loaded-module association."
        ),
    }
    backends, unavailable = {}, {}
    context_factory = nullcontext
    for name in args.backends:
        try:
            backend = load_backend(name)
            metadata = backend.metadata()
            backends[name] = backend
            report["backends"][name] = metadata
        except Exception as error:
            unavailable[name] = {"status": "unavailable", "exception_type": type(error).__name__, "stage": "backend_load"}
            report["backends"][name] = unavailable[name]
    if args.candidate:
        module_name, factory_name = args.candidate.split(":")
        module = importlib.import_module(module_name)
        function = getattr(module, factory_name)(**json.loads(args.candidate_kwargs))
        backends["candidate"] = Backend("cute", function, module, "kohakufa-cute")
    elif args.context:
        module_name, factory_name, *argument = args.context.split(":", 2)
        candidate_module = importlib.import_module(module_name)
        report["candidate"]["source_sha256"] = hashlib.sha256(Path(candidate_module.__file__).read_bytes()).hexdigest()
        factory = getattr(candidate_module, factory_name)
        context_factory = lambda: factory(*argument)
        backends["candidate"] = load_backend("cute")
    if "candidate" in backends:
        report["backends"]["candidate"] = backends["candidate"].metadata()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    for dtype_name in args.dtype:
        for sequence in args.seq:
            for dimension in args.dim:
                for mode in args.modes:
                    torch.manual_seed(2031)
                    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
                    q, k, v, dout = [
                        torch.randn(1, args.heads, sequence, dimension, device="cuda", dtype=dtype) for _ in range(4)
                    ]
                    inputs, causal = (q, k, v), mode == "causal"
                    references = [
                        fp64_reference(tuple(t[:, h : h + 1] for t in inputs), dout[:, h : h + 1], causal=causal)
                        for h in range(args.heads)
                    ]
                    row = {
                        "shape": list(q.shape),
                        "dtype": dtype_name,
                        "mode": mode,
                        "backends": {name: {p: dict(error) for p in args.phases} for name, error in unavailable.items()},
                    }
                    graphs, owners = {}, {}
                    for name, backend in backends.items():
                        active_scope = {
                            "dtype": dtype_name,
                            "sequence": sequence,
                            "dimension": dimension,
                            "mode": mode,
                            "backend": name,
                        }
                        row["backends"][name] = {}
                        try:
                            with context_factory() if name == "candidate" else nullcontext():
                                operands = [t.detach().requires_grad_(True) for t in inputs]
                                out = backend(*operands, is_causal=causal)
                                forward_errors = [metrics(out[:, h : h + 1], ref[0]) for h, ref in enumerate(references)]
                                limit = 0.01 if dtype == torch.bfloat16 else 0.005

                                def passed(errors):
                                    return all(
                                        e["finite"]
                                        and (
                                            e["relative_error"] < limit
                                            if e["reference_norm"]
                                            else e["max_absolute_error"] == 0
                                        )
                                        for e in errors
                                    )

                                gradients, backward_error = None, None
                                if any(p != "forward" for p in args.phases):
                                    try:
                                        gradients = torch.autograd.grad(out, operands, dout)
                                    except Exception as error:
                                        backward_error = {
                                            "status": "execution_error",
                                            "exception_type": type(error).__name__,
                                            "stage": "backward",
                                        }
                                for phase in args.phases:
                                    if phase != "forward" and backward_error:
                                        row["backends"][name][phase] = dict(backward_error)
                                        continue
                                    errors = list(forward_errors) if phase != "backward" else []
                                    rounded_errors = []
                                    if phase != "forward":
                                        for h, ref in enumerate(references):
                                            rounded = rounded_reference(
                                                tuple(t[:, h : h + 1] for t in inputs),
                                                dout[:, h : h + 1],
                                                causal=causal,
                                                output=out[:, h : h + 1],
                                            )
                                            errors.extend(
                                                metrics(g[:, h : h + 1], want) for g, want in zip(gradients, ref[1:])
                                            )
                                            rounded_errors.extend(
                                                metrics(g[:, h : h + 1], want) for g, want in zip(gradients, rounded[1:])
                                            )
                                    result = {
                                        "status": "pass" if passed(errors) else "numerical_failure",
                                        "fp64": errors,
                                        "rounded_backward": rounded_errors,
                                    }
                                    row["backends"][name][phase] = result
                                    if result["status"] != "pass":
                                        continue
                                    try:
                                        graph, storage = capture(backend, inputs, dout, causal, phase, keep_graph=True)
                                        resources = graph_kernels(graph)
                                    except Exception as error:
                                        result.update(
                                            status="execution_error", exception_type=type(error).__name__, stage="capture"
                                        )
                                    else:
                                        result["graph_kernels"] = resources
                                        graphs[name, phase], owners[name, phase] = graph, storage
                                        result["samples_ms"] = []
                        except Exception as error:
                            for phase in args.phases:
                                if phase not in row["backends"][name] or (
                                    row["backends"][name][phase]["status"] == "pass" and (name, phase) not in graphs
                                ):
                                    row["backends"][name][phase] = {
                                        "status": "unsupported" if isinstance(error, UnsupportedCase) else "execution_error",
                                        "exception_type": type(error).__name__,
                                        "stage": "forward_or_capture",
                                    }
                    order = list(backends)
                    for repeat in range(args.repeats):
                        offset = repeat % max(1, len(order))
                        for name in order[offset:] + order[:offset]:
                            for phase in args.phases:
                                graph = graphs.get((name, phase))
                                if graph is None:
                                    continue
                                pilot = measure(graph, 16)
                                count = max(16, min(8192, int(args.window_ms / pilot)))
                                row["backends"][name][phase]["samples_ms"].append(measure(graph, count))
                    for name, phases in row["backends"].items():
                        for result in phases.values():
                            if result.get("samples_ms"):
                                result["median_ms"] = statistics.median(result["samples_ms"])
                            if name in ("cute", "candidate") and result["status"] != "pass":
                                failures += 1
                    report["measurements"].append(row)
                    report["cute_failures"] = failures
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(
                        {
                            "shape": list(q.shape),
                            "dtype": dtype_name,
                            "mode": mode,
                            "backends": {
                                b: {p: {k: r[k] for k in ("status", "median_ms") if k in r} for p, r in phases.items()}
                                for b, phases in row["backends"].items()
                            },
                        },
                        flush=True,
                    )
                    del graphs, owners, references, row
                    gc.collect()
                    torch.cuda.empty_cache()
    report["loaded_libraries"] = loaded_libraries()
    report["device"]["board_after"] = board_state()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["unavailable_backends"] = list(unavailable)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return bool(failures or unavailable)


if __name__ == "__main__":
    raise SystemExit(main())
