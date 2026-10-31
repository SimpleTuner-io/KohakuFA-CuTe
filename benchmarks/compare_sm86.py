"""Compare RTX 3090 Ti schedules, their released control and default SDPA."""

import argparse
import csv
import ctypes
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
from contextlib import ExitStack, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import torch

from benchmarks.precision import evaluate
from benchmarks.reference import fp64_reference, metrics


def memory_state():
    free, total = torch.cuda.mem_get_info()
    return {
        "free_bytes": free,
        "total_bytes": total,
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
    }


def board_state():
    fields = [
        "name",
        "driver_version",
        "memory.total",
        "memory.used",
        "power.limit",
        "power.draw",
        "clocks.sm",
        "temperature.gpu",
    ]
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"], text=True
    )
    rows = list(csv.reader(output.splitlines()))
    if len(rows) != 1:
        raise RuntimeError("The comparison requires one visible physical GPU.")
    return dict(zip(fields, (value.strip() for value in rows[0])))


def launched_kernels(operation, path):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as p:
        outputs = operation()
        torch.cuda.synchronize()
    names = [event.name for event in p.events() if event.device_type == torch.autograd.DeviceType.CUDA]
    if not names:
        raise RuntimeError("The profiling pass did not identify actual CUDA kernels.")
    p.export_chrome_trace(str(path))
    del outputs
    return names


def loaded_libraries():
    names = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        candidate = line.split(maxsplit=5)[-1]
        name = Path(candidate).name
        if ".so" in name and any(
            word in name for word in ("cuda", "cudnn", "cublas", "torch", "kohakufa", "nvvm", "ptxjit")
        ):
            names.add(name)
    return sorted(names)


def driver_api_version():
    version = ctypes.c_int()
    status = ctypes.CDLL("libcuda.so.1").cuDriverGetVersion(ctypes.byref(version))
    if status:
        raise RuntimeError("Cannot query the active CUDA driver library.")
    return version.value


def reference_metrics(got, expected, dtype):
    error = metrics(got, expected)
    atol = rtol = 0.02 if dtype == torch.bfloat16 else 0.003
    difference = (got.detach().double() - expected).abs()
    failures = (difference > atol + rtol * expected.abs()) | ~torch.isfinite(difference)
    error.update(pointwise_violations=int(failures.sum()), pointwise_atol=atol, pointwise_rtol=rtol, elements=got.numel())
    norm_gate = 0.01 if dtype == torch.bfloat16 else 0.005
    error["passes_accuracy"] = bool(
        error["finite"]
        and (error["relative_error"] < norm_gate if error["reference_norm"] else error["max_absolute_error"] == 0)
        and not error["pointwise_violations"]
    )
    return error


def graph_metadata(graph):
    import cuda.bindings.driver as cuda

    status, nodes, count = cuda.cuGraphGetNodes(cuda.CUgraph(graph.raw_cuda_graph()), 256)
    assert int(status) == 0 and count <= 256
    records = []
    for node in nodes[:count]:
        status, kind = cuda.cuGraphNodeGetType(node)
        assert int(status) == 0
        if kind != cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            continue
        status, parameters = cuda.cuGraphKernelNodeGetParams(node)
        assert int(status) == 0
        status, symbol = cuda.cuFuncGetName(parameters.func)
        assert int(status) == 0
        record = {
            "symbol": symbol.decode(),
            "grid": [parameters.gridDimX, parameters.gridDimY, parameters.gridDimZ],
            "block": [parameters.blockDimX, parameters.blockDimY, parameters.blockDimZ],
            "dynamic_shared_bytes": parameters.sharedMemBytes,
            "attributes": {},
        }
        for name in ("NUM_REGS", "LOCAL_SIZE_BYTES", "SHARED_SIZE_BYTES", "BINARY_VERSION", "PTX_VERSION"):
            status, value = cuda.cuFuncGetAttribute(
                getattr(cuda.CUfunction_attribute, "CU_FUNC_ATTRIBUTE_" + name), parameters.func
            )
            assert int(status) == 0
            record["attributes"][name] = int(value)
        status, resident = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            parameters.func, parameters.blockDimX * parameters.blockDimY * parameters.blockDimZ, parameters.sharedMemBytes
        )
        assert int(status) == 0
        record["resource_limited_resident_ctas"] = int(resident)
        records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-device", required=True)
    parser.add_argument("--omit-released", action="store_true", help="Use three graph sets if memory is constrained.")
    parser.add_argument("--seq", nargs="+", type=int, default=[128, 1024, 4096])
    parser.add_argument("--dim", nargs="+", type=int, default=[16, 64, 128, 256, 512])
    parser.add_argument("--dtype", nargs="+", choices=("fp16", "bf16"), default=["bf16", "fp16"])
    parser.add_argument("--modes", nargs="+", choices=("dense", "causal"), default=["dense", "causal"])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--window-ms", type=int, default=200)
    parser.add_argument("--minimum-replays", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1081)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--export-code", action="store_true")
    parser.add_argument("--verify-activities", action="store_true", help="Initialize a profiler in this diagnostic process.")
    parser.add_argument("--profile-iterations", type=int, default=0)
    args = parser.parse_args()
    if args.export_code:
        args.profiles.mkdir(parents=True, exist_ok=True)
        (args.profiles / "compiler").mkdir(parents=True, exist_ok=True)
        os.environ.update(
            CUTE_DSL_KEEP="ptx,cubin",
            CUTE_DSL_DUMP_DIR=str((args.profiles / "compiler").resolve()),
            CUTE_DSL_DISABLE_FILE_CACHING="1",
        )
    from benchmarks.compare_ada import capture, measure
    from kohakufa_cute import kernels, native, scaled_dot_product_attention, sm86

    if not torch.cuda.is_available() or torch.cuda.get_device_name() != args.expect_device:
        parser.error("The visible GPU does not match the requested device.")
    if min(*args.seq, *args.dim, args.repeats, args.window_ms, args.minimum_replays) < 1:
        parser.error("Shapes and replay settings must be positive.")
    properties = torch.cuda.get_device_properties(0)
    if args.expect_device != "NVIDIA GeForce RTX 3090 Ti" or properties.multi_processor_count != 84:
        parser.error("Requires a full 84-SM RTX 3090 Ti.")
    source = Path(kernels.__file__).parent
    report = {
        "schema_version": 2,
        "date": datetime.now(timezone.utc).date().isoformat(),
        "base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "device": {
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "multiprocessors": properties.multi_processor_count,
            "memory_bytes": properties.total_memory,
            "partitioned": False,
            "board_before": board_state(),
        },
        "stack": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "cuda_driver_api_version": driver_api_version(),
            "torch_revision": torch.version.git_version,
            "cudnn": torch.backends.cudnn.version(),
            "packages": {
                name: importlib.metadata.version(name) for name in ("nvidia-cutlass-dsl", "apache-tvm-ffi", "triton")
            },
        },
        "source_sha256": {
            f"src/kohakufa_cute/{path.name}": hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source.iterdir())
            if path.suffix in (".py", ".cpp")
        },
        "benchmark_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in (
                "benchmarks/compare_sm86.py",
                "benchmarks/compare_ada.py",
                "benchmarks/reference.py",
            )
        },
        "scope": "Selected RTX 3090 Ti schedules vs precision-matched preceding schedules, pre-repair released computation and actual default SDPA, B1/H16 contiguous BHSD, FP16/BF16 dense/causal. CUDA-event latency across warmed per-call graph replays includes any replay/launch scheduling floor; forward retains autograd and joint includes all three gradients. CPU input preparation, compilation and full-model costs are excluded.",
        "protocol": {
            "seed": args.seed,
            "repeats": args.repeats,
            "target_window_ms": args.window_ms,
            "minimum_replays": args.minimum_replays,
            "ordering": "Rotate all backends each round.",
            "reference": "Full-length centered FP64 on identical rounded inputs, two-head chunks.",
            "accuracy_gate": "Each output/dQ/dK/dV norm-relative error below 1% BF16 or 0.5% FP16; exact zero when reference norm is zero. Every element also passes atol=rtol=0.02 BF16 or 0.003 FP16. Inaccurate previous/SDPA calls remain diagnostic, not qualified speed comparisons.",
            "profiler_during_timing": False,
            "telemetry_during_timing": False,
            "activity_profiler_initialized": args.verify_activities,
            "kernel_verification": (
                "Diagnostic process: eager activities before timing and graph activities afterward; timings may include persistent profiler callback overhead."
                if args.verify_activities
                else "CUDA driver graph-node symbols and resources only; eager activity verification belongs to a separate process."
            ),
            "phase_note": "Backward retains a forward graph. Separately timed medians need not sum to joint.",
        },
        "measurements": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.profiles.mkdir(parents=True, exist_ok=True)
    backends = {"sdpa": torch.nn.functional.scaled_dot_product_attention}
    if not args.omit_released:
        backends["released"] = scaled_dot_product_attention
    backends.update(previous=scaled_dot_product_attention, cute=scaled_dot_product_attention)
    phases = ("forward", "backward", "joint")
    original_eligible, original_schedule = sm86.eligible, sm86.schedule
    if not hasattr(kernels, "precision_schedule"):
        raise RuntimeError("The comparison requires the shared precision-policy boundary.")

    def backend_context(backend):
        if backend not in ("released", "previous"):
            return nullcontext()
        context = ExitStack()
        context.enter_context(patch.object(sm86, "eligible", return_value=False))
        if backend == "released":
            context.enter_context(patch.object(kernels, "precision_schedule", side_effect=lambda program, *a, **kw: program))
        return context

    report["control_scopes"] = {
        "released": "Current compiler with the computation and schedules of main 3e17149: disable the Ti selector and central precision repairs. Other device selectors cannot match this full SM86 Ti. This is not a byte-identical old cubin; failed numerical gates are diagnostic.",
        "previous": "Retain all shared precision repairs and disable only the Ti schedule selector.",
        "cute": "Final Ti schedule and shared precision repairs.",
        "sdpa": "Actual default native SDPA on identical rounded inputs and the same runtime stack.",
    }
    report["protocol"]["backends"] = list(backends)
    original_compile = kernels.compile_kernel
    compiled = []
    compile_context = {"label": "initial"}
    exports = []

    def compile_and_retain(*a, **kw):
        function = original_compile(*a, **kw)
        compiled.append(function)
        if args.export_code:
            prefix = args.profiles / "code" / (compile_context["label"] + "-" + str(len(compiled)))
            prefix.parent.mkdir(parents=True, exist_ok=True)
            record = {"label": compile_context["label"], "files": {}}
            for suffix in ("ptx", "cubin"):
                data = getattr(function, "__" + suffix + "__")
                if isinstance(data, str) and "\n" not in data and Path(data).is_file():
                    data = Path(data).read_bytes()
                if isinstance(data, (bytes, bytearray)) and data.startswith(b"/") and b"\n" not in data:
                    data = Path(data.decode()).read_bytes()
                if isinstance(data, str):
                    data = data.encode()
                if not data or not (b".entry" in data if suffix == "ptx" else data.startswith(b"\x7fELF")):
                    raise RuntimeError("Compiler did not retain the requested code export.")
                path = prefix.with_suffix("." + suffix)
                path.write_bytes(data)
                record["files"][suffix] = {"file": path.name, "sha256": hashlib.sha256(data).hexdigest()}
            for flag, suffix in (("--dump-sass", "sass"), ("--dump-resource-usage", "resources.txt")):
                result = subprocess.run(
                    ["cuobjdump", flag, str(prefix.with_suffix(".cubin"))], capture_output=True, text=True, check=True
                )
                path = prefix.with_suffix("." + suffix)
                path.write_text(result.stdout)
                record["files"][suffix] = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            exports.append(record)
        return function

    with (
        patch.object(sm86, "eligible", original_eligible),
        patch.object(sm86, "schedule", original_schedule),
        patch.object(kernels, "compile_kernel", compile_and_retain),
    ):
        for dtype_name in args.dtype:
            for sequence in args.seq:
                for dimension in args.dim:
                    for mode in args.modes:
                        dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
                        causal = mode == "causal"
                        torch.cuda.reset_peak_memory_stats()
                        memory_before_case = memory_state()
                        torch.manual_seed(args.seed)
                        q, k, v, dout = [
                            torch.randn(1, 16, sequence, dimension, device="cuda", dtype=dtype) for _ in range(4)
                        ]
                        parts = [
                            fp64_reference(
                                (q[:, head : head + 2], k[:, head : head + 2], v[:, head : head + 2]),
                                dout[:, head : head + 2],
                                causal=causal,
                            )
                            for head in range(0, 16, 2)
                        ]
                        reference = tuple(torch.cat([part[i] for part in parts], dim=1) for i in range(4))
                        del parts
                        graphs, storage, accuracy, symbols = {}, {}, {}, {}
                        captured_accuracy = {}
                        accuracy_passed = {}
                        memory_before_capture = memory_state()
                        for backend, function in backends.items():
                            compile_context["label"] = f"{dtype_name}-s{sequence}-d{dimension}-{mode}-{backend}"
                            native.extension().clear_cache()
                            kernels._compile_kernel.cache_clear()
                            with backend_context(backend):
                                actual = evaluate(function, (q, k, v), dout, None, causal)
                                errors = [
                                    reference_metrics(got, expected, dtype) for got, expected in zip(actual, reference)
                                ]
                                accuracy_passed[backend] = all(error["passes_accuracy"] for error in errors)
                                if not accuracy_passed[backend]:
                                    label = f"{dtype_name}-s{sequence}-d{dimension}-{mode}-{backend}"
                                    (args.profiles / (label + "-accuracy-failure.json")).write_text(
                                        json.dumps(errors, indent=2, allow_nan=False) + "\n"
                                    )
                                    if backend == "cute":
                                        raise AssertionError(
                                            "Selected CuTe failed the combined independent FP64 accuracy gates."
                                        )
                                accuracy[backend] = errors
                                label = f"{dtype_name}-s{sequence}-d{dimension}-{mode}-{backend}"
                                if args.verify_activities:
                                    eager_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in (q, k, v)]

                                    def eager_call():
                                        output = function(*eager_inputs, is_causal=causal)
                                        return output, *torch.autograd.grad(output, eager_inputs, dout)

                                    symbols[backend] = {
                                        "eager": launched_kernels(
                                            eager_call,
                                            args.profiles / f"{label}-eager.json",
                                        )
                                    }
                                for phase in phases:
                                    graphs[backend, phase], storage[backend, phase] = capture(
                                        function, (q, k, v), dout.detach().clone(), causal, phase, keep_graph=True
                                    )
                                    captured = storage[backend, phase][3]
                                    expected = (
                                        reference[:1]
                                        if phase == "forward"
                                        else reference[1:] if phase == "backward" else reference
                                    )
                                    phase_errors = [
                                        reference_metrics(got, want, dtype) for got, want in zip(captured, expected)
                                    ]
                                    captured_accuracy.setdefault(backend, {})[phase] = phase_errors
                                    accuracy_passed[backend] = accuracy_passed[backend] and all(
                                        error["passes_accuracy"] for error in phase_errors
                                    )
                                    if backend == "cute" and not accuracy_passed[backend]:
                                        raise AssertionError("Selected captured phase failed the independent FP64 gates.")
                                del captured, expected

                        memory_after_capture = memory_state()
                        # A separate FP64 check verifies changed captured data, not only eager agreement.
                        changed_operands, changed_dout, _, changed_outputs = storage["cute", "joint"]
                        with torch.no_grad():
                            for tensor in (*changed_operands, changed_dout):
                                tensor.copy_(torch.randn_like(tensor) * 0.5)
                            for tensor in changed_outputs:
                                tensor.fill_(float("nan"))
                        graphs["cute", "joint"].replay()
                        torch.cuda.synchronize()
                        changed_parts = [
                            fp64_reference(
                                tuple(tensor[:, head : head + 2] for tensor in changed_operands),
                                changed_dout[:, head : head + 2],
                                causal=causal,
                            )
                            for head in range(0, 16, 2)
                        ]
                        changed_reference = tuple(torch.cat([part[i] for part in changed_parts], dim=1) for i in range(4))
                        changed_errors = [
                            reference_metrics(got, expected, dtype)
                            for got, expected in zip(changed_outputs, changed_reference)
                        ]
                        if not all(error["passes_accuracy"] for error in changed_errors):
                            raise AssertionError("Changed-data poisoned joint graph failed the independent FP64 gates.")
                        with torch.no_grad():
                            for tensor, original in zip(changed_operands, (q, k, v)):
                                tensor.copy_(original)
                            changed_dout.copy_(dout)
                        graphs["cute", "joint"].replay()
                        torch.cuda.synchronize()
                        del changed_parts, changed_reference, changed_operands, changed_dout, changed_outputs
                        iterations = {
                            key: max(args.minimum_replays, min(100000, int(args.window_ms / max(measure(graph, 10), 0.001))))
                            for key, graph in graphs.items()
                        }
                        samples = {backend: {phase: [] for phase in phases} for backend in backends}
                        for repeat in range(args.repeats):
                            names = tuple(backends)
                            offset = repeat % len(names)
                            order = names[offset:] + names[:offset]
                            for backend in order:
                                for phase in phases:
                                    graph = graphs[backend, phase]
                                    for _ in range(8):
                                        graph.replay()
                                    samples[backend][phase].append(measure(graph, iterations[backend, phase]))
                        if args.verify_activities:
                            for backend in backends:
                                label = f"{dtype_name}-s{sequence}-d{dimension}-{mode}-{backend}"
                                symbols[backend]["graph"] = launched_kernels(
                                    graphs[backend, "joint"].replay, args.profiles / f"{label}-graph.json"
                                )
                        row = {
                            "shape": list(q.shape),
                            "dtype": dtype_name,
                            "mode": mode,
                            "samples_ms": samples,
                            "medians_ms": {
                                backend: {phase: statistics.median(values) for phase, values in record.items()}
                                for backend, record in samples.items()
                            },
                            "iterations": {
                                backend: {phase: iterations[backend, phase] for phase in phases} for backend in backends
                            },
                            "fp64_metrics_out_dq_dk_dv": accuracy,
                            "captured_fp64_metrics_by_backend_phase": captured_accuracy,
                            "passes_precision_by_backend": accuracy_passed,
                            "passes_precision": accuracy_passed,
                            "memory_bytes": {
                                "before_case": memory_before_case,
                                "before_graph_capture": memory_before_capture,
                                "after_graph_capture": memory_after_capture,
                                "peak_allocated_including_fp64_reference": torch.cuda.max_memory_allocated(),
                                "peak_reserved_including_fp64_reference": torch.cuda.max_memory_reserved(),
                            },
                            "accuracy_qualified_speed_comparison": accuracy_passed["cute"] and accuracy_passed["sdpa"],
                            "accuracy_qualified_schedule_comparison": accuracy_passed["cute"]
                            and accuracy_passed["previous"],
                            "changed_poisoned_joint_fp64_out_dq_dk_dv": changed_errors,
                            "graph_checks_passed": True,
                            "changed_input_forward_and_joint_graphs_checked": True,
                            "actual_launched_kernels": symbols,
                            "actual_graph_kernels": {
                                backend: graph_metadata(graphs[backend, "joint"]) for backend in backends
                            },
                        }
                        report["measurements"].append(row)
                        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                        print(json.dumps({key: row[key] for key in ("shape", "dtype", "mode", "medians_ms")}), flush=True)
                        if args.profile_iterations:
                            torch.cuda.cudart().cudaProfilerStart()
                            for backend in backends:
                                torch.cuda.nvtx.range_push(backend)
                                for _ in range(args.profile_iterations):
                                    graphs[backend, "joint"].replay()
                                torch.cuda.synchronize()
                                torch.cuda.nvtx.range_pop()
                            torch.cuda.cudart().cudaProfilerStop()
                        del graphs, storage, graph, actual, reference, q, k, v, dout
                        native.extension().clear_cache()
                        kernels._compile_kernel.cache_clear()
                        compiled.clear()
                        gc.collect()
    final_sources = {
        f"src/kohakufa_cute/{path.name}": hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source.iterdir())
        if path.suffix in (".py", ".cpp")
    }
    if final_sources != report["source_sha256"]:
        raise RuntimeError("Runtime source changed during comparison.")
    if any(
        hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest for name, digest in report["benchmark_sha256"].items()
    ):
        raise RuntimeError("Benchmark source changed during comparison.")
    report["runtime_and_benchmark_sources_unchanged"] = True
    report["device"]["board_after"] = board_state()
    report["code_exports"] = exports
    report["stack"]["loaded_libraries"] = loaded_libraries()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
