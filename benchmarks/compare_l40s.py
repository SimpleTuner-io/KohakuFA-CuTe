"""Compare L40S schedules, precision-matched/released controls and native SDPA."""

import argparse
import csv
import ctypes
import gc
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import cuda.bindings.driver as cuda
import torch

from benchmarks.compare_ada import capture, measure
from benchmarks.precision import evaluate
from benchmarks.reference import fp64_reference, metrics
from kohakufa_cute import kernels, l40s, native, scaled_dot_product_attention


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


def graph_kernels(graph):
    status, nodes, count = cuda.cuGraphGetNodes(cuda.CUgraph(graph.raw_cuda_graph()), 256)
    if int(status) or count > len(nodes):
        raise RuntimeError("Cannot inspect the captured CUDA graph.")
    records = []
    for node in nodes[:count]:
        status, kind = cuda.cuGraphNodeGetType(node)
        if int(status):
            raise RuntimeError("Cannot inspect a CUDA graph node.")
        if kind != cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            continue
        status, params = cuda.cuGraphKernelNodeGetParams(node)
        if int(status):
            raise RuntimeError("Cannot inspect a CUDA graph kernel.")
        status, symbol = cuda.cuFuncGetName(params.func)
        if int(status):
            raise RuntimeError("Cannot identify a CUDA graph kernel.")
        record = dict(
            symbol=symbol.decode(),
            grid=[params.gridDimX, params.gridDimY, params.gridDimZ],
            block=[params.blockDimX, params.blockDimY, params.blockDimZ],
            shared_bytes=params.sharedMemBytes,
        )
        for name in ("NUM_REGS", "LOCAL_SIZE_BYTES", "BINARY_VERSION", "PTX_VERSION"):
            status, value = cuda.cuFuncGetAttribute(
                getattr(cuda.CUfunction_attribute, "CU_FUNC_ATTRIBUTE_" + name), params.func
            )
            if int(status):
                raise RuntimeError("Cannot inspect CUDA kernel resources.")
            record[name] = int(value)
        records.append(record)
    if not records:
        raise RuntimeError("The captured graph contains no CUDA kernels.")
    return records


def qualify(actual, expected, dtype):
    tolerance, norm_limit = (0.02, 0.01) if dtype == torch.bfloat16 else (0.003, 0.005)
    errors = []
    for got, reference in zip(actual, expected):
        error = metrics(got, reference)
        error["pointwise_violations"] = int(((got.double() - reference).abs() > tolerance * (1 + reference.abs())).sum())
        error["passed"] = bool(
            error["finite"]
            and error["pointwise_violations"] == 0
            and (error["relative_error"] < norm_limit if error["reference_norm"] else error["max_absolute_error"] == 0)
        )
        errors.append(error)
    return errors


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-device", default="NVIDIA L40S", choices=("NVIDIA L40S",))
    parser.add_argument("--seq", nargs="+", type=int, default=[128, 1024, 4096])
    parser.add_argument("--dim", nargs="+", type=int, default=[16, 64, 128, 256, 512])
    parser.add_argument("--dtype", nargs="+", choices=("fp16", "bf16"), default=["bf16", "fp16"])
    parser.add_argument("--modes", nargs="+", choices=("dense", "causal"), default=["dense", "causal"])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--window-ms", type=int, default=200)
    parser.add_argument("--minimum-replays", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1081)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_name() != args.expect_device:
        parser.error("The visible GPU does not match the requested device.")
    if min(*args.seq, *args.dim, args.repeats, args.window_ms, args.minimum_replays) < 1:
        parser.error("Shapes and replay settings must be positive.")
    properties = torch.cuda.get_device_properties(0)
    if properties.multi_processor_count != 142:
        parser.error("The comparison requires a full 142-SM L40S.")
    from kohakufa_cute import sm86, sm120_schedule

    if not callable(getattr(kernels, "precision_schedule", None)):
        raise RuntimeError("The comparison requires the integrated final precision hook.")
    architecture_modules = (l40s, sm86, sm120_schedule)

    @contextmanager
    def backend_control(backend):
        with ExitStack() as contexts:
            if backend == "previous":
                contexts.enter_context(patch.object(l40s, "eligible", return_value=False))
            elif backend == "released":
                for module in architecture_modules:
                    contexts.enter_context(patch.object(module, "eligible", return_value=False))
                contexts.enter_context(
                    patch.object(kernels, "precision_schedule", side_effect=lambda program, dimension, **kwargs: program)
                )
            yield

    source = Path(kernels.__file__).parent
    report = {
        "schema_version": 3,
        "date": datetime.now(timezone.utc).date().isoformat(),
        "base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_state": "Runtime files identified by source_sha256; base_revision alone does not identify uncommitted files.",
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
                name: importlib.metadata.version(name)
                for name in ("nvidia-cutlass-dsl", "apache-tvm-ffi", "triton", "cuda-bindings")
            },
        },
        "source_sha256": {
            f"src/kohakufa_cute/{path.name}": hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source.iterdir())
            if path.suffix in (".py", ".cpp")
        },
        "benchmark_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in ("benchmarks/compare_l40s.py", "benchmarks/compare_ada.py", "benchmarks/reference.py")
        },
        "scope": "L40S final CuTe vs released arithmetic/schedules, precision-matched preceding schedules and default SDPA, B1/H16 contiguous BHSD, FP16/BF16 dense/causal. Resident GPU work through warmed CUDA graphs; forward retains autograd and joint includes all three gradients. Initial API dispatch, compilation and full-model costs excluded; CUDA-event spans can include host replay submission gaps.",
        "protocol": {
            "seed": args.seed,
            "repeats": args.repeats,
            "target_window_ms": args.window_ms,
            "minimum_replays": args.minimum_replays,
            "ordering": "Rotate all four backend orders each round.",
            "reference": "Full-length centered FP64 on identical rounded inputs, two-head chunks.",
            "accuracy_gate": "Independent output/dQ/dK/dV: pointwise atol=rtol .02 BF16 or .003 FP16; norm-relative error below 1% BF16 or 0.5% FP16; exact zero for zero reference.",
            "profiler_during_timing": False,
            "telemetry_during_timing": False,
            "kernel_verification": "Driver CUDA-graph function identity/resources only in this process; activity profilers run in separate processes.",
            "phase_note": "Backward retains a forward graph. Separately timed medians need not sum to joint.",
        },
        "controls": {
            "sdpa": "Default native PyTorch SDPA.",
            "released": "All new architecture selectors and final kernels.precision_schedule disabled. Reconstructs 3e17149 computation through this compiler, not a byte-identical old binary.",
            "previous": "Only L40S selector disabled; final shared precision retained.",
            "cute": "Final L40S selectors and shared precision enabled.",
            "architecture_predicates": [module.__name__ + ".eligible" for module in architecture_modules],
            "precision_hook": "kohakufa_cute.kernels.precision_schedule",
        },
        "measurements": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    backends = {
        "sdpa": torch.nn.functional.scaled_dot_product_attention,
        "released": scaled_dot_product_attention,
        "previous": scaled_dot_product_attention,
        "cute": scaled_dot_product_attention,
    }
    phases = ("forward", "backward", "joint")
    original_compile = kernels.compile_kernel
    compiled = []
    active_backend = {"name": None}
    control_audits = {}
    original_cute_compile = kernels.cute.compile

    def inspect_and_compile(program, *a, **kw):
        records, seen = [], set()

        def inspect(owner):
            if id(owner) in seen:
                return
            seen.add(id(owner))
            flags = {
                name: getattr(owner, name)
                for name in (
                    "compensated_keys",
                    "compensated_values",
                    "precise_values",
                    "scaled_key_residual",
                    "scaled_value_residual",
                )
                if isinstance(getattr(owner, name, None), bool)
            }
            if flags:
                records.append({"class": type(owner).__name__, "flags": flags})
                if active_backend["name"] == "released" and any(flags.values()):
                    raise RuntimeError("Released control retained a new precision policy.")
            for name in ("query", "keyvalue", "fallback", "tuned", "short", "compact", "checked", "complete", "precise"):
                child = getattr(owner, name, None)
                if child is not None:
                    inspect(child)

        inspect(program)
        control_audits.setdefault(active_backend["name"], []).append(records)
        return original_cute_compile(program, *a, **kw)

    def compile_and_retain(*a, **kw):
        function = original_compile(*a, **kw)
        compiled.append(function)
        return function

    with (
        patch.object(kernels, "compile_kernel", compile_and_retain),
        patch.object(kernels.cute, "compile", inspect_and_compile),
    ):
        for dtype_name in args.dtype:
            for sequence in args.seq:
                for dimension in args.dim:
                    for mode in args.modes:
                        dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
                        causal = mode == "causal"
                        torch.cuda.reset_peak_memory_stats()
                        free_before, total_memory = torch.cuda.mem_get_info()
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
                        graphs, storage, accuracy = {}, {}, {}
                        graph_accuracy = {}
                        control_audits.clear()
                        for backend, function in backends.items():
                            active_backend["name"] = backend
                            native.extension().clear_cache()
                            kernels._compile_kernel.cache_clear()
                            with backend_control(backend):
                                actual = evaluate(function, (q, k, v), dout, None, causal)
                                accuracy[backend] = qualify(actual, reference, dtype)
                                graph_accuracy[backend] = {}
                                for phase in phases:
                                    graphs[backend, phase], storage[backend, phase] = capture(
                                        function, (q, k, v), dout, causal, phase, keep_graph=True
                                    )
                                    expected = (
                                        reference[1:]
                                        if phase == "backward"
                                        else reference if phase == "joint" else reference[:1]
                                    )
                                    graph_accuracy[backend][phase] = qualify(storage[backend, phase][3], expected, dtype)
                        free_after_graphs, _ = torch.cuda.mem_get_info()
                        symbols = {
                            backend: {
                                "graph": [node["symbol"] for node in graph_kernels(graphs[backend, "joint"])],
                                "graph_resources": {phase: graph_kernels(graphs[backend, phase]) for phase in phases},
                            }
                            for backend in backends
                        }
                        iterations = {
                            key: max(args.minimum_replays, min(100000, int(args.window_ms / max(measure(graph, 10), 0.001))))
                            for key, graph in graphs.items()
                        }
                        samples = {backend: {phase: [] for phase in phases} for backend in backends}
                        for repeat in range(args.repeats):
                            names = tuple(backends)
                            order = names[repeat % len(names) :] + names[: repeat % len(names)]
                            for backend in order:
                                for phase in phases:
                                    graph = graphs[backend, phase]
                                    for _ in range(8):
                                        graph.replay()
                                    samples[backend][phase].append(measure(graph, iterations[backend, phase]))
                        row = {
                            "shape": list(q.shape),
                            "device_memory": {
                                "total_bytes": total_memory,
                                "free_before_bytes": free_before,
                                "free_after_graphs_bytes": free_after_graphs,
                                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                "reserved_bytes": torch.cuda.memory_reserved(),
                            },
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
                            "captured_graph_fp64_metrics": graph_accuracy,
                            "qualified_backends": {
                                backend: all(error["passed"] for error in accuracy[backend])
                                and all(error["passed"] for checks in graph_accuracy[backend].values() for error in checks)
                                for backend in backends
                            },
                            "compiled_precision_control_audit": dict(control_audits),
                            "graph_checks_passed": True,
                            "changed_input_forward_and_joint_graphs_checked": True,
                            "actual_launched_kernels": symbols,
                        }
                        report["measurements"].append(row)
                        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                        print(json.dumps({key: row[key] for key in ("shape", "dtype", "mode", "medians_ms")}), flush=True)
                        torch.cuda.synchronize()
                        del graph, graphs, storage, actual, expected, reference, q, k, v, dout
                        compiled.clear()
                        gc.collect()
    source_unchanged = report["source_sha256"] == {
        f"src/kohakufa_cute/{path.name}": hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source.iterdir())
        if path.suffix in (".py", ".cpp")
    }
    if not source_unchanged:
        raise RuntimeError("Runtime source changed during the benchmark.")
    report["qualification"] = {
        "source_unchanged_during_run": source_unchanged,
        "all_candidate_cases_passed": all(row["qualified_backends"]["cute"] for row in report["measurements"]),
        "failed_backend_timings_are_diagnostic": True,
    }
    report["device"]["board_after"] = board_state()
    report["stack"]["loaded_libraries"] = loaded_libraries()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not report["qualification"]["all_candidate_cases_passed"]:
        raise AssertionError("At least one candidate case failed the precision gates.")


if __name__ == "__main__":
    main()
