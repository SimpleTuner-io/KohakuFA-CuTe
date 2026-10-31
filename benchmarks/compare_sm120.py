"""Compare RTX 5090 outlier schedules with their preceding path and SDPA."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import torch

from benchmarks.compare_ada import capture, measure
from benchmarks.compare_devices import board_state, driver_api_version, loaded_libraries
from benchmarks.precision import evaluate
from benchmarks.reference import fp64_reference, metrics
from kohakufa_cute import kernels, l40s, native, scaled_dot_product_attention, sm86, sm120_schedule


def graph_nodes(graph):
    """Read the deployed graph functions without instrumenting timing samples."""
    import math

    from cuda.bindings import driver as cuda

    status, nodes, count = cuda.cuGraphGetNodes(cuda.CUgraph(graph.raw_cuda_graph()), 1024)
    assert int(status) == 0
    result = []
    for node in nodes[:count]:
        status, kind = cuda.cuGraphNodeGetType(node)
        assert int(status) == 0
        row = {"type": str(kind)}
        if kind == cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            status, params = cuda.cuGraphKernelNodeGetParams(node)
            assert int(status) == 0
            status, symbol = cuda.cuFuncGetName(params.func)
            assert int(status) == 0
            row.update(
                symbol=symbol.decode(),
                grid=[params.gridDimX, params.gridDimY, params.gridDimZ],
                block=[params.blockDimX, params.blockDimY, params.blockDimZ],
                dynamic_shared_bytes=params.sharedMemBytes,
            )
            row["attributes"] = {}
            for name in ("NUM_REGS", "LOCAL_SIZE_BYTES", "SHARED_SIZE_BYTES", "PTX_VERSION", "BINARY_VERSION"):
                status, value = cuda.cuFuncGetAttribute(
                    getattr(cuda.CUfunction_attribute, "CU_FUNC_ATTRIBUTE_" + name), params.func
                )
                assert int(status) == 0
                row["attributes"][name] = int(value)
            status, resident = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                params.func, math.prod(row["block"]), params.sharedMemBytes
            )
            assert int(status) == 0
            row["resident_cta_resource_bound"] = int(resident)
        result.append(row)
    return result


def precision_gate(actual, reference, dtype):
    """Apply the same independent component gates to eager and captured outputs."""
    assert len(actual) == len(reference)
    assert all(got.shape == want.shape for got, want in zip(actual, reference))
    threshold = 0.01 if dtype == torch.bfloat16 else 0.005
    tolerance = 0.02 if dtype == torch.bfloat16 else 0.003
    records = []
    for got, want in zip(actual, reference):
        error = metrics(got, want)
        error.update(
            pointwise_violations=int(((got.double() - want).abs() > tolerance + tolerance * want.abs()).sum().item()),
            pointwise_atol=tolerance,
            pointwise_rtol=tolerance,
            elements=got.numel(),
        )
        records.append(error)
    passed = all(
        error["finite"]
        and error["pointwise_violations"] == 0
        and (error["relative_error"] < threshold if error["reference_norm"] else error["max_absolute_error"] == 0)
        for error in records
    )
    return records, passed


def source_hashes(source):
    return {
        f"src/kohakufa_cute/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(source.iterdir())
        if p.suffix in (".py", ".cpp")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", nargs="+", type=int, default=[128, 1024, 4096])
    parser.add_argument("--dim", nargs="+", type=int, default=[16, 64, 128, 256, 512])
    parser.add_argument("--dtype", nargs="+", choices=("fp16", "bf16"), default=["bf16", "fp16"])
    parser.add_argument("--modes", nargs="+", choices=("dense", "causal"), default=["dense", "causal"])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--window-ms", type=int, default=200)
    parser.add_argument("--minimum-replays", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1213)
    parser.add_argument("--control", choices=("previous", "forward-only"), default="previous")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--omit-released", action="store_true", help="Use three graph sets if four exceed device headroom.")
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_name() != "NVIDIA GeForce RTX 5090":
        parser.error("This comparison requires a full RTX 5090.")
    if min(*args.seq, *args.dim, args.repeats, args.window_ms, args.minimum_replays) < 1:
        parser.error("Shapes and replay settings must be positive.")
    if not hasattr(kernels, "precision_schedule"):
        parser.error("This comparison requires the combined precision-dispatch base.")
    prop = torch.cuda.get_device_properties(0)
    if prop.multi_processor_count != 170 or not 31 * 1024**3 < prop.total_memory < 34 * 1024**3:
        parser.error("The measured path requires a full 170-SM, 32-GB RTX 5090.")
    source = Path(kernels.__file__).parent
    report = {
        "schema_version": 2,
        "date": datetime.now(timezone.utc).date().isoformat(),
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "device": {
            "name": prop.name,
            "compute_capability": [prop.major, prop.minor],
            "multiprocessors": prop.multi_processor_count,
            "memory_bytes": prop.total_memory,
            "partitioned": False,
            "board_before": board_state(),
            "properties": {
                name: getattr(prop, name)
                for name in (
                    "shared_memory_per_multiprocessor",
                    "shared_memory_per_block_optin",
                    "regs_per_multiprocessor",
                    "max_threads_per_multi_processor",
                    "total_memory",
                    "warp_size",
                    "max_threads_per_block",
                    "max_blocks_per_multiprocessor",
                )
                if hasattr(prop, name)
            },
        },
        "stack": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "cuda_driver_api_version": driver_api_version(),
            "torch_revision": torch.version.git_version,
            "cudnn": torch.backends.cudnn.version(),
            "packages": {
                **{name: importlib.metadata.version(name) for name in ("nvidia-cutlass-dsl", "apache-tvm-ffi", "triton")},
                **{
                    distribution.metadata["Name"]: distribution.version
                    for distribution in importlib.metadata.distributions()
                    if distribution.metadata["Name"].lower().startswith("nvidia-cutlass-dsl-libs-")
                },
            },
        },
        "source_sha256": source_hashes(source),
        "benchmark_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in (
                "benchmarks/compare_sm120.py",
                "benchmarks/compare_ada.py",
                "benchmarks/compare_devices.py",
                "benchmarks/reference.py",
            )
        },
        "scope": "Same-process pre-repair released computation, precision-matched previous schedule, selected CuTe and actual default SDPA; B1/H16 contiguous BHSD, FP16/BF16 dense/causal. Warmed resident GPU forward, retained-forward backward, and complete forward/backward. Compilation, allocation and graph capture excluded; device-event intervals include graph-entry/node scheduling and any submission stalls.",
        "protocol": {
            "seed": args.seed,
            "repeats": args.repeats,
            "target_window_ms": args.window_ms,
            "minimum_replays": args.minimum_replays,
            "ordering": "Rotate all selected backends each round; actual replay counts retained.",
            "control": (
                "Disable only sm120_schedule.eligible; shared precision repairs remain enabled. This is the precision-matched previous schedule."
                if args.control == "previous"
                else "Retain selected forward schedules and disable only new backward schedules; native workspace is unchanged."
            ),
            "released_control": "Disable sm120_schedule.eligible, sm86.eligible and l40s.eligible, and replace kernels.precision_schedule with identity accepting **kwargs. This reconstructs3e17149 computation under the current compiler, not the historical binary.",
            "reference": "Full-length centered FP64 on identical rounded inputs, two-head chunks.",
            "accuracy_gate": "Each output/dQ/dK/dV norm-relative error below 1% BF16 or 0.5% FP16; exact zero when reference norm is zero.",
            "pointwise_gate": "Each output/dQ/dK/dV element also meets atol=rtol=0.02 BF16 or 0.003 FP16. Failed released or SDPA rows remain diagnostic.",
            "profiler_during_timing": False,
            "telemetry_during_timing": False,
            "kernel_verification": "Actual graph functions/resources after uninstrumented timing. Activity profiling uses a separate process and is never initialized here.",
            "phase_note": "Backward retains a forward graph; separately timed medians need not sum to the complete-call median.",
        },
        "measurements": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    backends = ("sdpa", "previous", "cute") if args.omit_released else ("sdpa", "released", "previous", "cute")
    report["protocol"]["backends"] = list(backends)
    phases = ("forward", "backward", "joint")
    compiled = []
    original_compile = kernels.compile_kernel
    original_eligible = sm120_schedule.eligible

    def compile_and_retain(*a, **kw):
        function = original_compile(*a, **kw)
        compiled.append(function)
        return function

    def control(backend):
        context = ExitStack()
        if backend == "released":
            for selector in (sm120_schedule, sm86, l40s):
                context.enter_context(patch.object(selector, "eligible", return_value=False))
            context.enter_context(patch.object(kernels, "precision_schedule", side_effect=lambda program, *a, **kw: program))
        elif backend == "previous":
            if args.control == "previous":
                context.enter_context(patch.object(sm120_schedule, "eligible", return_value=False))
            else:
                context.enter_context(
                    patch.object(
                        sm120_schedule,
                        "eligible",
                        side_effect=lambda device, mode, *a: mode == "forward" and original_eligible(device, mode, *a),
                    )
                )
        return context

    with patch.object(kernels, "compile_kernel", compile_and_retain):
        for dtype_name in args.dtype:
            for sequence in args.seq:
                for dimension in args.dim:
                    for mode in args.modes:
                        dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
                        causal = mode == "causal"
                        torch.manual_seed(args.seed)
                        q, k, v, dout = [
                            torch.randn(1, 16, sequence, dimension, device="cuda", dtype=dtype) for _ in range(4)
                        ]
                        parts = [
                            fp64_reference(
                                (q[:, h : h + 2], k[:, h : h + 2], v[:, h : h + 2]), dout[:, h : h + 2], causal=causal
                            )
                            for h in range(0, 16, 2)
                        ]
                        reference = tuple(torch.cat([part[i] for part in parts], dim=1) for i in range(4))
                        del parts
                        torch.cuda.reset_peak_memory_stats()
                        graphs, storage, accuracy, qualified, graph_accuracy, graph_passed = {}, {}, {}, {}, {}, {}
                        for backend in backends:
                            native.extension().clear_cache()
                            kernels._compile_kernel.cache_clear()
                            function = (
                                torch.nn.functional.scaled_dot_product_attention
                                if backend == "sdpa"
                                else scaled_dot_product_attention
                            )
                            with control(backend):
                                actual = evaluate(function, (q, k, v), dout, None, causal)
                                errors, qualified[backend] = precision_gate(actual, reference, dtype)
                                accuracy[backend] = errors
                                graph_accuracy[backend], graph_passed[backend] = {}, {}
                                for phase in phases:
                                    graphs[backend, phase], storage[backend, phase] = capture(
                                        function, (q, k, v), dout, causal, phase, keep_graph=True
                                    )
                                    expected = (
                                        reference[:1]
                                        if phase == "forward"
                                        else reference[1:] if phase == "backward" else reference
                                    )
                                    records, passed = precision_gate(storage[backend, phase][3], expected, dtype)
                                    graph_accuracy[backend][phase] = records
                                    graph_passed[backend][phase] = passed
                                qualified[backend] = qualified[backend] and all(graph_passed[backend].values())
                                if backend in ("previous", "cute"):
                                    assert qualified[backend], (backend, errors, graph_accuracy[backend])
                        iterations = {
                            key: max(args.minimum_replays, min(100000, int(args.window_ms / max(measure(graph, 10), 0.001))))
                            for key, graph in graphs.items()
                        }
                        samples = {backend: {phase: [] for phase in phases} for backend in backends}
                        for repeat in range(args.repeats):
                            offset = repeat % len(backends)
                            for backend in backends[offset:] + backends[:offset]:
                                for phase in phases:
                                    graph = graphs[backend, phase]
                                    for _ in range(8):
                                        graph.replay()
                                    samples[backend][phase].append(measure(graph, iterations[backend, phase]))
                        symbols = {backend: {"graph_nodes": graph_nodes(graphs[backend, "joint"])} for backend in backends}
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
                            "passes_precision": qualified,
                            "graph_metrics_out_dq_dk_dv": graph_accuracy,
                            "graph_phase_passes_precision": graph_passed,
                            "comparisons_qualified": qualified["sdpa"] and qualified["cute"],
                            "qualified_comparisons": {
                                name + "_vs_cute": qualified[name] and qualified["cute"]
                                for name in backends
                                if name != "cute"
                            },
                            "released_role": "Pre-repair computation through current compiler; failed precision is diagnostic.",
                            "sdpa_role": (
                                "Qualified performance comparator; not the numerical oracle."
                                if qualified["sdpa"]
                                else "Numerically failed diagnostic; no qualified SDPA speedup claim."
                            ),
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                            "graph_validation": "Each captured phase meets strict FP64 component gates on identical inputs. Changed-input replay additionally agrees with fresh eager execution; independent changed-data FP64 checks are in the GPU qualification suite.",
                            "graph_checks_passed": True,
                            "changed_input_forward_and_joint_graphs_checked": True,
                            "actual_launched_kernels": symbols,
                        }
                        report["measurements"].append(row)
                        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                        print(json.dumps({key: row[key] for key in ("shape", "dtype", "mode", "medians_ms")}), flush=True)
                        del graphs, storage, actual, reference, q, k, v, dout, graph, expected
                        native.extension().clear_cache()
                        kernels._compile_kernel.cache_clear()
                        compiled.clear()
                        gc.collect()
    report["source_sha256_end"] = source_hashes(source)
    assert report["source_sha256_end"] == report["source_sha256"], "Runtime source changed during measurement."
    report["runtime_source_unchanged"] = True
    report["device"]["board_after"] = board_state()
    report["stack"]["loaded_libraries"] = loaded_libraries()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
