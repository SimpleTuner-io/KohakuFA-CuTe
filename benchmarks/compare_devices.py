"""Measure current CuTe and default SDPA on a qualified full GPU."""

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
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import torch

from benchmarks.compare_ada import capture, measure
from benchmarks.precision import evaluate
from benchmarks.reference import fp64_reference, metrics
from kohakufa_cute import kernels, native, scaled_dot_product_attention


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
        operation()
        torch.cuda.synchronize()
    names = [event.name for event in p.events() if event.device_type == torch.autograd.DeviceType.CUDA]
    if not names:
        raise RuntimeError("The profiling pass did not identify actual CUDA kernels.")
    p.export_chrome_trace(str(path))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-device", required=True)
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
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_name() != args.expect_device:
        parser.error("The visible GPU does not match the requested device.")
    if min(*args.seq, *args.dim, args.repeats, args.window_ms, args.minimum_replays) < 1:
        parser.error("Shapes and replay settings must be positive.")
    properties = torch.cuda.get_device_properties(0)
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
            for name in ("benchmarks/compare_devices.py", "benchmarks/compare_ada.py", "benchmarks/reference.py")
        },
        "scope": "Current main CuTe vs actual default SDPA, B1/H16 contiguous BHSD, FP16/BF16 dense/causal. Resident GPU work through warmed CUDA graphs; forward retains autograd and joint includes all three gradients. CPU dispatch, compilation and full-model costs excluded.",
        "protocol": {
            "seed": args.seed,
            "repeats": args.repeats,
            "target_window_ms": args.window_ms,
            "minimum_replays": args.minimum_replays,
            "ordering": "Alternate baseline/candidate order each round.",
            "reference": "Full-length centered FP64 on identical rounded inputs, two-head chunks.",
            "accuracy_gate": "Each output/dQ/dK/dV norm-relative error below 1% BF16 or 0.5% FP16; exact zero when reference norm is zero.",
            "profiler_during_timing": False,
            "telemetry_during_timing": False,
            "kernel_verification": "Separate eager and CUDA-graph profiler passes after uninstrumented timing.",
            "phase_note": "Backward retains a forward graph. Separately timed medians need not sum to joint.",
        },
        "measurements": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.profiles.mkdir(parents=True, exist_ok=True)
    backends = {"sdpa": torch.nn.functional.scaled_dot_product_attention, "cute": scaled_dot_product_attention}
    phases = ("forward", "backward", "joint")
    original_compile = kernels.compile_kernel
    compiled = []

    def compile_and_retain(*a, **kw):
        function = original_compile(*a, **kw)
        compiled.append(function)
        return function

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
                                (q[:, head : head + 2], k[:, head : head + 2], v[:, head : head + 2]),
                                dout[:, head : head + 2],
                                causal=causal,
                            )
                            for head in range(0, 16, 2)
                        ]
                        reference = tuple(torch.cat([part[i] for part in parts], dim=1) for i in range(4))
                        del parts
                        graphs, storage, accuracy = {}, {}, {}
                        for backend, function in backends.items():
                            actual = evaluate(function, (q, k, v), dout, None, causal)
                            errors = [metrics(got, expected) for got, expected in zip(actual, reference)]
                            for error in errors:
                                threshold = 0.01 if dtype == torch.bfloat16 else 0.005
                                if (
                                    not error["finite"]
                                    or (error["reference_norm"] and error["relative_error"] >= threshold)
                                    or (not error["reference_norm"] and error["max_absolute_error"] != 0)
                                ):
                                    raise AssertionError(f"Independent FP64 gate failed for {backend}.")
                            accuracy[backend] = errors
                            for phase in phases:
                                graphs[backend, phase], storage[backend, phase] = capture(
                                    function, (q, k, v), dout, causal, phase
                                )
                        iterations = {
                            key: max(args.minimum_replays, min(10000, int(args.window_ms / max(measure(graph, 10), 0.001))))
                            for key, graph in graphs.items()
                        }
                        samples = {backend: {phase: [] for phase in phases} for backend in backends}
                        for repeat in range(args.repeats):
                            order = ("sdpa", "cute") if repeat % 2 == 0 else ("cute", "sdpa")
                            for backend in order:
                                for phase in phases:
                                    graph = graphs[backend, phase]
                                    for _ in range(8):
                                        graph.replay()
                                    samples[backend][phase].append(measure(graph, iterations[backend, phase]))
                        symbols = {}
                        for backend, function in backends.items():
                            label = f"{dtype_name}-s{sequence}-d{dimension}-{mode}-{backend}"
                            symbols[backend] = {
                                "eager": launched_kernels(
                                    lambda: evaluate(function, (q, k, v), dout, None, causal),
                                    args.profiles / f"{label}-eager.json",
                                ),
                                "graph": launched_kernels(
                                    graphs[backend, "joint"].replay, args.profiles / f"{label}-graph.json"
                                ),
                            }
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
                            "graph_checks_passed": True,
                            "changed_input_forward_and_joint_graphs_checked": True,
                            "actual_launched_kernels": symbols,
                        }
                        report["measurements"].append(row)
                        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                        print(json.dumps({key: row[key] for key in ("shape", "dtype", "mode", "medians_ms")}), flush=True)
                        del graphs, storage, actual, reference, q, k, v, dout
                        gc.collect()
    report["device"]["board_after"] = board_state()
    report["stack"]["loaded_libraries"] = loaded_libraries()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
