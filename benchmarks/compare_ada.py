"""Compare RTX 4090 scheduling with its preceding CuTe path and native SDPA."""

import argparse
import hashlib
import importlib.metadata
import json
import statistics
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch

from benchmarks.precision import evaluate
from benchmarks.reference import fp64_reference, metrics
from kohakufa_cute import ada, kernels, native, scaled_dot_product_attention


def capture(function, inputs, dout, causal, phase):
    operands = [tensor.clone().requires_grad_(True) for tensor in inputs]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        saved = function(*operands, is_causal=causal) if phase == "backward" else None

        def run():
            if phase == "backward":
                return torch.autograd.grad(saved, operands, dout, retain_graph=True)
            output = function(*operands, is_causal=causal)
            return (output, *torch.autograd.grad(output, operands, dout)) if phase == "joint" else (output,)

        for _ in range(3):
            expected = run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = run()
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    torch.cuda.synchronize()
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=1e-5, rtol=0.03)
    if phase != "backward":
        with torch.no_grad():
            for tensor in operands:
                tensor.add_(0.01)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            changed = run()
        torch.cuda.current_stream().wait_stream(stream)
        graph.replay()
        torch.cuda.synchronize()
        for got, want in zip(actual, changed):
            torch.testing.assert_close(got, want, atol=1e-5, rtol=0.03)
        with torch.no_grad():
            for tensor, original in zip(operands, inputs):
                tensor.copy_(original)
    graph.replay()
    torch.cuda.synchronize()
    return graph, (operands, dout, saved, actual)


def measure(graph, iterations):
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) / iterations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", nargs="+", type=int, default=[1024, 4096])
    parser.add_argument("--dim", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--dtype", nargs="+", choices=("fp16", "bf16"), default=["bf16", "fp16"])
    parser.add_argument("--modes", nargs="+", choices=("dense", "causal"), default=["dense", "causal"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--window-ms", type=int, default=200)
    parser.add_argument("--minimum-replays", type=int, default=16)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument(
        "--control",
        choices=("previous", "unpacked"),
        default="previous",
        help="Use original scheduling or checked D128 causal backward as the CuTe control.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_name() != "NVIDIA GeForce RTX 4090":
        parser.error("This comparison requires an RTX 4090.")
    if min(*args.seq, *args.dim, args.repeats, args.window_ms, args.minimum_replays) < 1:
        parser.error("Shapes and replay settings must be positive.")
    backends = ("sdpa", args.control, "cute")
    phases = ("forward", "backward", "joint")
    functions = dict(sdpa=torch.nn.functional.scaled_dot_product_attention)
    compiled = []
    original_compile = kernels.compile_kernel
    original_schedule = ada.schedule

    def unpacked_schedule(program, mode, dimension, causal):
        result = original_schedule(program, mode, dimension, causal)
        if mode == "backward" and dimension == 128 and causal:
            result.tuned = result.tuned.checked
        return result

    def compile_and_retain(*a, **kw):
        function = original_compile(*a, **kw)
        compiled.append(function)
        return function

    report = {
        "device": torch.cuda.get_device_name(),
        "multiprocessors": torch.cuda.get_device_properties(0).multi_processor_count,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "versions": {
            package: importlib.metadata.version(package) for package in ("nvidia-cutlass-dsl", "apache-tvm-ffi", "triton")
        },
        "source_sha256": {
            f"src/kohakufa_cute/{path.name}": hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(kernels.__file__).parent.iterdir())
            if path.suffix in (".py", ".cpp")
        },
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "control_backend": args.control,
        "protocol": {
            "seed": args.seed,
            "repeats": args.repeats,
            "target_window_ms": args.window_ms,
            "minimum_replays": args.minimum_replays,
            "ordering": "Rotate SDPA, preceding CuTe and scheduled CuTe each round.",
            "timing": "CUDA events around warmed graph replays; milliseconds per operation.",
            "forward": "Retains autograd; includes preparations required by the API.",
            "backward": "Retained forward; computes all three gradients without accumulation.",
            "joint": "Includes forward and all three gradients.",
            "control": (
                "Disable only the RTX 4090 scheduling selector on the same sources and stack."
                if args.control == "previous"
                else "Use checked long D128 causal backward; retain selected forward, other policies and the same native workspace allocation."
            ),
            "profiler_during_timing": False,
            "telemetry_during_timing": False,
            "compilation_excluded": True,
        },
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with patch.object(kernels, "compile_kernel", compile_and_retain):
        for dtype_name in args.dtype:
            for sequence in args.seq:
                for dimension in args.dim:
                    for mode in args.modes:
                        dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
                        torch.manual_seed(args.seed)
                        q, k, v, dout = [
                            torch.randn(1, 16, sequence, dimension, device="cuda", dtype=dtype) for _ in range(4)
                        ]
                        causal = mode == "causal"
                        reference = evaluate(functions["sdpa"], (q, k, v), dout, None, causal)
                        # Keep the full-length reference within a consumer GPU's
                        # memory budget without changing any per-head equations.
                        parts = [
                            fp64_reference(
                                (q[:, head : head + 2], k[:, head : head + 2], v[:, head : head + 2]),
                                dout[:, head : head + 2],
                                causal=causal,
                            )
                            for head in range(0, q.shape[1], 2)
                        ]
                        mathematical = tuple(torch.cat([part[i] for part in parts], dim=1) for i in range(4))
                        del parts
                        accuracy = {}
                        graphs, storage, values, exact = {}, {}, {}, {}
                        for backend in backends:
                            native.extension().clear_cache()
                            kernels._compile_kernel.cache_clear()
                            context = (
                                patch.object(kernels, "_rtx4090_tuning", return_value=False)
                                if backend == "previous"
                                else (
                                    patch.object(ada, "schedule", unpacked_schedule)
                                    if backend == "unpacked"
                                    else nullcontext()
                                )
                            )
                            with context:
                                function = functions.get(backend, scaled_dot_product_attention)
                                values[backend] = evaluate(function, (q, k, v), dout, None, causal)
                                if mathematical is not None:
                                    accuracy[backend] = [
                                        metrics(got, want) for got, want in zip(values[backend], mathematical)
                                    ]
                                    for error in accuracy[backend]:
                                        threshold = 0.01 if dtype == torch.bfloat16 else 0.005
                                        if (
                                            not error["finite"]
                                            or (error["reference_norm"] and error["relative_error"] >= threshold)
                                            or (not error["reference_norm"] and error["max_absolute_error"] != 0)
                                        ):
                                            raise AssertionError("Independent FP64 accuracy gate failed.")
                                for got, want in zip(values[backend], reference):
                                    tolerance = 0.03 if dtype == torch.bfloat16 else 0.005
                                    torch.testing.assert_close(got, want, atol=tolerance, rtol=tolerance)
                                if backend == "cute":
                                    exact = [
                                        torch.equal(got.view(torch.uint8), want.view(torch.uint8))
                                        for got, want in zip(values[backend], values[args.control])
                                    ]
                                for phase in phases:
                                    key = (backend, phase)
                                    graphs[key], storage[key] = capture(function, (q, k, v), dout, causal, phase)
                        iterations = {
                            key: max(args.minimum_replays, min(10000, int(args.window_ms / max(measure(graph, 10), 0.001))))
                            for key, graph in graphs.items()
                        }
                        samples = {backend: {phase: [] for phase in phases} for backend in backends}
                        for repeat in range(args.repeats):
                            offset = repeat % len(backends)
                            for backend in backends[offset:] + backends[:offset]:
                                for phase in phases:
                                    key = (backend, phase)
                                    for _ in range(8):
                                        graphs[key].replay()
                                    samples[backend][phase].append(measure(graphs[key], iterations[key]))
                        row = {
                            "shape": list(q.shape),
                            "dtype": dtype_name,
                            "mode": mode,
                            "samples_ms": samples,
                            "medians_ms": {
                                backend: {phase: statistics.median(times) for phase, times in values.items()}
                                for backend, values in samples.items()
                            },
                            "iterations": {
                                backend: {phase: iterations[backend, phase] for phase in phases} for backend in backends
                            },
                            "graph_checks_passed": True,
                            "changed_input_forward_and_joint_graphs_checked": True,
                            f"byte_equal_{args.control}_out_dq_dk_dv": exact,
                            "fp64_metrics_out_dq_dk_dv": accuracy,
                            f"requires_byte_equal_{args.control}": False,
                            "precision_contract": "Independent full-length FP64; each output/gradient norm error below 1% BF16 or 0.5% FP16; exact zero when the reference norm is zero.",
                        }
                        report["rows"].append(row)
                        args.output.write_text(json.dumps(report, indent=2) + "\n")
                        print(json.dumps({key: row[key] for key in ("shape", "dtype", "mode", "medians_ms")}), flush=True)
                        del graphs, storage, values, reference


if __name__ == "__main__":
    main()
