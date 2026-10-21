"""Attribute attention GPU time to individual launches before changing its pipeline."""

import argparse
import hashlib
import json
import statistics
from contextlib import nullcontext
from pathlib import Path

import torch

from benchmarks.precision import evaluate, load_upstream_tuning, upstream_configurations, upstream_provenance
from kohakufa_cute import scaled_dot_product_attention


def launch_report(events, iterations):
    """Retain launch order as different CuTe stages can share a kernel name."""
    events = sorted(
        (
            event
            for event in events
            if event.device_type == torch.autograd.DeviceType.CUDA and not getattr(event, "is_user_annotation", False)
        ),
        key=lambda event: event.time_range.start,
    )
    if not events:
        raise RuntimeError("No CUDA activities were recorded; check that CUPTI profiling is available.")
    launches = [
        dict(name=event.name, duration_us=event.device_time_total, start_us=event.time_range.start) for event in events
    ]
    width = len(launches) // iterations
    uniform = (
        width > 0
        and len(launches) % iterations == 0
        and all(
            [launch["name"] for launch in launches[start : start + width]] == [launch["name"] for launch in launches[:width]]
            for start in range(0, len(launches), width)
        )
    )
    slots = []
    if uniform:
        for slot in range(width):
            samples = [launches[start + slot]["duration_us"] for start in range(0, len(launches), width)]
            slots.append(
                dict(
                    slot=slot,
                    name=launches[slot]["name"],
                    median_us=statistics.median(samples),
                    samples_us=samples,
                )
            )
    return dict(
        gpu_ms_per_iteration=sum(launch["duration_us"] for launch in launches) / iterations / 1000,
        uniform_launch_pattern=uniform,
        launch_slots=slots,
        launches=launches,
    )


def profile_phase(run, *, backend, phase, iterations, warmup, profiler, trace_dir, execution="eager", stream=None):
    """Use one stream for both phases so retained autograd nodes keep their stream."""
    if execution == "graph" and stream is None:
        raise ValueError("Graph profiling requires a shared forward/backward capture stream.")
    if stream is not None:
        stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream) if stream is not None else nullcontext():
        return _profile_on_stream(
            run,
            backend=backend,
            phase=phase,
            iterations=iterations,
            warmup=warmup,
            profiler=profiler,
            trace_dir=trace_dir,
            execution=execution,
            stream=stream,
        )


def _profile_on_stream(run, *, backend, phase, iterations, warmup, profiler, trace_dir, execution, stream):
    graph = None
    for _ in range(warmup):
        result = run()
    torch.cuda.synchronize()
    if execution == "graph":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured_result = run()

        def replay():
            graph.replay()
            return captured_result

        run = replay
        for _ in range(warmup):
            result = run()
        torch.cuda.synchronize()
    label = f"{backend}_{phase}"
    context = (
        torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
        if profiler == "torch"
        else nullcontext()
    )
    with context as profile:
        for _ in range(iterations):
            with torch.cuda.nvtx.range(label), torch.profiler.record_function(label):
                result = run()
        torch.cuda.synchronize()
    report = dict(phase=phase, nvtx_range=label, execution=execution)
    if profile is not None:
        report.update(launch_report(profile.events(), iterations))
        if trace_dir is not None:
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace = trace_dir / f"{label}.json"
            profile.export_chrome_trace(str(trace))
            report["trace"] = str(trace)
    # Keep graph-owned output storage alive until the caller finishes validation.
    return result, report, graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--backends", nargs="+", choices=("cute", "kohakufa", "sdpa"), default=["cute", "kohakufa"])
    parser.add_argument("--kohakufa-tuning", type=Path, nargs="+")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--profiler", choices=("torch", "nvtx"), default="torch")
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--execution", choices=("eager", "graph"), default="eager")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.seq, args.dim, args.heads, args.iterations, args.warmup) < 1:
        parser.error("Shapes, iterations and warmup must be positive.")
    if args.dim > 512:
        parser.error("Head dimensions must be at most 512.")
    if not torch.cuda.is_available():
        parser.error("Profiling requires CUDA.")
    if args.profiler == "nvtx" and args.trace_dir is not None:
        parser.error("NVTX-only mode exports markers for external tools; omit --trace-dir.")
    if args.kohakufa_tuning and "kohakufa" not in args.backends:
        parser.error("Upstream tuning tables require the kohakufa backend.")
    tuning = load_upstream_tuning(args.kohakufa_tuning)
    functions = dict(cute=scaled_dot_product_attention, sdpa=torch.nn.functional.scaled_dot_product_attention)
    if "kohakufa" in args.backends:
        from kohakufa import attention

        def upstream(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
            return attention(q, k, v, mask=attn_mask, causal=is_causal, scale=scale)

        functions["kohakufa"] = upstream
    torch.manual_seed(21)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    inputs = [torch.randn(1, args.heads, args.seq, args.dim, device="cuda", dtype=dtype) for _ in range(4)]
    reference = evaluate(functions["sdpa"], inputs[:3], inputs[3], None, args.causal)
    report = dict(
        device=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        shape=list(inputs[0].shape),
        dtype=args.dtype,
        causal=args.causal,
        iterations=args.iterations,
        warmup=args.warmup,
        profiler=args.profiler,
        execution=args.execution,
        cute_source_sha256={
            str(path.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((Path(__file__).resolve().parents[1] / "src" / "kohakufa_cute").glob("*"))
            if path.suffix in (".py", ".cpp")
        },
        protocol=f"{args.execution.capitalize()} launch attribution with profiling overhead; use benchmarks.speed for comparative timings.",
        backward_protocol="One retained forward graph; each backward computes all Q/K/V gradients without accumulation.",
        upstream_tuning=tuning,
        upstream_source=upstream_provenance() if "kohakufa" in args.backends else None,
        backends={},
    )
    for backend in args.backends:
        function = functions[backend]
        actual = evaluate(function, inputs[:3], inputs[3], None, args.causal)
        tolerance = 0.03 if dtype == torch.bfloat16 else 0.005
        for got, want in zip(actual, reference):
            torch.testing.assert_close(got, want, atol=tolerance, rtol=tolerance)
        del actual
        q, k, v = [tensor.detach().clone().requires_grad_(True) for tensor in inputs[:3]]

        def forward():
            return function(q, k, v, is_causal=args.causal)

        capture_stream = torch.cuda.Stream() if args.execution == "graph" else None
        out, forward_report, forward_graph = profile_phase(
            forward,
            backend=backend,
            phase="forward",
            iterations=args.iterations,
            warmup=args.warmup,
            profiler=args.profiler,
            trace_dir=args.trace_dir,
            execution=args.execution,
            stream=capture_stream,
        )

        def backward():
            return torch.autograd.grad(out, (q, k, v), inputs[3], retain_graph=True)

        grads, backward_report, backward_graph = profile_phase(
            backward,
            backend=backend,
            phase="backward",
            iterations=args.iterations,
            warmup=args.warmup,
            profiler=args.profiler,
            trace_dir=args.trace_dir,
            execution=args.execution,
            stream=capture_stream,
        )
        for got, want in zip((out, *grads), reference):
            torch.testing.assert_close(got, want, atol=tolerance, rtol=tolerance)
        if backend == "cute":
            from kohakufa_cute import kernels

            descriptors = tuple(kernels.descriptor(t) for t in (q, k, v, q, q, q, q, inputs[3]))
            configurations = {
                mode: list(kernels.kernel_configuration(q.device.index, mode, descriptors))
                for mode in ("forward", "q", "kv")
            }
            configurations["q"] = list(kernels.backward_query_configuration(q.device.index, args.causal, False, descriptors))
        else:
            configurations = upstream_configurations(q, k, args.causal) if backend == "kohakufa" else None
        report["backends"][backend] = dict(
            validated=True, configurations=configurations, forward=forward_report, backward=backward_report
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"{backend}: forward/backward attribution recorded", flush=True)
        del out, grads, forward_graph, backward_graph, q, k, v


if __name__ == "__main__":
    main()
