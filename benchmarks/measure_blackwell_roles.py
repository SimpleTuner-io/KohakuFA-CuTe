"""Measure sampled Blackwell role timelines with an uninstrumented control.

Use instrument_blackwell_roles.py first. D512 dense BF16 on sm100/sm103 is the
supported experiment. CUDA-event timings quantify observer overhead. Raw role
records are low-32-bit globaltimer nanoseconds and static source event IDs.
"""

import argparse
import hashlib
import json
import statistics
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

import torch

from benchmarks.compare_blackwell_codegen import load_source
from benchmarks.precision import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrumented-source", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.instrumented_source.with_suffix(".metadata.json").read_text())
    if hashlib.sha256(args.instrumented_source.read_bytes()).hexdigest() != metadata["instrumented_sha256"]:
        parser.error("Instrumented source does not match its metadata.")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        parser.error("Use Blackwell sm100 or sm103.")
    if args.seq % 128 or args.seq // 128 <= metadata["sample_step"] + 2:
        parser.error("Use a multiple of 128 with at least one full tile after the sampled iterations.")
    if min(args.heads, args.samples, args.iterations) < 1 or args.samples < 2:
        parser.error("Use positive heads/iterations and at least two samples.")
    from kohakufa_cute import blackwell_pipeline, kernels, native

    control = load_source(args.baseline_source)
    instrumented = load_source(args.instrumented_source)
    modules = {"control": control, "instrumented": instrumented}
    caches = {name: lru_cache(maxsize=128)(kernels._compile_kernel.__wrapped__) for name in modules}
    extension = native.extension()

    @contextmanager
    def context(name):
        extension.clear_cache()
        module = modules[name]
        try:
            with (
                patch.multiple(blackwell_pipeline, Forward=control.Forward, Query=module.Query, KeyValue=module.KeyValue),
                patch.object(kernels, "_compile_kernel", caches[name]),
            ):
                yield
        finally:
            extension.clear_cache()

    torch.manual_seed(21)
    q, k, v, dout = [torch.randn(1, args.heads, args.seq, 512, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
    scale = 512**-0.5
    with context("control"):
        out, maximum, logsum, blocks = extension.forward(q, k, v, None, scale, False)
    profile = torch.zeros(
        (
            2,
            (args.seq // 128 + metadata["tile_stride"] - 1) // metadata["tile_stride"],
            len(metadata["roles"]),
            metadata["capacity"],
            2,
        ),
        device="cuda",
        dtype=torch.int32,
    )
    reference = evaluate(torch.nn.functional.scaled_dot_product_attention, (q, k, v), dout, None, False)
    properties = torch.cuda.get_device_properties(0)
    report = {
        "device": properties.name,
        "multiprocessors": properties.multi_processor_count,
        "capability": list(torch.cuda.get_device_capability()),
        "shape": list(q.shape),
        "control_sha256": hashlib.sha256(args.baseline_source.read_bytes()).hexdigest(),
        "trace_metadata": metadata,
        "scope": "Sampled role-leader intervals including instrumentation overhead. Asynchronous issue/enqueue intervals are not accelerator execution durations. First raw sample is warmup.",
        "samples": [],
        "timings": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    expected = None
    try:
        for name in modules:
            with context(name):
                workspace = blocks if name == "control" else profile

                def call():
                    return extension.backward(q, k, v, None, maximum, logsum, dout, workspace, out, scale, False)

                actual = call()
                torch.cuda.synchronize()
                for got, want in zip((out, *actual), reference):
                    torch.testing.assert_close(got, want, atol=0.03, rtol=0.03)
                if name == "control":
                    expected = [tensor.clone() for tensor in actual]
                else:
                    for got, want in zip(actual, expected):
                        torch.testing.assert_close(got, want, atol=0, rtol=0)
                for _ in range(2):
                    call()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = call()
                graph.replay()
                torch.cuda.synchronize()
                for got, want in zip(actual, expected):
                    torch.testing.assert_close(got, want, atol=0, rtol=0)
                start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                timings = []
                for _ in range(4):
                    start.record()
                    for _ in range(args.iterations):
                        graph.replay()
                    end.record()
                    torch.cuda.synchronize()
                    timings.append(start.elapsed_time(end) / args.iterations)
                report["timings"][name] = {"backward_ms": timings, "median_ms": statistics.median(timings)}
                print(json.dumps({"backend": name, **report["timings"][name]}), flush=True)
                if name == "control":
                    continue
                for sample in range(args.samples):
                    profile.zero_()
                    actual = call()
                    torch.cuda.synchronize()
                    for got, want in zip(actual, expected):
                        torch.testing.assert_close(got, want, atol=0, rtol=0)
                    raw = profile.cpu().to(torch.int64)
                    rows = []
                    for mode in range(2):
                        for tile in range(profile.shape[1]):
                            for role in range(profile.shape[2]):
                                data = raw[mode, tile, role]
                                data = data[data[:, 0] != 0]
                                if len(data) == 0:
                                    if role != 4:
                                        raise RuntimeError(("Missing role trace", mode, tile, role))
                                    continue
                                data[:, 0] &= 0xFFFFFFFF
                                for index in range(1, len(data)):
                                    if data[index, 0] < data[index - 1, 0]:
                                        data[index:, 0] += 2**32
                                if int((data[:, 1] == 0).sum()) != 2 or int((data[:, 1] == 1).sum()) != 2:
                                    raise RuntimeError(("Incomplete/overflowing step records", mode, tile, role))
                                rows.append(
                                    {
                                        "mode": ("q", "kv")[mode],
                                        "tile": tile * metadata["tile_stride"],
                                        "role": role,
                                        "events": data.tolist(),
                                    }
                                )
                    report["samples"].append({"index": sample, "traces": rows})
                    args.output.write_text(json.dumps(report) + "\n")
        report["observer_overhead_fraction"] = (
            report["timings"]["instrumented"]["median_ms"] / report["timings"]["control"]["median_ms"] - 1
        )
        report.update(passed=True, bitwise_control=True, sdpa_checked=True, graph_bitwise_control=True)
        args.output.write_text(json.dumps(report) + "\n")
    finally:
        extension.clear_cache()
        for cache in caches.values():
            cache.cache_clear()


if __name__ == "__main__":
    main()
