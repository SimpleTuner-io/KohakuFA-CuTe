"""Complete eager inference calls: corrected 3D attention versus LTX's SDPA helper.

The external helper is supplied as a frozen source file. Raw timings, traces,
telemetry, and failures belong outside the tracked repository. This times the
attention operation including its preparation, not a complete decoder.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import random
import subprocess
import time
from itertools import product
from pathlib import Path

import torch

from benchmarks.neighborhood.benchmark import atomic_save, snapshot_gpu, summarize
from benchmarks.reference import metrics
from kohakufa_cute import neighborhood_attention_3d


def workloads():
    # 17-frame 512x512 output: 3x16x16 latent, two trailing latent frames,
    # temporal upsampling drops one leading frame at each factor-two stage.
    return [
        dict(name="stage1", spatial=(5, 16, 16), heads=32, window=(3, 7, 7), blocks=4),
        dict(name="stage2", spatial=(5, 32, 32), heads=16, window=(3, 7, 7), blocks=6),
        dict(name="stage3", spatial=(9, 32, 32), heads=8, window=(3, 5, 5), blocks=4),
        dict(name="stage4", spatial=(17, 64, 64), heads=8, window=(3, 5, 5), blocks=2),
        dict(name="stage5", spatial=(17, 128, 128), heads=4, window=(11, 11, 11), blocks=8),
        dict(name="stage5-49f-768x512", spatial=(49, 128, 192), heads=4, window=(11, 11, 11), blocks=None),
    ]


def heldout_workloads():
    """Independent geometry checks for the large-window scheduling family."""
    return [
        dict(name="heldout-13f", spatial=(13, 64, 96), heads=2, window=(11, 11, 11), blocks=None),
        dict(name="heldout-33f", spatial=(33, 96, 128), heads=1, window=(11, 11, 11), blocks=None),
        dict(name="fallback-tail", spatial=(13, 64, 65), heads=2, window=(11, 11, 11), blocks=None),
    ]


def positions(spatial, extra_queries=0, seed=0):
    depth, height, width = spatial
    count = math.prod(spatial)
    generator = random.Random(seed)
    return sorted(
        {0, width - 1, width, 63, 64, height * width - 1, height * width, count // 3, count // 2, count - 1}
        | {generator.randrange(count) for _ in range(extra_queries)}
    )


@torch.no_grad()
def sampled_reference(inputs, spatial, window, *, extra_queries=0, seed=0):
    """Native CPU FP64 at boundary/interior queries; no global edge table."""
    q, k, v = [tensor.flatten(1, 3) for tensor in inputs]
    selected = positions(spatial, extra_queries, seed)
    expected = []
    clipped = [min(size, extent) for size, extent in zip(window, spatial)]
    for index in selected:
        coords = (index // (spatial[1] * spatial[2]), index // spatial[2] % spatial[1], index % spatial[2])
        starts = [
            min(range(extent - size + 1), key=lambda candidate: abs(candidate + size // 2 - position))
            for position, size, extent in zip(coords, clipped, spatial)
        ]
        neighbors = [
            (t * spatial[1] + y) * spatial[2] + x
            for t, y, x in product(*(range(first, first + size) for first, size in zip(starts, clipped)))
        ]
        query = q[:, index].detach().cpu().double()
        keys, values = [tensor[:, neighbors].detach().cpu().double() for tensor in (k, v)]
        scores = torch.einsum("bhd,bnhd->bhn", query, keys)
        probability = (scores - scores.amax(-1, keepdim=True)).softmax(-1)
        expected.append(torch.einsum("bhn,bnhd->bhd", probability, values))
    return selected, torch.stack(expected, dim=1)


def sample(function, iterations):
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    torch.cuda.synchronize()
    before = time.perf_counter()
    start.record()
    for _ in range(iterations):
        output = function()
        del output
    end.record()
    end.synchronize()
    return dict(wall_ms=(time.perf_counter() - before) * 1000 / iterations, event_ms=start.elapsed_time(end) / iterations)


def profile(function, path):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as trace:
        output = function()
        torch.cuda.synchronize()
    del output
    trace.export_chrome_trace(str(path))
    return dict(
        trace_file=path.name,
        kernels=[event.name for event in trace.events() if event.device_type == torch.autograd.DeviceType.CUDA],
        sdpa_operators=sorted({event.key for event in trace.key_averages() if "scaled_dot_product" in event.key}),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ltx-helper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--target-ms", type=float, default=200)
    parser.add_argument("--cases", nargs="+", choices=[case["name"] for case in workloads() + heldout_workloads()])
    parser.add_argument("--dtypes", nargs="+", choices=("float16", "bfloat16"), default=("float16", "bfloat16"))
    parser.add_argument("--seed", type=int, default=521)
    parser.add_argument("--reference-queries", type=int, default=0, help="Extra independently sampled FP64 query positions.")
    parser.add_argument("--include-previous", action="store_true", help="Explicit original 3D C32/single-P forward control.")
    parser.add_argument(
        "--include-generic", action="store_true", help="Explicit generic owner with the current two-term P arithmetic."
    )
    parser.add_argument(
        "--warp-volume-tile", nargs=3, type=int, metavar=("T", "H", "W"), help="Explicit compact warp-MMA tile experiment."
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.output.resolve().is_relative_to(root) or args.samples < 3 or args.target_ms <= 0 or args.reference_queries < 0:
        parser.error("Use an external output directory, at least three samples, and a positive duration.")
    if args.warp_volume_tile and (min(args.warp_volume_tile) < 1 or math.prod(args.warp_volume_tile) not in (16, 32, 64)):
        parser.error("The experimental warp tile must contain 16, 32, or 64 positive-volume rows.")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if revision != args.source_revision or not torch.cuda.is_available():
        parser.error("CUDA and the declared source revision are required.")
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location("frozen_ltx_na", args.ltx_helper)
    ltx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ltx)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "performance.json"
    properties = torch.cuda.get_device_properties(0)
    report = dict(
        status="running",
        source_revision=revision,
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ltx_helper_sha256=hashlib.sha256(args.ltx_helper.read_bytes()).hexdigest(),
        torch_version=torch.__version__,
        cuda=torch.version.cuda,
        gpu=properties.name,
        sm_count=properties.multi_processor_count,
        capability=torch.cuda.get_device_capability(),
        memory_bytes=properties.total_memory,
        contract="B1, D64, contiguous LTX BTHWHD layout, normalized synthetic Q/K, Q pre-scaled by 0.125, scale=1.0, no autograd.",
        timing="Complete warm eager attention calls, including mask construction, stacking, copies, output layout and Python work. Synchronized wall time is primary; CUDA-event spans include host launch gaps. No graph capture or complete decoder timing.",
        precision="3D two-term P forward default; normalized dQ/reconstructed dK are backward work and are not executed here. The optional previous control uses the original C32/single-P forward.",
        seed=args.seed,
        extra_reference_queries=args.reference_queries,
        reference="Independent sampled native CPU FP64 on identical rounded inputs. Per-head output norm gates: FP16 0.5%, BF16 1%.",
        samples=args.samples,
        target_ms=args.target_ms,
        warp_volume_tile=args.warp_volume_tile,
        include_generic=args.include_generic,
        telemetry=[snapshot_gpu()],
        rows=[],
    )
    atomic_save(path, report)
    try:
        with torch.inference_mode():
            for dtype_name in args.dtypes:
                for case in workloads() + (heldout_workloads() if args.cases else []):
                    if args.cases and case["name"] not in args.cases:
                        continue
                    torch.cuda.empty_cache()
                    torch.manual_seed(args.seed)
                    dtype = getattr(torch, dtype_name)
                    shape = (1, *case["spatial"], case["heads"], 64)
                    inputs = [torch.randn(shape, dtype=dtype, device="cuda") for _ in range(3)]
                    for index, factor in ((0, 0.125), (1, 1.0)):
                        source = inputs[index].float()
                        inputs[index] = (source * torch.rsqrt(source.square().mean(-1, keepdim=True) + 1e-6) * factor).to(
                            dtype
                        )
                        del source
                    selected, expected = sampled_reference(
                        inputs, case["spatial"], case["window"], extra_queries=args.reference_queries, seed=args.seed + 1
                    )
                    from kohakufa_cute.neighborhood_kernels import set_experiment_volume

                    def execute(name):
                        selection = None
                        if name in ("previous", "generic_control"):
                            selection = False
                        elif name == "compact_candidate":
                            selection = (tuple(args.warp_volume_tile), False)
                        set_experiment_volume(selection, residual=False if name == "previous" else None)
                        if name == "ltx_helper":
                            return ltx.na3d(*inputs, case["window"], None, 1.0)
                        return neighborhood_attention_3d(*inputs, case["window"], None, 1.0)

                    functions = {
                        "kohakufa": lambda: execute("kohakufa"),
                        "ltx_helper": lambda: execute("ltx_helper"),
                    }
                    if args.include_previous:
                        functions["previous"] = lambda: execute("previous")
                    if args.include_generic:
                        functions["generic_control"] = lambda: execute("generic_control")
                    if args.warp_volume_tile:
                        functions["compact_candidate"] = lambda: execute("compact_candidate")
                    rows = {}
                    counts = {}
                    for name, function in functions.items():
                        row = dict(**case, dtype=dtype_name, implementation=name, status="warming", samples=[])
                        rows[name] = row
                        report["rows"].append(row)
                        atomic_save(path, report)
                        torch.cuda.synchronize()
                        allocated = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                        output = function()
                        torch.cuda.synchronize()
                        row["first_call_incremental_peak_allocated_bytes"] = torch.cuda.max_memory_allocated() - allocated
                        got = output.flatten(1, 3)[:, selected].cpu().double()
                        errors = [metrics(got[0, :, head], expected[0, :, head]) for head in range(case["heads"])]
                        row["accuracy"] = dict(
                            positions=selected,
                            per_head=errors,
                            full_output_finite=bool(torch.isfinite(output).all()),
                            output_rounding_floor=[
                                metrics(expected[0, :, head].to(dtype).double(), expected[0, :, head])
                                for head in range(case["heads"])
                            ],
                        )
                        limit = 0.005 if dtype == torch.float16 else 0.01
                        valid = row["accuracy"]["full_output_finite"] and all(
                            e["finite"] and e["relative_error"] < limit for e in errors
                        )
                        if not valid:
                            torch.save(
                                dict(inputs=[t.cpu() for t in inputs], output=output.cpu()),
                                args.output / (case["name"] + "-" + dtype_name + "-" + name + "-failure.pt"),
                            )
                            row["status"] = "numerical-failure"
                            atomic_save(path, report)
                            raise RuntimeError("Sampled FP64 gate failed.")
                        del output, got
                        # Separate compilation/first allocation from sustained warm timing.
                        sample(function, 2)
                        probe = sample(function, 1)
                        counts[name] = min(10000, max(1, math.ceil(args.target_ms / max(probe["wall_ms"], 0.001))))
                        sample(function, max(1, counts[name] * 2))
                        row["status"] = "measuring"
                        row["iterations_per_sample"] = counts[name]
                    report["telemetry"].append(snapshot_gpu())
                    for repeat in range(args.samples):
                        for name in list(functions) if repeat % 2 == 0 else list(reversed(functions)):
                            rows[name]["samples"].append(sample(functions[name], counts[name]))
                        atomic_save(path, report)
                    report["telemetry"].append(snapshot_gpu())
                    for name, function in functions.items():
                        row = rows[name]
                        row["wall_ms"] = summarize([s["wall_ms"] for s in row["samples"]])
                        row["event_ms"] = summarize([s["event_ms"] for s in row["samples"]])
                        row["profile"] = profile(
                            function, args.output / (case["name"] + "-" + dtype_name + "-" + name + ".trace.json")
                        )
                        row["status"] = "pass"
                        print(case["name"], dtype_name, name, round(row["wall_ms"]["median"], 4), "ms", flush=True)
                        atomic_save(path, report)
                    del functions, inputs, expected
        set_experiment_volume()
        report["status"] = "pass"
    except BaseException as error:
        report.update(status="execution-error", error_type=type(error).__name__)
        atomic_save(path, report)
        raise
    atomic_save(path, report)


if __name__ == "__main__":
    main()
