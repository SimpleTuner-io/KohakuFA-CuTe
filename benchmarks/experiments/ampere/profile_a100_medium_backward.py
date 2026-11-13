import argparse
import cProfile
import ctypes
import json
import pstats
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import torch

from benchmarks.experiments.ampere.a100_medium_group_controls import make_attention, make_direct, schedule
from benchmarks.experiments.ampere.validate_sm80_static import assess

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
p.add_argument("--profile", action="store_true")
args = p.parse_args()
root = args.output
root.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(4)
torch.manual_seed(839761)
inputs = [torch.randn(1, 8, 2048, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
dout = torch.randn_like(inputs[0])
functions = {
    "g1_api": make_attention(group=1),
    "g4_api": make_attention(group=4),
    "g1_runner": make_direct(group=1),
    "g4_runner": make_direct(group=4),
}
operations = {}
owners = []
report = {
    "scope": "Backward-only ladder on retained rounded forward outputs. Resident-flat controls omit autograd and allocation, include prepare/main/post. Diagnostic boundary only.",
    "rows": [],
    "optional_flat": {},
}
for name, function in functions.items():
    out = function(*inputs, is_causal=True)
    owners.append(out)
    operations[name] = lambda output=out: torch.autograd.grad(output, inputs, dout, retain_graph=True)
try:
    from kohakufa_cute.fa4_sm80._native import compile_kernel

    assert torch.cuda.current_stream().cuda_stream == 0, "Resident flat probe requires default stream in this diagnostic."
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
        flat_output = functions["g1_api"](*inputs, is_causal=True)
    owners.append((flat_output, saved))
    q, k, v, out, maximum, logsum = [t.detach() for t in saved]
    rounded = 2048
    delta = torch.empty(1, 8, rounded, device="cuda")
    lognorm = torch.empty_like(delta)
    scratch = torch.empty(1, 8, rounded * 128, device="cuda")
    dq, dk, dv = [torch.empty_like(q) for _ in range(3)]
    tensors = (q, k, v, q, out, maximum, logsum, dout.detach(), delta, dq, dk, dv, scratch, lognorm)
    for group in (1, 4):
        compiled = compile_kernel("backward", tensors, True, asdict(schedule(group)))
        ffi = compiled.__tvm_ffi_object__()

        def call(function=ffi):
            function(*tensors, 128**-0.5)
            return dq, dk, dv

        call()
        torch.cuda.synchronize()
        errors = assess(
            tuple(t.detach().cpu() for t in (out, dq, dk, dv)),
            tuple(t.detach().cpu() for t in inputs),
            dout.cpu(),
            causal=True,
            scale=None,
            directed=False,
            pointwise=False,
        )
        operations["g" + str(group) + "_resident_flat"] = call
        owners.append(compiled)
        report["optional_flat"][str(group)] = {"status": "pass", "errors": errors}
except Exception as error:
    report["optional_flat"]["setup"] = {
        "status": "unsupported_or_failed",
        "type": type(error).__name__,
        "message": str(error),
    }
for operation in operations.values():
    for _ in range(20):
        values = operation()
torch.cuda.synchronize()
if args.profile:
    torch.cuda.cudart().cudaProfilerStart()
    try:
        labels = list(operations)
        for repeat in range(3):
            for name in labels[repeat:] + labels[:repeat]:
                torch.cuda.nvtx.range_push(name + ":window:" + str(repeat))
                for i in range(10):
                    torch.cuda.nvtx.range_push(name + ":call:" + str(i))
                    values = operations[name]()
                    torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
    finally:
        torch.cuda.cudart().cudaProfilerStop()
else:
    labels = list(operations)
    samples = {name: [] for name in labels}
    counts = {}
    for name, operation in operations.items():
        torch.cuda.synchronize()
        begin = time.perf_counter()
        for _ in range(20):
            values = operation()
        torch.cuda.synchronize()
        counts[name] = max(20, int(0.1 / ((time.perf_counter() - begin) / 20)))
    for repeat in range(9):
        for name in labels[repeat % len(labels) :] + labels[: repeat % len(labels)]:
            torch.cuda.synchronize()
            begin = time.perf_counter()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(counts[name]):
                values = operations[name]()
            end.record()
            end.synchronize()
            samples[name].append(
                {
                    "wall_ms": (time.perf_counter() - begin) * 1000 / counts[name],
                    "event_ms": start.elapsed_time(end) / counts[name],
                }
            )
    report["samples"] = samples
    report["calls_per_window"] = counts
    for name in labels:
        profiler = cProfile.Profile()
        profiler.enable()
        for _ in range(100):
            values = operations[name]()
        torch.cuda.synchronize()
        profiler.disable()
        profiler.dump_stats(str(root / (name + ".prof")))
        with (root / (name + ".txt")).open("w") as log:
            pstats.Stats(profiler, stream=log).strip_dirs().sort_stats("cumtime").print_stats(35)
        report["rows"].append(
            {
                "label": name,
                "median_ms": {key: statistics.median(x[key] for x in samples[name]) for key in ("wall_ms", "event_ms")},
            }
        )
report["status"] = "pass"
(root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
