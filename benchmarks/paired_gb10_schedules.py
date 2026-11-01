"""Alternate complete GB10 M64/M32 calls and standalone FA2 in one process."""

import argparse
import hashlib
import itertools
import json
import math
import statistics
import time
from pathlib import Path

import torch

from benchmarks.experimental_gb10_shared import make_attention, make_public_attention
from benchmarks.flashattention_backends import load_backend
from benchmarks.validate_sm80_static import assess


def eager_call(function, operands, dout, *, causal, phase):
    """Build the complete call used by eager timing with explicit visibility."""
    saved = function(*operands, is_causal=causal) if phase == "backward" else None

    def execute():
        out = saved if phase == "backward" else function(*operands, is_causal=causal)
        return out, *torch.autograd.grad(out, operands, dout, retain_graph=phase == "backward")

    return execute


def graph_call(function, inputs, dout, *, causal, phase, capture_function=None):
    """Forward the same call contract to the existing graph capture helper."""
    if capture_function is None:
        from benchmarks.compare_ada import capture

        capture_function = capture
    return capture_function(function, inputs, dout, causal, phase, keep_graph=True)


def main():
    from benchmarks.compare_ada import measure
    from benchmarks.compare_l40s import board_state, graph_kernels, loaded_libraries

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq", nargs="+", type=int, default=[512, 1024, 4096])
    p.add_argument("--heads", nargs="+", type=int, default=[4, 16])
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--window-ms", type=float, default=100)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--controls", action="store_true")
    p.add_argument("--public", action="store_true", help="Include the actual bounded public API and native controls.")
    p.add_argument("--causal", action="store_true")
    args = p.parse_args()
    functions = {"m64": make_attention(), "m32": make_attention(dense_m32=True), "fa2": load_backend("fa2")}
    if args.controls:
        functions.update(m64_q1_vregs=make_attention(query_stages=1), m64_q1_sharedv=make_attention(stage1_shared_v=True))
    if args.public:
        functions.update(public=make_public_attention(), cute=load_backend("cute"), sdpa=load_backend("sdpa"))
    report = {
        "scope": "Complete eager wall time and captured device time, alternating identical rounded inputs.",
        "rows": [],
        "telemetry": [],
        "device": torch.cuda.get_device_name(),
        "source_sha256": {
            str(f.relative_to(Path(__file__).parent)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in (Path(__file__), Path(__file__).with_name("experimental_gb10_shared.py"))
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for sequence, heads, dtype in itertools.product(args.seq, args.heads, (torch.bfloat16, torch.float16)):
        torch.manual_seed(2031)
        inputs = tuple(torch.randn(1, heads, sequence, 128, device="cuda", dtype=dtype) for _ in range(3))
        dout = torch.randn_like(inputs[0])
        errors = {}
        for name, function in functions.items():
            operands = tuple(t.detach().requires_grad_(True) for t in inputs)
            out = function(*operands, is_causal=args.causal)
            actual = out, *torch.autograd.grad(out, operands, dout)
            errors[name] = assess(actual, operands, dout, causal=args.causal, scale=None, directed=False, pointwise=False)
        for phase in ("backward", "joint"):
            graphs, owners, eager = {}, {}, {}
            for name, function in functions.items():
                graphs[name], owners[name] = graph_call(function, inputs, dout, causal=args.causal, phase=phase)
                operands = tuple(t.detach().requires_grad_(True) for t in inputs)
                eager[name] = eager_call(function, operands, dout, causal=args.causal, phase=phase)
            for execution in eager.values():
                for _ in range(4):
                    execution()
            torch.cuda.synchronize()
            for timing in ("graph", "eager"):
                samples = {name: [] for name in functions}
                order = list(functions)
                for repeat in range(args.repeats):
                    for name in order[repeat % len(order) :] + order[: repeat % len(order)]:
                        if timing == "graph":
                            pilot = measure(graphs[name], 16)
                            count = max(16, min(8192, math.ceil(args.window_ms / pilot)))
                            sample = measure(graphs[name], count)
                        else:
                            torch.cuda.synchronize()
                            start = time.perf_counter()
                            for _ in range(4):
                                eager[name]()
                            torch.cuda.synchronize()
                            pilot = (time.perf_counter() - start) * 250
                            count = max(4, min(8192, math.ceil(args.window_ms / pilot)))
                            start = time.perf_counter()
                            for _ in range(count):
                                eager[name]()
                            torch.cuda.synchronize()
                            sample = (time.perf_counter() - start) * 1000 / count
                        samples[name].append(sample)
                    report["telemetry"].append(
                        {
                            "sequence": sequence,
                            "heads": heads,
                            "dtype": str(dtype),
                            "timing": timing,
                            "phase": phase,
                            "repeat": repeat,
                            "board": board_state(),
                            "cpu_load": Path("/proc/loadavg").read_text().strip(),
                        }
                    )
                row = {
                    "shape": list(inputs[0].shape),
                    "causal": args.causal,
                    "dtype": str(dtype),
                    "phase": phase,
                    "timing": timing,
                    "samples_ms": samples,
                    "median_ms": {n: statistics.median(v) for n, v in samples.items()},
                    "errors": errors,
                    "resources": {n: graph_kernels(g) for n, g in graphs.items()},
                }
                report["rows"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print({k: row[k] for k in ("shape", "dtype", "phase", "timing", "median_ms")}, flush=True)
            del graphs, owners, eager
    report["loaded_libraries"] = loaded_libraries()
    report["status"] = "pass"
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
