"""Controlled immutable-device classification probe and complete-call factories."""

import argparse
import cProfile
import hashlib
import importlib.util
import json
import pstats
import statistics
import time
from pathlib import Path


def previous_module(control_file):
    spec = importlib.util.spec_from_file_location("kohakufa_cute.fa4_sm80._gb10_cache_control", control_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_attention(*, previous=False, direct=False, control_file="controls/gb10_before_cache.py"):
    from kohakufa_cute.fa4_sm80 import gb10
    from kohakufa_cute.fa4_sm80._host import native_runner

    module = previous_module(control_file) if previous else gb10
    if not direct:
        return module.scaled_dot_product_attention

    def run(q, k, v, *, is_causal=False, scale=None, enable_gqa=False):
        schedule = gb10.schedule(q.shape[2], is_causal, q.shape[1], q.dtype)
        return native_runner(schedule)(q, k, v, causal=is_causal, scale=scale)

    return run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control-file", type=Path, default=Path("controls/gb10_before_cache.py"))
    p.add_argument("--profile", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import torch

    from benchmarks.flashattention_backends import load_backend
    from kohakufa_cute.fa4_sm80 import gb10

    old = previous_module(args.control_file)
    inputs = tuple(torch.randn(1, 4, 512, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
    dout = torch.randn_like(inputs[0])
    report = {
        "baseline_sha256": hashlib.sha256(args.control_file.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(Path(gb10.__file__).read_bytes()).hexdigest(),
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.profile:
        operations = {
            "uncached": old.scaled_dot_product_attention,
            "cached": gb10.scaled_dot_product_attention,
            "direct": make_attention(direct=True),
            "fa2": load_backend("fa2"),
        }
        for phase in ("forward", "joint"):
            for name, operation in operations.items():

                def execute():
                    out = operation(*inputs)
                    return (out, *torch.autograd.grad(out, inputs, dout)) if phase == "joint" else (out,)

                for _ in range(10):
                    actual = execute()
                torch.cuda.synchronize()
                profile = cProfile.Profile()
                profile.enable()
                for _ in range(64):
                    actual = execute()
                profile.disable()
                torch.cuda.synchronize()
                profile.dump_stats(str(args.output.with_name(name + "-" + phase + ".pstats")))
                stats = pstats.Stats(profile)
                records = []
                for (filename, line, function), (primitive, total, own, cumulative, callers) in stats.stats.items():
                    records.append(
                        {
                            "file": Path(filename).name,
                            "line": line,
                            "function": function,
                            "primitive_calls": primitive,
                            "total_calls": total,
                            "own_seconds": own,
                            "cumulative_seconds": cumulative,
                        }
                    )
                report["rows"].append(
                    {
                        "operation": name,
                        "phase": phase,
                        "functions": sorted(records, key=lambda x: x["cumulative_seconds"], reverse=True)[:25],
                    }
                )
    else:
        functions = {"uncached": old.compatible, "cached": gb10.compatible}
        for function in functions.values():
            assert function(*inputs)
            for _ in range(50):
                function(*inputs)
        samples = {name: [] for name in functions}
        for repeat in range(11):
            order = list(functions) if repeat % 2 == 0 else list(reversed(functions))
            for name in order:
                start = time.perf_counter_ns()
                for _ in range(10000):
                    assert functions[name](*inputs)
                samples[name].append((time.perf_counter_ns() - start) / 10000 / 1000)
        report.update(
            scope="Hot predicate wall microseconds; alternating11rounds10000calls, real resident GPU tensors.",
            samples_us=samples,
            median_us={n: statistics.median(v) for n, v in samples.items()},
        )
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
