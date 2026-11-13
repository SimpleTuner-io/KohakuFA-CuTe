"""Measure backward sensitivity to delays in diagnostic KV role modules.

The same graph reads a device delay for every level. All gradients must remain
bitwise equal to production. Zero-delay overhead is reported separately: this
is intervention sensitivity, not accelerator utilization or a speedup forecast.
Run GPU jobs serially against immutable source snapshots.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, nargs="+", required=True)
    p.add_argument("--delays", type=int, nargs="+", default=[0, 800, 3200])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if min(a.repeats, a.iterations) < 1 or min(a.delays) < 0 or 0 not in a.delays:
        p.error("Use positive repetitions/iterations and nonnegative delays including zero.")
    if max(a.delays) >= 2**31 or len(set(a.delays)) != len(a.delays):
        p.error("Delay values must be distinct signed 32-bit nanosecond counts.")
    names = [path.stem for path in a.source]
    if len(set(names)) != len(names) or "baseline" in names:
        p.error("Source stems must be distinct and cannot be 'baseline'.")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["CUTE_DSL_KEEP"] = "ptx,cubin"
    os.environ["CUTE_DSL_DUMP_DIR"] = str(a.output.parent / "delay-dumps")
    os.environ["CUTE_DSL_DISABLE_FILE_CACHING"] = "1"
    Path(os.environ["CUTE_DSL_DUMP_DIR"]).mkdir(exist_ok=True)
    import torch

    from benchmarks.precision import evaluate
    from kohakufa_cute import blackwell_register_aux as aux
    from kohakufa_cute import kernels, native

    if torch.cuda.get_device_capability() != (10, 3):
        p.error("This profiler requires the SM103 register-auxiliary path.")
    classes = {"baseline": aux.AdaptiveBackward}
    modules = []
    caches = {}
    captures = {}
    functions = []
    for i, path in enumerate(a.source):
        name = "kohakufa_cute._sensitivity_" + str(i)
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        modules.append(m)
        classes[path.stem] = m.AdaptiveBackward
    torch.manual_seed(21)
    q, k, v, do = [torch.randn(1, 16, 4096, 512, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
    ext = native.extension()
    out, mx, ls, _ = ext.forward(q, k, v, None, 512**-0.5, True)
    profile = torch.zeros((1, 1, 1, 4), device="cuda", dtype=torch.int32)
    base = ext.backward(q, k, v, None, mx, ls, do, profile, out, 512**-0.5, True)
    ref = evaluate(torch.nn.functional.scaled_dot_product_attention, (q, k, v), do, None, True)[1:]
    for x, y in zip(base, ref):
        torch.testing.assert_close(x, y, atol=0.03, rtol=0.03)
    jobs = []
    report = {
        "scope": "Causal backward-only native dynamic-shape/cache dispatch. Per-role busy waits once per KV inner iteration; delay is a device tensor read by the same compiled graph for all delay levels. Every gradient bitwise checked. Differences measure sensitivity to injected latency, not hardware stall fractions.",
        "shape": list(q.shape),
        "device": torch.cuda.get_device_name(),
        "runtime_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(kernels.__file__).parent.iterdir()
            if p.suffix in (".py", ".cpp")
        },
        "sources": {p.stem: hashlib.sha256(p.read_bytes()).hexdigest() for p in a.source},
        "delays_requested_ns": a.delays,
        "samples": [],
        "zero_delay_overhead": {},
        "medians": {},
    }
    compiler = kernels.compile_kernel
    for name, cls in classes.items():
        captures[name] = []
        caches[name] = lru_cache(maxsize=128)(kernels._compile_kernel.__wrapped__)

        def record(*args, **kwargs):
            result = compiler(*args, **kwargs)
            captures[name].append((args[1], result))
            functions.append(result)
            return result

        ext.clear_cache()
        profile.zero_()
        print("compile", name, flush=True)
        with ExitStack() as stack:
            stack.enter_context(patch.object(aux, "AdaptiveBackward", cls))
            stack.enter_context(patch.object(kernels, "_compile_kernel", caches[name]))
            stack.enter_context(patch.object(kernels, "compile_kernel", record))

            def call():
                return ext.backward(q, k, v, None, mx, ls, do, profile, out, 512**-0.5, True)

            actual = call()
            call()
            torch.cuda.synchronize()
            for x, y in zip(actual, base):
                torch.testing.assert_close(x, y, atol=0, rtol=0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = call()
            graph.replay()
            torch.cuda.synchronize()
            for x, y in zip(captured, base):
                torch.testing.assert_close(x, y, atol=0, rtol=0)
            jobs.append((name, graph, actual, captured))
        ext.clear_cache()
        for mode, function in captures[name]:
            directory = a.output.parent / "sensitivity-assembly" / name / mode
            directory.mkdir(parents=True, exist_ok=True)
            for suffix in ("ptx", "cubin"):
                payload = getattr(function, "__" + suffix + "__")
                if isinstance(payload, str) and "\n" not in payload and Path(payload).is_file():
                    payload = Path(payload).read_bytes()
                if isinstance(payload, str):
                    payload = payload.encode()
                assert payload and (b".entry" in payload if suffix == "ptx" else payload.startswith(b"\x7fELF"))
                (directory / ("kernel." + suffix)).write_bytes(payload)
    cases = [(job, ns) for job in jobs for ns in ([0] if job[0] == "baseline" else a.delays)]
    for repeat in range(a.repeats):
        offset = repeat * 3 % len(cases)
        for (name, graph, actual, captured), ns in cases[offset:] + cases[:offset]:
            profile.zero_()
            profile[0, 0, 0, 0] = ns
            for _ in range(2):
                graph.replay()
            torch.cuda.synchronize()
            for x, y in zip(captured, base):
                torch.testing.assert_close(x, y, atol=0, rtol=0)
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start.record()
            for _ in range(a.iterations):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            for x, y in zip(captured, base):
                torch.testing.assert_close(x, y, atol=0, rtol=0)
            sample = {
                "role": name,
                "requested_ns": ns,
                "observed_sampled_wait_ns": int(profile[0, 0, 0, 1]) if name != "baseline" else None,
                "repeat": repeat,
                "backward_ms": start.elapsed_time(end) / a.iterations,
                "bitwise_checked": True,
            }
            report["samples"].append(sample)
            a.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(sample), flush=True)
    for name in classes:
        report["medians"][name] = {
            str(ns): statistics.median(
                s["backward_ms"] for s in report["samples"] if s["role"] == name and s["requested_ns"] == ns
            )
            for ns in ([0] if name == "baseline" else a.delays)
        }
        if name != "baseline":
            report["zero_delay_overhead"][name] = report["medians"][name]["0"] / report["medians"]["baseline"]["0"] - 1
    report["passed"] = True
    a.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
