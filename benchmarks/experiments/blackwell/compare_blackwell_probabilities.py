"""Compare Blackwell probability tile changes against an explicit baseline source."""

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

import torch

from benchmarks.precision import evaluate, load_upstream_tuning, timing, upstream_provenance
from kohakufa_cute import scaled_dot_product_attention


def load_baseline(path):
    name = "kohakufa_cute._probability_baseline"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--kohakufa-tuning", type=Path, nargs="+", required=True)
    parser.add_argument("--seq", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--dim", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        parser.error("This comparison requires Blackwell sm_100 or sm_103.")
    if min(*args.seq, *args.dim, args.iterations, args.repeats) < 1:
        parser.error("Shapes, iterations and repetitions must be positive.")
    if any(dimension not in (64, 128, 256, 512) for dimension in args.dim):
        parser.error("Blackwell pipeline dimensions are 64, 128, 256 and 512.")
    from kohakufa import attention

    from kohakufa_cute import blackwell_pipeline, kernels, native

    baseline = load_baseline(args.baseline_source)
    candidate = {name: getattr(blackwell_pipeline, name) for name in ("Forward", "Query", "KeyValue")}
    variants = ["baseline-64", "candidate-64", "baseline-128", "candidate-128", "gluon"]
    compile_function = kernels._compile_kernel.__wrapped__
    compile_caches = {name: lru_cache(maxsize=128)(compile_function) for name in variants[:-1]}
    upstream_tuning = load_upstream_tuning(args.kohakufa_tuning)

    def upstream(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
        return attention(q, k, v, mask=attn_mask, causal=is_causal, scale=scale)

    @contextmanager
    def context(backend):
        if backend == "gluon":
            yield
            return
        table = (
            kernels._SM100_FUSED_CONFIGURATIONS if backend.endswith("128") else kernels._BLACKWELL_PIPELINE_CONFIGURATIONS
        )
        classes = candidate if backend.startswith("candidate") else {name: getattr(baseline, name) for name in candidate}

        def configuration(device, mode, descriptors):
            return table[(descriptors[0][1][-1], mode)]

        native.extension().clear_cache()
        try:
            with (
                patch.multiple(blackwell_pipeline, **classes),
                patch.object(kernels, "kernel_configuration", configuration),
                patch.object(kernels, "_compile_kernel", compile_caches[backend]),
            ):
                yield
        finally:
            native.extension().clear_cache()

    properties = torch.cuda.get_device_properties(0)
    report = {
        "device": properties.name,
        "capability": list(torch.cuda.get_device_capability()),
        "multiprocessors": properties.multi_processor_count,
        "memory_bytes": properties.total_memory,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "dtype": args.dtype,
        "causal": args.causal,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "protocol": "Rotate five backends each repetition; identical inputs; all outputs and gradients checked against SDPA; candidate checked bitwise against its matching baseline; graph outputs verified against eager.",
        "policies": {"64": "current sm_103 tile table", "128": "sm_100 tile table on the actual device"},
        "baseline_sha256": hashlib.sha256(args.baseline_source.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(Path(blackwell_pipeline.__file__).read_bytes()).hexdigest(),
        "upstream_tuning": upstream_tuning,
        "upstream_source": upstream_provenance(),
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    tolerance = 0.03 if dtype == torch.bfloat16 else 0.005
    for sequence in args.seq:
        for dimension in args.dim:
            torch.manual_seed(21)
            q, k, v, dout = [torch.randn(1, 16, sequence, dimension, device="cuda", dtype=dtype) for _ in range(4)]
            reference = evaluate(torch.nn.functional.scaled_dot_product_attention, (q, k, v), dout, None, args.causal)
            baselines = {}
            row = {"shape": list(q.shape), "samples": {backend: [] for backend in variants}}
            for repeat in range(args.repeats):
                offset = repeat % len(variants)
                for backend in variants[offset:] + variants[:offset]:
                    fn = upstream if backend == "gluon" else scaled_dot_product_attention
                    with context(backend):
                        actual = evaluate(fn, (q, k, v), dout, None, args.causal)
                        for got, want in zip(actual, reference):
                            torch.testing.assert_close(got, want, atol=tolerance, rtol=tolerance)
                        if backend.startswith("baseline"):
                            baselines[backend] = tuple(tensor.detach() for tensor in actual)
                        elif backend.startswith("candidate"):
                            for got, want in zip(actual, baselines[backend.replace("candidate", "baseline")]):
                                torch.testing.assert_close(got, want, atol=0, rtol=0)
                        sample = timing(fn, (q, k, v), dout, None, args.causal, args.iterations, verify_graph=True)
                        row["samples"][backend].append({"repeat": repeat, **sample})
                        print(json.dumps({"shape": row["shape"], "backend": backend, **sample}), flush=True)
            row["medians"] = {
                backend: {
                    key: statistics.median(sample[key] for sample in samples)
                    for key in ("forward_ms", "forward_backward_ms")
                }
                for backend, samples in row["samples"].items()
            }
            row["candidate_bitwise_matches_baseline"] = True
            report["rows"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
