"""Compare production Blackwell code generation with a pinned baseline source."""

import argparse
import hashlib
import importlib.util
import inspect
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


def load_source(path):
    name = "kohakufa_cute._codegen_baseline"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def baseline_factory(module, name):
    cls = getattr(module, name)
    parameter = "head_dimension" if name == "Forward" else "operand_chunk"
    constructor = module.Forward if name == "Forward" else module.Backward
    if parameter in inspect.signature(constructor.__init__).parameters:
        return cls

    def construct(*args, **kwargs):
        # Preserve the older source's operand policy when the current host
        # passes a forward head dimension or backward operand-chunk hint.
        kwargs.pop(parameter, None)
        return cls(*args, **kwargs)

    return construct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--kohakufa-tuning", type=Path, nargs="+", required=True)
    parser.add_argument("--seq", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--dim", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--backward-only", action="store_true", help="Hold the forward kernel at its baseline source.")
    parser.add_argument("--bitwise", action="store_true", help="Require zero-tolerance candidate/baseline agreement.")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        parser.error("This comparison requires Blackwell sm_100 or sm_103.")
    if min(*args.seq, *args.dim, args.iterations, args.repeats) < 1:
        parser.error("Shapes, iterations and repetitions must be positive.")
    from kohakufa import attention

    from kohakufa_cute import blackwell_pipeline, kernels, native

    baseline = load_source(args.baseline_source)
    names = ("Forward", "Query", "KeyValue")
    classes = {
        "baseline": {name: baseline_factory(baseline, name) for name in names},
        "candidate": {name: getattr(blackwell_pipeline, name) for name in names},
    }
    if args.backward_only:
        classes["candidate"]["Forward"] = baseline.Forward
    caches = {name: lru_cache(maxsize=128)(kernels._compile_kernel.__wrapped__) for name in classes}

    @contextmanager
    def context(backend):
        if backend == "gluon":
            yield
            return
        native.extension().clear_cache()
        try:
            with (
                patch.multiple(blackwell_pipeline, **classes[backend]),
                patch.object(kernels, "_compile_kernel", caches[backend]),
            ):
                yield
        finally:
            native.extension().clear_cache()

    def upstream(q, k, v, attn_mask=None, is_causal=False, enable_gqa=False):
        return attention(q, k, v, mask=attn_mask, causal=is_causal)

    properties = torch.cuda.get_device_properties(0)
    report = {
        "device": properties.name,
        "capability": list(torch.cuda.get_device_capability()),
        "multiprocessors": properties.multi_processor_count,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "dtype": args.dtype,
        "causal": args.causal,
        "backward_only": args.backward_only,
        "bitwise_required": args.bitwise,
        "baseline_sha256": hashlib.sha256(args.baseline_source.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(Path(blackwell_pipeline.__file__).read_bytes()).hexdigest(),
        "candidate_source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(kernels.__file__).parent.iterdir())
            if path.suffix in (".py", ".cpp")
        },
        "upstream_tuning": load_upstream_tuning(args.kohakufa_tuning),
        "upstream_source": upstream_provenance(),
        "iterations": args.iterations,
        "repeats": args.repeats,
        "protocol": "Rotate baseline/candidate/Gluon; identical inputs; outputs and gradients checked against SDPA; graph replays verified against eager.",
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    tolerance = 0.03 if dtype == torch.bfloat16 else 0.005
    variants = ["baseline", "candidate", "gluon"]
    for sequence in args.seq:
        for dimension in args.dim:
            torch.manual_seed(21)
            q, k, v, dout = [torch.randn(1, 16, sequence, dimension, device="cuda", dtype=dtype) for _ in range(4)]
            reference = evaluate(torch.nn.functional.scaled_dot_product_attention, (q, k, v), dout, None, args.causal)
            baseline_outputs = None
            row = {"shape": list(q.shape), "samples": {name: [] for name in variants}, "validation": []}
            report["rows"].append(row)
            for repeat in range(args.repeats):
                offset = repeat % len(variants)
                for backend in variants[offset:] + variants[:offset]:
                    fn = upstream if backend == "gluon" else scaled_dot_product_attention
                    with context(backend):
                        actual = evaluate(fn, (q, k, v), dout, None, args.causal)
                        for got, want in zip(actual, reference):
                            torch.testing.assert_close(got, want, atol=tolerance, rtol=tolerance)
                        validation = {"backend": backend, "repeat": repeat, "sdpa_checked": True}
                        if backend == "baseline":
                            baseline_outputs = actual
                        elif backend == "candidate" and baseline_outputs is not None:
                            differences = [
                                float((got.float() - want.float()).abs().max())
                                for got, want in zip(actual, baseline_outputs)
                            ]
                            validation["baseline_max_abs_out_dq_dk_dv"] = differences
                            if args.bitwise:
                                for got, want in zip(actual, baseline_outputs):
                                    torch.testing.assert_close(got, want, atol=0, rtol=0)
                        sample = timing(fn, (q, k, v), dout, None, args.causal, args.iterations, verify_graph=True)
                    row["validation"].append(validation)
                    row["samples"][backend].append({"repeat": repeat, **sample})
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(json.dumps({"shape": row["shape"], "backend": backend, **sample}), flush=True)
            row["medians"] = {
                name: {
                    key: statistics.median(sample[key] for sample in samples)
                    for key in ("forward_ms", "forward_backward_ms")
                }
                for name, samples in row["samples"].items()
            }
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["passed"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
