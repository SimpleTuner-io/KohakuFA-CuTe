"""Compare complete native compile/dispatch variants and pinned Gluon kernels."""

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-kernels", type=Path, required=True)
    parser.add_argument("--baseline-register-aux", type=Path)
    parser.add_argument("--kohakufa-tuning", type=Path, nargs="+", required=True)
    parser.add_argument("--seq", type=int, nargs="+", default=[2048, 4096])
    parser.add_argument("--dim", type=int, nargs="+", default=[512])
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--bitwise", action="store_true")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(*args.seq, *args.dim, args.heads, args.batch, args.iterations, args.repeats) < 1:
        parser.error("Shapes, iterations and repetitions must be positive.")

    import torch
    from kohakufa import attention as gluon_attention

    from benchmarks.precision import evaluate, load_upstream_tuning, timing, upstream_configurations, upstream_provenance
    from kohakufa_cute import kernels, native, scaled_dot_product_attention

    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        parser.error("This comparison requires SM100 or SM103.")
    spec = importlib.util.spec_from_file_location("kohakufa_cute._native_baseline_kernels", args.baseline_kernels)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    baseline_aux = None
    if args.baseline_register_aux:
        spec = importlib.util.spec_from_file_location("kohakufa_cute._native_baseline_aux", args.baseline_register_aux)
        baseline_aux = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline_aux
        spec.loader.exec_module(baseline_aux)
    backends = {"baseline": baseline, "candidate": kernels}

    @contextmanager
    def context(name):
        if name == "gluon":
            yield
        else:
            native.extension().clear_cache()
            try:
                with ExitStack() as stack:
                    stack.enter_context(patch.object(native, "kernels", backends[name]))
                    if name == "baseline" and baseline_aux is not None:
                        from kohakufa_cute import blackwell_register_aux

                        stack.enter_context(
                            patch.object(blackwell_register_aux, "AdaptiveBackward", baseline_aux.AdaptiveBackward)
                        )
                    yield
            finally:
                native.extension().clear_cache()

    def upstream(q, k, v, attn_mask=None, is_causal=False, enable_gqa=False):
        return gluon_attention(q, k, v, mask=attn_mask, causal=is_causal)

    properties = torch.cuda.get_device_properties(0)
    report = {
        "scope": "Real native fake-tensor/dynamic-sequence compilation and C++ cache dispatch. Identical inputs; rotated baseline/candidate/Gluon graph timings; outputs and gradients checked against SDPA; graph outputs verified against eager.",
        "device": properties.name,
        "multiprocessors": properties.multi_processor_count,
        "capability": list(torch.cuda.get_device_capability()),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "dtype": args.dtype,
        "causal": args.causal,
        "bitwise_required": args.bitwise,
        "baseline_kernels_sha256": hashlib.sha256(args.baseline_kernels.read_bytes()).hexdigest(),
        "baseline_register_aux_sha256": (
            hashlib.sha256(args.baseline_register_aux.read_bytes()).hexdigest() if args.baseline_register_aux else None
        ),
        "candidate_source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(kernels.__file__).parent.iterdir())
            if path.suffix in (".py", ".cpp")
        },
        "upstream_tuning": load_upstream_tuning(args.kohakufa_tuning),
        "upstream_source": upstream_provenance(),
        "iterations": args.iterations,
        "repeats": args.repeats,
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    tolerance = 0.03 if dtype == torch.bfloat16 else 0.005
    names = ["baseline", "candidate", "gluon"]
    for sequence in args.seq:
        for dimension in args.dim:
            torch.manual_seed(21)
            q, k, v, dout = [
                torch.randn(args.batch, args.heads, sequence, dimension, device="cuda", dtype=dtype) for _ in range(4)
            ]
            reference = evaluate(torch.nn.functional.scaled_dot_product_attention, (q, k, v), dout, None, args.causal)
            baseline_outputs = None
            row = {
                "shape": list(q.shape),
                "samples": {name: [] for name in names},
                "validation": [],
                "upstream_configurations": upstream_configurations(q, k, args.causal),
            }
            report["rows"].append(row)
            for repeat in range(args.repeats):
                offset = repeat % len(names)
                for name in names[offset:] + names[:offset]:
                    function = upstream if name == "gluon" else scaled_dot_product_attention
                    with context(name):
                        actual = evaluate(function, (q, k, v), dout, None, args.causal)
                        for got, want in zip(actual, reference):
                            torch.testing.assert_close(got, want, atol=tolerance, rtol=tolerance)
                        validation = {"backend": name, "repeat": repeat, "sdpa_checked": True}
                        if name == "baseline":
                            baseline_outputs = actual
                        elif name == "candidate" and baseline_outputs is not None:
                            validation["baseline_max_abs_out_dq_dk_dv"] = [
                                float((got.float() - want.float()).abs().max())
                                for got, want in zip(actual, baseline_outputs)
                            ]
                            if args.bitwise:
                                for got, want in zip(actual, baseline_outputs):
                                    torch.testing.assert_close(got, want, atol=0, rtol=0)
                        sample = timing(function, (q, k, v), dout, None, args.causal, args.iterations, verify_graph=True)
                    row["validation"].append(validation)
                    row["samples"][name].append({"repeat": repeat, **sample})
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(json.dumps({"shape": row["shape"], "backend": name, **sample}), flush=True)
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
