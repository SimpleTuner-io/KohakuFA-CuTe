"""Measure separately generated matrix-warp instrumentation on Blackwell."""

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

import torch

from benchmarks.compare_blackwell_probabilities import load_baseline
from benchmarks.instrument_blackwell_barriers import FIELDS
from benchmarks.precision import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrumented-source", type=Path, required=True)
    parser.add_argument("--dim", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--columns", type=int, choices=[64, 128], default=128)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        parser.error("Instrumentation requires Blackwell sm_100 or sm_103.")
    if min(args.seq, args.samples, *args.dim) < 1 or args.samples < 2:
        parser.error("Use positive dimensions and at least two samples (the first is warmup).")
    from kohakufa_cute import blackwell_pipeline, kernels, native

    module = load_baseline(args.instrumented_source)
    cache = lru_cache(maxsize=128)(kernels._compile_kernel.__wrapped__)

    def configuration(device, mode, descriptors):
        selected = kernels._BLACKWELL_PIPELINE_CONFIGURATIONS[(descriptors[0][1][-1], mode)]
        return selected if mode == "forward" else (args.columns, *selected[1:])

    properties = torch.cuda.get_device_properties(0)
    report = {
        "device": properties.name,
        "multiprocessors": properties.multi_processor_count,
        "capability": list(torch.cuda.get_device_capability()),
        "fields": FIELDS,
        "columns": args.columns,
        "source_sha256": hashlib.sha256(args.instrumented_source.read_bytes()).hexdigest(),
        "scope": "Matrix-warp clock32 deltas including measurement overhead; no hardware stall-counter attribution. Per-CTA intervals must be shorter than signed 32-bit clock wrap.",
        "rows": [],
    }
    native.extension().clear_cache()
    try:
        with (
            patch.multiple(blackwell_pipeline, Query=module.Query, KeyValue=module.KeyValue),
            patch.object(kernels, "kernel_configuration", configuration),
            patch.object(kernels, "_compile_kernel", cache),
        ):
            for dimension in args.dim:
                torch.manual_seed(21)
                q, k, v, dout = [
                    torch.randn(1, 16, args.seq, dimension, device="cuda", dtype=torch.bfloat16) for _ in range(4)
                ]
                scale = dimension**-0.5
                extension = native.extension()
                out, maximum, logsum, _ = extension.forward(q, k, v, None, scale, False)
                profile = torch.zeros(
                    (2, 16, ((args.seq + 127) // 128) * ((dimension + 127) // 128), 9), device="cuda", dtype=torch.int32
                )
                reference = evaluate(torch.nn.functional.scaled_dot_product_attention, (q, k, v), dout, None, False)
                samples = []
                for _ in range(args.samples):
                    profile.zero_()
                    gradients = extension.backward(q, k, v, None, maximum, logsum, dout, profile, out, scale, False)
                    torch.cuda.synchronize()
                    for actual, expected in zip((out, *gradients), reference):
                        torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
                    samples.append(profile.cpu().tolist())
                raw = torch.tensor(samples, dtype=torch.float64)
                summary = {}
                for name, index in (("q", 0), ("kv", 1)):
                    data = raw[1:, index].flatten(0, 2)
                    duration = data[:, -1]
                    if not bool((duration > 0).all()) or not bool((data[:, :-1] >= 0).all()):
                        raise RuntimeError("Incomplete or overflowing clock instrumentation.")
                    summary[name] = {
                        "median_cycles": {field: float(data[:, i].median()) for i, field in enumerate(FIELDS)},
                        "median_fraction_of_MMA_role": {
                            field: float((data[:, i] / duration).median()) for i, field in enumerate(FIELDS[:-1])
                        },
                        "ctas_per_sample": profile.shape[1] * profile.shape[2],
                    }
                row = {"dimension": dimension, "seq": args.seq, "passed": True, "summary": summary, "samples": samples}
                report["rows"].append(row)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({"dimension": dimension, "summary": summary}), flush=True)
    finally:
        native.extension().clear_cache()
        cache.cache_clear()


if __name__ == "__main__":
    main()
