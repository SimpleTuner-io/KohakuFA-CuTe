"""Measure attention forward and forward/backward using rotating CUDA-graph replays."""

import argparse
import json
import statistics
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch

from benchmarks.precision import evaluate, load_upstream_tuning, timing, upstream_configurations, upstream_provenance
from kohakufa_cute import automatic_scaled_dot_product_attention, scaled_dot_product_attention


@contextmanager
def configuration_context(backend, initial, upstream_defaults):
    if backend == "cute-initial":
        from kohakufa_cute import kernels, native

        original = kernels.kernel_configuration
        original_query = kernels.backward_query_configuration
        original_options = kernels.backward_options

        def configuration(device, mode, descriptors):
            return initial[(descriptors[0][1][-1], mode)]

        native.extension().clear_cache()
        try:
            kernels.kernel_configuration = configuration
            kernels.backward_query_configuration = lambda device, causal, masked, descriptors: configuration(
                device, "q", descriptors
            )
            # New wide tiles require their column layout even when replaying a
            # configuration table; archived tables predate shared statistics.
            kernels.backward_options = lambda *args: (original_options(*args)[0], False)
            yield
        finally:
            kernels.kernel_configuration = original
            kernels.backward_query_configuration = original_query
            kernels.backward_options = original_options
            native.extension().clear_cache()
    elif backend == "kohakufa-default":
        from kohakufa.sm100 import config, tune

        original = tune._CACHE.copy()
        original_chosen = config._CHOSEN.copy()
        try:
            tune._CACHE.clear()
            tune._CACHE.update(upstream_defaults)
            config._CHOSEN.clear()
            yield
        finally:
            tune._CACHE.clear()
            tune._CACHE.update(original)
            config._CHOSEN.clear()
            config._CHOSEN.update(original_chosen)
    else:
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--dim", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument(
        "--kohakufa-tuning", type=Path, nargs="+", help="Load upstream device tuning tables before comparing."
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument(
        "--initial-cute-configurations",
        type=Path,
        help="Compare the initial configurations recorded by a CuTe tuning report.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["sdpa", "cudnn", "cute", "cute-auto", "cute-initial", "kohakufa", "kohakufa-default"],
        default=["sdpa", "cudnn", "cute"],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.kohakufa_tuning and not {"kohakufa", "kohakufa-default"}.intersection(args.backends):
        parser.error("Upstream tuning tables require the kohakufa backend.")
    upstream_defaults = None
    if "kohakufa-default" in args.backends:
        from kohakufa.sm100 import tune

        tune._load_shipped()
        upstream_defaults = tune._CACHE.copy()
    upstream_tuning = load_upstream_tuning(args.kohakufa_tuning)
    initial = {}
    if "cute-initial" in args.backends:
        if args.initial_cute_configurations is None:
            parser.error("cute-initial requires --initial-cute-configurations.")
        calibration = json.loads(args.initial_cute_configurations.read_text())
        for trial in calibration["trials"]:
            initial.setdefault((trial["dimension"], trial["mode"]), tuple(trial["configuration"]))
    if not torch.cuda.is_available():
        parser.error("Timing requires CUDA.")
    if initial and calibration["device"] != torch.cuda.get_device_name():
        parser.error("Initial configurations must come from the same GPU model.")
    if min(*args.seq, *args.dim, args.heads, args.iterations, args.repeats) < 1:
        parser.error("Shapes, iterations and repeats must be positive.")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    functions = {
        "sdpa": torch.nn.functional.scaled_dot_product_attention,
        "cudnn": torch.nn.functional.scaled_dot_product_attention,
        "cute": scaled_dot_product_attention,
        "cute-auto": automatic_scaled_dot_product_attention,
        "cute-initial": scaled_dot_product_attention,
    }
    if {"kohakufa", "kohakufa-default"}.intersection(args.backends):
        from kohakufa import attention

        def upstream(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
            return attention(q, k, v, mask=attn_mask, causal=is_causal, scale=scale)

        functions["kohakufa"] = upstream
        functions["kohakufa-default"] = upstream
    rows = []
    for sequence in args.seq:
        for dimension in args.dim:
            torch.manual_seed(21)
            q, k, v, dout = [torch.randn(1, args.heads, sequence, dimension, device="cuda", dtype=dtype) for _ in range(4)]
            reference = evaluate(functions["sdpa"], (q, k, v), dout, None, args.causal)
            samples = {backend: [] for backend in args.backends}
            configurations = {}
            unsupported = {}
            for repeat in range(args.repeats):
                offset = repeat % len(args.backends)
                for backend in args.backends[offset:] + args.backends[:offset]:
                    if backend in unsupported:
                        continue
                    context = (
                        torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION)
                        if backend == "cudnn"
                        else nullcontext()
                    )
                    try:
                        with context, configuration_context(backend, initial, upstream_defaults):
                            actual = evaluate(functions[backend], (q, k, v), dout, None, args.causal)
                            for got, want in zip(actual, reference):
                                torch.testing.assert_close(
                                    got,
                                    want,
                                    atol=0.03 if dtype == torch.bfloat16 else 0.005,
                                    rtol=0.03 if dtype == torch.bfloat16 else 0.005,
                                )
                            sample = timing(
                                functions[backend], (q, k, v), dout, None, args.causal, args.iterations, verify_graph=True
                            )
                            if backend in ("cute", "cute-auto", "cute-initial"):
                                from kohakufa_cute import kernels

                                descriptors = tuple(kernels.descriptor(t) for t in (q, k, v, q, q, q, q, dout))
                                selected = {
                                    mode: list(kernels.kernel_configuration(q.device.index, mode, descriptors))
                                    for mode in ("forward", "q", "kv")
                                }
                                selected["q"] = list(
                                    kernels.backward_query_configuration(q.device.index, args.causal, False, descriptors)
                                )
                                columns, statistics = kernels.backward_options(
                                    q.device.index,
                                    args.causal,
                                    False,
                                    descriptors,
                                    tuple(selected["q"]),
                                    tuple(selected["kv"]),
                                )
                                selected["backward_options"] = dict(
                                    query_column_warps=columns, shared_kv_statistics=statistics
                                )
                            elif backend in ("kohakufa", "kohakufa-default"):
                                selected = upstream_configurations(q, k, args.causal)
                            else:
                                selected = None
                            if selected is not None:
                                if backend in configurations and configurations[backend] != selected:
                                    raise RuntimeError(f"{backend} configurations changed between repetitions.")
                                configurations[backend] = selected
                    except RuntimeError as error:
                        if backend != "cudnn" or "No available kernel" not in str(error):
                            raise
                        unsupported[backend] = "No available cuDNN kernel."
                        continue
                    samples[backend].append(sample)
            medians = {
                backend: {
                    key: statistics.median(sample[key] for sample in values) for key in ("forward_ms", "forward_backward_ms")
                }
                for backend, values in samples.items()
                if values
            }
            row = {
                "shape": list(q.shape),
                "samples": samples,
                "medians": medians,
                "unsupported": unsupported,
                "configurations": configurations,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    report = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "dtype": args.dtype,
        "causal": args.causal,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "ordering": "rotate backend order by one each repetition",
        "timing": "CUDA event milliseconds per graph replay; forward_backward includes forward",
        "upstream_tuning": upstream_tuning,
        "upstream_source": upstream_provenance() if {"kohakufa", "kohakufa-default"}.intersection(args.backends) else None,
        "initial_cute_configurations": {
            f"{dimension}:{mode}": list(config) for (dimension, mode), config in initial.items()
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
