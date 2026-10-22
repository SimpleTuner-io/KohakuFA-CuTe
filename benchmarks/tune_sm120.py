"""Tune Blackwell attention tiles with precision checks and CUDA-graph timings."""

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path

import torch

from benchmarks.precision import evaluate, timing
from benchmarks.reference import fp64_reference, metrics
from kohakufa_cute import attention, scaled_dot_product_attention


def candidates(mode, dimension, baseline, shared_limit, tcgen=False, sm103=False):
    widths = sorted(
        {baseline[1], min(dimension, 128), min(dimension, 256), dimension}
        if dimension == 512 and not (tcgen or sm103)
        else {baseline[1], min(dimension, 128), min(dimension, 256)}
    )
    result = {baseline}
    for columns, width, warps, tma, pipelined in itertools.product(
        (16, 32, 64), widths, (1, 2, 4), (False, True), (False, True)
    ):
        if tma and dimension not in ((64, 128, 256, 512) if tcgen else (64, 128, 256)):
            continue
        if pipelined and not tma:
            continue
        column_warps = 2 if mode == "kv" and width > 128 else 1
        full_forward = mode == "forward" and (columns, width, warps, tma, pipelined) == (32, 512, 4, False, False)
        wide_query = (
            mode == "q"
            and not (tcgen or sm103)
            and dimension == 512
            and columns == 32
            and not (tma or pipelined)
            and (width, warps) in ((256, 2), (512, 4))
        )
        wide_kv = (
            mode == "kv"
            and not (tcgen or sm103)
            and dimension == 512
            and (columns, width, warps, tma, pipelined) == (32, 512, 4, False, False)
        )
        if full_forward:
            column_warps = 2
        elif wide_query or wide_kv:
            column_warps = warps
        if warps < column_warps:
            continue
        if column_warps > 1 and columns < 32:
            continue
        rows = 16 * (warps // column_warps)
        padded = math.ceil(dimension / width) * width
        shared = 2 * padded * (rows * (1 if mode == "forward" else 2) + (4 if pipelined else 2) * columns)
        if tma:
            shared += 1024  # barriers and alignment padding
        if full_forward or wide_query:
            # P (and dS) reuse the consumed source tile; only row reductions
            # need additional shared storage for these N-warp layouts.
            shared += 8 * column_warps * rows
        elif column_warps > 1:
            shared += 4 * rows * columns
        # Keep tiles inside the existing 64x64 mask classification cells.
        if rows <= 64 and shared <= shared_limit:
            option = (columns, width, warps, False, mode != "forward", tma, pipelined)
            result.add(option)
            if (
                tcgen
                and rows == 64
                and warps == 4
                and columns in (32, 64)
                and shared + 4 * rows * columns + 128 <= shared_limit
            ):
                result.add((*option, True))
    if sm103:
        for rows, columns, width, tma, pipelined in itertools.product(
            (64, 128), (32, 64, 128), sorted({64, *widths}), (False, True), (False, True)
        ):
            if pipelined and (not tma or mode == "kv"):
                continue
            padded = math.ceil(dimension / width) * width
            shared = 2 * padded * (rows * (1 if mode == "forward" else 2) + (4 if pipelined else 2) * columns)
            shared += 2 * rows * columns * (1 if mode == "forward" else 2) + 4096
            if shared <= shared_limit:
                option = (columns, width, 4, False, mode != "forward", tma, pipelined, False, rows)
                old_columns = 2 * (columns + width)
                if mode == "forward" or old_columns <= 512:
                    result.add(option)
                if mode != "forward":
                    new_columns = width + max(width, 2 * columns)
                    old_allocation = 1 << (old_columns - 1).bit_length()
                    new_allocation = 1 << (new_columns - 1).bit_length()
                    if new_columns <= 512 and min(512 // new_allocation, shared_limit // shared) > min(
                        512 // old_allocation, shared_limit // shared
                    ):
                        result.add((*option, True))
    return sorted(result)


def validate(dtype, dimension, heads, random_saturated=False):
    torch.manual_seed(127)
    q = torch.randn(1, heads, 137, dimension, device="cuda", dtype=dtype)
    k, v = [torch.randn(1, heads, 193, dimension, device="cuda", dtype=dtype) for _ in range(2)]
    dout = torch.randn_like(q)
    reference = fp64_reference((q, k, v), dout)
    actual = evaluate(scaled_dot_product_attention, (q, k, v), dout, None, False)
    for got, expected in zip(actual, reference):
        torch.testing.assert_close(got.double(), expected, atol=0.02, rtol=0.02)
    # A normal-logit comparison alone misses gradients rounded to zero at saturation.
    q = torch.zeros(1, heads, 17, dimension, device="cuda", dtype=dtype)
    k = torch.zeros(1, heads, 19, dimension, device="cuda", dtype=dtype)
    v, dout = torch.zeros_like(k), torch.zeros_like(q)
    q[..., 0], k[..., 0], k[:, :, 0, 0] = 10, 5 if dtype == torch.bfloat16 else 9, 10
    v[..., 0], v[:, :, 0, 0], dout[..., 0] = -1, 1, 1
    reference = fp64_reference((q, k, v), dout, scale=1)
    inputs = [tensor.requires_grad_() for tensor in (q, k, v)]
    out = attention(*inputs, scale=1)
    actual = (out, *torch.autograd.grad(out, inputs, dout))
    for got, expected in zip(actual, reference):
        torch.testing.assert_close(got.double(), expected, atol=0, rtol=0.03)
    if random_saturated:
        v, dout = torch.randn_like(k), torch.randn_like(q)
        reference = fp64_reference((q, k, v), dout, scale=1)
        inputs = [tensor.detach().requires_grad_() for tensor in (q, k, v)]
        out = attention(*inputs, scale=1)
        actual = (out, *torch.autograd.grad(out, inputs, dout))
        for got, expected in zip(actual, reference):
            error = metrics(got, expected)
            if not error["finite"] or error["reference_norm"] and error["relative_error"] > 0.03:
                raise AssertionError(f"Random saturated gradient failed: {error}")
            if not error["reference_norm"]:
                torch.testing.assert_close(got.double(), expected, atol=0, rtol=0)


def main(capability=(12, 0)):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", nargs="+", type=int, choices=(64, 128, 256, 512), default=[64, 128, 256, 512])
    parser.add_argument("--seq", nargs="+", type=int, default=[1024, 4096])
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--max-candidates", type=int, default=0, help="Limit each stage to an evenly spaced shortlist; zero searches all."
    )
    parser.add_argument(
        "--sample-ms", type=float, default=10.0, help="Target timed replay duration; iterations sets the cap."
    )
    parser.add_argument("--resume", action="store_true", help="Reuse completed trials from the output report.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != capability:
        parser.error(f"Tuning requires sm_{capability[0]}{capability[1]}.")
    if min(*args.seq, args.heads, args.iterations, args.repeats) < 1:
        parser.error("Lengths, heads, iterations and repeats must be positive.")
    if args.max_candidates < 0 or args.max_candidates == 1:
        parser.error("Candidate limit must be zero or at least two.")
    if not math.isfinite(args.sample_ms) or args.sample_ms <= 0:
        parser.error("Sample duration must be finite and positive.")
    from kohakufa_cute import kernels, native

    properties = torch.cuda.get_device_properties(0)
    report = dict(
        device=properties.name,
        capability=list(capability),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        dtype="bf16",
        selection="geometric mean over lengths; require 2% improvement over the current winner",
        settings=vars(args).copy(),
        trials=[],
        selected={},
    )
    report["settings"]["output"] = str(args.output)
    if args.resume:
        if not args.output.exists():
            parser.error("Resume requires an existing output report.")
        report = json.loads(args.output.read_text())
        if report["device"] != properties.name or report["torch"] != str(torch.__version__):
            parser.error("Resume requires the same device and PyTorch version.")
        if any(report["settings"][key] != getattr(args, key) for key in ("dim", "seq", "heads")):
            parser.error("Resume requires the same dimensions, sequence lengths and head count.")
        report.setdefault("dtype", "bf16")
        report.setdefault("selection", "geometric mean over lengths; require 2% improvement over the current winner")
        report.setdefault("timing_runs", []).append(
            dict(
                completed_before_resume=len(report["trials"]),
                iterations=args.iterations,
                repeats=args.repeats,
                sample_ms=args.sample_ms,
            )
        )
    original = kernels.kernel_configuration
    original_query = kernels.backward_query_configuration
    overrides = {}

    def configuration(device, mode, descriptors):
        return overrides.get(mode, original(device, mode, descriptors))

    def checkpoint():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)

    try:
        kernels.kernel_configuration = configuration
        kernels.backward_query_configuration = lambda device, causal, masked, descriptors: configuration(
            device, "q", descriptors
        )
        for dimension in args.dim:
            overrides.clear()
            torch.manual_seed(128)
            cases = [
                [torch.randn(1, args.heads, length, dimension, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
                for length in args.seq
            ]
            q, k, v, dout = cases[0]
            descriptors = tuple(kernels.descriptor(t) for t in (q, k, v, q, q, q, q, dout))
            for mode in ("forward", "q", "kv"):
                baseline = original_query(0, False, False, descriptors) if mode == "q" else original(0, mode, descriptors)
                selected, best = baseline, float("inf")
                options = candidates(
                    mode,
                    dimension,
                    baseline,
                    properties.shared_memory_per_block_optin,
                    tcgen=capability in ((10, 0), (10, 3)),
                    sm103=capability == (10, 3),
                )
                options.remove(baseline)
                if args.max_candidates and len(options) >= args.max_candidates:
                    count = args.max_candidates - 1
                    options = [options[round(i * (len(options) - 1) / max(1, count - 1))] for i in range(count)]
                options.insert(0, baseline)
                baseline_score = None
                for option in options:
                    overrides[mode] = option
                    previous = next(
                        (
                            t
                            for t in report["trials"]
                            if t["dimension"] == dimension and t["mode"] == mode and tuple(t["configuration"]) == option
                        ),
                        None,
                    )
                    if previous is not None:
                        if previous["status"] != "ok" and option == baseline:
                            raise RuntimeError("The saved baseline failed validation; start a new report after fixing it.")
                        if previous["status"] == "ok":
                            score = previous["score_ms"]
                            if baseline_score is None:
                                baseline_score = score
                            if score < best * 0.98:
                                best, selected = score, option
                        continue
                    native.extension().clear_cache()
                    trial = dict(dimension=dimension, mode=mode, configuration=option)
                    try:
                        validate(torch.bfloat16, dimension, args.heads, random_saturated=capability == (10, 3))
                        samples = []
                        for q, k, v, dout in cases:
                            probe = timing(scaled_dot_product_attention, (q, k, v), dout, None, False, 1)
                            iterations = min(
                                args.iterations, max(5, math.ceil(args.sample_ms / probe["forward_backward_ms"]))
                            )
                            values = [
                                timing(scaled_dot_product_attention, (q, k, v), dout, None, False, iterations)
                                for _ in range(args.repeats)
                            ]
                            samples.append(dict(sequence=q.shape[2], iterations=iterations, probe=probe, repetitions=values))
                        field = "forward_ms" if mode == "forward" else "forward_backward_ms"
                        score = math.exp(
                            statistics.mean(math.log(statistics.median(s[field] for s in c["repetitions"])) for c in samples)
                        )
                        trial.update(status="ok", score_ms=score, samples=samples)
                        if baseline_score is None:
                            baseline_score = score
                        if score < best * 0.98:
                            best, selected = score, option
                    except Exception as error:
                        trial.update(status="error", error=str(error))
                        fatal = any(
                            text in str(error).lower()
                            for text in ("illegal memory", "device-side assert", "launch failure", "misaligned address")
                        )
                        if option == baseline or fatal:
                            report["trials"].append(trial)
                            checkpoint()
                            raise
                    report["trials"].append(trial)
                    checkpoint()
                    print(json.dumps({k: v for k, v in trial.items() if k != "samples"}), flush=True)
                overrides[mode] = selected
                report["selected"][f"{dimension}:{mode}"] = dict(
                    configuration=selected, baseline_score_ms=baseline_score, score_ms=best
                )
                checkpoint()
    finally:
        kernels.kernel_configuration = original
        kernels.backward_query_configuration = original_query
        native.extension().clear_cache()


if __name__ == "__main__":
    main()
