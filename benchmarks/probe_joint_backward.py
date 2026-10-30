"""Compare the experimental centered joint backward on a qualified RTX 4090."""

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--dim", type=int, choices=(64, 128), default=128)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--cache-global", action="store_true")
    parser.add_argument("--direct-dq", action="store_true")
    parser.add_argument("--query-column-warps", choices=(2, 4), type=int, default=2)
    parser.add_argument("--gradient-column-warps", choices=(2, 4), type=int, default=2)
    parser.add_argument("--packed-weights", action="store_true")
    parser.add_argument("--overlap-query", action="store_true")
    parser.add_argument("--query-row-scores", action="store_true")
    parser.add_argument("--resident-kv", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--reduce-score-mass", action="store_true")
    parser.add_argument("--interleave-accumulator", action="store_true")
    parser.add_argument("--early-probability", action="store_true")
    parser.add_argument("--opaque-accumulator", action="store_true")
    parser.add_argument("--delta-columns", type=int, choices=(32, 64, 128), default=32)
    parser.add_argument(
        "--case", choices=("random", "shifted", "sharp", "negative", "subnormal", "moment"), default="random"
    )
    correction = parser.add_mutually_exclusive_group()
    correction.add_argument("--single-component", action="store_true")
    correction.add_argument("--pivot-correction", action="store_true")
    parser.add_argument("--seed", type=int, default=1009)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--window-ms", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--profile-iterations", type=int, default=0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if min(args.seq, args.repeats, args.window_ms) < 1:
        parser.error("Lengths, repetitions and timing windows must be positive.")

    if args.dump_dir is not None:
        args.dump_dir.mkdir(parents=True, exist_ok=True)
        os.environ.update(CUTE_DSL_KEEP="ptx,cubin", CUTE_DSL_DUMP_DIR=str(args.dump_dir.resolve()))

    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    import torch
    from cutlass.cute.runtime import from_dlpack

    from benchmarks.compare_ada import measure
    from benchmarks.experimental_joint_backward import JointBackward
    from benchmarks.reference import fp64_reference, metrics
    from kohakufa_cute import native

    prop = torch.cuda.get_device_properties(0)
    if prop.name != "NVIDIA GeForce RTX 4090" or prop.multi_processor_count != 128 or prop.total_memory > 28 * 1024**3:
        parser.error("Requires a standard full RTX 4090.")
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    shape = (1, 16, args.seq, args.dim)
    q, k, v, dout = [torch.randn(shape, device="cuda", dtype=dtype) for _ in range(4)]
    if args.case == "shifted":
        q[..., 0], k[..., 0] = 0, 1024
    elif args.case in ("sharp", "subnormal"):
        for tensor in (q, k, v, dout):
            tensor.zero_()
        q[..., 0] = 10 * math.sqrt(args.dim)
        k[..., 0], k[:, :, 0, 0] = 9 if dtype == torch.float16 else 5, 10
        v[..., 0], v[:, :, 0, 0], dout[..., 0] = -1, 1, 1
        if args.case == "subnormal":
            dout.mul_(1e-18)
    elif args.case == "moment":
        if args.dim != 64 or args.seq < 3 or dtype != torch.bfloat16:
            parser.error("The moment case requires BF16 D64 and at least three keys.")
        for tensor in (q, k, v, dout):
            tensor.zero_()
        q[..., 0] = 8
        k[..., 0] = -1000
        k[:, :, :3, 0] = torch.tensor([0, -4.59375, -50], device="cuda", dtype=dtype)
        v[:, :, 1, :2], v[:, :, 2, 0] = 1, 1
        dout[..., 0], dout[..., 1] = 1, -1
    elif args.case == "negative":
        q.mul_(0.3)
        k.mul_(0.3)
        q[..., 0], k[..., 0] = 20, -20
    scale = args.dim**-0.5
    extension = native.extension()
    out, maximum, logsum, blocks = extension.forward(q, k, v, None, scale, args.causal)
    dq, dk, dv = [torch.empty(shape, device="cuda", dtype=dtype) for _ in range(3)]
    base = torch.empty_like(maximum)
    delta = torch.empty((1, 16, args.seq * (2 if args.pivot_correction else 1)), device="cuda", dtype=torch.float32)
    pivots = torch.empty_like(maximum, dtype=torch.int32)
    scratch_length = ((args.seq + 63) // 64) * 64 if args.opaque_accumulator else args.seq
    scratch_shape = (1, 16, scratch_length, args.dim)
    scratch = (
        torch.empty((1, scratch_length, 16, args.dim), device="cuda", dtype=torch.float32).transpose(1, 2)
        if args.interleave_accumulator
        else torch.empty(scratch_shape, device="cuda", dtype=torch.float32)
    )
    tensors = (q, k, v, pivots, out, maximum, logsum, dout, delta, dq, dk, dv, scratch, base)
    arguments = [from_dlpack(tensor, assumed_align=16) for tensor in tensors]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    program = JointBackward(
        args.causal,
        args.dim,
        compensated=not (args.single_component or args.pivot_correction),
        pivot_correction=args.pivot_correction,
        cache_global=args.cache_global,
        direct_dq=args.direct_dq,
        query_column_warps=args.query_column_warps,
        gradient_column_warps=args.gradient_column_warps,
        packed_weights=args.packed_weights,
        overlap_query=args.overlap_query,
        query_row_scores=args.query_row_scores,
        resident_kv=args.resident_kv,
        reduce_score_mass=args.reduce_score_mass,
        early_probability=args.early_probability,
        opaque_accumulator=args.opaque_accumulator,
        delta_columns=args.delta_columns,
    )
    print("Compiling joint prototype", args.dim, args.dtype, args.causal, args.case, flush=True)
    function = cute.compile(program, *arguments, stream)
    if args.dump_dir is not None:
        for suffix in ("ptx", "cubin"):
            data = getattr(function, "__" + suffix + "__")
            if isinstance(data, str) and "\n" not in data and Path(data).is_file():
                data = Path(data).read_bytes()
            if isinstance(data, (bytes, bytearray)) and data.startswith(b"/") and b"\n" not in data:
                data = Path(data.decode()).read_bytes()
            if isinstance(data, str):
                data = data.encode()
            assert data
            (args.dump_dir / ("joint." + suffix)).write_bytes(data)
        for flag, suffix in (("--dump-sass", "sass"), ("--dump-resource-usage", "resources.txt")):
            result = subprocess.run(
                ["cuobjdump", flag, str(args.dump_dir / "joint.cubin")], capture_output=True, text=True, check=True
            )
            (args.dump_dir / ("joint." + suffix)).write_text(result.stdout)

    def candidate():
        function(*arguments, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        return dq, dk, dv

    def production():
        return extension.backward(q, k, v, None, maximum, logsum, dout, blocks, out, scale, args.causal)

    sdpa_inputs = tuple(tensor.detach().requires_grad_(True) for tensor in (q, k, v))
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        sdpa_saved = torch.nn.functional.scaled_dot_product_attention(*sdpa_inputs, is_causal=args.causal)
    torch.cuda.current_stream().wait_stream(capture_stream)

    def sdpa():
        return torch.autograd.grad(sdpa_saved, sdpa_inputs, dout, retain_graph=True)

    def reference():
        parts = [
            fp64_reference((q[:, i : i + 2], k[:, i : i + 2], v[:, i : i + 2]), dout[:, i : i + 2], causal=args.causal)
            for i in range(0, 16, 2)
        ]
        return tuple(torch.cat([p[j] for p in parts], dim=1) for j in (1, 2, 3))

    def errors(actual, expected):
        return [
            {
                key: value if not isinstance(value, float) or math.isfinite(value) else None
                for key, value in metrics(got, want).items()
            }
            for got, want in zip(actual, expected)
        ]

    def valid(checks):
        gate = 0.01 if dtype == torch.bfloat16 else 0.005
        return all(
            check["finite"]
            and (check["relative_error"] < gate if check["reference_norm"] else check["max_absolute_error"] == 0)
            for check in checks
        )

    report = {
        "scope": "Experimental backward only with fixed production forward statistics; includes scratch clear, centered prepass, joint gradients, FP32 dQ reduction and final conversion. No API dispatch change.",
        "device": prop.name,
        "multiprocessors": prop.multi_processor_count,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "shape": list(shape),
        "dtype": args.dtype,
        "causal": args.causal,
        "case": args.case,
        "seed": args.seed,
        "compensated_dq": not (args.single_component or args.pivot_correction),
        "pivot_correction": args.pivot_correction,
        "cache_global": args.cache_global,
        "direct_dq": args.direct_dq,
        "query_column_warps": args.query_column_warps,
        "gradient_column_warps": args.gradient_column_warps,
        "packed_weights": args.packed_weights,
        "overlap_query": args.overlap_query,
        "query_row_scores": args.query_row_scores,
        "resident_kv": args.resident_kv,
        "reduce_score_mass": args.reduce_score_mass,
        "early_probability": args.early_probability,
        "opaque_accumulator": args.opaque_accumulator,
        "delta_columns": args.delta_columns,
        "scratch_bytes": scratch.numel() * scratch.element_size(),
        "scratch_strides": list(scratch.stride()),
        "source_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in (
                "benchmarks/experimental_joint_backward.py",
                "benchmarks/probe_joint_backward.py",
                "src/kohakufa_cute/kernels.py",
            )
        },
        "rounds": args.repeats,
        "target_window_ms": args.window_ms,
        "profile_capture_requested": args.profile_iterations > 0,
        "results": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    want = reference()
    jobs = []
    for name, call in (("production", production), ("joint", candidate), ("sdpa", sdpa)):
        actual = call()
        torch.cuda.synchronize()
        checks = errors(actual, want)
        report["results"][name] = {
            "fp64_dq_dk_dv": checks,
            "passes_precision": valid(checks),
            "role": (
                "performance comparator, not a numerical oracle" if name == "sdpa" else "CuTe implementation under test"
            ),
        }
        save()
        print(name, "accuracy", json.dumps(checks), flush=True)
        if not valid(checks) and name != "sdpa":
            continue
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                call()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=capture_stream):
            captured = call()
        torch.cuda.current_stream().wait_stream(capture_stream)
        expected = [tensor.detach().clone() for tensor in actual]
        graph.replay()
        torch.cuda.synchronize()
        if name == "sdpa":
            report["results"][name]["graph_replay_vs_eager"] = errors(captured, expected)
            save()
            if report["results"][name]["passes_precision"]:
                assert valid(errors(captured, want))
        else:
            assert valid(errors(captured, want))
        jobs.append((name, graph, captured))
    if not all(report["results"][name]["passes_precision"] for name in ("production", "joint")):
        print("Precision rejected; no successful implementation comparison recorded.", flush=True)
        return

    # Change dO on the captured storage: both graphs must recompute the center,
    # all gradient products, and scratch clearing on every replay.
    saved_dout = dout.clone()
    dout.copy_(torch.randn_like(dout))
    changed_reference = reference()
    for name, graph, captured in jobs:
        scratch.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        changed = errors(captured, changed_reference)
        report["results"][name]["changed_dout_fp64_dq_dk_dv"] = changed
        if name == "sdpa":
            report["results"][name]["changed_dout_graph_vs_eager"] = errors(captured, sdpa())
            save()
            if report["results"][name]["passes_precision"]:
                assert valid(changed)
        else:
            save()
            assert valid(changed)
    dout.copy_(saved_dout)
    torch.cuda.synchronize()
    # Fresh forward state must accompany changed Q/K/V. SDPA's saved forward
    # is intentionally excluded here because changing only its leaves would
    # violate the retained-forward autograd contract.
    retained = [tensor.clone() for tensor in (q, k, v, out, maximum, logsum)]
    for tensor in (q, k, v):
        tensor.copy_(torch.randn_like(tensor))
    refreshed = extension.forward(q, k, v, None, scale, args.causal)
    for target, value in zip((out, maximum, logsum), refreshed[:3]):
        target.copy_(value)
    changed_reference = reference()
    for name, graph, captured in jobs:
        if name == "sdpa":
            continue
        for tensor in (scratch, delta, base, dq, dk, dv):
            tensor.fill_(float("nan"))
        pivots.fill_(2147483647)
        graph.replay()
        torch.cuda.synchronize()
        changed = errors(captured, changed_reference)
        report["results"][name]["changed_qkv_fresh_statistics_fp64_dq_dk_dv"] = changed
        assert valid(changed)
    for target, value in zip((q, k, v, out, maximum, logsum), retained):
        target.copy_(value)
    torch.cuda.synchronize()
    save()
    if args.validate_only:
        report["validation_only"] = True
        save()
        print("Validation passed; no timing requested.", flush=True)
        return
    samples = {name: [] for name, *_ in jobs}
    counts = {name: max(16, min(10000, int(args.window_ms / measure(graph, 10)))) for name, graph, _ in jobs}
    for repeat in range(args.repeats):
        for index in range(len(jobs)):
            name, graph, _ = jobs[(repeat + index) % len(jobs)]
            for _ in range(8):
                graph.replay()
            samples[name].append(measure(graph, counts[name]))
    for name in samples:
        report["results"][name].update(
            samples_ms=samples[name], median_ms=statistics.median(samples[name]), graph_replays_per_sample=counts[name]
        )
    for name, graph, _ in jobs:
        status, nodes, count = cuda.cuGraphGetNodes(cuda.CUgraph(graph.raw_cuda_graph()), 256)
        assert int(status) == 0
        records = []
        for node in nodes[:count]:
            status, kind = cuda.cuGraphNodeGetType(node)
            assert int(status) == 0
            if kind != cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
                continue
            status, parameters = cuda.cuGraphKernelNodeGetParams(node)
            assert int(status) == 0
            status, symbol = cuda.cuFuncGetName(parameters.func)
            assert int(status) == 0
            record = {
                "symbol": symbol.decode(),
                "grid": [parameters.gridDimX, parameters.gridDimY, parameters.gridDimZ],
                "block": [parameters.blockDimX, parameters.blockDimY, parameters.blockDimZ],
                "dynamic_shared_bytes": parameters.sharedMemBytes,
                "attributes": {},
            }
            for attribute in ("NUM_REGS", "LOCAL_SIZE_BYTES", "BINARY_VERSION", "PTX_VERSION"):
                status, value = cuda.cuFuncGetAttribute(
                    getattr(cuda.CUfunction_attribute, "CU_FUNC_ATTRIBUTE_" + attribute), parameters.func
                )
                assert int(status) == 0
                record["attributes"][attribute] = int(value)
            records.append(record)
        report["results"][name]["actual_graph_kernels"] = records
    save()
    print("Medians", {name: record["median_ms"] for name, record in report["results"].items()}, flush=True)
    if args.profile_iterations:
        torch.cuda.cudart().cudaProfilerStart()
        for name, graph, _ in jobs:
            torch.cuda.nvtx.range_push(name)
            for _ in range(args.profile_iterations):
                graph.replay()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
