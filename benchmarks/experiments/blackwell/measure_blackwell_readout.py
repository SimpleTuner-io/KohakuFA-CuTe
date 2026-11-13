"""Measure frozen backward prototypes against production and SDPA.

A prototype exports JointBackward(causal) with the native backward argument order.
Its optional WORKSPACE_PLANES declares FP32 [planes, *K.shape] storage, zeroed on
both eager calls and graph replays. Zero declares an unused four-byte placeholder.
Use --dq-only-source explicitly for ablations that invalidate dK and dV. Those
measurements diagnose a cost and never qualify an attention implementation.
"""

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--dq-only-source", type=Path, nargs="*", default=[])
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.seq, args.iterations, args.repeats) < 1:
        parser.error("Sequence length, iterations and repetitions must be positive.")
    sources = [path.resolve(strict=True) for path in args.source]
    diagnostic = {path.resolve(strict=True) for path in args.dq_only_source}
    if len(set(sources)) != len(sources) or len({path.stem for path in sources}) != len(sources):
        parser.error("Use distinct source paths and filename stems.")
    if not diagnostic.issubset(sources):
        parser.error("Each --dq-only-source must also occur in --source.")

    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    import torch
    from cutlass.cute.runtime import from_dlpack

    from benchmarks.precision import evaluate
    from kohakufa_cute import kernels, native

    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        parser.error("This diagnostic requires SM100 or SM103.")
    torch.manual_seed(21)
    shape = (1, 16, args.seq, 512)
    q, k, v, dout = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
    scale = shape[-1] ** -0.5
    extension = native.extension()
    out, maximum, logsum, blocks = extension.forward(q, k, v, None, scale, args.causal)
    reference = evaluate(torch.nn.functional.scaled_dot_product_attention, (q, k, v), dout, None, args.causal)[1:]
    properties = torch.cuda.get_device_properties(0)
    report = {
        "scope": "Backward only; fixed native forward statistics, rotated graph replays. Workspace zeroing is timed. DQ-only ablations produce invalid dK/dV and are not complete backward kernels.",
        "shape": list(shape),
        "causal": args.causal,
        "device": properties.name,
        "multiprocessors": properties.multi_processor_count,
        "capability": list(torch.cuda.get_device_capability()),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "runtime_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(kernels.__file__).parent.iterdir())
            if path.suffix in (".py", ".cpp")
        },
        "variants": {},
        "samples": [],
    }
    jobs = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def prepare(name, call, tensors, valid_gradients, metadata, keepalive):
        actual = call()
        torch.cuda.synchronize()
        checked = actual[:valid_gradients]
        for got, want in zip(checked, reference):
            torch.testing.assert_close(got, want, atol=0.03, rtol=0.03)
        expected = [tensor.clone() for tensor in checked]
        call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = call()
        graph.replay()
        torch.cuda.synchronize()
        for got, want in zip(captured[:valid_gradients], expected):
            torch.testing.assert_close(got, want, atol=0.03, rtol=0.03)
        report["variants"][name] = {
            **metadata,
            "sdpa_checked_gradients": ["dq", "dk", "dv"][:valid_gradients],
            "graph_checked": True,
            "sdpa_max_abs_error": [float((got.float() - want.float()).abs().max()) for got, want in zip(checked, reference)],
        }
        # Retain every source argument, module and compiled function until all
        # measurements finish. Graphs do not own their external tensor storage.
        jobs.append((name, graph, call, tensors, actual, captured, expected, keepalive))
        save()

    def production():
        return extension.backward(q, k, v, None, maximum, logsum, dout, blocks, out, scale, args.causal)

    prepare("production", production, (q, k, v, out, maximum, logsum, dout, blocks), 3, {"production": True}, extension)
    for index, path in enumerate(sources):
        name = f"kohakufa_cute._readout_probe_{index}"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        planes = getattr(module, "WORKSPACE_PLANES", 3)
        if not isinstance(planes, int) or planes < 0:
            raise ValueError(f"Invalid WORKSPACE_PLANES in {path}")
        workspace = torch.empty((planes, *k.shape) if planes else (1,), device="cuda", dtype=torch.float32)
        base, delta = torch.empty_like(maximum), torch.empty_like(maximum)
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        tensors = (q, k, v, q, out, maximum, logsum, dout, delta, dq, dk, dv, workspace, base)
        arguments = [from_dlpack(tensor, assumed_align=16) for tensor in tensors]
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        print("compiling", path.name, flush=True)
        function = cute.compile(module.JointBackward(args.causal), *arguments, scale, stream)

        def call(function=function, arguments=arguments, workspace=workspace, planes=planes, outputs=(dq, dk, dv)):
            if planes:
                workspace.zero_()
            function(*arguments, scale, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
            return outputs

        prepare(
            path.stem,
            call,
            tensors,
            1 if path in diagnostic else 3,
            {
                "production": False,
                "source": str(path),
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "dq_only_ablation": path in diagnostic,
                "workspace_planes": planes,
                "workspace_bytes": workspace.numel() * workspace.element_size(),
            },
            (module, function, arguments),
        )
    for repeat in range(args.repeats):
        offset = repeat % len(jobs)
        for name, graph, *_ in jobs[offset:] + jobs[:offset]:
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start.record()
            for _ in range(args.iterations):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            sample = {"name": name, "repeat": repeat, "backward_ms": start.elapsed_time(end) / args.iterations}
            report["samples"].append(sample)
            save()
            print(json.dumps(sample), flush=True)
    for name, metadata in report["variants"].items():
        metadata["median_ms"] = statistics.median(
            sample["backward_ms"] for sample in report["samples"] if sample["name"] == name
        )
    report["passed"] = True
    save()


if __name__ == "__main__":
    main()
