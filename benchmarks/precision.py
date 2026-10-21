"""Compare attention outputs and Q/K/V gradients with FP64 on identical inputs."""

import argparse
import dataclasses
import hashlib
import json
import math
import subprocess
from contextlib import nullcontext
from pathlib import Path

import torch


def make_inputs(logit_scale, *, device, dtype, seq, dim, seed, heads=2):
    generator = torch.Generator(device=device).manual_seed(seed)
    centers = torch.randn(1, heads, 16, dim, generator=generator, device=device)
    centers = centers / centers.norm(dim=-1, keepdim=True)
    choices = torch.randint(16, (1, heads, seq), generator=generator, device=device)
    base = centers.gather(2, choices[..., None].expand(-1, -1, -1, dim))
    radius = math.sqrt(logit_scale * math.sqrt(dim))
    q, k = [
        (radius * (base + 0.03 * torch.randn(base.shape, generator=generator, device=device))).to(dtype) for _ in range(2)
    ]
    v, dout = [torch.randn(base.shape, generator=generator, device=device).to(dtype) for _ in range(2)]
    return q, k, v, dout


def evaluate(fn, inputs, dout, mask, causal):
    q, k, v = [t.detach().clone().requires_grad_(True) for t in inputs]
    out = fn(q, k, v, attn_mask=mask, is_causal=causal, enable_gqa=q.shape[1] != k.shape[1])
    grads = torch.autograd.grad(out, (q, k, v), dout)
    return out.detach(), *(g.detach() for g in grads)


from benchmarks.reference import fp64_reference, metrics, row_metrics


def load_upstream_tuning(paths):
    if not paths:
        return []
    from kohakufa.sm100 import config, tune

    records = []
    for path in paths:
        tune.load(str(path))
        records.append(dict(name=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    config._CHOSEN.clear()
    return records


def upstream_provenance():
    import kohakufa

    package = Path(kohakufa.__file__).resolve().parent
    repository = package.parent.parent
    record = {
        "source_sha256": {
            str(path.relative_to(package.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(package.rglob("*.py"))
        }
    }
    if (repository / ".git").exists():
        record["revision"] = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
        diff = subprocess.check_output(["git", "-C", str(repository), "diff", "HEAD", "--binary"])
        record["modified"] = bool(diff)
        record["diff_sha256"] = hashlib.sha256(diff).hexdigest()
    return record


def upstream_configurations(q, k, causal):
    from kohakufa.sm100 import config

    dimension, length = q.shape[-1], max(q.shape[2], k.shape[2])
    return {
        kind: dataclasses.asdict(getattr(config, f"{kind}_config")(dimension, q.element_size(), causal, length))
        for kind in ("forward", "backward")
    }


def timing(fn, inputs, dout, mask, causal, iterations, *, verify_graph=False):
    q, k, v = [t.detach().clone().requires_grad_(True) for t in inputs]

    def run(backward):
        out = fn(q, k, v, attn_mask=mask, is_causal=causal, enable_gqa=q.shape[1] != k.shape[1])
        return (out, *torch.autograd.grad(out, (q, k, v), dout)) if backward else out

    result = {}
    for backward in (False, True):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                if verify_graph:
                    expected = run(backward)
                else:
                    run(backward)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = run(backward)
        for _ in range(iterations):
            graph.replay()
        torch.cuda.synchronize()
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        stop.record()
        stop.synchronize()
        result["forward_backward_ms" if backward else "forward_ms"] = start.elapsed_time(stop) / iterations
        if verify_graph:
            for actual, want in zip(outputs if backward else (outputs,), expected if backward else (expected,)):
                tolerance = 0.03 if q.dtype == torch.bfloat16 else 0.005
                absolute = 1e-5 if q.dtype == torch.bfloat16 else 1e-6
                torch.testing.assert_close(actual, want, atol=absolute, rtol=tolerance)
            del expected
        del outputs, graph
    if verify_graph:
        result["graph_verified"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--layout", choices=["bhsd", "bshd"], help="Physical Q/K/V storage order; logical inputs remain BHSD."
    )
    parser.add_argument("--scales", default="10,1000,100000,1000000,10000000")
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["auto", "math", "flash", "efficient", "cudnn", "kohakufa-cute", "kohakufa-cute-auto", "kohakufa"],
        default=["auto", "cudnn", "kohakufa-cute"],
    )
    parser.add_argument("--causal", action="store_true")
    parser.add_argument(
        "--kohakufa-tuning", type=Path, nargs="+", help="Load upstream device tuning tables before comparing."
    )
    parser.add_argument("--row-errors", action="store_true", help="Retain per-row errors for distribution plots.")
    parser.add_argument(
        "--timing-iterations",
        type=int,
        default=0,
        help="CUDA graph replay iterations for warmup and measurement; zero disables timing.",
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        help="Tensor-only .pt mapping: q, k, v, dout, optional boolean mask; BHSD layout. Preserves saved dtypes.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.kohakufa_tuning and "kohakufa" not in args.backends:
        parser.error("Upstream tuning tables require the kohakufa backend.")
    upstream_tuning = load_upstream_tuning(args.kohakufa_tuning)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    sdpa = torch.nn.functional.scaled_dot_product_attention
    kernels = {
        "math": torch.nn.attention.SDPBackend.MATH,
        "flash": torch.nn.attention.SDPBackend.FLASH_ATTENTION,
        "efficient": torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
        "cudnn": torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
    }
    saved = torch.load(args.inputs, map_location=args.device, weights_only=True) if args.inputs else None
    rows = []
    for scale in [None] if saved is not None else [float(x) for x in args.scales.split(",")]:
        q, k, v, dout = (
            [saved[key] for key in ("q", "k", "v", "dout")]
            if saved is not None
            else make_inputs(
                scale,
                device=args.device,
                dtype=dtype,
                seq=args.seq,
                dim=args.dim,
                seed=args.seed,
                heads=args.heads,
            )
        )
        if args.layout == "bhsd":
            q, k, v, dout = [tensor.contiguous() for tensor in (q, k, v, dout)]
        elif args.layout == "bshd":
            q, k, v, dout = [tensor.transpose(1, 2).contiguous().transpose(1, 2) for tensor in (q, k, v, dout)]
        mask = saved.get("mask") if saved is not None else None
        reference = fp64_reference((q, k, v), dout, mask, args.causal)
        scores = q.double() @ k.double().repeat_interleave(q.shape[1] // k.shape[1], dim=1).transpose(-1, -2)
        max_logit = float(scores.abs().max() / math.sqrt(q.shape[-1]))
        for backend in args.backends:
            row = dict(
                backend=backend,
                requested_scale=scale,
                max_absolute_logit=max_logit,
                dtype=str(q.dtype),
                shape=list(q.shape),
                strides={name: list(tensor.stride()) for name, tensor in zip(("q", "k", "v", "dout"), (q, k, v, dout))},
            )
            try:
                fn = sdpa
                if backend == "kohakufa-cute":
                    from kohakufa_cute import scaled_dot_product_attention as fn
                elif backend == "kohakufa-cute-auto":
                    from kohakufa_cute import automatic_scaled_dot_product_attention as fn
                elif backend == "kohakufa":
                    from kohakufa import attention

                    def fn(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
                        return attention(q, k, v, mask=attn_mask, causal=is_causal, scale=scale)

                context = torch.nn.attention.sdpa_kernel(kernels[backend]) if backend in kernels else nullcontext()
                with context:
                    got = evaluate(fn, (q, k, v), dout, mask, args.causal)
                    if args.timing_iterations:
                        row["timing"] = timing(fn, (q, k, v), dout, mask, args.causal, args.timing_iterations)
                row["metrics"] = {name: metrics(g, r) for name, g, r in zip(("out", "dq", "dk", "dv"), got, reference)}
                if backend == "kohakufa":
                    row["configurations"] = upstream_configurations(q, k, args.causal)
                if args.row_errors:
                    row["row_metrics"] = {
                        name: row_metrics(g, r) for name, g, r in zip(("out", "dq", "dk", "dv"), got, reference)
                    }
                row["status"] = "ok"
            except (RuntimeError, ValueError, ImportError) as error:
                row.update(status="error", error=str(error))
            rows.append(row)
            print(json.dumps(row), flush=True)
    report = {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name() if args.device.startswith("cuda") else args.device,
        "causal": args.causal,
        "input_source": "captured" if saved is not None else "synthetic",
        "reference": "FP64 centered analytical backward on identical rounded inputs",
        "upstream_tuning": upstream_tuning,
        "upstream_source": upstream_provenance() if "kohakufa" in args.backends else None,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
