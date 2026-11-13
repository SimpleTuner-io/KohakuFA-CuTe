"""Measure the complete GB10 long D64 forward-only API against its preceding policy."""

import argparse
import hashlib
import itertools
import json
import math
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import torch

from benchmarks.reference import metrics
from kohakufa_cute.fa4_sm80._attention import Schedule


@torch.no_grad()
def references(inputs, causal=False, scale=None):
    cpu = [t.detach().cpu() for t in inputs]
    expected = []
    for h in range(16):
        q, k, v = [t[:, h : h + 1].double() for t in cpu]
        score = (q @ k.transpose(-1, -2)) * (q.shape[-1] ** -0.5 if scale is None else scale)
        if causal:
            score.masked_fill_(~torch.ones(score.shape[-2:], dtype=torch.bool).tril(), -torch.inf)
        expected.append(torch.softmax(score, -1) @ v)
    return expected


def assess_output(output, expected):
    output = output.detach().cpu()
    limit = 0.01 if output.dtype == torch.bfloat16 else 0.005
    records = []
    for head, reference in enumerate(expected):
        error = metrics(output[:, head : head + 1], reference)
        assert error["finite"], (head, error)
        if error["reference_norm"]:
            assert error["relative_error"] < limit, (head, error)
        else:
            assert error["max_absolute_error"] == 0, (head, error)
        records.append(error)
    return records


def retain_tensor(tensor, root):
    cpu = tensor.detach().cpu().resolve_conj().resolve_neg().contiguous()
    h = hashlib.sha256((str(cpu.dtype) + repr(list(cpu.shape))).encode())
    h.update(cpu.view(torch.uint8).numpy().tobytes())
    digest = h.hexdigest()
    folder = Path(root) / "tensors"
    folder.mkdir(exist_ok=True)
    path = folder / (digest + ".pt")
    if not path.exists():
        torch.save(cpu, path)
    return dict(file="tensors/" + path.name, raw_sha256=digest, shape=list(cpu.shape), dtype=str(cpu.dtype))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-source", type=Path, required=True)
    parser.add_argument("--seq", nargs="+", type=int, choices=(4096,), default=[4096])
    parser.add_argument("--heads", type=int, choices=(16,), default=16)
    parser.add_argument("--dim", nargs="+", type=int, choices=(64,), default=[64])
    parser.add_argument("--dtype", nargs="+", choices=("bf16", "fp16"), default=["bf16", "fp16"])
    parser.add_argument("--seed", type=int, default=1179659)
    parser.add_argument("--rounds", type=int, default=13)
    parser.add_argument("--window-ms", type=float, default=150)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-batch", type=int, default=32)
    args = parser.parse_args()
    if min(*args.seq, args.heads, args.rounds, args.window_ms) <= 0:
        parser.error("Shapes, rounds and windows must be positive.")

    from benchmarks.experiments.ada.compare_l40s import graph_kernels
    from benchmarks.flashattention_backends import load_backend
    from kohakufa_cute.fa4_sm80 import gb10

    if not gb10._qualified_device(torch.cuda.current_device()):
        raise ValueError("The experiment requires the full 48-SM GB10.")
    torch.set_num_threads(4)
    fa2 = load_backend("fa2")
    from benchmarks.experiments.gb10.gb10_width_controls import parent as load_parent

    parent = load_parent(str(args.parent_source))

    def identity(graph):
        import cuda.bindings.driver as driver

        status, nodes, count = driver.cuGraphGetNodes(driver.CUgraph(graph.raw_cuda_graph()), 256)
        assert int(status) == 0
        values = []
        for node in nodes[:count]:
            status, kind = driver.cuGraphNodeGetType(node)
            assert int(status) == 0
            if kind == driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
                status, params = driver.cuGraphKernelNodeGetParams(node)
                assert int(status) == 0
                values.append(int(params.func))
        return values

    report = {
        "scope": (
            "Warmed NoGrad device-event graph time"
            if args.graph
            else "Complete warmed NoGrad eager wall time; graph event medians divide captured batches by their real attention-call count"
        ),
        "device": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "fa2": fa2.metadata(),
        "rounds": args.rounds,
        "window_ms": args.window_ms,
        "graph": args.graph,
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": {
            str(p.relative_to(Path(__file__).resolve().parents[3])): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__).resolve().parents[3] / "src/kohakufa_cute/native.cpp",
                Path(__file__).resolve().parents[3] / "src/kohakufa_cute/native.py",
                *sorted((Path(__file__).resolve().parents[3] / "src/kohakufa_cute/fa4_sm80").glob("*.py")),
            ]
        },
        "seed": args.seed,
        "policy_selection_seeds": [1048583],
        "preceding_policy": parent.width_baseline_provenance,
        "scope_note": "Actual public GB10 new and preceding policy plus raw, generic, FA2 and SDPA controls.",
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for sequence, dimension, dtype_name, causal in itertools.product(args.seq, args.dim, args.dtype, (False, True)):
        torch.manual_seed(args.seed)
        dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
        inputs = [torch.randn(1, args.heads, sequence, dimension, device="cuda", dtype=dtype) for _ in range(3)]
        runner = gb10._long_d64_runner(causal)
        functions = {
            "parent": lambda: parent.scaled_dot_product_attention(*inputs, is_causal=causal),
            "raw_inference": lambda: runner.inference(*inputs, causal=causal),
            "selected": lambda: gb10.scaled_dot_product_attention(*inputs, is_causal=causal),
            "fa2": lambda: fa2(*inputs, is_causal=causal),
            "sdpa": lambda: torch.nn.functional.scaled_dot_product_attention(*inputs, is_causal=causal),
        }
        input_refs = [retain_tensor(t, args.output.parent) for t in inputs]
        (args.output.parent / (args.output.stem + f"-s{sequence}-{dtype_name}-c{int(causal)}-inputs.json")).write_text(
            json.dumps(input_refs, indent=2) + "\n"
        )
        expected = references(inputs, causal)
        actual_graphs, owners, execution, errors, identities = {}, [], {}, {}, {}
        with torch.no_grad():
            for name, function in functions.items():
                for _ in range(3):
                    output = function()
                errors[name] = assess_output(output, expected)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(graph):
                    for _ in range(args.capture_batch):
                        output = function()
                actual_graphs[name] = graph_kernels(graph)
                identities[name] = identity(graph)
                owners.append((graph, output))
                execution[name] = graph.replay if args.graph else function
            torch.cuda.synchronize()
            started = time.perf_counter()
            while time.perf_counter() - started < 1:
                for _ in range(4):
                    execution["raw_inference"]()
                torch.cuda.synchronize()
            counts = {}
            for name, function in execution.items():
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(20):
                    function()
                torch.cuda.synchronize()
                counts[name] = max(
                    1 if args.graph else 20, math.ceil(args.window_ms / ((time.perf_counter() - start) * 1000 / 20))
                )
            order, samples = list(execution), {name: [] for name in execution}
            for repeat in range(args.rounds):
                offset = repeat % len(order)
                for name in order[offset:] + order[:offset]:
                    torch.cuda.synchronize()
                    if args.graph:
                        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(counts[name]):
                            execution[name]()
                        stop.record()
                        stop.synchronize()
                        elapsed = start.elapsed_time(stop)
                    else:
                        start = time.perf_counter()
                        for _ in range(counts[name]):
                            execution[name]()
                        torch.cuda.synchronize()
                        elapsed = (time.perf_counter() - start) * 1000
                    samples[name].append(elapsed / counts[name] / (args.capture_batch if args.graph else 1))
            paths = {}
            for name, function in functions.items():
                with torch.profiler.profile(
                    activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA),
                    profile_memory=True,
                ) as profiler:
                    function()
                torch.cuda.synchronize()
                paths[name] = {
                    "cuda_kernels": sorted(
                        {event.name for event in profiler.events() if "cuda" in str(event.device_type).lower()}
                    ),
                    "allocation_operators": {
                        event.key: event.count for event in profiler.key_averages() if "empty" in event.key
                    },
                }
        assert identities["selected"] == identities["raw_inference"]
        row = {
            "selected_raw_same_cuda_functions": True,
            "shape": list(inputs[0].shape),
            "input_tensor_refs": input_refs,
            "dtype": dtype_name,
            "causal": causal,
            "schedule": dict(
                asdict(
                    Schedule(forward_m=128, forward_n=64, forward_stages=1 if causal else 2, query_stages=1, dout_stages=1)
                ),
                forward_threads=128,
            ),
            "inference_selected": gb10.inference_compatible(*inputs, causal=causal),
            "errors": errors,
            "graph_node_counts": {name: len(values) for name, values in actual_graphs.items()},
            "unique_graph_kernels": {
                name: list({json.dumps(r, sort_keys=True): r for r in values}.values())
                for name, values in actual_graphs.items()
            },
            "eager_paths": paths,
            "replays_per_window": counts,
            "calls_per_window": {name: n * (args.capture_batch if args.graph else 1) for name, n in counts.items()},
            "capture_batch": args.capture_batch,
            "samples_ms": samples,
            "median_ms": {name: statistics.median(values) for name, values in samples.items()},
        }
        report["rows"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print({key: row[key] for key in ("shape", "dtype", "causal", "median_ms")}, flush=True)


if __name__ == "__main__":
    main()
