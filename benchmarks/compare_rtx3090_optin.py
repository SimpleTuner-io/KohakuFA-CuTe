"""Measure the bounded public RTX 3090 opt-in entry."""

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--source-revision", required=True)
args = parser.parse_args()
root = args.output.parent
root.mkdir(parents=True, exist_ok=True)
os.environ.update(CUTE_DSL_KEEP="ptx,cubin", CUTE_DSL_DUMP_DIR=str(root / "code"))
from benchmarks.compare_ada import capture, measure
from benchmarks.compare_flashattention import source_metadata
from benchmarks.compare_l40s import board_state, driver_api_version, graph_kernels, loaded_libraries
from benchmarks.compiled_artifacts import retain_code
from benchmarks.flashattention_backends import load_backend
from benchmarks.reference import fp64_reference, metrics, rounded_reference
from kohakufa_cute import kernels, rtx3090_attention

report = {
    "hypothesis": "Measure the bounded public RTX3090 opt-in against native CuTe, standalone FA2 and SDPA with identical inputs and complete training calls.",
    "protocol": "Same-process paired graph and synchronized eager windows; seven rotating 100-ms rounds; training forward, retained-forward backward and complete forward+backward; one-second candidate warmup; profiling separate.",
    "arithmetic": "Maintained rounded-P/dS, actual-O delta, FP32 accumulations and late scale. Independent per-head/component FP64 limits: 1% BF16, 0.5% FP16.",
    "candidate_sha256": hashlib.sha256(
        (Path(__file__).resolve().parents[1] / "src/kohakufa_cute/rtx3090.py").read_bytes()
    ).hexdigest(),
    "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "board_before": board_state(),
    "device": {
        "name": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "sm": torch.cuda.get_device_properties(0).multi_processor_count,
    },
    "torch": str(torch.__version__),
    "cuda": torch.version.cuda,
    "driver_api": driver_api_version(),
    "packages": {
        name: importlib.metadata.version(name)
        for name in ("nvidia-cutlass-dsl", "apache-tvm-ffi", "triton", "quack-kernels", "flash-attn")
    },
    "rows": [],
    "compiler_artifacts": [],
}
scope = {}
original_compile = kernels.cute.compile


def observe(program, *a, **kw):
    function = original_compile(program, *a, **kw)
    report["compiler_artifacts"].append(
        {
            "scope": dict(scope),
            "program": type(program).__module__ + "." + type(program).__name__,
            "artifacts": retain_code(function, root / "retained"),
        }
    )
    return function


kernels.cute.compile = observe
report.update(source_metadata(Path(__file__).resolve().parents[1], Path(kernels.__file__).parent, args.source_revision))


def save():
    args.output.write_text(json.dumps(report, indent=2) + "\n")


def retain_failure(kind, error, trace):
    try:
        if "inputs" in globals():
            torch.save(
                {"inputs": [t.detach().cpu() for t in inputs], "dout": dout.cpu(), "causal": causal, "config": config},
                root / "failed-operands.pt",
            )
    finally:
        sys.__excepthook__(kind, error, trace)


sys.excepthook = retain_failure
save()
for dtype, dimension in itertools.product((torch.bfloat16, torch.float16), (64, 128)):
    config = {"entry": "rtx3090_attention"}
    functions = {
        "native": load_backend("cute"),
        "fa2": load_backend("fa2"),
        "sdpa": load_backend("sdpa"),
        "public": rtx3090_attention,
    }
    report["backends"] = {name: function.metadata() for name, function in functions.items() if hasattr(function, "metadata")}
    for sequence in (1024, 4096):
        for causal in (False, True):
            torch.manual_seed(927)
            q, k, v, dout = [torch.randn(1, 16, sequence, dimension, dtype=dtype, device="cuda") for _ in range(4)]
            inputs = (q, k, v)
            row = {
                "sequence": sequence,
                "dimension": dimension,
                "causal": causal,
                "dtype": str(dtype),
                "config": config,
                "results": {},
            }
            references = [
                fp64_reference(tuple(t[:, h : h + 1] for t in inputs), dout[:, h : h + 1], causal=causal) for h in range(16)
            ]
            graphs, owners = {}, {}
            for name, function in functions.items():
                scope.update(name=name, dtype=str(dtype), dimension=dimension, sequence=sequence, causal=causal)
                operands = [t.detach().requires_grad_(True) for t in inputs]
                out = function(*operands, is_causal=causal)
                actual = (out, *torch.autograd.grad(out, operands, dout))
                errors, rounding = [], []
                for h, ref in enumerate(references):
                    rounded = rounded_reference(
                        tuple(t[:, h : h + 1] for t in inputs), dout[:, h : h + 1], causal=causal, output=out[:, h : h + 1]
                    )
                    for got, want, low in zip(actual, ref, rounded):
                        e = metrics(got[:, h : h + 1], want)
                        assert e["finite"] and (
                            e["relative_error"] < (0.01 if dtype == torch.bfloat16 else 0.005)
                            if e["reference_norm"]
                            else e["max_absolute_error"] == 0
                        ), (name, dimension, sequence, causal, e)
                        errors.append(e)
                        rounding.append(metrics(got[:, h : h + 1], low))
                row["results"][name] = {"fp64": errors, "rounded": rounding, "phases": {}}
                for phase in ("forward", "backward", "joint"):
                    graph, storage = capture(function, inputs, dout, causal, phase, keep_graph=True)
                    key = name, phase
                    graphs[key], owners[key] = graph, storage
                    row["results"][name]["phases"][phase] = {"kernels": graph_kernels(graph), "samples_ms": []}
            keys = list(graphs)
            warm_until = time.perf_counter() + 1
            while time.perf_counter() < warm_until:
                for _ in range(8):
                    graphs["public", "joint"].replay()
                torch.cuda.synchronize()
            counts = {key: max(2, min(4096, int(100 / measure(graphs[key], 3)))) for key in keys}
            for repeat in range(7):
                order = keys[repeat % len(keys) :] + keys[: repeat % len(keys)]
                for key in order:
                    row["results"][key[0]]["phases"][key[1]]["samples_ms"].append(measure(graphs[key], counts[key]))
            for result in row["results"].values():
                for phase in result["phases"].values():
                    phase["median_ms"] = statistics.median(phase["samples_ms"])
            eager, eager_owners = {}, []
            for name, function in functions.items():
                leaves = [t.detach().requires_grad_(True) for t in inputs]
                saved_out = function(*leaves, is_causal=causal)
                eager_owners.append((leaves, saved_out))
                for phase in ("forward", "backward", "joint"):

                    def execute(function=function, leaves=leaves, saved_out=saved_out, phase=phase):
                        if phase == "backward":
                            return torch.autograd.grad(saved_out, leaves, dout, retain_graph=True)
                        output = function(*leaves, is_causal=causal)
                        return (output, *torch.autograd.grad(output, leaves, dout)) if phase == "joint" else output

                    for _ in range(3):
                        execute()
                    eager[name, phase] = execute
                    row["results"][name].setdefault("eager", {})[phase] = {"samples_ms": []}
            eager_counts = {}
            for key, execute in eager.items():
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(3):
                    execute()
                torch.cuda.synchronize()
                per_call = (time.perf_counter() - start) * 1000 / 3
                eager_counts[key] = max(3, min(4096, math.ceil(100 / per_call)))
            eager_keys = list(eager)
            for repeat in range(7):
                order = eager_keys[repeat % len(eager_keys) :] + eager_keys[: repeat % len(eager_keys)]
                for key in order:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(eager_counts[key]):
                        eager[key]()
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - start) * 1000 / eager_counts[key]
                    row["results"][key[0]]["eager"][key[1]]["samples_ms"].append(elapsed)
            for result in row["results"].values():
                for phase in result["eager"].values():
                    phase["median_ms"] = statistics.median(phase["samples_ms"])
            row["board_after"] = board_state()
            report["rows"].append(row)
            save()
            print(
                str(dtype),
                dimension,
                sequence,
                causal,
                {
                    name: {phase: round(value["median_ms"], 6) for phase, value in result["phases"].items()}
                    for name, result in row["results"].items()
                },
                flush=True,
            )
            del graphs, owners, references, actual, operands, out, eager, eager_owners
            torch.cuda.empty_cache()
report["board_after"] = board_state()
report["loaded_libraries"] = loaded_libraries()
save()
