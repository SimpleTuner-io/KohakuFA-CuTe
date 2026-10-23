"""Capture and compare real text-only Clef attention and model forwards.

Use an immutable local Hugging Face snapshot. Raw tensor captures and profiler
traces belong in the explicitly supplied output directory outside the source tree.
The native MHA inference fastpath remains enabled in every variant.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

REVISION = "17f0b0ad64efb65d273590632833508766b2aae6"
BACKENDS = ("native", "cute-backbone", "cute-head", "cute-all")


def fp64_forward_chunks(q, k, v, mask=None, *, causal=False, scale=None, chunk=128):
    """Bound oracle memory while retaining absolute causal query positions."""
    import torch

    q, k, v = (tensor.double() for tensor in (q, k, v))
    groups = q.shape[1] // k.shape[1]
    k, v = (tensor.repeat_interleave(groups, dim=1) for tensor in (k, v))
    result = torch.empty_like(q)
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    for start in range(0, q.shape[2], chunk):
        stop = min(q.shape[2], start + chunk)
        scores = (q[:, :, start:stop] @ k.transpose(-1, -2)) * scale
        if mask is not None:
            visible = mask.expand(q.shape[0], q.shape[1], q.shape[2], k.shape[2])[:, :, start:stop]
            scores.masked_fill_(~visible, -torch.inf)
        if causal:
            visible = (
                torch.arange(k.shape[2], device=q.device)[None, :] <= torch.arange(start, stop, device=q.device)[:, None]
            )
            scores.masked_fill_(~visible, -torch.inf)
        valid = torch.isfinite(scores).any(-1, keepdim=True)
        probability = scores.masked_fill(~valid, 0).softmax(-1).masked_fill(~valid, 0)
        result[:, :, start:stop] = probability @ v
    return result


def replay_captures(args):
    """Measure captured calls through the public API without loading model weights."""
    import torch
    import torch.nn.functional as functional

    from kohakufa_cute import scaled_dot_product_attention

    if min(args.rounds, args.calls_per_graph, args.window_ms) <= 0:
        raise ValueError("Positive rounds, graph sizes and timing windows are required")
    if args.preheat_seconds < 0:
        raise ValueError("Preheat duration must be nonnegative")
    report = json.loads((args.captures / "report.json").read_text())
    result = {
        "format": "clef-attention-replay-v1",
        "model": report["model"],
        "revision": report["revision"],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "protocol": {
            "rounds": args.rounds,
            "calls_per_graph": args.calls_per_graph,
            "target_window_ms": args.window_ms,
            "rotated_backends": True,
            "forward_only": True,
            "preheat_seconds": args.preheat_seconds,
        },
        "cases": [],
    }
    with torch.inference_mode():
        for case in report["cases"]:
            for call in case["calls"]:
                if args.attention_dim is not None and call["q"]["shape"][-1] not in args.attention_dim:
                    continue
                data = torch.load(args.captures / call["tensor_file"], map_location="cpu", weights_only=True)
                inputs = []
                for name in ("q", "k", "v"):
                    spec = call[name]
                    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.dtype, data[name].dtype)
                    size = spec["storage_offset"] + sum((n - 1) * s for n, s in zip(spec["shape"], spec["stride"])) + 1
                    storage = torch.empty(size, device="cuda", dtype=dtype)
                    tensor = storage.as_strided(spec["shape"], spec["stride"], spec["storage_offset"])
                    tensor.copy_(data[name])
                    inputs.append(tensor)
                q, k, v = inputs
                mask = None if data["mask"] is None else data["mask"].to("cuda")
                kwargs = dict(
                    attn_mask=mask,
                    dropout_p=call["dropout_p"],
                    is_causal=call["is_causal"],
                    scale=call["scale"],
                    enable_gqa=call["enable_gqa"],
                )
                if call["dropout_p"] or mask is not None and mask.dtype != torch.bool:
                    raise ValueError("Replay requires dropout-free boolean-mask attention")
                reference = fp64_forward_chunks(q, k, v, mask, causal=call["is_causal"], scale=call["scale"])
                tolerance = 0.02 if q.dtype == torch.bfloat16 else 0.003
                functions = {"sdpa": functional.scaled_dot_product_attention, "cute": scaled_dot_product_attention}
                graphs, outputs, repeats, samples, errors = {}, {}, {}, {}, {}

                def duration(graph, count):
                    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                    start.record()
                    for _ in range(count):
                        graph.replay()
                    stop.record()
                    stop.synchronize()
                    return start.elapsed_time(stop) / count / args.calls_per_graph

                for name, function in functions.items():
                    for _ in range(3):
                        expected = function(q, k, v, **kwargs)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(args.calls_per_graph):
                            observed = function(q, k, v, **kwargs)
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(observed, expected, atol=0, rtol=0)
                    torch.testing.assert_close(observed.double(), reference, atol=tolerance, rtol=tolerance)
                    errors[name] = {
                        "max_abs": (observed.double() - reference).abs().max().item(),
                        "relative_l2": ((observed.double() - reference).norm() / reference.norm()).item(),
                    }
                    saved_v = v.clone()
                    v.add_(1)
                    changed = function(q, k, v, **kwargs)
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(observed, changed, atol=0, rtol=0)
                    if torch.equal(expected, changed):
                        raise AssertionError("Changed values did not change the attention output")
                    v.copy_(saved_v)
                    graph.replay()
                    torch.cuda.synchronize()
                    graphs[name], outputs[name], samples[name] = graph, observed, []
                    estimate = duration(graph, 3)
                    repeats[name] = max(1, math.ceil(args.window_ms / (estimate * args.calls_per_graph)))
                preheat_until = time.perf_counter() + args.preheat_seconds
                while time.perf_counter() < preheat_until:
                    for name, graph in graphs.items():
                        duration(graph, repeats[name])
                for index in range(args.rounds):
                    names = list(graphs)
                    for name in names[index % len(names) :] + names[: index % len(names)]:
                        samples[name].append(duration(graphs[name], repeats[name]))
                result["cases"].append(
                    {
                        "record": case["id"],
                        "q": descriptor(q),
                        "k": descriptor(k),
                        "v": descriptor(v),
                        "mask": descriptor(mask),
                        "causal": call["is_causal"],
                        "dtype_cast": args.dtype,
                        "graph_changed_values_verified": True,
                        "fp64": errors,
                        "samples_ms": samples,
                        "medians_ms": {name: statistics.median(values) for name, values in samples.items()},
                    }
                )
                (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
                print(case["id"], q.shape[2], k.shape[2], result["cases"][-1]["medians_ms"], flush=True)
                del outputs, graphs, reference
    return result


def make_record(index, questions, state_words):
    """Match the infrastructure's existing four-choice text probe recipe."""
    words = ("invoice router customer risk policy payment service contract evidence priority").split()
    return {
        "id": f"bench-{index}",
        "state": {
            "case_id": f"bench-{index}",
            "summary": " ".join(words[i % len(words)] for i in range(state_words)),
            "amount": 1250 + index,
            "currency": "USD",
            "signals": {"escalated": index % 2 == 0, "region": "us", "latency_ms": 217},
        },
        "questions": {
            f"field_{q:03d}": {
                "type": "choice",
                "instructions": f"Choose the best route for field {q}.",
                "criteria": {f"option_{o:02d}": f"Evidence pattern {o} for field {q}." for o in range(4)},
            }
            for q in range(questions)
        },
    }


def descriptor(tensor):
    if tensor is None:
        return None
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "storage_offset": tensor.storage_offset(),
    }


def import_joint(root):
    spec = importlib.util.spec_from_file_location("clef_forward_joint", root / "joint_schema_model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def attention_scope(model, backend="native", capture=None, validation=None):
    """Instrument only capture passes; warm timing uses a minimal SDPA adapter."""
    import torch
    import torch.nn.functional as functional

    if backend == "native" and capture is None:
        yield
        return
    native = functional.scaled_dot_product_attention
    cute = None
    if backend != "native":
        from kohakufa_cute import scaled_dot_product_attention

        cute = scaled_dot_product_attention
    owner = ["unknown"]
    handles = []
    modules = model.named_modules() if capture is not None or validation is not None or backend != "cute-all" else []
    for name, module in modules:
        if (
            name == "head"
            or name.endswith("self_attn")
            or (name.startswith("head.") and isinstance(module, torch.nn.MultiheadAttention))
        ):

            def before(module, args, name=name):
                owner.append(name)

            def after(module, args, output):
                owner.pop()

            handles.extend(
                (
                    module.register_forward_pre_hook(before),
                    module.register_forward_hook(after, always_call=True),
                )
            )

    def dispatch(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
        scope = "head" if owner[-1].startswith("head") else "backbone"
        selected = backend == "cute-all" or backend == "cute-" + scope
        if capture is not None:
            item = {
                "owner": owner[-1],
                "q": descriptor(q),
                "k": descriptor(k),
                "v": descriptor(v),
                "mask": descriptor(attn_mask),
                "dropout_p": dropout_p,
                "is_causal": is_causal,
                "scale": kwargs.get("scale"),
                "enable_gqa": kwargs.get("enable_gqa", False),
            }
        result = (cute if selected else native)(q, k, v, attn_mask, dropout_p, is_causal, **kwargs)
        if capture is not None:
            item["output"] = descriptor(result)
            capture(item, (q, k, v, attn_mask))
        if selected and validation is not None:
            expected = native(q, k, v, attn_mask, dropout_p, is_causal, **kwargs)
            difference = result.float() - expected.float()
            passed = True
            try:
                torch.testing.assert_close(result, expected, atol=0.02, rtol=0.02)
            except AssertionError:
                passed = False
            validation.append(
                {
                    "owner": owner[-1],
                    "passed": passed,
                    "max_abs": difference.abs().max().item(),
                    "relative_l2": (difference.norm() / expected.float().norm().clamp_min(1e-30)).item(),
                    "native_output": descriptor(expected),
                    "cute_output": descriptor(result),
                }
            )
        return result

    functional.scaled_dot_product_attention = dispatch
    try:
        yield
    finally:
        functional.scaled_dot_product_attention = native
        for handle in handles:
            handle.remove()


def measure(fn, repeat):
    import torch

    samples, wall = [], []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        now = time.perf_counter()
        start.record()
        output = fn()
        stop.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(stop))
        wall.append((time.perf_counter() - now) * 1000)
        del output
    return {
        "cuda_event_ms": samples,
        "cuda_event_median_ms": statistics.median(samples),
        "synchronized_wall_ms": wall,
        "synchronized_wall_median_ms": statistics.median(wall),
    }


def compare(reference, actual):
    import torch

    maximum = 0.0
    decisions = flips = 0
    passed = True
    scores = []
    for expected_record, actual_record in zip(reference, actual, strict=True):
        for expected, observed in zip(expected_record, actual_record, strict=True):
            try:
                torch.testing.assert_close(observed, expected, atol=0.02, rtol=0.02)
            except AssertionError:
                passed = False
            maximum = max(maximum, (observed - expected).float().abs().max().item())
            decisions += 1
            flips += int(observed.argmax().item() != expected.argmax().item())
            scores.append({"native": expected.float().tolist(), "candidate": observed.float().tolist()})
    return {
        "passed": passed and flips == 0,
        "logits_max_abs": maximum,
        "decisions_checked": decisions,
        "flips": flips,
        "scores": scores,
    }


def run(args):
    import torch

    output = args.output
    if output.resolve().is_relative_to(Path(__file__).resolve().parent.parent):
        raise ValueError("Keep raw captures outside the source tree")
    if min(args.repeat, args.rounds, args.max_length, *args.questions, *args.batches) < 1 or args.warmup < 0:
        raise ValueError("Positive timing counts, token limits and sizes are required")
    if min(args.state_words) < 0:
        raise ValueError("State word counts must be nonnegative")
    if args.captures is None and args.model_id != "Cloudflare/clef-flash" and args.revision is None:
        raise ValueError("Provide an immutable --revision for the selected model identifier")
    output.mkdir(parents=True, exist_ok=False)
    if args.captures is not None:
        return replay_captures(args)
    joint = import_joint(args.model)
    model, processor = joint.load_release_model(args.model, device="cuda", dtype=torch.bfloat16, attn_implementation="sdpa")
    report = {
        "format": "clef-forward-v1",
        "model": args.model_id,
        "revision": args.revision or REVISION,
        "source_hashes": {
            name: hashlib.sha256((args.model / name).read_bytes()).hexdigest()
            for name in ("config.json", "joint_head_config.json", "joint_schema_model.py")
        },
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "mha_fastpath_enabled": torch.backends.mha.get_fastpath_enabled(),
        "timing_scope": "encoded_batch_to_device_logits; excludes_tokenization_and_collation",
        "precision": "unmodified_bfloat16_weights; no_autocast_or_TF32_changes",
        "correctness": {"logits_atol": 0.02, "logits_rtol": 0.02, "decision_flips": 0},
        "cases": [],
    }
    with torch.inference_mode():
        for words in args.state_words:
            for questions in args.questions:
                for batch_size in args.batches:
                    records = [make_record(i + 1, questions, words) for i in range(batch_size)]
                    encoded = [
                        joint.encode_record(processor.tokenizer, row, processor=processor, max_length=2**31 - 1)
                        for row in records
                    ]
                    lengths = [len(row.input_ids) for row in encoded]
                    if max(lengths) > args.max_length:
                        continue
                    batch = joint.collate_records(encoded, processor.tokenizer.pad_token_id, torch.device("cuda"))
                    label = f"w{words}-q{questions}-b{batch_size}"
                    case = {
                        "id": label,
                        "state_words": words,
                        "questions": questions,
                        "batch_size": batch_size,
                        "tokens": lengths,
                        "calls": [],
                        "backends": {},
                    }
                    seen = {}

                    def capture(item, tensors):
                        signature = json.dumps({k: v for k, v in item.items() if k != "owner"}, sort_keys=True)
                        if signature in seen:
                            seen[signature]["count"] += 1
                            seen[signature]["owners"].append(item["owner"])
                            return
                        item.update(count=1, owners=[item.pop("owner")])
                        seen[signature] = item
                        if args.capture_tensors:
                            payload = f"{label}-call{len(case['calls'])}.pt"
                            torch.save(
                                {name: None if t is None else t.cpu() for name, t in zip(("q", "k", "v", "mask"), tensors)},
                                output / payload,
                            )
                            item["tensor_file"] = payload
                        case["calls"].append(item)

                    with attention_scope(model, capture=capture):
                        reference = model(batch)
                    if args.profile:
                        with torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                            record_shapes=True,
                        ) as prof:
                            model(batch)
                        prof.export_chrome_trace(str(output / f"{label}-native.trace.json"))
                        case["native_operators"] = [
                            {"name": row.key, "count": row.count, "cuda_us": row.device_time_total}
                            for row in prof.key_averages()
                            if "attention" in row.key
                        ]
                        case["native_copy_operators"] = [
                            {
                                "name": row.key,
                                "count": row.count,
                                "cuda_us": row.device_time_total,
                                "input_shapes": row.input_shapes,
                            }
                            for row in prof.key_averages(group_by_input_shape=True)
                            if row.key in {"aten::clone", "aten::contiguous", "aten::copy_"}
                        ]
                    for backend in args.backends:
                        try:
                            attention_validation = []
                            with attention_scope(model, backend, validation=attention_validation):
                                checks = compare(reference, model(batch))
                            checks["attention_checks"] = attention_validation
                            if not checks["passed"] or not all(item["passed"] for item in attention_validation):
                                case["backends"][backend] = {"status": "numerical_failure", **checks}
                                continue
                            with attention_scope(model, backend):
                                for _ in range(args.warmup):
                                    model(batch)
                            case["backends"][backend] = {"status": "ok", **checks, "rounds": []}
                        except Exception as exc:
                            traceback.print_exc()
                            case["backends"][backend] = {
                                "status": "failed",
                                "error_type": type(exc).__name__,
                                "attention_checks": attention_validation,
                            }
                    for round_index in range(args.rounds):
                        rotation = round_index % len(args.backends)
                        order = args.backends[rotation:] + args.backends[:rotation]
                        for backend in order:
                            measured = case["backends"][backend]
                            if measured["status"] != "ok":
                                continue
                            with attention_scope(model, backend):
                                timings = measure(lambda: model(batch), args.repeat)
                            measured["rounds"].append({"index": round_index, **timings})
                    for measured in case["backends"].values():
                        if measured["status"] == "ok":
                            for metric in ("cuda_event", "synchronized_wall"):
                                samples = [value for row in measured["rounds"] for value in row[metric + "_ms"]]
                                measured[metric + "_ms"] = samples
                                measured[metric + "_median_ms"] = statistics.median(samples)
                    if args.nsys:
                        for backend, measured in case["backends"].items():
                            if measured["status"] != "ok":
                                continue
                            with attention_scope(model, backend):
                                torch.cuda.synchronize()
                                torch.cuda.cudart().cudaProfilerStart()
                                try:
                                    with torch.cuda.nvtx.range(f"clef.{label}.{backend}"):
                                        model(batch)
                                        torch.cuda.synchronize()
                                finally:
                                    torch.cuda.cudart().cudaProfilerStop()
                    report["cases"].append(case)
                    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                    print(label, lengths, {name: row["status"] for name, row in case["backends"].items()}, flush=True)
                    del reference, batch
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path)
    source.add_argument("--captures", type=Path, help="Replay report.json and tensor captures without loading the model")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--model-id", default="Cloudflare/clef-flash", help="Model identifier recorded in capture metadata")
    result.add_argument("--revision", help="Immutable snapshot revision; defaults to the pinned Clef-Flash snapshot")
    result.add_argument("--state-words", type=int, nargs="+", default=[16, 256, 2048])
    result.add_argument("--questions", type=int, nargs="+", default=[3, 64])
    result.add_argument("--batches", type=int, nargs="+", default=[1])
    result.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    result.add_argument("--max-length", type=int, default=16384)
    result.add_argument("--repeat", type=int, default=5)
    result.add_argument("--rounds", type=int, default=3)
    result.add_argument("--warmup", type=int, default=3)
    result.add_argument("--capture-tensors", action="store_true")
    result.add_argument("--profile", action="store_true")
    result.add_argument("--nsys", action="store_true", help="Emit CUDA profiler ranges after warm timings")
    result.add_argument("--attention-dim", type=int, nargs="+", help="Filter captured head dimensions")
    result.add_argument("--dtype", choices=("bf16", "fp16"), help="Optional dtype cast for captured attention replay")
    result.add_argument("--calls-per-graph", type=int, default=32)
    result.add_argument("--window-ms", type=float, default=100)
    result.add_argument("--preheat-seconds", type=float, default=5, help="Untimed graph replay after correctness checks")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
