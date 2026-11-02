"""Experimental SM80 width enabling; public dispatch stays unchanged."""

import argparse
import itertools
import json
import traceback
from dataclasses import asdict
from pathlib import Path

import torch


def configuration(preset, group=1):
    from kohakufa_cute.fa4_sm80._attention import Schedule

    schedule = Schedule(
        forward_m=64,
        backward_n=32 if preset == "n32-resident" else 64,
        query_stages=1,
        dout_stages=1,
        score_m_warps=4,
        kv_m_warps=2 if preset == "n64-shared-fa2" else 1,
        dq_m_warps=2,
        v_in_regs=preset == "n32-resident",
        causal_head_group=group,
    )
    return 256, {**asdict(schedule), "accumulator_head_dim": 0, "forward_threads": 128}


def make_attention(*, preset="n32-resident", group=1):
    from kohakufa_cute.native import extension

    dimension, settings = configuration(preset, group)
    runner = extension().AttentionRunner("warp", settings)

    def operation(q, k, v, *, is_causal=False, scale=None):
        if q.shape[-1] != dimension:
            raise ValueError("The explicit width candidate requires its declared dimension.")
        return runner(q, k, v, causal=is_causal, scale=scale)

    return operation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("n32-resident", "n64-shared-kv1", "n64-shared-fa2"), required=True)
    parser.add_argument("--group", type=int, choices=(1, 4), default=1)
    parser.add_argument("--resource-only", action="store_true")
    parser.add_argument("--cpu-reference", action="store_true")
    parser.add_argument("--agreement", action="store_true")
    parser.add_argument("--preserved-case", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from benchmarks.compare_ada import capture
    from benchmarks.compare_l40s import graph_kernels
    from benchmarks.device_rounded_reference import assess_device_intermediates
    from benchmarks.validate_sm80_static import statistics

    assert torch.cuda.get_device_capability() == (8, 0)
    torch.set_num_threads(4)
    dimension, settings = configuration(args.preset, args.group)
    report = {
        "preset": args.preset,
        "dimension": dimension,
        "configuration": settings,
        "shared_capacity": torch.cuda.get_device_properties(0).shared_memory_per_block_optin,
        "resource_only": args.resource_only,
        "cpu_reference": args.cpu_reference,
        "rows": [],
        "status": "running",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def copied(values):
        return tuple(t.detach().cpu() for t in values)

    def check(actual, inputs, dout, causal, scale, directed=False):
        return assess_device_intermediates(
            actual, inputs, dout, causal=causal, scale=scale, directed=directed, agreement=args.agreement
        )

    operation = make_attention(preset=args.preset, group=args.group)
    actual, saved = (), []
    fixtures = (
        ((torch.bfloat16, 129, False),)
        if args.resource_only
        else itertools.product((torch.bfloat16, torch.float16), (128, 129, 257), (False, True))
    )
    try:
        for number, (dtype, sequence, causal) in enumerate(fixtures):
            torch.manual_seed(4201 + number)
            inputs = [torch.randn(1, 2, sequence, dimension, device="cuda", dtype=dtype) for _ in range(3)]
            dout = torch.randn_like(inputs[0])
            torch.save(
                {"inputs": copied(inputs), "dout": dout.detach().cpu(), "configuration": settings, "causal": causal},
                args.output.with_name(args.output.stem + ".input-" + str(number) + ".pt"),
            )
            row = {"shape": list(inputs[0].shape), "dtype": str(dtype), "causal": causal}
            report["current"] = row
            save()
            if args.resource_only:
                graph, storage = capture(operation, inputs, dout, causal, "joint", keep_graph=True)
                row["kernels"] = graph_kernels(graph)
                assert len(row["kernels"]) == 4 and all(r["BINARY_VERSION"] == 80 for r in row["kernels"])
                assert all(r["shared_bytes"] <= report["shared_capacity"] for r in row["kernels"])
            else:
                operands = [t.detach().requires_grad_(True) for t in inputs]
                saved = []
                with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
                    output = operation(*operands, is_causal=causal)
                assert len(saved) == 6
                device = "cpu" if args.cpu_reference else "cuda"
                if args.agreement:
                    statistics(operands, saved[-2], saved[-1], causal, dimension**-0.5)
                row["statistics"] = statistics(
                    tuple(t.detach().to(device) for t in operands),
                    saved[-2].detach().to(device),
                    saved[-1].detach().to(device),
                    causal,
                    dimension**-0.5,
                )
                actual = output, *torch.autograd.grad(output, operands, dout)
                row["errors"] = check(actual, operands, dout, causal, None)
                graph, storage = capture(operation, inputs, dout, causal, "joint", keep_graph=True)
                row["kernels"] = graph_kernels(graph)
                operands, _, _, actual = storage
                row["changed_graphs"] = []
                for _ in range(2):
                    with torch.no_grad():
                        for t in (*operands, dout):
                            t.copy_(torch.randn_like(t))
                        for t in actual:
                            t.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize()
                    row["changed_graphs"].append(check(actual, operands, dout, causal, None))
            report["rows"].append(row)
            report.pop("current", None)
            save()
        if not args.resource_only and args.preserved_case is not None:
            data = torch.load(args.preserved_case, map_location="cpu", weights_only=False)
            inputs = tuple(t.cuda() for t in data["inputs"])
            operands = tuple(t.detach().requires_grad_(True) for t in inputs)
            dout = data["dout"].cuda()
            causal, scale = data["case"]["causal"], data["case"]["scale"]
            report["current"] = {"case": "exact-preserved-common-offset", **data["case"]}
            save()
            output = operation(*operands, is_causal=causal, scale=scale)
            actual = output, *torch.autograd.grad(output, operands, dout)
            report["rows"].append(
                {
                    "case": "exact-preserved-common-offset",
                    "errors": check(actual, operands, dout, causal, scale, directed=True),
                }
            )
            report.pop("current", None)
        report["status"] = "supported_resource_pending_math" if args.resource_only else "qualified_math"
    except Exception as error:
        message = str(error).lower()
        report["status"] = (
            "unsupported_resource"
            if any(
                x in message
                for x in (
                    "shared memory capacity",
                    "shared memory exceeds",
                    "exceeds the available shared",
                    "storage exceeds the actual device limit",
                )
            )
            else "implementation_failure"
        )
        try:
            torch.save(
                {
                    "inputs": copied(inputs),
                    "dout": dout.detach().cpu(),
                    "actual": copied(actual),
                    "saved": copied(saved),
                    "configuration": settings,
                    "case": report.get("current"),
                },
                args.output.with_suffix(".failed.pt"),
            )
        except Exception as retain_error:
            report["failure_output_retention_error"] = type(retain_error).__name__
        report["exception_type"] = type(error).__name__
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
        save()
        raise SystemExit(0 if report["status"] == "unsupported_resource" else 1)
    save()


if __name__ == "__main__":
    main()
