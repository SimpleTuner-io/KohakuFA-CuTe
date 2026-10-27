"""Run open-issue regressions in isolated processes, including crash controls."""

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

CASES = (
    "grid_limits",
    "masked_grid_limits",
    "live_kv_views_ignore_capacity_tail",
    "cached_strides_and_batch_changes",
    "nonpositive_scales_with_masks",
    "single_head_single_query",
    "single_query_gqa_statistics",
    "zero_stride_output_gradient",
)


def frontend_plan():
    """NVIDIA issue 1071's planning-only shape; caller isolates SIGSEGV."""
    import cudnn

    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    values = {
        name: torch.zeros(1, 1, length, 64, device="cuda", dtype=torch.bfloat16)
        for name, length in (("q", 1), ("k", 128), ("v", 128), ("o", 1), ("dO", 1))
    }
    values["stats"] = torch.zeros(1, 1, 1, 1, device="cuda", dtype=torch.float32)
    attrs = {name: graph.tensor_like(value) for name, value in values.items()}
    grads = graph.sdpa_backward(**attrs, attn_scale=0.125, use_causal_mask=False)
    for attr, name in zip(grads, ("q", "k", "v")):
        attr.set_output(True).set_dim(values[name].shape).set_stride(values[name].stride())
    graph.validate()
    graph.build_operation_graph()
    print("before plan creation", flush=True)
    graph.create_execution_plans([cudnn.heur_mode.A])
    print("after plan creation", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backends", nargs="+", choices=("cute", "cudnn"), default=["cute", "cudnn"])
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--frontend-plan", action="store_true", help="Add an isolated planning-only cuDNN control")
    parser.add_argument("--plan-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.plan_worker:
        frontend_plan()
        return 0
    if args.output is None:
        parser.error("--output is required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "source_issues": [1573, 1567, 1570, 624, 1428, 1435, 1071, 1083, 100],
        "dependencies": {
            name: importlib.metadata.version(name) for name in ("nvidia-cutlass-dsl", "apache-tvm-ffi", "triton")
        },
        "rows": [],
    }
    jobs = [
        (
            backend,
            case,
            [
                sys.executable,
                "-m",
                "unittest",
                (
                    "tests.test_launch_limits.LaunchLimitTests"
                    if case in ("grid_limits", "masked_grid_limits")
                    else "tests.test_tracker_regressions.TrackerRegressionTests"
                )
                + f".test_{case}",
                "-v",
            ],
        )
        for backend in args.backends
        for case in args.cases
    ]
    if args.frontend_plan:
        jobs.append(
            ("frontend", "planning_only_1071", [sys.executable, "-m", "benchmarks.tracker_regressions", "--plan-worker"])
        )
    for backend, case, command in jobs:
        log = args.output.with_name(args.output.stem + f"-{backend}-{case}.log")
        env = dict(os.environ, KOH_CUTE_TRACKER_BACKEND=backend)
        start = time.monotonic()
        with log.open("w") as stream:
            try:
                result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=args.timeout)
                status = "pass" if result.returncode == 0 else "signal" if result.returncode < 0 else "failed"
                if result.returncode == 0 and "OK (skipped=" in log.read_text():
                    status = "skipped"
                row = {"backend": backend, "case": case, "status": status, "returncode": result.returncode}
            except subprocess.TimeoutExpired:
                row = {"backend": backend, "case": case, "status": "timeout"}
        row.update(seconds=time.monotonic() - start, log=log.name)
        report["rows"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(backend, case, row["status"], flush=True)
    return int(any(row["backend"] == "cute" and row["status"] not in ("pass", "skipped") for row in report["rows"]))


if __name__ == "__main__":
    sys.exit(main())
