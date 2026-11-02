"""Create the compact, repository-relative A40 D256 qualification report."""

import argparse
import hashlib
import json
from pathlib import Path


def summarize(arguments):
    root = arguments.input
    measured = json.loads((root / "paired-public.json").read_text())
    if getattr(arguments, "short", False):
        nograd = json.loads((root / "paired-nograd.json").read_text())
        assert len(measured["rows"]) == 24 and len(nograd["rows"]) == 8
        for row in measured["rows"]:
            row["workload"] = "training"
        for row in nograd["rows"]:
            row["workload"] = "no_grad"
        measured["rows"] += nograd["rows"]
    state = json.loads((root / "state.json").read_text())
    assert len(state["stages"]) == 12 and all(s["exit_code"] == 0 for s in state["stages"])
    report = {
        "scope": "Actual explicit A40 public API versus its recorded controls, ordinary CuTe, FA2 and torch SDPA. Complete implementation comparisons; matching rounded input tensors, distinct backend arithmetic.",
        "device": {
            "name": measured["device"],
            "capability": [8, 6],
            "sm_count": 84,
            "memory_gb": 48,
            "shared_capacity_bytes": 101376,
            "power_limit_w": 300,
        },
        "stack": {
            "python": "3.13.18",
            "torch": measured["torch"],
            "cutlass_dsl": "4.8.1",
            "tvm_ffi": "0.1.15.post0",
            "triton": "3.9.0",
            "quack": "0.6.5",
            "dlpack": "0.1.5",
            "fa2": "2.8.4",
            "fa4_source": "4616f5116d33e3ba3cbb71d582caffcfcb49649c",
        },
        "source_snapshot_sha256": measured["source_revision"],
        "driver_sha256": measured["driver_sha256"],
        "source_sha256": measured["source_sha256"],
        "protocol": measured["protocol"],
        "forward_gradients": measured["forward_gradients"],
        "event_scope": measured["event_scope"],
        "precision": "Single input-dtype P conversion before dS, one dS conversion, FP32 products/accumulation, actual rounded O delta and late gradient scale; unchanged separate literal score maximum and logarithmic normalizer.",
        "selection": {
            "batch": 1,
            "heads": 16,
            "sequences": sorted({row["shape"][2] for row in measured["rows"]}),
            "dimension": 256,
            "dtypes": ["bf16", "fp16"],
            "visibility": ["dense", "causal"],
            "forward_tile": [64, 64],
            "backward_tile": [64, 32],
            "query_stages": 1,
            "dout_stages": 1,
            "mma_warps": [4, 1, 2],
            "v_in_regs": True,
        },
        "stages": [{"name": s["label"], "exit_code": s["exit_code"]} for s in state["stages"]],
        "prior_schedule_qualification": {
            "source_sha256": "a6d92906747659d0152b8735642476e5452ad2a33dfd66d03468be93c14bb910",
            "directed_cases_per_width": 53,
            "forward_widths": [16, 32, 64],
            "directed_status": [0, 0, 0],
            "memcheck_widths": [16, 32, 64],
            "racecheck_synccheck_initcheck_widths": [32, 64],
            "status": 0,
            "earlier_metadata_failure": "Pre-math missing metadata filename; retained failure, repaired lookup only in immutable continuation.",
        },
        "operations": measured["operations"],
        "acceptance": {
            "bf16_per_head_component_norm": 0.01,
            "fp16_per_head_component_norm": 0.005,
            "reference": "Independent FP64 output/dQ/dK/dV, plus rounded equations using actual O; directed fixtures retain their qualified gates.",
        },
        "rows": [],
        "bindings": {},
    }
    for row in measured["rows"]:
        result = {
            k: row[k]
            for k in (
                "shape",
                "dtype",
                "causal",
                "phase",
                "mode",
                "calls_per_window",
                "samples",
                "median_ms",
                "graph_resources",
            )
        }
        if "workload" in row:
            result["workload"] = row["workload"]
        result["accuracy"] = {}
        for backend, checks in row["qualification"].items():
            result["accuracy"][backend] = {}
            for component in ("out", "dq", "dk", "dv"):
                subset = [v for v in checks if v["component"] == component]
                result["accuracy"][backend][component] = {
                    "fp64_relative_error_by_head": [v["fp64"]["relative_error"] for v in subset],
                    "rounded_relative_error_by_head": [v["rounded"]["relative_error"] for v in subset],
                    "finite": all(v["fp64"]["finite"] and v["rounded"]["finite"] for v in subset),
                }
        for key in ("accuracy", "graph_resources"):
            value = result.pop(key)
            identifier = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
            shared = report.setdefault(key + "_sets", {})
            if identifier in shared:
                assert shared[identifier] == value
            else:
                shared[identifier] = value
            result[key + "_id"] = identifier
        report["rows"].append(result)
    oracle = json.loads((root / "public-oracle.json").read_text())
    report["public_oracle"] = {
        "status": oracle["status"],
        "cases": [
            {k: case[k] for k in ("case", "shape", "dtype", "causal", "scale", "selected")} for case in oracle["cases"]
        ],
        "changed_poisoned_graph_cases": sum("changed_poisoned_graph" in case for case in oracle["cases"]),
        "saved_literal_statistics": True,
        "reference_device": "cpu",
    }
    for backend in ("candidate", "fa2"):
        for visibility in ("dense", "causal"):
            name = "code-" + backend + "-" + visibility
            binding = json.loads((root / name / "binding.json").read_text())
            report["bindings"][name] = {
                k: binding[k]
                for k in (
                    "backend",
                    "shape",
                    "dtype",
                    "causal",
                    "graph",
                    "symbol",
                    "loaded_modules",
                    "exact_symbol_ptx",
                    "library",
                    "ptx_status",
                    "compiler_binding_required",
                    "scope",
                )
            }
    report["retention"] = {
        "raw_operands_and_loaded_code": "Retained outside the repository with per-file SHA256 verification.",
        "raw_report_sha256": hashlib.sha256((root / "paired-public.json").read_bytes()).hexdigest(),
    }
    if getattr(arguments, "short", False):
        report["forward_gradients"] = {"training": True, "no_grad": False}
        report["no_grad_report_sha256"] = hashlib.sha256((root / "paired-nograd.json").read_bytes()).hexdigest()
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--short", action="store_true", help="Include the separately measured S128 no-grad forward rows."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    summarize(parser.parse_args())


if __name__ == "__main__":
    main()
