"""Summarize the qualified L40S causal ordering and render its current chart."""

import argparse
import collections
import hashlib
import json
import re
import statistics
from pathlib import Path

from benchmarks.summarize_l40s_training import binding, digest


def query_loops(path):
    instructions = []
    for line in path.read_text().splitlines():
        match = re.match(r"\s*/\*([0-9a-f]+)\*/\s+(.*?)\s*;", line)
        if match:
            instructions.append((int(match[1], 16), match[2]))
    loops = []
    for pc, text in instructions:
        branch = re.search(r"\bBRA\s+0x([0-9a-f]+)", text)
        if not branch or int(branch[1], 16) >= pc:
            continue
        first = int(branch[1], 16)
        counts = collections.Counter(
            re.sub(r"^@!?[A-Z]+\d*\s+", "", instruction).split()[0]
            for address, instruction in instructions
            if first <= address <= pc
        )
        if sum(count for name, count in counts.items() if name.startswith("HMMA")) > 20:
            loops.append({"pc": [hex(first), hex(pc)], "instructions": sum(counts.values()), "counts": dict(counts)})
    return loops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--verified-checkpoint", type=Path, required=True)
    parser.add_argument("--source-tar", type=Path, required=True)
    parser.add_argument("--publication-cpu", type=Path)
    parser.add_argument("--publication-proof", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/l40s-causal-groups.json"))
    parser.add_argument("--plot", type=Path, default=Path("results/l40s-causal-groups.svg"))
    args = parser.parse_args()
    root = args.captures
    health = json.loads((args.verified_checkpoint / "health.json").read_text())
    assert health["complete"] and not health["missing"]
    assert health["source_job"] == root.name
    inventory = json.loads((args.verified_checkpoint / "expected-inventory.json").read_text())
    assert health["verified_files"] == len(inventory)
    for item in inventory:
        path = root.parent / item["file"]
        assert path.stat().st_size == item["bytes"] and digest(path) == item["sha256"], item["file"]
    state = json.loads((root / "state.json").read_text())
    assert len(state["stages"]) == 45 and all(stage["exit_code"] == 0 for stage in state["stages"])
    raw = json.loads((root / "evaluation.json").read_text())
    assert len(raw["rows"]) == 96 and raw["protocol"]["seed"] == 9713
    preflight = json.loads((root / "preflight.log").read_text())
    qualification = json.loads((root / "qualification.json").read_text())
    assert qualification["status"] == "pass" and len(qualification["rows"]) == 96
    assert len(qualification["statistics_checks"]) == 60
    report = {
        "schema_version": 1,
        "status": "Qualified proposed dispatch; publication and merge state are separate.",
        "date": "2026-11-02",
        "device": {
            "name": preflight["device"],
            "sm": preflight["sm_count"],
            "memory_bytes": preflight["total_memory"],
            "power_limit_w": 350,
        },
        "stack": {
            key: preflight[key] for key in ("python", "torch", "cuda", "packages", "fa2_loaded_elf_sha256", "hardware")
        },
        "precision": "Separate raw maximum/log norm, input-dtype P and dS once, FP32 accumulation/actual rounded O delta, late dQ/dK scale. Independent per-head 1% BF16/0.5% FP16 gates unchanged.",
        "scope": "Only causal B1/H16 training: D64/S4096 group8; D128/S1024 or4096 group16. Existing eligibility and fallback guards unchanged. Dense and intervening lengths are not new selections.",
        "controls": {
            "public": "Actual scaled_dot_product_attention public API",
            "g1": "Explicit native group1 control with a shorter Python guard; not the preceding complete public API or original generic CUDA backend",
            "fa2": "Standalone FA2 complete implementation",
            "sdpa": "PyTorch SDPA complete implementation",
        },
        "protocol": raw["protocol"],
        "timing_scope": raw["scope"],
        "event_scope": raw["event_scope"],
        "source_manifest_sha256": raw["source_revision"],
        "source_tar_sha256": digest(args.source_tar),
        "source_sha256": raw["source_sha256"],
        "driver_sha256": raw["driver_sha256"],
        "operations": raw["operations"],
        "capture_sha256": {
            name: digest(root / name) for name in ("evaluation.json", "qualification.json", "public-gpu.json", "state.json")
        },
        "verified_retention": {
            "files": health["verified_files"],
            "bytes": health["expected_bytes"],
            "inventory_sha256": digest(args.verified_checkpoint / "expected-inventory.json"),
            "off_host": True,
            "scope": "Pilot and held-out operands, final reports, all closed code bindings/binaries/PTX/SASS and failing operands if present; raw files outside the tracked tree.",
        },
        "qualification": {
            "directed_rows": 96,
            "saved_statistics_checks": 60,
            "public_gpu": json.loads((root / "public-gpu.json").read_text()),
            "sanitizers": {},
        },
        "machine_code": {},
        "graph_resource_sets": {},
        "accuracy": {},
        "limitations": [
            "Short eager joint calls include public Python overhead; native G1 is not an isolated old-public-API eager baseline.",
            "GPU ordering/resource comparisons are not runtime counter measurements. Nsight Compute counters were denied on this device.",
            "Separate dated November1 dense N32/allocation-control reports remain unchanged; the historical full-device grid is a different workload/runtime scope.",
        ],
        "rows": [],
    }
    if args.publication_cpu or args.publication_proof:
        assert args.publication_cpu and args.publication_proof
        proof = json.loads(args.publication_proof.read_text())
        cpu_state = json.loads((args.publication_cpu / "state.json").read_text())
        assert all(stage["exit_code"] == 0 for stage in cpu_state["stages"])
        cpu_text = (args.publication_cpu / "cpu.log").read_text()
        guard_text = (args.publication_cpu / "guards.log").read_text()
        report["publication_validation"] = {
            "base_revision": proof["publication_base"],
            "identical_runtime_files": proof["identical_runtime_files"],
            "unrelated_upstream_changes": proof["changed_unrelated_files"],
            "scope": proof["review"],
            "cpu_tests": int(re.search(r"Ran (\d+) tests", cpu_text)[1]),
            "cpu_skips": int(re.search(r"skipped=(\d+)", cpu_text)[1]),
            "guard_tests": int(re.search(r"Ran (\d+) tests", guard_text)[1]),
            "receipt_sha256": digest(args.publication_cpu / "state.json"),
            "proof_sha256": digest(args.publication_proof),
        }
    assert report["qualification"]["public_gpu"]["tests_run"] == 4
    assert all(report["qualification"]["public_gpu"][key] == 0 for key in ("failures", "errors", "skips"))
    for tool in ("memcheck", "racecheck", "synccheck", "initcheck"):
        data = json.loads((root / (tool + ".json")).read_text())
        assert data["status"] == "pass" and len(data["rows"]) == 8
        text = (root / (tool + ".log")).read_text()
        summaries = re.findall(
            r"(?:ERROR SUMMARY: (\d+) errors|RACECHECK SUMMARY: \d+ hazards displayed \((\d+) errors, (\d+) warnings\))",
            text,
        )
        assert summaries and all(int(value) == 0 for row in summaries for value in row if value)
        report["qualification"]["sanitizers"][tool] = {
            "exit_code": 0,
            "rows": 8,
            "summary_errors": 0,
            "summary_warnings": 0,
            "report_sha256": digest(root / (tool + ".json")),
            "log_sha256": digest(root / (tool + ".log")),
        }
    for path in sorted((root / "code").glob("*/binding.json")):
        record = binding(path)
        if not path.parent.name.startswith("forward-") and not path.parent.name.endswith("-fa2"):
            record["steady_query_loops"] = query_loops(path.parent / "loaded/selected-0.sass")
        assert record["ptx_status"] == "exact_symbol_retained" and record["loaded_cubin_sha256"] and record["ptx_sha256"]
        report["machine_code"][path.parent.name] = record
    assert len(report["machine_code"]) == 30
    for name, record in report["machine_code"].items():
        if name.startswith("forward-") and name.endswith("-public"):
            control = report["machine_code"][name.removesuffix("public") + "g1"]
            for key in ("loaded_cubin_sha256", "sass_sha256", "ptx_sha256"):
                assert record[key] == control[key], (name, key)
    for row in raw["rows"]:
        result = {key: row[key] for key in ("shape", "dtype", "causal", "phase", "mode", "samples", "median_ms")}
        result["graph_resource_ids"] = {}
        for backend, resources in row["graph_resources"].items():
            resource_id = hashlib.sha256(json.dumps(resources, sort_keys=True).encode()).hexdigest()
            report["graph_resource_sets"][resource_id] = resources
            result["graph_resource_ids"][backend] = resource_id
        result["selected_group"] = (
            (8 if row["shape"][-1] == 64 else 16)
            if row["causal"] and not (row["shape"][-1] == 64 and row["shape"][2] == 1024)
            else 1
        )
        result["paired_g1_over_public"] = {
            "minimum": min(
                a["cuda_event_ms"] / b["cuda_event_ms"] for a, b in zip(row["samples"]["g1"], row["samples"]["public"])
            ),
            "median": statistics.median(
                a["cuda_event_ms"] / b["cuda_event_ms"] for a, b in zip(row["samples"]["g1"], row["samples"]["public"])
            ),
            "maximum": max(
                a["cuda_event_ms"] / b["cuda_event_ms"] for a, b in zip(row["samples"]["g1"], row["samples"]["public"])
            ),
        }
        result["maximum_per_head_error"] = {}
        for backend, checks in row["qualification"].items():
            errors = {}
            for component in ("out", "dq", "dk", "dv"):
                values = [check for check in checks if check["component"] == component]
                assert len(values) == 16
                limit = 0.01 if row["dtype"] == "bf16" else 0.005
                assert all(
                    check["fp64"]["finite"]
                    and (
                        check["fp64"]["relative_error"] < limit
                        if check["fp64"]["reference_norm"]
                        else check["fp64"]["max_absolute_error"] == 0
                    )
                    for check in values
                )
                errors[component] = {
                    policy: max(check[policy]["relative_error"] for check in values) for policy in ("fp64", "rounded")
                }
            result["maximum_per_head_error"][backend] = errors
        accuracy_id = f"s{row['shape'][2]}-d{row['shape'][-1]}-{row['dtype']}-{'causal' if row['causal'] else 'dense'}"
        accuracy = result.pop("maximum_per_head_error")
        if accuracy_id in report["accuracy"]:
            assert report["accuracy"][accuracy_id] == accuracy
        report["accuracy"][accuracy_id] = accuracy
        result["accuracy_id"] = accuracy_id
        report["rows"].append(result)
    collections_to_compact = ("machine_code", "graph_resource_sets", "accuracy", "rows")
    header = {key: value for key, value in report.items() if key not in collections_to_compact}
    compact = json.dumps(header, indent=2)[:-2]
    for key in collections_to_compact:
        values = report[key]
        if isinstance(values, dict):
            compact += ",\n  " + json.dumps(key) + ": {\n    "
            compact += ",\n    ".join(
                json.dumps(name) + ": " + json.dumps(value, separators=(",", ":")) for name, value in values.items()
            )
            compact += "\n  }"
        else:
            compact += ",\n  " + json.dumps(key) + ": [\n    "
            compact += ",\n    ".join(json.dumps(value, separators=(",", ":")) for value in values)
            compact += "\n  ]"
    compact += "\n}\n"
    assert json.loads(compact) == report
    args.output.write_text(compact)
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "kohakufa-l40s-causal-groups"})
    import matplotlib.pyplot as plt
    import numpy as np

    cases = sorted(
        {(row["shape"][2], row["shape"][-1], row["dtype"]) for row in report["rows"] if row["selected_group"] > 1}
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for axis, (mode, phase) in zip(
        axes.flat, (("graph", "backward"), ("graph", "joint"), ("eager", "backward"), ("eager", "joint"))
    ):
        rows = {
            (row["shape"][2], row["shape"][-1], row["dtype"]): row
            for row in report["rows"]
            if row["causal"] and row["selected_group"] > 1 and row["mode"] == mode and row["phase"] == phase
        }
        y = np.arange(len(cases))
        maximum = 1.7
        for offset, backend, color, title in (
            (-0.23, "g1", "#375f7b", "Native G1 control / public"),
            (0, "fa2", "#c87834", "FA2 / public"),
            (0.23, "sdpa", "#777777", "SDPA / public"),
        ):
            ratios = [
                rows[key]["median_ms"][backend]["cuda_event_ms"] / rows[key]["median_ms"]["public"]["cuda_event_ms"]
                for key in cases
            ]
            maximum = max(maximum, max(ratios) * 1.05)
            axis.barh(y + offset, ratios, height=0.22, label=title, color=color)
        axis.axvline(1, color="#444444", linewidth=0.8)
        axis.set_yticks(y, [f"S{s} D{d} {dtype.upper()}" for s, d, dtype in cases], fontsize=9)
        axis.invert_yaxis()
        axis.set_xlim(0.8, maximum)
        axis.set_title(f"{mode.title()} · {'backward' if phase == 'backward' else 'forward + backward'}")
        axis.set_xlabel("Median latency ratio · above 1 favors public")
        axis.grid(axis="x", alpha=0.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=9)
    figure.suptitle(
        "L40S causal CTA groups · bounded public API · B1/H16 · 350 W · 2026-11-02\n"
        "Native G1 uses a shorter guard; short eager is not an old-public-API comparison.",
        fontsize=11,
    )
    figure.savefig(args.plot, metadata={"Date": None, "Creator": "KohakuFA-CuTe benchmarks.summarize_l40s_causal_groups"})
    if args.plot.suffix.lower() == ".svg":
        args.plot.write_text("\n".join(line.rstrip() for line in args.plot.read_text().splitlines()) + "\n")
    plt.close(figure)


if __name__ == "__main__":
    main()
