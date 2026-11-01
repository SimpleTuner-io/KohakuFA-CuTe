"""Build the compact L40S training report and separate candidate chart."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def selected(row):
    return row["shape"][2] in (1024, 4096) and not (row["causal"] and row["shape"][-1] == 64 and row["shape"][2] == 1024)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding(path):
    data = json.loads(path.read_text())
    return {
        "symbol": data["symbol"],
        "graph": data["graph"],
        "loaded_cubin_sha256": [entry["loaded_cubin_sha256"] for entry in data["loaded_modules"]],
        "sass_sha256": [entry["sass_sha256"] for entry in data["loaded_modules"]],
        "ptx_status": data["ptx_status"],
        "ptx_sha256": [entry["sha256"] for entry in data["exact_symbol_ptx"]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/l40s-warp-training.json"))
    parser.add_argument("--plot", type=Path, default=Path("results/l40s-warp-training.svg"))
    args = parser.parse_args()
    root = args.captures
    name = "l40s-stage22-cached-final-matrix.json"
    raw = json.loads((root / name).read_text())
    assert len(raw["rows"]) == 96
    assert (root / "l40s-stage22-cached-final-matrix.status").read_text().strip() == "0"
    report = {
        "schema_version": 1,
        "status": "qualified PR candidate; merge state is separate",
        "date": "2026-11-01",
        "device": {
            "name": "NVIDIA L40S",
            "capability": [8, 9],
            "sm": 142,
            "memory_bytes": 47667740672,
            "power_limit_w": 350,
        },
        "stack": {
            "python": "3.13.18",
            "torch": raw["torch"],
            "cuda": "13.2",
            "driver": "595.99.02",
            "cutlass_dsl": "4.8.1",
            "tvm_ffi": "0.1.15.post0",
            "quack": "0.6.5",
        },
        "base_revision": "224781bb6fc15c62208a34c94e988fd3165bbc31",
        "precision": "Separate unscaled maxima and shifted log normalization; P and dS rounded once to input dtype; FP32 products/actual-O delta; late dQ/dK scale. Unchanged per-head 1% BF16/0.5% FP16 gates.",
        "scope": "Training B1/H16 square aligned contiguous FP16/BF16 MHA, S1024 or4096; D64 causal only4096. Inference, other lengths, masks, GQA and other dimensions retain preceding dispatch.",
        "protocol": raw["protocol"],
        "timing_scope": raw["scope"],
        "event_scope": raw["event_scope"],
        "fa2": raw["operations"]["fa2"],
        "source_sha256": raw["source_sha256"],
        "driver_sha256": raw["driver_sha256"],
        "capture_sha256": {name: digest(root / name)},
        "rows": [],
        "machine_code": {},
    }
    for row in raw["rows"]:
        entry = {key: row[key] for key in ("shape", "dtype", "causal", "phase", "mode")}
        entry["selected"] = selected(row)
        entry["kernel_counts"] = {key: len(value) for key, value in row["graph_resources"].items()}
        entry["median_event_ms"] = {key: value["cuda_event_ms"] for key, value in row["median_ms"].items()}
        entry["median_wall_ms"] = {key: value["wall_ms"] for key, value in row["median_ms"].items()}
        entry["samples_event_ms"] = {
            key: [value["cuda_event_ms"] for value in values] for key, values in row["samples"].items()
        }
        entry["fa2_over_public"] = entry["median_event_ms"]["fa2"] / entry["median_event_ms"]["public"]
        entry["previous_over_public"] = entry["median_event_ms"]["previous"] / entry["median_event_ms"]["public"]
        report["rows"].append(entry)
    for dimension in (64, 128):
        for causal in (False, True):
            label = f"D{dimension}-" + ("causal" if causal else "dense")
            relative = f"l40s-stage14-public-binding/{label}/binding.json"
            if dimension == 128 and not causal:
                relative = "l40s-stage20-final-code/D128-dense-backward-bf16/binding.json"
            report["machine_code"][label + "-candidate"] = binding(root / relative)
            report["capture_sha256"][relative] = digest(root / relative)
            relative = f"l40s-stage11-profile/{label}-fa2/binding.json"
            report["machine_code"][label + "-fa2"] = binding(root / relative)
            report["capture_sha256"][relative] = digest(root / relative)
    for dtype in ("bf16", "fp16"):
        for backend in ("candidate", "fa2"):
            label = f"D128-dense-forward-{dtype}-{backend}"
            relative = f"l40s-stage20-final-code/{label}/binding.json"
            report["machine_code"][label] = binding(root / relative)
            report["capture_sha256"][relative] = digest(root / relative)
    report["qualification"] = {}
    for name in ("l40s-stage12-native-qualification.json", "l40s-stage18-forward128-qualification.json"):
        data = json.loads((root / name).read_text())
        assert data["status"] == "pass" and len(data["rows"]) == 96
        report["qualification"][name] = {
            "rows": len(data["rows"]),
            "statistics_checks": len(data["statistics_checks"]),
            "status": data["status"],
        }
        maximum_errors = {}
        for row in data["rows"]:
            reference = "rounded" if row["directed_rounded_gate"] else "fp64"
            group = maximum_errors.setdefault(row["dtype"] + ":" + reference, {})
            for error in row["errors"]:
                value = error[reference]["relative_error"]
                if value is not None:
                    group[error["component"]] = max(group.get(error["component"], 0), value)
        report["qualification"][name]["max_required_relative_error"] = maximum_errors
        report["capture_sha256"][name] = digest(root / name)
    for stage, variant in ((12, "native"), (18, "forward128")):
        for tool in ("memcheck", "racecheck", "synccheck", "initcheck"):
            name = f"l40s-stage{stage}-{variant}-{tool}.json"
            data = json.loads((root / name).read_text())
            assert data["status"] == "pass" and len(data["rows"]) == 6
            log = (root / name).with_suffix(".log").read_text()
            summary = [line for line in log.splitlines() if "SUMMARY:" in line][-1]
            if tool == "racecheck":
                assert re.search(r"RACECHECK SUMMARY: 0 hazards displayed \(0 errors, 0 warnings\)", summary)
            else:
                assert re.search(r"ERROR SUMMARY: 0 errors(?:\s|$)", summary)
            report["qualification"][name] = {
                "rows": 6,
                "status": "pass",
                "reference_device": data.get("reference_device", "cuda"),
                "tool_summary": [line for line in log.splitlines() if "SUMMARY:" in line][-1],
            }
            report["capture_sha256"][name] = digest(root / name)
    for suffix in ("guard-tests", "cpu", "gpu-tests"):
        stage = 21 if suffix == "gpu-tests" else 25
        name = f"l40s-stage{stage}-{suffix}.log"
        text = (root / name).read_text()
        assert "\nOK" in text and "FAILED" not in text
        skips = re.findall(r"OK \(skipped=(\d+)\)", text)
        report["qualification"][name] = {
            "tests": int(re.findall(r"Ran (\d+) tests", text)[-1]),
            "skipped": int(skips[-1]) if skips else 0,
            "status": "pass",
            "scope": "GPU public API" if suffix == "gpu-tests" else "local CPU checks",
        }
        report["capture_sha256"][name] = digest(root / name)
    report["counter_limit"] = (
        "Nsight Compute ERR_NVGPUCTRPERM; no counter-derived occupancy, cache, stall or utilization claims."
    )
    header = {key: value for key, value in report.items() if key != "rows"}
    compact = (
        json.dumps(header, indent=2)[:-2]
        + ',\n  "rows": [\n    '
        + ",\n    ".join(json.dumps(row, separators=(",", ":")) for row in report["rows"])
        + "\n  ]\n}\n"
    )
    assert json.loads(compact) == report
    args.output.write_text(compact)

    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "kohakufa-l40s-warp-training"})
    import matplotlib.pyplot as plt
    import numpy as np

    cases = sorted({(r["shape"][2], r["shape"][-1], r["causal"], r["dtype"]) for r in report["rows"] if r["selected"]})
    figure, axes = plt.subplots(2, 2, figsize=(12, 10), layout="constrained")
    for axis, (mode, phase) in zip(
        axes.flat, (("graph", "backward"), ("graph", "joint"), ("eager", "backward"), ("eager", "joint"))
    ):
        rows = {
            (r["shape"][2], r["shape"][-1], r["causal"], r["dtype"]): r
            for r in report["rows"]
            if r["mode"] == mode and r["phase"] == phase
        }
        y = np.arange(len(cases))
        for offset, backend, color, title in (
            (-0.16, "previous", "#375f7b", "Preceding CuTe / candidate"),
            (0.16, "fa2", "#c87834", "FA2 / candidate"),
        ):
            ratio = [rows[key][backend + "_over_public"] for key in cases]
            axis.barh(y + offset, ratio, height=0.3, color=color, label=title)
        axis.axvline(1, color="#444444", lw=0.8)
        axis.set_yticks(
            y, [f"S{s} D{d} {'causal' if causal else 'dense'} {dtype.upper()}" for s, d, causal, dtype in cases], fontsize=8
        )
        axis.invert_yaxis()
        maximum_ratio = max(max(rows[key]["previous_over_public"], rows[key]["fa2_over_public"]) for key in cases)
        axis.set_xlim(0.7, max(1.7, maximum_ratio * 1.05))
        axis.set_title(f"{mode.title()} · {'backward' if phase == 'backward' else 'forward + backward'}")
        axis.set_xlabel("Median latency ratio · above 1 favors candidate")
        axis.grid(axis="x", alpha=0.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, fontsize=9, loc="outside lower center", ncol=2)
    figure.suptitle("L40S training PR candidate · exact qualified lengths · B1/H16 · 350 W · 2026-11-01")
    figure.savefig(args.plot, metadata={"Date": None, "Creator": "KohakuFA-CuTe benchmarks.summarize_l40s_training"})
    if args.plot.suffix.lower() == ".svg":
        args.plot.write_text("\n".join(line.rstrip() for line in args.plot.read_text().splitlines()) + "\n")
    plt.close(figure)


if __name__ == "__main__":
    main()
