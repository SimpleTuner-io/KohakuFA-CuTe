"""Build the measured L40S host-guard report and paired-variability chart."""

import argparse
import json
import re
import statistics
from pathlib import Path

from benchmarks.summarize_l40s_training import binding, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-tar", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/l40s-native-metadata.json"))
    parser.add_argument("--plot", type=Path, default=Path("results/l40s-native-metadata.svg"))
    parser.add_argument("--integration", type=Path)
    parser.add_argument("--integration-checkpoint", type=Path)
    parser.add_argument("--integration-source-tar", type=Path)
    args = parser.parse_args()
    if any((args.integration, args.integration_checkpoint, args.integration_source_tar)) and not all(
        (args.integration, args.integration_checkpoint, args.integration_source_tar)
    ):
        parser.error("Supply all three integration inputs together.")
    root = args.captures
    health = json.loads((args.checkpoint / "health.json").read_text())
    inventory = json.loads((args.checkpoint / "expected-inventory.json").read_text())
    assert health["complete"] and health["verified_files"] == len(inventory) and not health["missing"]
    for item in inventory:
        path = root.parent / item["file"]
        assert path.stat().st_size == item["bytes"] and digest(path) == item["sha256"]
    state = json.loads((root / "state.json").read_text())
    assert len(state["stages"]) == 34 and all(row["exit_code"] == 0 for row in state["stages"])
    raw = json.loads((root / "evaluation.json").read_text())
    assert len(raw["rows"]) == 96 and raw["protocol"]["seed"] == 21331
    gpu = json.loads((root / "native-parity-public.json").read_text())
    assert gpu["tests_run"] == 12 and all(gpu[key] == 0 for key in ("failures", "errors", "skips"))
    identity = json.loads((root / "code-identity.json").read_text())
    assert identity["all_current_parent_identical"] and len(identity["pairs"]) == 12
    assert all(pair["same_as_PR69"] for pair in identity["pairs"])
    bridge = json.loads((root / "host-bridge-binding.json").read_text())
    assert bridge and all(
        digest(root / "host-bridge" / (row["loaded_bridge_sha256"] + ".so")) == row["loaded_bridge_sha256"]
        for row in bridge
    )
    cpu = (root / "cpu.log").read_text()
    guards = (root / "guards.log").read_text()
    runtime = json.loads((root / "preflight.log").read_text())
    runtime["tools"] = {name: Path(path).name if path else None for name, path in runtime["tools"].items()}
    report = {
        "schema_version": 1,
        "date": "2026-11-02",
        "status": "Qualified proposed host-guard change; merge state is separate.",
        "scope": "Actual L40S public API native live-metadata check vs pristine preceding public Python namespace, sharing unchanged GPU methods/native runner. Original eligibility, precision and fallbacks remain unchanged.",
        "device": {"name": raw["device"], "sm": 142, "power_limit_w": 350},
        "runtime": runtime,
        "precision": "Original separate raw-max/log-normalization; P and dS input-dtype once, FP32 products and actual-rounded-O delta, late dQ/dK scale. Independent per-head 1% BF16/0.5% FP16 gates unchanged.",
        "protocol": raw["protocol"],
        "timing_scope": raw["scope"],
        "event_scope": raw["event_scope"],
        "parent_revision": "3f6ba0465a2f0887a0992831f0caa904beb939b3",
        "controls": {
            "public": "Actual public API with native metadata check",
            "parent": "Pristine preceding public API namespace and Python predicate; same native runner/GPU methods",
            "fa2": "Standalone FA2 complete implementation",
            "sdpa": "PyTorch SDPA complete implementation",
        },
        "source_manifest_sha256": raw["source_revision"],
        "source_tar_sha256": digest(args.source_tar),
        "source_sha256": raw["source_sha256"],
        "driver_sha256": raw["driver_sha256"],
        "operations": raw["operations"],
        "capture_sha256": {
            name: digest(root / name)
            for name in (
                "state.json",
                "evaluation.json",
                "native-parity-public.json",
                "code-identity.json",
                "host-bridge-binding.json",
            )
        },
        "qualification": {
            "cpu_total": int(re.search(r"Ran (\d+) tests", cpu)[1]),
            "cpu_expected_gpu_skips": int(re.search(r"skipped=(\d+)", cpu)[1]),
            "guard_tests": int(re.search(r"Ran (\d+) tests", guards)[1]),
            "actual_public_gpu": gpu,
            "live_metadata_parity": "More than 400 actual comparisons, plus parameters/subclasses/parametrizations, mutable storage offsets/gradient flags, cold/warm, required-loader errors, fallback and compilation checks.",
            "strict_gpu_bindings": 24,
            "identical_current_parent_pairs": 12,
            "identical_PR69_pairs": 12,
            "sanitizers": "PR69 all four passing sanitizer runs retained: every selected current CUDA cubin/PTX/SASS pair is byte-identical, GPU-method source/ownership unchanged; no redundant tool rerun.",
        },
        "host_bridge": [
            {key: row[key] for key in ("library", "loaded_bridge_sha256", "native_source_sha256", "scope")}
            for row in bridge
        ],
        "verified_retention": {
            "files": health["verified_files"],
            "bytes": health["expected_bytes"],
            "inventory_sha256": digest(args.checkpoint / "expected-inventory.json"),
            "off_host": True,
            "scope": "All final actual artifacts, input tensors, native host bridge ELF/build command, loaded CUDA binaries/PTX/SASS and receipts physically verified off-host; raw files outside tracked tree.",
        },
        "limitations": [
            "Benefits are case-specific. Short FP16 eager calls show the clearest gains; BF16 short calls include ties and variable paired samples.",
            "Long differences are mixed and include dense D128 regressions below 1%. No uniform long-call or GPU graph speedup is claimed.",
            "Difference of latency medians and median of paired differences need not agree; both and all samples are retained.",
            "The preceding public namespace uses pristine Python policy/API bytes with explicit unchanged native-runner/GPU-module aliases to isolate the guard. It is not a shorter direct-runner proxy.",
        ],
        "machine_code": {},
        "accuracy": {},
        "rows": [],
    }
    for path in sorted((root / "code").glob("*/binding.json")):
        report["machine_code"][path.parent.name] = binding(path)
    assert len(report["machine_code"]) == 24
    for row in raw["rows"]:
        result = {key: row[key] for key in ("shape", "dtype", "causal", "phase", "mode", "samples", "median_ms")}
        pairs = [
            (before["wall_ms"] - after["wall_ms"]) * 1000
            for before, after in zip(row["samples"]["parent"], row["samples"]["public"])
        ]
        result["parent_minus_public_wall_us"] = {
            "latency_median_difference": (row["median_ms"]["parent"]["wall_ms"] - row["median_ms"]["public"]["wall_ms"])
            * 1000,
            "paired_minimum": min(pairs),
            "paired_median": statistics.median(pairs),
            "paired_maximum": max(pairs),
        }
        result["public_parent_graph_resources_identical"] = (
            row["graph_resources"]["public"] == row["graph_resources"]["parent"]
        )
        errors = {}
        for backend, checks in row["qualification"].items():
            errors[backend] = {}
            for component in ("out", "dq", "dk", "dv"):
                values = [check for check in checks if check["component"] == component]
                limit = 0.01 if row["dtype"] == "bf16" else 0.005
                assert len(values) == 16 and all(
                    check["fp64"]["finite"] and check["fp64"]["relative_error"] < limit for check in values
                )
                errors[backend][component] = {
                    policy: max(check[policy]["relative_error"] for check in values) for policy in ("fp64", "rounded")
                }
        key = f"s{row['shape'][2]}-d{row['shape'][-1]}-{row['dtype']}-{'causal' if row['causal'] else 'dense'}"
        if key in report["accuracy"]:
            assert report["accuracy"][key] == errors
        report["accuracy"][key] = errors
        result["accuracy_id"] = key
        report["rows"].append(result)
    compact_keys = ("machine_code", "accuracy", "rows")
    if args.integration:
        final = args.integration
        retained = json.loads((args.integration_checkpoint / "health.json").read_text())
        inventory = json.loads((args.integration_checkpoint / "expected-inventory.json").read_text())
        assert retained["complete"] and retained["verified_files"] == len(inventory) and not retained["missing"]
        for item in inventory:
            path = final.parent / item["file"]
            assert path.stat().st_size == item["bytes"] and digest(path) == item["sha256"]
        state = json.loads((final / "state.json").read_text())
        assert len(state["stages"]) == 5 and all(row["exit_code"] == 0 for row in state["stages"])
        actual = json.loads((final / "integration.json").read_text())
        assert actual["tests_run"] == 6 and all(actual[key] == 0 for key in ("failures", "errors", "skips"))
        measured = json.loads((final / "integration-timing.json").read_text())
        assert len(measured["rows"]) == 16
        cpu = (final / "cpu.log").read_text()
        guards = (final / "guards.log").read_text()
        report["merged_main_integration"] = {
            "source_manifest_sha256": state["source_manifest_sha256"],
            "source_tar_sha256": digest(args.integration_source_tar),
            "scope": "Independent focused current-main integration; the original 96-row evaluation remains primary. Merged A40 D256 and GB10 helpers retained.",
            "cpu_suite": {
                "total": int(re.search(r"Ran (\d+) tests", cpu)[1]),
                "skipped": int(re.search(r"skipped=(\d+)", cpu)[1]),
                "failures": 0,
            },
            "publication_guard_tests": int(re.search(r"Ran (\d+) tests", guards)[1]),
            "actual_gpu_api_tests": actual["tests_run"],
            "physical_raw_files_verified": retained["verified_files"],
            "physical_raw_bytes_verified": retained["expected_bytes"],
            "inventory_sha256": digest(args.integration_checkpoint / "expected-inventory.json"),
            "rows": [
                {
                    key: row[key]
                    for key in ("shape", "dtype", "causal", "phase", "mode", "calls_per_window", "samples", "median_ms")
                }
                for row in measured["rows"]
            ],
        }
        compact_keys += ("merged_main_integration",)

    text = json.dumps({key: value for key, value in report.items() if key not in compact_keys}, indent=2)[:-2]
    for key in compact_keys:
        values = report[key]
        if isinstance(values, dict):
            text += (
                ",\n  "
                + json.dumps(key)
                + ": {\n    "
                + ",\n    ".join(
                    json.dumps(name) + ": " + json.dumps(value, separators=(",", ":")) for name, value in values.items()
                )
                + "\n  }"
            )
        else:
            text += (
                ",\n  "
                + json.dumps(key)
                + ": [\n    "
                + ",\n    ".join(json.dumps(value, separators=(",", ":")) for value in values)
                + "\n  ]"
            )
    text += "\n}\n"
    assert json.loads(text) == report
    args.output.write_text(text)

    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "kohakufa-l40s-native-metadata"})
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    for axis, sequence in zip(axes, (1024, 4096)):
        rows = [
            row
            for row in report["rows"]
            if row["phase"] == "joint" and row["mode"] == "eager" and row["shape"][2] == sequence
        ]
        rows.sort(key=lambda row: (row["shape"][-1], row["dtype"], row["causal"]))
        for y, row in enumerate(rows):
            samples = [
                (a["wall_ms"] - b["wall_ms"]) * 1000 for a, b in zip(row["samples"]["parent"], row["samples"]["public"])
            ]
            color = "#375f7b" if row["dtype"] == "bf16" else "#c87834"
            axis.plot((min(samples), max(samples)), (y, y), color=color, linewidth=0.7, alpha=0.55)
            axis.scatter(samples, [y + (index - 4) * 0.018 for index in range(9)], color=color, s=18, alpha=0.65)
            axis.scatter(
                [row["parent_minus_public_wall_us"]["latency_median_difference"]],
                [y],
                color="black",
                marker="D",
                s=24,
                zorder=3,
            )
        labels = [
            f"D{row['shape'][-1]} {'causal' if row['causal'] else 'dense'} {row['dtype'].upper()}"
            + (" · fallback" if sequence == 1024 and row["shape"][-1] == 64 and row["causal"] else "")
            for row in rows
        ]
        axis.set_yticks(range(len(rows)), labels, fontsize=9)
        axis.invert_yaxis()
        axis.axvline(0, color="#444444", linewidth=0.8)
        axis.set_xlim(-70, 60)
        axis.set_title(f"S{sequence} · complete eager forward + backward")
        axis.set_xlabel("Parent − native guard wall time (µs) · positive favors native")
        axis.grid(axis="x", alpha=0.15)
    figure.suptitle(
        "L40S native metadata · actual public API comparison · B1/H16 · 350 W · 2026-11-02\nDiamonds: difference of latency medians. Dots: nine paired rounds. Mixed samples and regressions retained.",
        fontsize=11,
    )
    figure.savefig(
        args.plot, metadata={"Date": None, "Creator": "KohakuFA-CuTe benchmarks.summarize_l40s_native_metadata"}
    )
    if args.plot.suffix.lower() == ".svg":
        args.plot.write_text("\n".join(line.rstrip() for line in args.plot.read_text().splitlines()) + "\n")
    plt.close(figure)


if __name__ == "__main__":
    main()
