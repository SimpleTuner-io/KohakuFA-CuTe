"""Build the bounded L40S short-training report from verified retained captures."""

import argparse
import collections
import hashlib
import html
import json
import re
import statistics
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified(root, record):
    health = load(record / "health.json")
    inventory = load(record / "expected-inventory.json")
    assert health["complete"] and health["verified_files"] == len(inventory)
    for row in inventory:
        path = root / row["file"]
        assert path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], row["file"]
    return {
        "files": len(inventory),
        "bytes": sum(r["bytes"] for r in inventory),
        "inventory_sha256": sha(record / "expected-inventory.json"),
    }


def compact(report):
    rows = []
    for row in report["rows"]:
        errors = {}
        for policy, components in row["qualification"].items():
            for component in components:
                error = component["fp64"]
                assert error["finite"] and error["relative_error"] < (0.01 if row["dtype"] == "bf16" else 0.005)
                key = policy + ":" + component["component"]
                errors[key] = max(errors.get(key, 0), error["relative_error"])
        key = "cuda_event_ms" if row["mode"] == "graph" else "wall_ms"
        differences = [1000 * (p[key] - c[key]) for p, c in zip(row["samples"]["parent"], row["samples"]["public"])]
        rows.append(
            {
                **{
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
                },
                "max_per_head_fp64_relative_error": errors,
                "parent_minus_public_paired_us": {
                    "min": min(differences),
                    "median": statistics.median(differences),
                    "max": max(differences),
                },
            }
        )
    return rows


def chart(rows, output):
    colors = {"public": "#6b47ba", "parent": "#777f8a", "fa2": "#087f8c", "sdpa": "#c97812"}
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630">',
        '<rect width="1200" height="630" fill="white"/>',
    ]

    def text(x, y, value, size=14, fill="#253041", anchor="start"):
        parts.append(
            f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{size}" fill="{fill}" text-anchor="{anchor}">{html.escape(str(value))}</text>'
        )

    text(35, 38, "L40S public short training: B1 H16 S128 D16", 25)
    text(35, 64, "November 2, 2026 · held-out seed · 9 alternating rounds · complete forward + backward", 15)
    for column, mode, limit, title in (
        (0, "graph", 20, "Graph replay / CUDA events (µs)"),
        (1, "eager", 600, "Complete eager call / wall time (µs)"),
    ):
        left = 150 + 560 * column
        width = 400
        text(left, 104, title, 18)
        for tick in range(5):
            x = left + width * tick / 4
            parts.append(f'<path d="M{x} 120 V535" stroke="#e4e8ed"/>')
            text(x, 558, f"{limit*tick/4:g}", 12, anchor="middle")
        selected = [r for r in rows if r["phase"] == "joint" and r["mode"] == mode]
        for index, row in enumerate(selected):
            top = 143 + index * 98
            if column == 0:
                text(35, top + 28, row["dtype"].upper(), 14)
                text(35, top + 48, "causal" if row["causal"] else "dense", 14)
            metric = "cuda_event_ms" if mode == "graph" else "wall_ms"
            for backend_index, backend in enumerate(colors):
                y = top + backend_index * 18
                value = row["median_ms"][backend][metric] * 1000
                samples = [s[metric] * 1000 for s in row["samples"][backend]]
                x = left + width * value / limit
                parts.append(
                    f'<rect x="{left}" y="{y}" width="{width*value/limit}" height="12" fill="{colors[backend]}" opacity=".85"/>'
                )
                low, high = [left + width * v / limit for v in (min(samples), max(samples))]
                parts.append(
                    f'<path d="M{low} {y+6} H{high} M{low} {y+2} V{y+10} M{high} {y+2} V{y+10}" stroke="#1f2937"/>'
                )
                text(x + 6, y + 11, f"{value:.2f}", 11)
    for index, (backend, color) in enumerate(colors.items()):
        x = 120 + index * 260
        parts.append(f'<rect x="{x}" y="588" width="16" height="12" fill="{color}"/>')
        text(
            x + 24,
            600,
            {
                "public": "L40S short owner",
                "parent": "Preceding public API",
                "fa2": "Standalone FA2",
                "sdpa": "PyTorch SDPA",
            }[backend],
            13,
        )
    text(
        35,
        622,
        "Bars: medians; whiskers: min/max rounds. Eager includes host overhead. This chart covers only the qualified training shape.",
        12,
    )
    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "archive",
        "public",
        "completion",
        "probe",
        "probe-record",
        "probe-source",
        "public-record",
        "completion-record",
        "source",
        "output",
        "svg",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--failed-record", action="append", type=Path, default=[])
    args = parser.parse_args()
    raw, completion, probe = [args.archive / p for p in (args.public, args.completion, args.probe)]
    inventory = {
        "probe": verified(args.archive, args.probe_record),
        "public": verified(args.archive, args.public_record),
        "completion": verified(args.archive, args.completion_record),
    }
    failed_attempts = []
    for index, record in enumerate(args.failed_record):
        inventory["failed_" + str(index + 1)] = verified(args.archive, record)
        state_file = next(
            r["file"] for r in load(record / "expected-inventory.json") if r["file"].endswith("/state.json")
        )
        failed_state = load(args.archive / state_file)
        assert any(s.get("exit_code", 0) != 0 for s in failed_state["stages"])
        failed_attempts.append(
            {
                "job": Path(state_file).parts[0],
                "stages": [{"stage": s["label"], "exit_code": s.get("exit_code")} for s in failed_state["stages"]],
                "scope": "Retained packaging or offline-statistics failure, before any new GPU math; never counted as a successful gate.",
            }
        )
    state = load(raw / "state.json")
    assert all(s["exit_code"] == 0 for s in state["stages"] if s["label"] != "systems")
    assert next(s for s in state["stages"] if s["label"] == "systems")["exit_code"] == 1
    completed = load(completion / "state.json")
    assert completed.get("finished") and all(s["exit_code"] == 0 for s in completed["stages"])
    qualification = load(raw / "qualification.json")
    assert (
        qualification["status"] == "pass"
        and len(qualification["rows"]) == 18
        and len(qualification["diagnostics"]) == 1
    )
    gates = load(raw / "public-gates.json")
    assert gates == {"tests": 13, "failures": 0, "errors": 0, "skips": 0, "cpu_reference_threads": 4}
    tools = []
    for name in ("memcheck", "racecheck", "synccheck", "initcheck"):
        report = load(probe / (name + ".json"))
        assert report["status"] == "pass" and len(report["rows"]) == 12
        tools.append(
            {
                "tool": name,
                "cases": 12,
                "device_errors": 0,
                "report_sha256": sha(probe / (name + ".json")),
                "scope": "Prior L40S pilot; exact unchanged GPU source, not a rerun of changed memory ownership.",
            }
        )
    bindings = []
    for folder in sorted(raw.glob("code-*-*-*-*")):
        binding = load(folder / "binding.json")
        assert (
            binding["loaded_modules"]
            and binding["exact_symbol_ptx"]
            and binding["ptx_status"] == "exact_symbol_retained"
        )
        sass = list((folder / "loaded").glob("*.sass"))
        assert len(sass) == 1
        counts = collections.Counter(
            re.findall(r"/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9]*)", sass[0].read_text())
        )
        bindings.append(
            {
                "label": folder.name,
                "symbol": binding["symbol"],
                "graph": binding["graph"],
                "loaded_cubin_sha256": [m["loaded_cubin_sha256"] for m in binding["loaded_modules"]],
                "sass_sha256": sha(sass[0]),
                "exact_ptx_sha256": [p["sha256"] for p in binding["exact_symbol_ptx"]],
                "loaded_library": binding["library"],
                "static_instruction_sites": {k: counts[k] for k in ("HMMA", "LDSM", "BAR", "LDL", "STL")},
            }
        )
    assert len(bindings) == 16
    systems = load(completion / "systems/systems-proof.json")
    mapped = load(completion / "mapped-public/host-bridge-binding.json")
    assert len(mapped["rows"]) == 4 and sha(completion / "mapped-public/native-host.so") == mapped["library_sha256"]
    manifest = load(args.source / "source-manifest.json")
    assert mapped["native_source_sha256"] == manifest["src/kohakufa_cute/native.cpp"]
    prior_manifest = load(args.probe_source / "source-manifest.json")
    gpu_files = [
        "short_owned.py",
        "kernels.py",
        "native.py",
        "ada.py",
        "l40s.py",
        "weight_precision.py",
        "short_forward.py",
        "precision.py",
    ]
    for name in gpu_files:
        name = "src/kohakufa_cute/" + name
        assert manifest[name] == prior_manifest[name], name
    cpu_log = (raw / "cpu.log").read_text()
    cpu_total = int(re.search(r"Ran (\d+) tests", cpu_log)[1])
    cpu_skips = int(re.search(r"OK \(skipped=(\d+)\)", cpu_log)[1])
    assert cpu_total == 394 and cpu_skips == 190
    primary, nograd, long, confirmation = [
        compact(load(p))
        for p in (
            raw / "paired/report.json",
            raw / "nograd/report.json",
            raw / "long/report.json",
            completion / "paired/report.json",
        )
    ]
    assert [len(r) for r in (primary, nograd, long, confirmation)] == [24, 8, 8, 8]
    report = {
        "scope": "Full142SM NVIDIA L40S B1H16S128D16 FP16/BF16 dense/causal training only; exact basic exclusive-owner GPU methods from PR76. No precision-policy change.",
        "date": "2026-11-02",
        "runtime": {k: load(raw / "paired/report.json")[k] for k in ("device", "torch", "driver_sha256", "protocol")},
        "source": {
            "manifest_sha256": sha(args.source / "source-manifest.json"),
            "tar_sha256": sha(args.source / "source.tar.gz"),
            "runtime_files": {k: v for k, v in manifest.items() if k.startswith("src/")},
        },
        "qualification": {
            "finite_cases": 18,
            "expected_cast_overflow_diagnostics": 1,
            "actual_API_tests": gates,
            "sanitizers": tools,
            "CPU_total": cpu_total,
            "CPU_expected_GPU_skips": cpu_skips,
            "publication_tests": 24,
        },
        "stage_receipts": {
            "public": [{"stage": s["label"], "exit_code": s["exit_code"]} for s in state["stages"]],
            "completion": [{"stage": s["label"], "exit_code": s["exit_code"]} for s in completed["stages"]],
            "limitations": "Original Systems statistics command failed on stale SQLite. Its old export/failure remain retained; offline re-export and correlated launch analysis passed separately. Native mapping receipt comes from a new four-case verification, not the earlier timed process.",
        },
        "retained_failed_attempts": failed_attempts,
        "archive": inventory,
        "primary_rows": primary,
        "no_grad_controls": nograd,
        "long_controls": long,
        "independent_confirmation": confirmation,
        "bindings": bindings,
        "systems": {
            "capture_sha256": systems["retained_capture_sha256"],
            "eager": systems["eager"],
            "graph_scope": systems["graph_scope"],
        },
        "mapped_bridge": {k: v for k, v in mapped.items() if k not in ("mapped_lines", "rows")},
        "instruction_scope": "Whole selected function sites, not executed instruction counts, transferred bytes, stall percentages or runtime fractions.",
    }
    with args.output.open("w") as handle:
        handle.write("{\n")
        for index, (key, value) in enumerate(report.items()):
            handle.write("  " + json.dumps(key) + ": ")
            if isinstance(value, list):
                handle.write("[\n")
                for item_index, item in enumerate(value):
                    handle.write(
                        "    "
                        + json.dumps(item, separators=(",", ":"))
                        + ("," if item_index + 1 < len(value) else "")
                        + "\n"
                    )
                handle.write("  ]")
            else:
                handle.write(json.dumps(value, separators=(",", ":")))
            handle.write(("," if index + 1 < len(report) else "") + "\n")
        handle.write("}\n")
    chart(primary, args.svg)


if __name__ == "__main__":
    main()
