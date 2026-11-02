"""Summarize the independently qualified L40S public short-kernel policy."""

import argparse
import collections
import copy
import json
import re
import sqlite3
import statistics
from pathlib import Path

from benchmarks.summarize_l40s_short import chart, load, sha, verified


def compact(report):
    rows = []
    for row in report["rows"]:
        errors = {}
        for label, components in row["qualification"].items():
            for component in components:
                error = component["fp64"]
                assert error["finite"] and error["relative_error"] < (0.01 if row["dtype"] == "bf16" else 0.005)
                key = label + ":" + component["component"]
                errors[key] = max(errors.get(key, 0), error["relative_error"])
        metric = "cuda_event_ms" if row["mode"] == "graph" else "wall_ms"
        differences = [
            1000 * (a[metric] - b[metric])
            for a, b in zip(row["samples"]["basic_default_false"], row["samples"]["public"])
        ]
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
                "basic_false_minus_public_paired_us": {
                    "min": min(differences),
                    "median": statistics.median(differences),
                    "max": max(differences),
                },
            }
        )
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "public", "probe", "public-record", "probe-record", "source", "output", "svg"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    raw, probe = args.archive / args.public, args.archive / args.probe
    archive = {"public": verified(args.archive, args.public_record), "probe": verified(args.archive, args.probe_record)}
    state = load(raw / "state.json")
    assert state.get("finished") and len(state["stages"]) == 14 and all(s["exit_code"] == 0 for s in state["stages"])
    probe_state = load(probe / "state.json")
    assert probe_state.get("finished") and all(s["exit_code"] == 0 for s in probe_state["stages"])
    qualification = load(raw / "qualification.json")
    assert (
        qualification["status"] == "pass"
        and len(qualification["rows"]) == 18
        and len(qualification["diagnostics"]) == 1
    )
    gates = load(raw / "public-gates.json")
    assert gates["tests"] == 15 and gates["errors"] == gates["failures"] == gates["skips"] == 0
    cpu_log = (raw / "cpu.log").read_text()
    cpu_total = int(re.search(r"Ran (\d+) tests", cpu_log)[1])
    cpu_skipped = int(re.search(r"OK \(skipped=(\d+)\)", cpu_log)[1])
    equivalence = load(raw / "code-equivalence.json")
    assert len(equivalence["rows"]) == 4 and all(r["loaded_cubin_sass_ptx_equal_probe"] for r in equivalence["rows"])
    tools = []
    for tool in ("memcheck", "racecheck", "synccheck", "initcheck"):
        result = load(probe / (tool + ".json"))
        assert result["status"] == "pass" and len(result["rows"]) == 12
        footer = (probe / (tool + ".log")).read_text()
        expected = (
            "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
            if tool == "racecheck"
            else "ERROR SUMMARY: 0 errors"
        )
        assert expected in footer, tool
        tools.append(
            {
                "tool": tool,
                "cases": 12,
                "device_errors": 0,
                "report_sha256": sha(probe / (tool + ".json")),
                "scope": "Private L40S flag-policy probe; retained only after four exact loaded cubin/PTX/SASS equivalence checks.",
            }
        )
    primary, nograd, long = [compact(load(raw / folder / "report.json")) for folder in ("paired", "nograd", "long")]
    assert [len(rows) for rows in (primary, nograd, long)] == [24, 8, 8]
    original = load(raw / "original-pair/report.json")
    balanced = load(raw / "balanced/report.json")
    assert original["status"] == balanced["status"] == "pass" and len(original["rows"]) == len(balanced["rows"]) == 8
    for row in original["rows"]:
        a, b = row["resources"]["original"], row["resources"]["optimized"]
        assert a["operand_sha256"] == b["operand_sha256"] and a["native_sha256"] == b["native_sha256"]
    bindings = []
    for folder in sorted(raw.glob("code-*-*-*-*")):
        b = load(folder / "binding.json")
        assert b["loaded_modules"] and b["exact_symbol_ptx"] and b["ptx_status"] == "exact_symbol_retained"
        sass = list((folder / "loaded").glob("*.sass"))
        assert len(sass) == 1
        instructions = collections.Counter(
            re.findall(r"/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9]*)", sass[0].read_text())
        )
        bindings.append(
            {
                "label": folder.name,
                "symbol": b["symbol"],
                "graph": b["graph"],
                "loaded_cubin_sha256": [m["loaded_cubin_sha256"] for m in b["loaded_modules"]],
                "sass_sha256": sha(sass[0]),
                "exact_ptx_sha256": [t["sha256"] for t in b["exact_symbol_ptx"]],
                "library": b["library"],
                "static_instruction_sites": {k: instructions[k] for k in ("HMMA", "LDSM", "LDS", "BAR", "LDL", "STL")},
            }
        )
    assert len(bindings) == 16
    manifest = load(args.source / "source-manifest.json")
    assert (
        manifest["src/kohakufa_cute/short_owned.py"]
        == "97913a8c623edbde19ceacc324fd32159fbee8b10cf35b3a687c355a1b748dfa"
    )
    native = load(raw / "native-host.json")
    assert (
        native["cpp_source_sha256"] == manifest["src/kohakufa_cute/native.cpp"]
        and sha(raw / "native-host.so") == native["sha256"]
    )
    database = sqlite3.connect(raw / "short-systems.sqlite")
    eager_sequences = []
    for start, end, label in database.execute(
        'select start,end,text from NVTX_EVENTS where text like "%eager-joint%" order by start'
    ):
        launched = database.execute(
            "select s.value from CUPTI_ACTIVITY_KIND_KERNEL k join CUPTI_ACTIVITY_KIND_RUNTIME r using(correlationId) join StringIds s on s.id=k.shortName where r.start>=? and r.end<=? order by r.start",
            (start, end),
        ).fetchall()
        names = [name for (name,) in launched]
        assert len(names) == 10 and all(names[i : i + 2] == names[:2] for i in range(0, 10, 2)), label
        eager_sequences.append({"range": label, "calls": 5, "kernels_per_call": 2, "sequence": names[:2]})
    assert len(eager_sequences) == 8
    graph_inventory = load(raw / "graph-kernels.json")
    assert len(graph_inventory) == 8 and all(len(row["graph"]) == 2 for row in graph_inventory)
    report = {
        "scope": "Full NVIDIA L40S,142SM/SM89,B1H16S128D16,FP16/BF16,dense/causal training only. Actual public policy independently qualified; no precision change.",
        "date": "2026-11-02",
        "policy": {
            "BF16_dense": {"narrow_product": False, "prefetch_statistics": True},
            "FP16_dense": {"narrow_product": True, "prefetch_statistics": True},
            "causal_both_dtypes": {"narrow_product": True, "prefetch_statistics": False},
        },
        "baseline_scope": "Primary same-process baseline uses the actual PR81 Python namespace with shared-class flags disabled. It is not relabeled as the pristine original class. Eight separate persistent-process actual-public controls retain the pristine original class under matching native host and identical rounded input bytes.",
        "runtime": {k: load(raw / "paired/report.json")[k] for k in ("device", "torch", "driver_sha256", "protocol")},
        "source": {
            "tar_sha256": sha(args.source / "source.tar.gz"),
            "manifest_sha256": sha(args.source / "source-manifest.json"),
            "runtime_files": {k: v for k, v in manifest.items() if k.startswith("src/")},
        },
        "qualification": {
            "finite_directed_cases": 18,
            "expected_cast_overflow_diagnostics": 1,
            "public_API_tests": gates,
            "CPU_total": cpu_total,
            "CPU_expected_skips": cpu_skipped,
            "publication_tests": 24,
            "sanitizers": tools,
        },
        "stages": [{"stage": s["label"], "exit_code": s["exit_code"]} for s in state["stages"]],
        "archive": archive,
        "primary_rows": primary,
        "no_grad_controls": nograd,
        "long_controls": long,
        "shared_storage_controls": balanced,
        "pristine_original_class_controls": original,
        "code_equivalence": equivalence,
        "bindings": bindings,
        "host_bridge": native,
        "systems": {
            "capture_sha256": sha(raw / "short-systems.nsys-rep"),
            "sqlite_sha256": sha(raw / "short-systems.sqlite"),
            "graph_inventory": graph_inventory,
            "eager_runtime_correlated_sequences": eager_sequences,
            "scope": "Actual eager/graph selection proof, separate from uninstrumented performance medians.",
        },
        "static_instruction_scope": "Whole selected function sites; no runtime fractions, transferred-byte or stall-percentage inference.",
    }

    # Compiler export metadata contains only filenames/hashes, never local paths.
    def safe(value):
        if isinstance(value, dict):
            return {k: safe(v) for k, v in value.items() if k != "compiled_programs"}
        if isinstance(value, list):
            return [safe(v) for v in value]
        return value

    report = safe(report)
    with args.output.open("w") as f:
        f.write("{\n")
        for i, (key, value) in enumerate(report.items()):
            f.write("  " + json.dumps(key) + ": ")
            if isinstance(value, list):
                f.write("[\n")
                for j, row in enumerate(value):
                    f.write(
                        "    " + json.dumps(row, separators=(",", ":")) + ("," if j + 1 < len(value) else "") + "\n"
                    )
                f.write("  ]")
            else:
                f.write(json.dumps(value, separators=(",", ":")))
            f.write(("," if i + 1 < len(report) else "") + "\n")
        f.write("}\n")
    chart_rows = copy.deepcopy(primary)
    for row in chart_rows:
        for key in ("samples", "median_ms"):
            row[key]["parent"] = row[key].pop("basic_default_false")
    chart(chart_rows, args.svg)
    text = (
        args.svg.read_text()
        .replace("L40S public short training", "L40S optimized public short training")
        .replace("Preceding public API", "Basic, flags disabled")
        .replace("L40S short owner", "L40S optimized owner")
    )
    args.svg.write_text(text)


if __name__ == "__main__":
    main()
