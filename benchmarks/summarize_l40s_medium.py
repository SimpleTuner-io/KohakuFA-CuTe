"""Generate the bounded medium-causal report from verified retained artifacts."""

import argparse
import gzip
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    opener = gzip.open if not path.exists() else open
    actual = path if path.exists() else Path(str(path) + ".gz")
    with opener(actual, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(archive, record):
    health = read(record / "health.json")
    rows = read(record / "expected-inventory.json")
    assert health["complete"] and health["verified_files"] == len(rows) and not health["missing"]
    assert all(digest(archive / row["file"]) == row["sha256"] for row in rows)
    return {
        "files": len(rows),
        "bytes": sum(row["bytes"] for row in rows),
        "inventory_sha256": digest(record / "expected-inventory.json"),
        "physical_SHA_verified": True,
    }


def render(value, depth=0):
    """Keep dictionaries readable and retained sample rows compact."""
    padding = "  " * depth
    child = "  " * (depth + 1)
    if isinstance(value, dict):
        return (
            "{\n"
            + ",\n".join(child + json.dumps(key) + ": " + render(item, depth + 1) for key, item in value.items())
            + "\n"
            + padding
            + "}"
        )
    if isinstance(value, list):
        return "[\n" + ",\n".join(child + json.dumps(item, separators=(",", ":")) for item in value) + "\n" + padding + "]"
    return json.dumps(value)


def compact(report):
    rows = []
    for row in report["rows"]:
        rows.append({k: row[k] for k in ("shape", "dtype", "causal", "phase", "mode", "median_ms", "samples")})
    return {"protocol": report["protocol"], "rows": rows}


def chart(report, target):
    rows = [row for row in report["rows"] if row["phase"] == "joint"]
    labels = [
        ("old_public_PR88", "Preceding public", "#8290a8"),
        ("public", "Selected public", "#20b99d"),
        ("fa2", "FA2", "#edaf56"),
        ("sdpa", "SDPA", "#8d7ce8"),
    ]
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="590" viewBox="0 0 1100 590">',
        '<rect width="1100" height="590" fill="#101724"/>',
        '<g font-family="sans-serif" fill="#e8eef8">',
        '<text x="36" y="38" font-size="23">L40S · causal S1024 D64 training · B1/H16</text>',
        '<text x="36" y="64" font-size="14">November 2 · held-out public calls · lower is faster · compilation and profiling excluded</text>',
    ]
    for index, (_, name, color) in enumerate(labels):
        x = 36 + index * 250
        svg += [
            f'<rect x="{x}" y="85" width="14" height="14" fill="{color}"/>',
            f'<text x="{x+22}" y="98" font-size="13">{escape(name)}</text>',
        ]
    for panel, mode in enumerate(("graph", "eager")):
        panel_rows = [row for row in rows if row["mode"] == mode]
        left, top, width = 36 + panel * 540, 150, 465
        maximum = (
            max(
                row["median_ms"][key]["cuda_event_ms" if mode == "graph" else "wall_ms"] * 1000
                for row in panel_rows
                for key, _, _ in labels
            )
            * 1.18
        )
        svg += [
            f'<text x="{left}" y="135" font-size="18">{"Graph replay" if mode == "graph" else "Complete eager wall time"} (µs)</text>'
        ]
        for group, row in enumerate(panel_rows):
            y = top + group * 175
            svg.append(f'<text x="{left}" y="{y}" font-size="14">{row["dtype"].upper()}</text>')
            for index, (key, _, color) in enumerate(labels):
                value = row["median_ms"][key]["cuda_event_ms" if mode == "graph" else "wall_ms"] * 1000
                bar_y = y + 12 + index * 30
                svg += [
                    f'<rect x="{left}" y="{bar_y}" width="{value/maximum*width:.2f}" height="20" rx="3" fill="{color}"/>',
                    f'<text x="{left+value/maximum*width+6:.2f}" y="{bar_y+15}" font-size="12">{value:.2f}</text>',
                ]
    svg += [
        '<text x="36" y="548" font-size="13">Nine rotating 100 ms windows; independent seed 801779. Graph and eager use separate scales.</text>',
        '<text x="36" y="571" font-size="13">Preceding control uses the original PR88 mapped host binary and Python eligibility policy.</text>',
        "</g></svg>",
    ]
    target.write_text("\n".join(svg) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--initial", required=True)
    parser.add_argument("--continuation", required=True)
    parser.add_argument("--initial-record", type=Path, required=True)
    parser.add_argument("--continuation-record", type=Path, required=True)
    parser.add_argument("--control-code", default="l40s-medium-causal-code-v1-58719868-e2d6c8dde2")
    parser.add_argument("--output", type=Path, required=True, help="Write campaign output outside the checkout.")
    args = parser.parse_args()
    original, continuation = args.archive / args.initial, args.archive / args.continuation
    states = {
        label: read(folder / "state.json") for label, folder in [("initial", original), ("continuation", continuation)]
    }
    initial_status = {row["label"]: row.get("exit_code") for row in states["initial"]["stages"]}
    assert initial_status == {
        "preflight": 0,
        "cpu": 0,
        "guards": 0,
        "public-gates": 0,
        "qualification": 0,
        "code": 0,
        "memcheck": 86,
    }
    assert states["initial"]["stop"]
    assert states["continuation"].get("finished")
    assert all(row.get("exit_code") == 0 for row in states["continuation"]["stages"])
    frozen_sources = {}
    for label, name in (("initial", args.initial), ("continuation", args.continuation)):
        folder = args.archive / "frozen-sources" / name
        receipt = read(folder / "launch.json")
        assert digest(folder / "source.tar.gz") == receipt["tar_sha256"]
        frozen_sources[label] = {
            "source_sha256": receipt["source_sha256"],
            "tar_sha256": receipt["tar_sha256"],
            "payload_files": receipt["verified_payload_files"],
        }
    archives = {
        "initial": verify(args.archive, args.initial_record),
        "continuation": verify(args.archive, args.continuation_record),
    }
    report = read(continuation / "paired/report.json")
    repository = Path(__file__).resolve().parents[1]
    assert all(digest(repository / name) == checksum for name, checksum in report["source_sha256"].items())
    assert len(report["rows"]) == 12 and report["protocol"]["seed"] == 801779
    qualification = read(original / "qualification.json")
    assert qualification["status"] == "pass" and len(qualification["rows"]) == 10
    tools = {}
    for tool in ("memcheck", "racecheck", "synccheck", "initcheck"):
        q = read(continuation / (tool + ".json"))
        assert q["status"] == "pass" and len(q["rows"]) == 6
        log = (continuation / (tool + ".log")).read_text()
        if tool == "racecheck":
            assert re.search(r"RACECHECK SUMMARY: 0 hazards displayed \(0 errors, 0 warnings\)", log)
        else:
            assert "ERROR SUMMARY: 0 errors" in log
        tools[tool] = {"status": 0, "numerical_rows": 6, "log_sha256": digest(continuation / (tool + ".log"))}
    code = []
    for path in sorted(original.glob("code-*/binding.json")):
        binding = read(path)
        assert binding["loaded_modules"] and binding["symbol"]
        selected = next(row for row in binding["graph"] if row["symbol"] == binding["symbol"])
        for index, module in enumerate(binding["loaded_modules"]):
            assert digest(path.parent / "loaded" / f"selected-{index}.cubin") == module["loaded_cubin_sha256"]
            assert digest(path.parent / "loaded" / f"selected-{index}.sass") == module["sass_sha256"]
        code.append(
            {
                "capture": path.parent.name,
                "binding_sha256": digest(path),
                "selected": selected,
                "loaded_modules": binding["loaded_modules"],
                "PTX": binding["exact_symbol_ptx"],
            }
        )
    assert len(code) == 8
    matched_code_audit = []
    control_proof = verify(args.archive, args.archive / "native-metadata-analysis/medium-causal-code-raw")
    for dtype, stages in (("bf16", 2), ("fp16", 1)):
        entries = {}
        for label, folder in (
            ("G1", args.archive / args.control_code / f"code-n64-q{stages}-{dtype}"),
            ("G16", original / f"code-candidate-backward-{dtype}"),
        ):
            bound = read(folder / "binding.json")
            selected = next(row for row in bound["graph"] if row["symbol"] == bound["symbol"])
            assert digest(folder / "loaded/selected-0.cubin") == bound["loaded_modules"][0]["loaded_cubin_sha256"]
            sass = (folder / "loaded/selected-0.sass").read_text()
            counts = Counter()
            for line in sass.splitlines():
                match = re.match(r"\s*/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)\s", line)
                if match:
                    counts[match[1].split(".")[0]] += 1
            entries[label] = {
                "loaded_cubin_sha256": bound["loaded_modules"][0]["loaded_cubin_sha256"],
                "sass_sha256": digest(folder / "loaded/selected-0.sass"),
                "resources": selected,
                "whole_function_static_sites": {
                    key: counts[key] for key in ("HMMA", "LDSM", "LDGSTS", "LDS", "STS", "BAR", "DEPBAR")
                },
            }
        matched_code_audit.append(
            {
                "dtype": dtype,
                "query_stages": stages,
                "entries": entries,
                "scope": "SameN64 geometry/inputwork; wholefunction static sites, not executed totals/runtimepercentages. G16 changes CTAordering; no measured occupancy/cache-counter claim.",
            }
        )
    host = []
    for path in sorted((continuation / "host-bindings").glob("*/binding.json")):
        binding = read(path)
        for key, value in binding.items():
            assert value["mapped"] and digest(path.parent / (key + ".so")) == value["sha256"]
        assert binding["parent"]["sha256"] == "6325331ded4254c59f5ff13197244e560e3718310cef1696139bbdc57dbeb38e"
        host.append(binding)
    assert len(host) >= 3, "Actual paired/noGrad/legacy process mappings required"
    paired_variation = []
    for row in report["rows"]:
        if row["phase"] != "joint":
            continue
        metric = "cuda_event_ms" if row["mode"] == "graph" else "wall_ms"
        selected_samples = [sample[metric] * 1000 for sample in row["samples"]["public"]]
        parent_samples = [sample[metric] * 1000 for sample in row["samples"]["old_public_PR88"]]
        differences = [a - b for a, b in zip(selected_samples, parent_samples)]
        paired_variation.append(
            {
                "dtype": row["dtype"],
                "mode": row["mode"],
                "metric": metric,
                "selected_minus_parent_median_us": statistics.median(selected_samples) - statistics.median(parent_samples),
                "median_round_delta_us": statistics.median(differences),
                "round_delta_us": differences,
                "rounds_favoring_selected": sum(value < 0 for value in differences),
                "selected_range_us": [min(selected_samples), max(selected_samples)],
                "parent_range_us": [min(parent_samples), max(parent_samples)],
            }
        )
    output = {
        "scope": "Public B1/H16/S1024/D64 causal BF16/FP16 training, full142SM L40S; unchanged rounded policy.",
        "device": report["device"],
        "torch": report["torch"],
        "source_revision": report["source_revision"],
        "source_sha256": report["source_sha256"],
        "jobs": {"initial": args.initial, "continuation": args.continuation},
        "stages": {
            label: {row["label"]: row.get("exit_code") for row in state["stages"]} for label, state in states.items()
        },
        "tool_limitation": "Initial memcheck86:14 unsupported cuGetProcAddress API-version capability queries13041>driver13040;6 numerical smoke rows passed. Exact-source continuation explicitly disables API-error reporting only; device memory/race/sync/init instrumentation remains.",
        "archives": archives,
        "frozen_sources": frozen_sources,
        "qualification": {"rows": 10, "status": 0},
        "public_gates": read(original / "public-gates.json"),
        "tools": tools,
        "loaded_code": code,
        "matched_G1_G16_code": matched_code_audit,
        "control_code_archive": control_proof,
        "timing_process_host_bindings": host,
        "paired_joint_variation": paired_variation,
        "heldout": compact(report),
        "nograd": compact(read(continuation / "nograd/report.json")),
        "legacy": compact(read(continuation / "legacy/report.json")),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(output) + "\n")
    assert read(args.output) == output
    chart(report, args.output.with_suffix(".svg"))
    print("Verified and generated", args.output)


if __name__ == "__main__":
    main()
