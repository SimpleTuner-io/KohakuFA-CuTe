"""Bounded SM120 standalone TMA4D collector with explicit raw failure statuses."""

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
source = Path.cwd() / "benchmarks/diagnostics/tma4d_store.py"
spec = importlib.util.spec_from_file_location("reviewed_tma4d", source)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
start = time.monotonic()
summary = {
    "scope": "SM120 standalone rank4 TMA functional diagnostic; no H200 qualification",
    "rows": [],
    "collection_completed": False,
}
summary["sanitizer_version"] = subprocess.check_output(["compute-sanitizer", "--version"], text=True)
summary["instrumented_loaded_capture"] = "Explicitly disabled; only SHA-matched uninstrumented controls are directly bound."


def checkpoint():
    (a.output / "matrix.json").write_text(json.dumps(summary, indent=2) + "\n")


def invoke(argv, label, case=None):
    remaining = 240 - (time.monotonic() - start)
    if remaining < 5:
        raise TimeoutError("overall diagnostic deadline reached")
    with (a.output / (label + ".log")).open("w") as log:
        result = helper.execute_process(argv, log, min(55, remaining))
    row = {"label": label, "argv": argv, "process": result}
    if case is not None:
        report = a.output / label / "result.json"
        if report.exists():
            row["result"] = json.loads(report.read_text())
        else:
            row["result"] = {"physical_verification": "missing", "code_binding": "missing"}
    summary["rows"].append(row)
    checkpoint()
    if result.get("signal"):
        raise InterruptedError("collector was interrupted")
    return row


def case(mode, initialization, rows, tool):
    label = f"{mode}-{initialization}-{rows}-{tool}"
    argv = [
        sys.executable,
        str(source),
        "run",
        "--build",
        str(a.output / "build"),
        "--mode",
        mode,
        "--initialization",
        initialization,
        "--rows",
        str(rows),
        "--tool",
        tool,
        "--timeout",
        "45",
        "--output",
        str(a.output / label),
    ]
    if tool != "none":
        control = a.output / f"{mode}-{initialization}-{rows}-none" / "result.json"
        argv += ["--capture", "disabled", "--control-result", str(control)]
    return invoke(argv, label, True)


def physically_bound(row):
    r = row.get("result", {})
    binding = r.get("code_binding") == "passed" or (
        r.get("code_binding") == "unavailable_under_sanitizer" and bool(r.get("matching_uninstrumented_control"))
    )
    return r.get("physical_verification") == "passed" and binding


try:
    build = invoke([sys.executable, str(source), "build", "--arch", "sm_120", "--output", str(a.output / "build")], "build")
    if build["process"]["returncode"]:
        raise RuntimeError("build failed; no GPU test launched")
    smoke = [
        case(mode, initialization, rows, "none")
        for mode in ("read", "full", "ordinary")
        for rows in (256, 257)
        for initialization in ("uninitialized", "poison")
    ]
    if not all(physically_bound(row) and row["process"]["returncode"] == 0 for row in smoke):
        raise RuntimeError("smoke physical/loaded-code qualification failed; sanitizer stages skipped")
    init = []
    for mode in ("read", "full", "ordinary"):
        init.extend(
            case(mode, initialization, rows, "initcheck")
            for rows in (256, 257)
            for initialization in ("uninitialized", "poison")
        )
    if all(physically_bound(row) for row in init):
        for tool in ("memcheck", "racecheck", "synccheck"):
            for mode in ("read", "full", "ordinary"):
                case(mode, "uninitialized", 257, tool)
    else:
        summary["safety_stage_skipped"] = (
            "Instrumented execution/physical/code-binding limitation; no invented TMA classification."
        )
    summary["collection_completed"] = True
except Exception as error:
    summary["collection_error"] = str(error)
finally:
    summary["elapsed_seconds"] = time.monotonic() - start
    summary["unmodified_init_clean"] = all(
        row["process"]["returncode"] == 0 for row in summary["rows"] if row["label"].endswith("initcheck")
    ) and any(row["label"].endswith("initcheck") for row in summary["rows"])
    statuses = [row["process"]["returncode"] for row in summary["rows"]]
    summary["terminal_status"] = (
        124
        if 124 in statuses
        else 2 if summary.get("collection_error") else 99 if 99 in statuses else 2 if any(statuses) else 0
    )
    checkpoint()
raise SystemExit(summary["terminal_status"])
