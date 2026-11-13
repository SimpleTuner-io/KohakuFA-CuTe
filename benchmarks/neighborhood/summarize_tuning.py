"""Export compact public evidence from matched neighborhood tuning reports."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from benchmarks.neighborhood.summarize_performance import (
    accuracy_summary,
    public_token,
    source_metadata,
    timing_summary,
    workload_metadata,
)


def summarize_report(path):
    raw = json.loads(path.read_text())
    if raw.get("schema") != "neighborhood-tuning-v1":
        raise ValueError("Expected a retained neighborhood tuning report.")
    protocol = raw["protocol"]
    rows = []
    for original in raw["rows"]:
        row = dict(original)
        case = row["workload"]
        row["shape"] = [case["batch"], case["heads"], math.prod(case["spatial"]), 64]
        item = dict(
            workload=workload_metadata(row),
            variant=public_token(row["variant"]),
            status=public_token(row["status"]),
            configuration=row["configuration"],
            effective_schedule=row.get("effective_schedule"),
            accuracy=accuracy_summary(row),
            baseline_equivalence=row.get("baseline_equivalence"),
            schedule_equivalence=row.get("schedule_equivalence"),
            timings={
                key: timing_summary(row, key, protocol["samples"], protocol["capture_batch"]) for key in row["timings"]
            },
        )
        rows.append(item)
    return dict(
        status=raw["status"],
        report_sha256=hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
        source=source_metadata(raw),
        device=raw["device"],
        protocol=protocol,
        evaluation=raw.get("evaluation"),
        rows=rows,
    )


def markdown(data):
    lines = [
        "# Matched H100 neighborhood tuning tables",
        "",
        "Resident complete-call CUDA graph medians in microseconds. Each cell uses",
        "retained paired samples and requires independent accuracy, changed-input",
        "graph validation and an actual kernel trace. These are operator measurements.",
        "Eager results, ranges, samples, source hashes and all component errors remain",
        "in the accompanying JSON. Screen samples select candidates; full and held-out",
        "measurements evaluate those selections separately.",
    ]
    for label, report in data["reports"].items():
        lines += [
            "",
            "## " + label,
            "",
            "| case | variant | forward | forward + backward | training range |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
        for row in report["rows"]:
            times = row["timings"]
            values = []
            for phase in ("forward", "forward_backward"):
                timing = times.get(phase + "/graph", {})
                values.append(timing.get("device_us") if timing.get("eligible_for_ratio") else None)
            forward, training = values
            display = lambda value: "unavailable" if value is None else f"{value['median']:.2f}"
            interval = "unavailable" if training is None else f"{training['minimum']:.2f}–{training['maximum']:.2f}"
            lines.append(
                f"| {row['workload']['name']} | {row['variant']} | {display(forward)} | {display(training)} | {interval} |"
            )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    reports = {}
    for argument in args.report:
        label, name = argument.split("=", 1)
        public_token(label)
        if label in reports:
            raise ValueError("Each report label must be unique.")
        reports[label] = summarize_report(Path(name))
    result = dict(
        schema="neighborhood-tuning-summary-v1",
        arithmetic="docs/precision.md; unchanged P/dS conversions and FP32 accumulation.",
        scope="Shared qualified PyTorch 2.13 stack; no claim against newer FlexAttention releases.",
        reports=reports,
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    args.markdown.write_text(markdown(result))


if __name__ == "__main__":
    main()
