"""Plot sampled warp-role intervals and export Chrome/Perfetto trace events."""

import argparse
import json
from pathlib import Path


def intervals(row, metadata):
    for (start, before), (end, after) in zip(row["events"], row["events"][1:]):
        if before >= 2 and before % 2 == 0 and after == before + 1:
            yield start, end, metadata["events"][str(before // 2)]


def category(site):
    operation, code = site["operation"], site["code"]
    if operation == "statistics payload":
        return "Statistics loads"
    if "mbarrier_wait" in operation:
        return "Barrier wait"
    if "reconstruct_probabilities" in operation:
        return "Probability compute"
    if operation in ("self.issue", "self.issue_pv"):
        return "MMA enqueue"
    if "box_rows" in operation:
        return "TMA enqueue"
    if "read_matrix" in operation or "write_matrix" in operation:
        return "TMEM access"
    if "barrier" in operation:
        return "Worker sync"
    return code.split("\n")[0]


def relative_clock(rows):
    """Align roles across a low-32-bit wrap; one launch must span <2**31 ns."""
    anchor = rows[0]["events"][0][0]

    def offset(stamp):
        return (stamp - anchor + 2**31) % 2**32 - 2**31

    origin = min(offset(row["events"][0][0]) for row in rows)
    return lambda stamp: (offset(stamp) - origin) / 1000


def chrome_trace(report):
    events = []
    metadata = report["trace_metadata"]
    for sample in report["samples"]:
        # Each sample is a separate launch pair; timestamps wrap at 2**32 ns.
        relative = relative_clock(sample["traces"])
        for row in sample["traces"]:
            pid = sample["index"] * 2 + (row["mode"] == "kv")
            tid = row["tile"] * len(metadata["roles"]) + row["role"]
            events.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": pid,
                    "tid": tid,
                    "args": {"name": f"{row['mode']} tile {row['tile']}: {metadata['roles'][row['role']]}"},
                }
            )
            for start, end, site in intervals(row, metadata):
                events.append(
                    {
                        "name": category(site),
                        "cat": site["operation"],
                        "ph": "X",
                        "pid": pid,
                        "tid": tid,
                        "ts": relative(start),
                        "dur": (end - start) / 1000,
                        "args": {"source_line": site["line"], "code": site["code"]},
                    }
                )
    return {"displayTimeUnit": "ns", "traceEvents": events, "scope": report["scope"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--mode", choices=["q", "kv"], default="kv")
    parser.add_argument("--tile", type=int, default=8)
    parser.add_argument("--sample", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chrome-trace", type=Path)
    args = parser.parse_args()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    reports = [json.loads(path.read_text()) for path in (args.control, args.candidate)]
    colors = {
        "Statistics loads": "#e18b23",
        "Barrier wait": "#cdc9df",
        "Probability compute": "#3b9866",
        "MMA enqueue": "#286eae",
        "TMA enqueue": "#41a5a8",
        "TMEM access": "#a25b9a",
        "Worker sync": "#da6858",
    }
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.8), sharex=True)
    for axis, name, report in zip(axes, ("Control", "Independent statistics warp"), reports):
        metadata = report["trace_metadata"]
        rows = [
            row for row in report["samples"][args.sample]["traces"] if row["mode"] == args.mode and row["tile"] == args.tile
        ]
        if not rows:
            parser.error("No trace for this mode/tile/sample.")
        clock = relative_clock(rows)
        anchor_row = next(row for row in rows if row["role"] == 0)
        anchor = clock(anchor_row["events"][0][0])
        relative = lambda stamp: clock(stamp) - anchor
        axis.axvline(0, color="#555555", linewidth=0.6, alpha=0.5)
        for row in rows:
            role = row["role"]
            starts = []
            for start, end, site in intervals(row, metadata):
                label = category(site)
                axis.broken_barh(
                    [(relative(start), max(end - start, 8) / 1000)],
                    (role - 0.32, 0.64),
                    facecolors=colors.get(label, "#888888"),
                    linewidth=0,
                )
            for stamp, event in row["events"]:
                if event == 0:
                    starts.append(stamp)
            for index, stamp in enumerate(starts):
                axis.text(relative(stamp), role - 0.39, str(metadata["sample_step"] + index), fontsize=7)
        axis.set_yticks(range(len(metadata["roles"])), metadata["roles"])
        axis.invert_yaxis()
        axis.set_title(name, loc="left", fontsize=11, weight="bold")
        axis.grid(axis="x", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Microseconds relative to pointwise half 0 entering the first sampled iteration")
    fig.suptitle(f"{args.mode.upper()} backward: sampled inner iterations, tile {args.tile}", fontsize=14, weight="bold")
    fig.legend(
        handles=[Patch(color=color, label=label) for label, color in colors.items()],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.075),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    fig.text(
        0.5,
        0.012,
        "Enqueue intervals show the issuing warp, not asynchronous hardware execution. Gaps include unmarked work.\n"
        "Instrumentation adds shared memory and instructions; use uninstrumented measurements for performance claims.",
        fontsize=8,
        ha="center",
    )
    fig.subplots_adjust(left=0.17, right=0.985, top=0.90, bottom=0.23, hspace=0.32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    if args.chrome_trace:
        args.chrome_trace.parent.mkdir(parents=True, exist_ok=True)
        sample = {**reports[1], "samples": [reports[1]["samples"][args.sample]]}
        args.chrome_trace.write_text(json.dumps(chrome_trace(sample)) + "\n")


if __name__ == "__main__":
    main()
