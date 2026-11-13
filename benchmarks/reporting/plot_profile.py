"""Render ordered CUDA launches from profile_attention reports, with idle gaps intact."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def stage(name):
    for token, label in (
        ("OutputDelta", "Center"),
        ("Query_", "dQ pass"),
        ("KeyValue_", "dK/dV pass"),
        ("Forward_", "Forward"),
        ("attention_fwd_kernel", "Forward"),
        ("attention_bwd_kernel", "Fused dQ/dK/dV"),
        ("_prepare_kernel", "Prepare"),
        ("_dq_convert_kernel", "Convert dQ"),
        ("FillFunctor", "Initialize"),
    ):
        if token in name:
            return label
    return name


def representative(phase, iterations):
    """Select an actual iteration by median total kernel time; never join slot medians."""
    launches = phase["launches"]
    if not phase["uniform_launch_pattern"] or len(launches) % iterations:
        raise ValueError("Rendering requires a verified repeated launch pattern.")
    width = len(launches) // iterations
    if not width:
        raise ValueError("No launches in an iteration.")
    groups = [launches[i : i + width] for i in range(0, len(launches), width)]
    totals = [sum(item["duration_us"] for item in group) for group in groups]
    median = statistics.median(totals)
    index = min(range(iterations), key=lambda i: abs(totals[i] - median))
    origin = groups[index][0]["start_us"]
    events = [
        dict(stage=stage(e["name"]), start_us=e["start_us"] - origin, duration_us=e["duration_us"]) for e in groups[index]
    ]
    end = max(e["start_us"] + e["duration_us"] for e in events)
    covered, previous = 0, 0
    for e in sorted(events, key=lambda e: e["start_us"]):
        finish = e["start_us"] + e["duration_us"]
        covered += max(0, finish - max(previous, e["start_us"]))
        previous = max(previous, finish)
    return dict(iteration=index, events=events, span_us=end, busy_union_us=covered, idle_us=end - covered)


def render(paths, output, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams["svg.hashsalt"] = "kohakufa-ordered-profile"
    colors = dict(
        zip(
            ("Center", "dQ pass", "dK/dV pass", "Forward", "Fused dQ/dK/dV", "Prepare", "Convert dQ", "Initialize"),
            ("#989898", "#ed8b31", "#ce5252", "#387db5", "#27997f", "#9467bd", "#b18c64", "#b6bdc9"),
        )
    )
    fig, axes = plt.subplots(len(paths), 1, figsize=(13, max(4.5, 2.5 * len(paths))), squeeze=False)
    summary = dict(
        title=title,
        protocol="Actual representative eager/graph CUDA launch timestamps; not an intra-kernel trace.",
        reports=[],
    )
    for ax, path in zip(axes[:, 0], paths):
        report = json.loads(path.read_text())
        entry = dict(source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=[])
        rows = [(backend, phase) for phase in ("forward", "backward") for backend in report["backends"]]
        for y, (backend, phase) in enumerate(rows):
            result = representative(report["backends"][backend][phase], report["iterations"])
            entry["rows"].append(dict(backend=backend, phase=phase, **result))
            for event in result["events"]:
                ax.broken_barh(
                    [(event["start_us"] / 1000, event["duration_us"] / 1000)],
                    (y - 0.32, 0.64),
                    facecolors=colors.get(event["stage"], "#333333"),
                )
            ax.text(
                result["span_us"] / 1000 + 0.025,
                y,
                f'{result["busy_union_us"] / 1000:.3f} GPU ms\n{result["span_us"] / 1000:.3f} span ms',
                va="center",
                fontsize=9,
            )
        ax.set_yticks(range(len(rows)), [f"{backend} {phase}" for backend, phase in rows])
        ax.invert_yaxis()
        ax.set_xlim(0, max(row["span_us"] for row in entry["rows"]) / 1000 * 1.25)
        ax.set_title(
            f'{report["device"]}, S={report["shape"][2]}, D={report["shape"][3]}, {report["dtype"]}, {report.get("execution", "eager")}'
        )
        ax.set_xlabel("Milliseconds from first GPU launch in each row (gaps retained)")
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        summary["reports"].append(entry)
    fig.suptitle(title + "\nWhite gaps include submission/profiling overhead; spans are not benchmark times", fontsize=13)
    fig.legend(handles=[Patch(color=c, label=n) for n, c in colors.items()], loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.12 if len(paths) == 1 else 0.07, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, metadata={"Date": None})
    plt.close(fig)
    if output.suffix.lower() == ".svg":
        output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Ordered CUDA launches (profiling overhead included)")
    args = parser.parse_args()
    render(args.profiles, args.output, args.title)


if __name__ == "__main__":
    main()
