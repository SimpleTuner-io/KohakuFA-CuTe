"""Plot the measured SM120 pipeline change and observed backward launches."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def save(fig, path):
    fig.savefig(path, bbox_inches="tight", metadata={"Date": None})
    if path.suffix == ".svg":
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/sm120-pipeline-profiling"))
    args = parser.parse_args()
    report = json.loads((args.results / "paired-summary.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    colors = ("#9aa4b2", "#187c9c")
    for ax, sequence in zip(axes, (1024, 4096)):
        for i, dtype in enumerate(("bf16", "fp16")):
            row = next(x for x in report["selected_cases"] if x["seq"] == sequence and x["dtype"] == dtype)
            for offset, key, color in zip((-0.18, 0.18), ("baseline", "candidate"), colors):
                value = row[key]["forward_backward_ms"]
                ax.bar(i + offset, value, width=0.32, color=color)
                ax.text(i + offset, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
            ax.text(
                i,
                row["baseline"]["forward_backward_ms"] * 1.13,
                f"{row['fb_latency_reduction_percent']:.2f}% lower",
                ha="center",
                color=colors[1],
                fontsize=10,
            )
        ax.set_xticks((0, 1), ("BF16", "FP16"))
        ax.set_title(f"S{sequence}, D256, 16 heads")
        ax.set_ylabel("Forward + backward (ms)")
        ax.set_ylim(top=ax.get_ylim()[1] * 1.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    fig.legend(
        handles=[Patch(color=colors[0], label="main cc3475b"), Patch(color=colors[1], label="16-column TMA stages")],
        loc="outside lower center",
        ncols=2,
        frameon=False,
    )
    fig.suptitle("RTX 5090: dense D256 native graph timing", fontsize=13)
    save(fig, args.results / "paired-gains.svg")

    report = json.loads((args.results / "nsys-summary.json").read_text())
    fig, ax = plt.subplots(figsize=(10, 3.8), layout="constrained")
    colors = ("#9aa4b2", "#de8f05", "#187c9c")
    labels = []
    for i, capture in enumerate(report["captures"]):
        replay = capture["phases"]["backward"]["observed_replay"]
        events = replay["events"]
        assert len(events) == 3
        origin = events[0]["start"]
        for event, color in zip(events, colors):
            ax.barh(i, (event["end"] - event["start"]) / 1e6, left=(event["start"] - origin) / 1e6, color=color)
        elapsed = replay["elapsed_us"] / 1000
        ax.text(elapsed + 0.08, i, f"{elapsed:.3f} ms", va="center", fontsize=9)
        labels.append(f"{capture['mask']} / {'main' if capture['variant'] == 'before' else 'candidate'}")
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(right=ax.get_xlim()[1] + 0.6)
    ax.set_xlabel("Time from first backward kernel (ms)")
    ax.set_title("RTX 5090, BF16, S4096 D256: observed backward graph replays")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(
        handles=[Patch(color=color, label=label) for color, label in zip(colors, ("Output delta", "dQ", "dK/dV"))],
        loc="lower right",
        frameon=False,
        ncols=3,
    )
    save(fig, args.results / "backward-launches.svg")


if __name__ == "__main__":
    main()
