"""Plot public and raw GB10 S128 latency from the retained evaluation data."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/gb10-inference.json"))
    parser.add_argument("--output", type=Path, default=Path("results/gb10-inference.svg"))
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    plt.rcParams.update({"font.size": 9, "svg.fonttype": "none", "svg.hashsalt": "gb10-inference"})
    methods = [
        ("previous", "Previous public", "#858e98"),
        ("raw_inference", "Raw native inference", "#b2cdbb"),
        ("selected", "Selected public", "#258151"),
        ("fa2", "FA2", "#7962a8"),
        ("sdpa", "SDPA", "#3e87ad"),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))
    width = 0.15
    for axis, mode, title in zip(
        axes, ("graph", "eager"), ("CUDA graph: device event time", "Eager: complete warmed wall time")
    ):
        rows = [row for row in data["measurements"][mode] if row["shape"][2] == 128]
        positions = np.arange(len(rows))
        for index, (key, label, color) in enumerate(methods):
            values = [row["median_ms"][key] * 1000 for row in rows]
            bars = axis.bar(positions + (index - 2) * width, values, width, label=label, color=color)
            axis.bar_label(bars, fmt="%.1f", fontsize=6.5, padding=2)
        axis.set_xticks(
            positions,
            [row["dtype"].upper() + "\n" + ("causal" if row["causal"] else "dense") for row in rows],
        )
        axis.set_title(title, fontsize=10)
        axis.set_ylabel("Microseconds per attention call")
        axis.set_ylim(0, axis.get_ylim()[1] * 1.12)
        axis.grid(axis="y", alpha=0.18)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("GB10 attention, B1 / H16 / S128 / D128", y=0.99, fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=5, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.84))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, metadata={"Date": None})
    plt.close(figure)


if __name__ == "__main__":
    main()
