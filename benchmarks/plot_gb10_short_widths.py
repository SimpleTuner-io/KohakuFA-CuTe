"""Plot complete-call GB10 short-forward measurements from retained data."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/gb10-short-forward.json"))
    parser.add_argument("--output", type=Path, default=Path("results/gb10-short-forward.svg"))
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    plt.rcParams.update({"font.size": 8, "svg.fonttype": "none", "svg.hashsalt": "gb10-short-forward"})
    methods = [
        ("parent", "Preceding public", "#89929c"),
        ("selected", "Selected public", "#258151"),
        ("raw_inference", "Raw inference", "#a9ceb6"),
        ("generic", "Ordinary CuTe", "#d4a55c"),
        ("fa2", "FA2", "#7962a8"),
        ("sdpa", "SDPA", "#3e87ad"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    width = 0.13
    for row_index, mode in enumerate(("graph", "eager")):
        for column, dimension in enumerate((16, 64, 256)):
            axis = axes[row_index, column]
            rows = [row for row in data["measurements"][mode] if row["shape"][3] == dimension]
            positions = np.arange(len(rows))
            for index, (key, label, color) in enumerate(methods):
                values = [row["median_ms"][key] * 1000 for row in rows]
                bars = axis.bar(positions + (index - 2.5) * width, values, width, label=label, color=color)
                if key == "selected":
                    axis.bar_label(bars, fmt="%.2f", fontsize=6, padding=2)
            axis.set_xticks(
                positions,
                [row["dtype"].upper() + "\n" + ("causal" if row["causal"] else "dense") for row in rows],
            )
            axis.set_title(f"D{dimension}: " + ("graph device events" if mode == "graph" else "complete eager wall time"))
            axis.set_ylabel("Microseconds per call")
            axis.set_ylim(0, axis.get_ylim()[1] * 1.12)
            axis.grid(axis="y", alpha=0.18)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("GB10 forward-only attention, B1 / H16 / S128", fontsize=13)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=6, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, metadata={"Date": None})
    if args.output.suffix.lower() == ".svg":
        args.output.write_text("\n".join(line.rstrip() for line in args.output.read_text().splitlines()) + "\n")
    plt.close(figure)


if __name__ == "__main__":
    main()
