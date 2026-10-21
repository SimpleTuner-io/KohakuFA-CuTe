"""Reproduce paired-speed.svg from the retained 32-case timing samples."""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/sm120-sdpa-comparison"))
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    directory = args.results
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "svg.hashsalt": "kohakufa-sm120-sdpa"})
    data = json.loads((directory / "paired-timings.json").read_text())
    assert len(data["rows"]) == 32
    figure, axes = plt.subplots(2, 2, figsize=(13, 7), sharey=True)
    maximum = 1.0
    for row in data["rows"]:
        for samples in row["samples"].values():
            assert len(samples) == 7 and all(sample["graph_verified"] for sample in samples)
    for axis, (dtype, causal) in zip(axes.flat, (("bf16", False), ("bf16", True), ("fp16", False), ("fp16", True))):
        rows = sorted(
            (row for row in data["rows"] if row["dtype"] == dtype and row["causal"] == causal),
            key=lambda row: (row["dim"], row["seq"]),
        )
        baseline, candidate, labels = [], [], []
        for row in rows:
            values = {
                backend: statistics.median(sample["forward_backward_ms"] for sample in samples)
                for backend, samples in row["samples"].items()
            }
            baseline.append(values["pr20"] / values["sdpa"])
            candidate.append(values["final_v5"] / values["sdpa"])
            labels.append(f"D{row['dim']}\n{row['seq'] // 1024}K")
        maximum = max(maximum, *baseline, *candidate)
        axis.bar([index - 0.19 for index in range(8)], baseline, 0.36, color="#c4c9cf", label="Control d5d152d")
        axis.bar([index + 0.19 for index in range(8)], candidate, 0.36, color="#286fac", label="CuTe selected")
        axis.axhline(1, color="#333333", linestyle="--", linewidth=1)
        axis.set_xticks(range(8), labels, fontsize=9)
        axis.set_title(f"{dtype.upper()} · {'Implicit causal' if causal else 'Dense'}", loc="left", fontsize=11)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes.flat:
        axis.set_ylim(0, maximum * 1.12)
    for axis in axes[:, 0]:
        axis.set_ylabel("F+B time / PyTorch SDPA time", fontsize=10)
    figure.suptitle("RTX 5090: closing the backward performance gap", x=0.06, ha="left", fontsize=15, y=0.985)
    figure.text(0.06, 0.935, "B1 H16 · 7 rotated graph repeats · lower is faster · dashed line = SDPA", fontsize=10)
    figure.text(0.06, 0.015, "Default SDPA: FlashAttention-2 at D64/128/256; efficient attention at D512.", fontsize=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.97, 0.975), frameon=False, ncol=2)
    figure.tight_layout(rect=(0.025, 0.045, 0.99, 0.91), h_pad=2.4)
    figure.savefig(directory / "paired-speed.svg", bbox_inches="tight", metadata={"Date": None})
    if args.preview:
        figure.savefig(args.preview, dpi=140, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
