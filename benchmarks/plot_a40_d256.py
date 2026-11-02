"""Plot the separately qualified A40 D256 public training matrix."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short", action="store_true", help="Plot the S128 training and no-grad forward panels.")
    parser.add_argument("--input", type=Path, default=Path("results/a40-d256.json"))
    parser.add_argument("--output", type=Path, default=Path("results/a40-d256.svg"))
    args = parser.parse_args()
    import matplotlib.pyplot as plt
    import numpy as np

    rows = json.loads(args.input.read_text())["rows"]
    if args.short:
        rows = [dict(row, phase="no_grad" if row.get("workload") == "no_grad" else row["phase"]) for row in rows]
    plt.rcParams.update({"font.size": 9, "svg.hashsalt": "a40-d256-public"})
    fig, axes = plt.subplots(2, 2 if args.short else 3, figsize=(12, 6.2), constrained_layout=True)
    implementations = [
        ("public", "A40 public", "#009a88"),
        ("cute", "Ordinary CuTe", "#777777"),
        ("fa2", "FA2", "#d47b13"),
        ("sdpa", "torch SDPA", "#397bb3"),
    ]
    groups = [(dtype, causal) for dtype in ("bf16", "fp16") for causal in (False, True)]
    panels = (
        [
            (128, "forward", "Training forward"),
            (128, "backward", "Backward"),
            (128, "joint", "Forward + backward"),
            (128, "no_grad", "No-grad forward"),
        ]
        if args.short
        else [
            (sequence, phase, title)
            for sequence in (1024, 4096)
            for phase, title in (("forward", "Forward"), ("backward", "Backward"), ("joint", "Forward + backward"))
        ]
    )
    for index_panel, (sequence, phase, title) in enumerate(panels):
        i, j = divmod(index_panel, 2 if args.short else 3)
        ax = axes[i, j]
        for index, (key, label, color) in enumerate(implementations):
            values = [
                next(
                    row
                    for row in rows
                    if row["shape"][2] == sequence
                    and row["dtype"] == dtype
                    and row["causal"] == causal
                    and row["phase"] == phase
                    and row["mode"] == "graph"
                )["median_ms"][key]["cuda_event_ms"]
                for dtype, causal in groups
            ]
            ax.bar(np.arange(4) + (index - 1.5) * 0.19, values, 0.19, label=label, color=color)
        ax.set_xticks(range(4), ["BF16\ndense", "BF16\ncausal", "FP16\ndense", "FP16\ncausal"])
        ax.set_title(f"S{sequence}: {title}")
        ax.set_ylabel("CUDA graph latency (ms)")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("A40 D256 · B1 H16 · actual public API · lower is faster", fontsize=13)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.savefig(args.output, metadata={"Date": None})


if __name__ == "__main__":
    main()
