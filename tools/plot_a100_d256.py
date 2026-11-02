"""Plot retained D256 joint complete-call and graph measurements."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/a100-d256-training.json"))
    parser.add_argument("--output", type=Path, default=Path("results/a100-d256-training.svg"))
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["svg.hashsalt"] = "a100-d256-training"
    report = json.loads(args.input.read_text())
    rows = [r for r in report["rows"] if r["phase"] == "joint"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 12), sharey=True)
    colors = {"candidate_public": "#377eb8", "fa2": "#4daf4a", "sdpa": "#ff9f40"}
    for ax, mode, title in zip(axes, ("eager", "graph"), ("Complete eager wall time", "CUDA-graph replay")):
        selected = [r for r in rows if r["mode"] == mode]
        key = "wall_ms" if mode == "eager" else "cuda_event_ms"
        labels = []
        for index, row in enumerate(selected):
            heads, length = row["shape"][1:3]
            labels.append(f"H{heads}  L{length}  {row['dtype']}  {'causal' if row['causal'] else 'dense'}")
            automatic = heads in report["policy"]["automatic_shapes"].get(str(length), [])
            if automatic:
                ax.axhspan(index - 0.45, index + 0.45, color="#377eb8", alpha=0.07, lw=0)
            baseline = row["median_ms"]["previous_public"][key]
            for offset, (backend, color) in enumerate(colors.items()):
                values = row["samples_ms"][backend]
                paired = [a / b for a, b in zip(row["samples_ms"]["previous_public"], values)]
                ratio = baseline / row["median_ms"][backend][key]
                y = index + (offset - 1) * 0.22
                ax.barh(y, ratio, height=0.19, color=color, label=backend if index == 0 else None)
                ax.plot((min(paired), max(paired)), (y, y), color="#333333", lw=0.6)
        ax.axvline(1, color="#555555", lw=1, ls="--")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Previous public latency / backend latency (higher is faster)")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=8)
    axes[0].invert_yaxis()
    handles, _ = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        ("N32 candidate", "FlashAttention-2", "PyTorch SDPA"),
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.58, 0.955),
    )
    fig.suptitle("A100 D256 joint attention — actual public API comparisons", y=0.98)
    fig.text(
        0.05,
        0.02,
        "Shaded rows: final automatic selection. Other rows: explicit evaluation only.\nBars: ratios of medians; whiskers: seven paired-window ratios. Same inputs and stack; B1, full 108-SM A100.",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.2, right=0.98, top=0.88, bottom=0.08, wspace=0.08)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, metadata={"Date": None, "Creator": "KohakuFA-CuTe measured D256 chart"})
    if args.output.suffix == ".svg":
        args.output.write_text("\n".join(line.rstrip() for line in args.output.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    main()
