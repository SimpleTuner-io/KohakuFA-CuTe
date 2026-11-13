"""Render the observed Nsight kernel sequence from the readout profiling report."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask", choices=("dense", "causal"), default="causal")
    parser.add_argument("--candidate-label", default="Register auxiliary")
    parser.add_argument("--baseline-label", default="Prior PR17")
    parser.add_argument("--seq", type=int, default=4096)
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    data = json.loads(args.input.read_text())
    colors = {
        "forward": "#8795a5",
        "prepare": "#e5aa42",
        "query": "#258b79",
        "keyvalue": "#3672b5",
        "backward": "#8656a3",
        "convert": "#ce7974",
        "other": "#c0c5cd",
    }
    labels = {
        "forward": "Forward",
        "prepare": "Statistics",
        "query": "dQ",
        "keyvalue": "dK / dV",
        "backward": "Joint backward",
        "convert": "Convert dQ",
        "other": "Other kernels",
    }
    backends = {"baseline": args.baseline_label, "candidate": args.candidate_label, "gluon": "Upstream Gluon"}
    fig, ax = plt.subplots(figsize=(12, 3.6), layout="constrained")
    for y, timeline in enumerate(data["timelines"]):
        for kernel in timeline["kernels"]:
            start, end = kernel["start_ms"], kernel["end_ms"]
            role = kernel["role"]
            ax.barh(y, end - start, left=start, color=colors[role], height=0.56)
            if end - start > 0.7:
                ax.text(
                    (start + end) / 2,
                    y,
                    f"{labels[role]}\n{end - start:.2f} ms",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=10,
                )
    ax.set_yticks(range(len(data["timelines"])), [backends[t["backend"]] for t in data["timelines"]])
    ax.invert_yaxis()
    ax.set_xlabel("Elapsed GPU time within one observed replay (ms)")
    ax.set_title(
        f"B300 MIG · {args.mask} BF16 · B1/H16/S{args.seq}/D512\n"
        "Nsight Systems observer pass; use paired measurements for performance comparisons",
        loc="left",
        fontsize=12,
        pad=16,
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    used = {kernel["role"] for timeline in data["timelines"] for kernel in timeline["kernels"]}
    ax.legend(
        handles=[Patch(facecolor=colors[role], label=label) for role, label in labels.items() if role in used],
        ncols=len(used),
        loc="upper left",
        bbox_to_anchor=(0, -0.22),
        frameon=False,
        fontsize=9,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    if args.output.suffix.lower() == ".svg":
        args.output.write_text("\n".join(line.rstrip() for line in args.output.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    main()
