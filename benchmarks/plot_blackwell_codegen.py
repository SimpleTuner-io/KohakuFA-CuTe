"""Render paired latency reductions and captured CUDA-graph kernel timelines."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparisons", type=Path, nargs="+", required=True)
    parser.add_argument("--timeline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"svg.fonttype": "none", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    args.output.mkdir(parents=True, exist_ok=True)
    reports = [json.loads(path.read_text()) for path in args.comparisons]
    dimensions = [64, 128, 256, 512]
    rows = []
    for report in reports:
        for sequence in sorted({row["shape"][2] for row in report["rows"]}):
            rows.append((report, sequence))
    values = np.full((len(rows), len(dimensions)), np.nan)
    times = values.copy()
    labels = []
    for y, (report, sequence) in enumerate(rows):
        labels.append(f"{report['dtype'].upper()}  S={sequence}" + (" causal" if report["causal"] else ""))
        for row in report["rows"]:
            if row["shape"][2] != sequence:
                continue
            x = dimensions.index(row["shape"][3])
            before = row["medians"]["baseline"]["forward_backward_ms"]
            after = row["medians"]["candidate"]["forward_backward_ms"]
            values[y, x], times[y, x] = 100 * (1 - after / before), after
    limit = max(5, np.ceil(np.nanmax(np.abs(values)) / 5) * 5)
    fig, ax = plt.subplots(figsize=(8.2, 1.2 + len(rows) * 0.75))
    palette = plt.get_cmap("RdYlGn").copy()
    palette.set_bad("#ededed")
    colors = ax.imshow(values, cmap=palette, vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(dimensions)), [f"D{dimension}" for dimension in dimensions])
    ax.set_yticks(range(len(rows)), labels)
    for y in range(len(rows)):
        for x in range(len(dimensions)):
            label = "—" if np.isnan(values[y, x]) else f"{values[y, x]:+.1f}%\n{times[y, x]:.3f} ms"
            ax.text(x, y, label, ha="center", va="center", fontsize=10)
    ax.set_title("B300 MIG · 18 SMs · B1/H16\nForward + backward latency reduction from main")
    fig.colorbar(colors, ax=ax, label="Lower latency (%)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(args.output / "paired-gains.svg", bbox_inches="tight")
    plt.close(fig)

    if args.timeline is None:
        return
    data = json.loads(args.timeline.read_text())
    dimensions = sorted({row["dimension"] for row in data["timelines"]})
    roles = {
        "forward": ("Forward", "#4d9078"),
        "prepare": ("Prepare", "#a1a8ad"),
        "query": ("dQ", "#d88832"),
        "keyvalue": ("dK/dV", "#477aa5"),
        "backward": ("Fused backward", "#8961a8"),
        "convert": ("dQ convert", "#737b80"),
    }
    labels = {"baseline": "CuTe main", "candidate": "CuTe candidate", "gluon": "Gluon"}
    fig, axes = plt.subplots(len(dimensions), 1, figsize=(10, 1.6 + len(dimensions) * 2.3), squeeze=False)
    for ax, dimension in zip(axes[:, 0], dimensions):
        for y, backend in enumerate(labels):
            row = next(row for row in data["timelines"] if row["dimension"] == dimension and row["backend"] == backend)
            for kernel in row["kernels"]:
                ax.barh(
                    y,
                    kernel["end_ms"] - kernel["start_ms"],
                    left=kernel["start_ms"],
                    color=roles[kernel["role"]][1],
                    height=0.5,
                )
            end = max(kernel["end_ms"] for kernel in row["kernels"])
            ax.text(end, y, f"  {end:.2f} ms", va="center", fontsize=9)
        ax.set_yticks(range(len(labels)), labels.values())
        ax.invert_yaxis()
        ax.set_xlabel("Elapsed GPU time (ms)")
        ax.set_title(f"D{dimension} · one observed CUDA-graph forward/backward replay")
        ax.set_xlim(right=ax.get_xlim()[1] * 1.1)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for _, color in roles.values()]
    fig.legend(handles, [label for label, _ in roles.values()], loc="lower center", ncol=6, frameon=False)
    fig.suptitle("Nsight Systems CUDA kernel sequence · B300 MIG, BF16 B1/H16/S4096", y=1.01)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(args.output / "kernel-timeline.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
