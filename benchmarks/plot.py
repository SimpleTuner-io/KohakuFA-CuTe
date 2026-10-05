"""Render the measured H100/L40S precision sweeps."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "kohakufa-cute"})
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    figure, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    names = {"auto": "Native SDPA", "cudnn": "cuDNN", "kohakufa-cute": "CuTe"}
    colors = {"auto": "#d97706", "cudnn": "#dc2626", "kohakufa-cute": "#059669"}
    for row, gpu in enumerate(("h100", "l40s")):
        report = json.loads((args.results / f"{gpu}-precision.json").read_text())
        for column, gradient in enumerate(("dq", "dk", "dv")):
            axis = axes[row, column]
            for backend, label in names.items():
                values = sorted(
                    (item["max_absolute_logit"], item["metrics"][gradient]["relative_error"])
                    for item in report["rows"]
                    if item["backend"] == backend
                    and item["status"] == "ok"
                    and item["metrics"][gradient]["relative_error"] is not None
                    and item["metrics"][gradient]["relative_error"] > 0
                )
                if values:
                    x, y = zip(*values)
                    axis.loglog(x, y, marker="o", linewidth=1.8, color=colors[backend], label=label)
            axis.set_title(f"{gpu.upper()} · {gradient}")
            axis.grid(True, which="major", alpha=0.2)
            if row == 1:
                axis.set_xlabel("Maximum absolute logit")
            if column == 0:
                axis.set_ylabel("Relative L2 error vs FP64")
            if row == 0 and column == 0:
                axis.legend(fontsize=8)
    figure.suptitle("BF16 attention backward · B=1, H=2, S=256, D=128", fontsize=13)
    figure.text(
        0.5,
        0.015,
        "At the largest logits, the FP64 dQ norm is 5.04e−33; the JSON retains norms and absolute errors.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.95))
    path = args.results / "precision.svg"
    figure.savefig(path, metadata={"Creator": "KohakuFA-CuTe benchmarks.plot", "Date": None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(figure)


if __name__ == "__main__":
    main()
