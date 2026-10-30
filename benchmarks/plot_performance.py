"""Plot the latest retained RTX and L40S unmasked comparison grids."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "kohakufa-performance"})
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    report = json.loads((args.results / "performance.json").read_text())
    datasets = {dataset["id"]: dataset for dataset in report["datasets"]}
    latest_4090 = json.loads((args.results / "rtx4090.json").read_text())
    datasets["rtx4090-current"] = {
        "date": latest_4090["date"],
        "torch": latest_4090["stack"]["torch"],
        "rows": [
            {
                "shape": row["shape"],
                "dtype": row["dtype"],
                "mask": row["mode"],
                "medians_ms": {
                    backend: {"forward_ms": times["forward"], "forward_backward_ms": times["joint"]}
                    for backend, times in row["medians_ms"].items()
                },
            }
            for row in latest_4090["measurements"]
        ],
    }
    devices = (
        ("rtx5090-current", "RTX 5090"),
        ("rtx4090-current", "RTX 4090"),
        ("rtx3090ti-current", "RTX 3090 Ti · 285 W"),
        ("l40s-current", "L40S"),
    )
    row_counts = [len({(r["dtype"], r["shape"][2], r["mask"]) for r in datasets[key]["rows"]}) for key, _ in devices]
    figure, axes = plt.subplots(
        len(devices),
        2,
        figsize=(11.5, 0.34 * sum(row_counts) + 1.4 * len(devices) + 1),
        height_ratios=[count + 3 for count in row_counts],
        layout="constrained",
    )
    for pair, (dataset_id, device) in zip(axes, devices):
        dataset = datasets[dataset_id]
        rows = dataset["rows"]
        dimensions = sorted({r["shape"][3] for r in rows})
        sequences = sorted({r["shape"][2] for r in rows})
        keys = [(dtype, seq, mask) for dtype in ("bf16", "fp16") for seq in sequences for mask in ("dense", "causal")]
        lookup = {(r["dtype"], r["shape"][2], r["mask"], r["shape"][3]): r for r in rows}
        for axis, phase, title in zip(pair, ("forward_ms", "forward_backward_ms"), ("Forward", "Forward + backward")):
            ratios = np.array(
                [
                    [
                        lookup[(*key, dim)]["medians_ms"]["sdpa"][phase] / lookup[(*key, dim)]["medians_ms"]["cute"][phase]
                        for dim in dimensions
                    ]
                    for key in keys
                ]
            )
            axis.pcolormesh(
                np.arange(len(dimensions) + 1) - 0.5,
                np.arange(len(keys) + 1) - 0.5,
                np.log2(ratios),
                cmap="BrBG",
                norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=2),
                shading="flat",
            )
            axis.invert_yaxis()
            for (i, j), value in np.ndenumerate(ratios):
                axis.text(j, i, f"{value:.2f}×", ha="center", va="center", color="white" if value > 2 else "#172b34")
            axis.set_xticks(range(len(dimensions)), [f"D{dim}" for dim in dimensions])
            axis.set_yticks(
                range(len(keys)), [f"{dtype.upper()} · S{seq} · {mask}" for dtype, seq, mask in keys], fontsize=9
            )
            axis.set_title(f"{device} · {title}\n{dataset['date']} · PyTorch {dataset['torch']}", fontsize=11, pad=10)
            axis.tick_params(length=0)
            for spine in axis.spines.values():
                spine.set_visible(False)
    figure.suptitle("Default SDPA time / CuTe time · B1 H16", fontsize=14)
    figure.supxlabel(
        "Above 1×: CuTe faster (teal). Below 1×: SDPA faster (brown).\n"
        "D64–256: FlashAttention-2; D512: efficient attention. RTX 4090 also includes D16 and S128.\n"
        "Each cell compares backends on the same GPU and stack; dates and framework versions differ between GPUs.",
        fontsize=10,
    )
    output = args.results / "performance.svg"
    figure.savefig(output, metadata={"Date": None, "Creator": "KohakuFA-CuTe benchmarks.plot_performance"})
    output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")
    plt.close(figure)


if __name__ == "__main__":
    main()
