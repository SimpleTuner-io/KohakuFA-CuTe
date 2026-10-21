"""Plot the complete retained RTX 5090 unmasked comparison grid."""

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
    rows = [
        row
        for dataset in report["datasets"]
        if dataset["id"] in ("rtx5090-small-heads", "rtx5090-d512")
        for row in dataset["rows"]
    ]
    keys = [(dtype, seq, mask) for dtype in ("bf16", "fp16") for seq in (1024, 4096) for mask in ("dense", "causal")]
    dimensions = (64, 128, 256, 512)
    lookup = {(r["dtype"], r["shape"][2], r["mask"], r["shape"][3]): r for r in rows}
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), layout="constrained")
    for axis, phase, title in zip(axes, ("forward_ms", "forward_backward_ms"), ("Forward", "Forward + backward")):
        ratios = np.array(
            [
                [
                    lookup[(*key, dim)]["medians_ms"]["sdpa"][phase] / lookup[(*key, dim)]["medians_ms"]["cute"][phase]
                    for dim in dimensions
                ]
                for key in keys
            ]
        )
        axis.imshow(np.log2(ratios), cmap="BrBG", norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=2), aspect="auto")
        for (i, j), value in np.ndenumerate(ratios):
            axis.text(j, i, f"{value:.2f}×", ha="center", va="center", color="white" if value > 2 else "#172b34")
        axis.set_xticks(range(4), [f"D{dim}" for dim in dimensions])
        axis.set_yticks(range(8), [f"{dtype.upper()} · S{seq} · {mask}" for dtype, seq, mask in keys], fontsize=9)
        axis.set_title(title, fontsize=12, pad=10)
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
    figure.suptitle("RTX 5090 · default SDPA time / CuTe time · B1 H16", fontsize=14)
    figure.supxlabel(
        "Above 1×: CuTe faster (teal). Below 1×: SDPA faster (brown).\n"
        "D64–256: FlashAttention-2; D512: efficient attention. Matched ratios within each row; two host runs.",
        fontsize=10,
    )
    output = args.results / "performance.svg"
    figure.savefig(output, metadata={"Date": None, "Creator": "KohakuFA-CuTe benchmarks.plot_performance"})
    output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")
    plt.close(figure)


if __name__ == "__main__":
    main()
