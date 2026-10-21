"""Plot unprofiled D512 RTX 5090 timings against main and default SDPA."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def allocation_cliff(results):
    report = json.loads((results / "allocation-cliff.json").read_text())
    variants = [report["variants"][key] for key in ("v1", "v2")]
    control = variants[1]["profiled_backends"]["control"]
    records = [control, *(v["profiled_backends"]["candidate"] for v in variants)]
    colors = ("#9aa4b2", "#d48b25", "#187c9c")
    labels = ("Main", "Barriers first\n50,432 B dynamic", "Barriers last\n49,536 B dynamic")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), layout="constrained")
    for ax, phase in zip(axes, ("query", "complete")):
        for i, (record, color) in enumerate(zip(records, colors)):
            if phase == "query":
                kernel = record["kernels"][2]
                value, samples = kernel["median_ms"], kernel["per_range_median_ms"]
            else:
                value, samples = record["event_median_ms"], record["event_samples_ms"]
            ax.bar(i, value, color=color, width=0.65)
            ax.errorbar(i, value, yerr=[[value - min(samples)], [max(samples) - value]], color="#253343", capsize=3)
            ax.text(i, value, f"{value:.3f}", ha="center", va="bottom")
        ax.set_xticks(range(3), labels)
        ax.set_ylabel("Query kernel (ms)" if phase == "query" else "Forward + backward (ms)")
        ax.set_ylim(top=ax.get_ylim()[1] * 1.14)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    fig.suptitle("D128 BF16 S4096: a controlled shared-allocation change", fontsize=13)
    fig.supxlabel("Both prototypes: 106 registers, zero stack; driver CTA capacity per SM: 1 → 2", fontsize=10)
    output = results / "allocation-cliff.svg"
    fig.savefig(output, bbox_inches="tight", metadata={"Date": None})
    output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/sm120-wide-tiles"))
    args = parser.parse_args()
    report = json.loads((args.results / "paired-summary.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    backends = ("control", "candidate", "sdpa")
    colors = ("#9aa4b2", "#187c9c", "#d48b25")
    for ax, sequence in zip(axes, (1024, 4096)):
        cases = [case for case in report["cases"] if case["seq"] == sequence]
        for i, case in enumerate(cases):
            for offset, backend, color in zip((-0.25, 0, 0.25), backends, colors):
                value = case["forward_backward_ms"][backend]
                low, high = case["ranges_ms"]["forward_backward"][backend]
                ax.bar(i + offset, value, width=0.23, color=color)
                ax.errorbar(i + offset, value, yerr=[[value - low], [high - value]], color="#253343", capsize=2)
            ax.text(
                i,
                max(case["forward_backward_ms"].values()) * 1.035,
                f"{case['latency_reduction_vs_sdpa_percent']:.0f}% below SDPA",
                ha="center",
                color=colors[1],
                fontsize=9,
            )
        ax.set_xticks(
            range(len(cases)), [f"{case['dtype'].upper()}\n{'Causal' if case['causal'] else 'Dense'}" for case in cases]
        )
        ax.set_title(f"S{sequence}, D512, 16 heads")
        ax.set_ylabel("Forward + backward (ms)")
        ax.set_ylim(top=ax.get_ylim()[1] * 1.14)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    fig.legend(
        handles=[
            Patch(color=color, label=label)
            for color, label in zip(colors, ("Main eb1c2f2", "Full-width CuTe", "Default SDPA (efficient)"))
        ],
        loc="outside lower center",
        ncols=3,
        frameon=False,
    )
    fig.suptitle("RTX 5090: native CUDA graphs; median and full sample range", fontsize=13)
    output = args.results / "paired-speed.svg"
    fig.savefig(output, bbox_inches="tight", metadata={"Date": None})
    output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")
    plt.close(fig)
    allocation_cliff(args.results)


if __name__ == "__main__":
    main()
