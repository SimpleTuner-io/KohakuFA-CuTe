"""Plot complete graph medians from a retained short-attention comparison."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(args.input.read_text())
    rows = [r for r in report["timing_rows"] if r["mode"] == "graph" and r["phase"] == "joint"]
    if len(rows) != 4:
        raise ValueError("Expected four dtype/visibility complete graph cases.")
    dimension = rows[0]["shape"][-1]
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "short-attention-comparison"})
    figure, axes = plt.subplots(figsize=(10.4, 4.5), layout="constrained")
    variants = (
        ("parent", "Previous CuTe", "#9ca3af"),
        ("fa2", "FA2", "#f59e0b"),
        ("sdpa", "SDPA", "#10b981"),
        ("public", "New CuTe", "#3b82f6"),
    )
    width = 0.19
    for index, (key, label, color) in enumerate(variants):
        values = [r["median_ms"][key]["cuda_event_ms"] * 1000 for r in rows]
        low = [v - min(s["cuda_event_ms"] * 1000 for s in r["samples"][key]) for v, r in zip(values, rows)]
        high = [max(s["cuda_event_ms"] * 1000 for s in r["samples"][key]) - v for v, r in zip(values, rows)]
        locations = [i + (index - 1.5) * width for i in range(4)]
        bars = axes.bar(locations, values, width, label=label, color=color, yerr=(low, high), capsize=2)
        axes.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    axes.set_xticks(range(4), [r["dtype"].upper() + (" causal" if r["causal"] else " dense") for r in rows])
    axes.set_ylabel("Complete graph forward + backward (µs)")
    axes.set_title(f"RTX 3090 · B1 H16 S128 D{dimension} · lower is better")
    axes.set_ylim(0, max(r["median_ms"][key]["cuda_event_ms"] * 1000 for r in rows for key, _, _ in variants) * 1.22)
    axes.legend(ncol=4, loc="upper center", frameon=False)
    axes.spines[["top", "right"]].set_visible(False)
    axes.set_axisbelow(True)
    axes.grid(axis="y", alpha=0.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        args.output,
        metadata={
            "Title": f"RTX 3090 short D{dimension} complete graph comparison",
            "Description": "Explicit attention entry; identical rounded operands. Allocation and capture precede timing. Whiskers show min/max of seven windows.",
            "Creator": "KohakuFA-CuTe benchmark plot",
            "Date": None,
        },
    )
    if args.output.suffix.lower() == ".svg":
        args.output.write_text("\n".join(line.rstrip() for line in args.output.read_text().splitlines()) + "\n")
    plt.close(figure)


if __name__ == "__main__":
    main()
