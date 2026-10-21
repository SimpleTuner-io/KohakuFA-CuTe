"""Render per-role latency sensitivity relative to each zero-delay control."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    report = json.loads(args.input.read_text())
    labels = {"loader": "Operand loader", "mma": "MMA issuer", "pointwise": "Pointwise", "auxiliary": "Auxiliary"}
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for role, label in labels.items():
        values = report["medians"]["delay-" + role]
        delays = sorted(map(int, values))
        ax.plot(
            [delay / 1000 for delay in delays],
            [values[str(delay)] - values["0"] for delay in delays],
            marker="o",
            label=label,
        )
    ax.set_xlabel("Injected delay per KV inner iteration (µs)")
    ax.set_ylabel("Additional backward latency (ms)")
    ax.set_title("B300 MIG · causal BF16 · B1/H16/S4096/D512", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.supxlabel("Sensitivity to added latency; not utilization or a speedup forecast", fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    if args.output.suffix.lower() == ".svg":
        args.output.write_text("\n".join(line.rstrip() for line in args.output.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    main()
