"""Render paired probability-tile speed changes and actual instrumented CTA intervals."""

import argparse
import hashlib
import json
from pathlib import Path


def save_svg(figure, path):
    figure.savefig(path, format="svg", bbox_inches="tight")
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def main():
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--fp16", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = {dtype: json.loads(getattr(args, dtype).read_text()) for dtype in ("bf16", "fp16")}
    shapes = sorted((row["shape"][2], row["shape"][3]) for row in reports["bf16"]["rows"])
    values, labels = [], []
    for dtype, report in reports.items():
        rows = {(row["shape"][2], row["shape"][3]): row for row in report["rows"]}
        if set(rows) != set(shapes):
            raise ValueError("The paired reports must cover identical shapes.")
        for columns in (64, 128):
            values.append(
                [
                    (
                        1
                        - rows[shape]["medians"][f"candidate-{columns}"]["forward_backward_ms"]
                        / rows[shape]["medians"][f"baseline-{columns}"]["forward_backward_ms"]
                    )
                    * 100
                    for shape in shapes
                ]
            )
            labels.append(f"{dtype.upper()} · {columns} columns")
    values = np.asarray(values)
    figure, ax = plt.subplots(figsize=(11, 3.9))
    bound = max(float(np.abs(values).max()), 1.0)
    artist = ax.imshow(values, cmap="RdBu", norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound), aspect="auto")
    ax.set_xticks(range(len(shapes)), [f"S{sequence}\nD{dimension}" for sequence, dimension in shapes])
    ax.set_yticks(range(len(labels)), labels)
    for row in range(len(labels)):
        for col in range(len(shapes)):
            ax.text(
                col,
                row,
                f"{values[row, col]:+.1f}%",
                ha="center",
                va="center",
                color="white" if abs(values[row, col]) > 0.55 * bound else "black",
                fontsize=9,
            )
    ax.set_title("B300 MIG: less forward + backward time after interior-tile checks", pad=14)
    figure.colorbar(artist, ax=ax, label="Time reduction (%)", shrink=0.8)
    figure.tight_layout()
    save_svg(figure, args.output_dir / "paired-speed.svg")
    plt.close(figure)

    groups = (
        ("Operand-ready waits", (0, 6, 7), "#457b9d"),
        ("P-ready wait", (1,), "#e76f51"),
        ("dS-ready wait", (5,), "#9b5de5"),
        ("Other measured waits", (2, 3, 4), "#e9c46a"),
        ("Other role time", (), "#adb5bd"),
    )
    barrier_reports = {name: json.loads(getattr(args, name).read_text()) for name in ("before", "after")}
    selections, bar_labels, fractions = [], [], []
    for dimension in (128, 512):
        for mode, index in (("dQ", 0), ("dK/dV", 1)):
            for variant, report in barrier_reports.items():
                row = next(row for row in report["rows"] if row["dimension"] == dimension)
                raw = np.asarray(row["samples"], dtype=np.int64)[1:, index]
                data = raw.reshape(-1, raw.shape[-1])
                selected = int(np.abs(data[:, -1] - np.median(data[:, -1])).argmin())
                interval = data[selected]
                if interval[-1] <= 0 or interval[:-1].min() < 0 or interval[:-1].sum() > interval[-1]:
                    raise ValueError("Invalid representative clock intervals.")
                parts = [float(interval[list(fields)].sum() / interval[-1] * 100) for _, fields, _ in groups[:-1]]
                parts.append(100 - sum(parts))
                fractions.append(parts)
                bar_labels.append(f"D{dimension} {mode} · {variant}")
                selections.append(
                    {
                        "dimension": dimension,
                        "mode": mode,
                        "variant": variant,
                        "sample": 1 + selected // (raw.shape[1] * raw.shape[2]),
                        "head": selected // raw.shape[2] % raw.shape[1],
                        "tile_and_slice": selected % raw.shape[2],
                        "clock_cycles": interval.tolist(),
                    }
                )
    fractions = np.asarray(fractions)
    figure, ax = plt.subplots(figsize=(11, 6.3))
    positions = np.arange(len(bar_labels))
    left = np.zeros(len(bar_labels))
    for index, (label, _, color) in enumerate(groups):
        ax.barh(positions, fractions[:, index], left=left, label=label, color=color, height=0.7)
        left += fractions[:, index]
    ax.set_yticks(positions, bar_labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of one actual instrumented matrix-warp interval")
    ax.set_title("B300 MIG: representative matrix-warp wait intervals", pad=58)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=9)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    figure.subplots_adjust(top=0.76, left=0.22, right=0.98, bottom=0.12)
    save_svg(figure, args.output_dir / "barrier-intervals.svg")
    figure.savefig(args.output_dir / "barrier-intervals.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    metadata = {
        "source_sha256": {
            name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()
            for name in ("bf16", "fp16", "before", "after")
        },
        "representative_ctas": selections,
        "selection": "Actual CTA interval nearest median total time; no joining of independent per-field medians; first sample excluded as warmup.",
        "scope": "Instrumented matrix-warp intervals include timer overhead; these are not PM-counter stall percentages or GPU utilization.",
    }
    (args.output_dir / "plot-data.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
