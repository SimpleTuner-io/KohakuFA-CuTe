"""Render held-out joint-call speedups without pooling host latency regimes."""

import argparse
import html
import json
import math
from pathlib import Path

COLORS = {"parent-public": "#728197", "fa2": "#2c9b78", "sdpa": "#4d78d0"}
LABELS = {"parent-public": "Prior CuTe", "fa2": "FA2 2.8.4", "sdpa": "torch SDPA"}
CASES = [(64, "bf16", "D64 BF16"), (64, "fp16", "D64 FP16"), (128, "bf16", "D128 BF16")]


def render(data):
    pieces = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="940" height="440" viewBox="0 0 940 440">',
        "<title>RTX 5090 causal S1024 forward plus backward</title>",
        "<desc>Speedup of selected public CuTe over three same-process references. Bars are geometric means of two independent held-out seed ratios; whiskers show their range. Eager includes host work; graph replay captures 32 complete calls.</desc>",
        '<rect width="940" height="440" fill="#fff"/>',
        '<g font-family="sans-serif" fill="#233044">',
        '<text x="470" y="27" text-anchor="middle" font-size="19">RTX 5090 · causal B1/H16/S1024 · forward + all gradients</text>',
        '<text x="470" y="49" text-anchor="middle" font-size="12">Reference time / selected public CuTe time · higher is faster</text>',
    ]
    top, bottom, maximum = 98, 329, 2.5

    def y(value):
        return bottom - value / maximum * (bottom - top)

    for panel, mode in enumerate(("graph", "eager")):
        left = 64 + panel * 465
        title = "Captured device path" if mode == "graph" else "Complete warmed eager calls"
        pieces.append(f'<text x="{left + 190}" y="78" text-anchor="middle" font-size="15">{title}</text>')
        for tick in (0, 0.5, 1, 1.5, 2, 2.5):
            dash = ' stroke-dasharray="4 3"' if tick == 1 else ""
            pieces.append(f'<path d="M{left} {y(tick):.2f}H{left + 385}" stroke="#ccd4df"{dash}/>')
            pieces.append(f'<text x="{left - 9}" y="{y(tick) + 4:.2f}" text-anchor="end" font-size="11">{tick:g}×</text>')
        for case_index, (dimension, dtype, label) in enumerate(CASES):
            center = left + 64 + case_index * 128
            for reference_index, reference in enumerate(COLORS):
                ratios = []
                for run in data["holdout_runs"]:
                    if run["mode"] != mode:
                        continue
                    row = next(
                        row
                        for row in run["rows"]
                        if row["shape"][3] == dimension and row["dtype"] == dtype and row["phase"] == "joint"
                    )
                    ratios.append(row["medians_ms"][reference] / row["medians_ms"]["public-cute"])
                value = math.exp(sum(math.log(ratio) for ratio in ratios) / len(ratios))
                low, high = min(ratios), max(ratios)
                x = center - 35 + reference_index * 25
                pieces.append(
                    f'<rect x="{x}" y="{y(value):.2f}" width="20" height="{bottom - y(value):.2f}" fill="{COLORS[reference]}"/>'
                )
                pieces.append(
                    f'<path d="M{x + 10} {y(low):.2f}V{y(high):.2f}M{x + 6} {y(low):.2f}h8M{x + 6} {y(high):.2f}h8" stroke="#26354a"/>'
                )
                pieces.append(
                    f'<text x="{x + 10}" y="{y(high) - 6:.2f}" text-anchor="middle" font-size="10">{value:.2f}</text>'
                )
            pieces.append(f'<text x="{center}" y="351" text-anchor="middle" font-size="12">{html.escape(label)}</text>')
    for index, reference in enumerate(COLORS):
        x = 270 + index * 145
        pieces.append(f'<rect x="{x}" y="374" width="13" height="13" fill="{COLORS[reference]}"/>')
        pieces.append(f'<text x="{x + 20}" y="385" font-size="12">{LABELS[reference]}</text>')
    pieces += [
        '<text x="470" y="411" text-anchor="middle" font-size="11">Two held-out seeds; whiskers retain their range. D64 eager is mixed versus prior CuTe.</text>',
        '<text x="470" y="428" text-anchor="middle" font-size="11">FP16 D128 retains its preceding dispatch. Synthetic inputs; not complete-model measurements.</text>',
        "</g></svg>",
    ]
    return "\n".join(pieces) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/rtx5090-medium-causal.json"))
    parser.add_argument("--output", type=Path, default=Path("results/rtx5090-medium-causal.svg"))
    args = parser.parse_args()
    args.output.write_text(render(json.loads(args.input.read_text())))


if __name__ == "__main__":
    main()
