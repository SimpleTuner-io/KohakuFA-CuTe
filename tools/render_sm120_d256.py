"""Render held-out joint-call speedups without pooling host latency regimes."""

import argparse
import html
import json
import math
from pathlib import Path

COLORS = {"prior-public": "#728197", "fa2": "#2c9b78", "sdpa": "#4d78d0"}
LABELS = {"prior-public": "Prior CuTe", "fa2": "FA2 2.8.4", "sdpa": "torch SDPA"}
CASES = [
    (length, dtype, causal, f"{length} {short} {'C' if causal else 'D'}")
    for length in (1024, 4096)
    for dtype, short in (("bf16", "BF"), ("fp16", "FP"))
    for causal in (False, True)
]


def render(data):
    pieces = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1320" height="500" viewBox="0 0 1320 500">',
        "<title>RTX 5090 D256 forward plus backward</title>",
        "<desc>Speedup of selected public CuTe over three same-process references. Bars are geometric means of two independent held-out seed ratios; whiskers show their range. Eager includes host work; graph replay captures one complete call.</desc>",
        '<rect width="1320" height="500" fill="#fff"/>',
        '<g font-family="sans-serif" fill="#233044">',
        '<text x="660" y="27" text-anchor="middle" font-size="19">RTX 5090 · B1/H16/D256 · forward + all gradients</text>',
        '<text x="660" y="49" text-anchor="middle" font-size="12">Reference time / selected public CuTe time · higher is faster</text>',
    ]
    top, bottom, maximum = 98, 350, 2.5

    def y(value):
        return bottom - value / maximum * (bottom - top)

    for panel, mode in enumerate(("graph", "eager")):
        left = 64 + panel * 655
        title = "Captured device path" if mode == "graph" else "Complete warmed eager calls"
        pieces.append(f'<text x="{left + 280}" y="78" text-anchor="middle" font-size="15">{title}</text>')
        for tick in (0, 0.5, 1, 1.5, 2, 2.5):
            dash = ' stroke-dasharray="4 3"' if tick == 1 else ""
            pieces.append(f'<path d="M{left} {y(tick):.2f}H{left + 570}" stroke="#ccd4df"{dash}/>')
            pieces.append(f'<text x="{left - 9}" y="{y(tick) + 4:.2f}" text-anchor="end" font-size="11">{tick:g}×</text>')
        for case_index, (length, dtype, causal, label) in enumerate(CASES):
            center = left + 38 + case_index * 72
            for reference_index, reference in enumerate(COLORS):
                ratios = []
                for run in data["holdout_runs"]:
                    row = next(
                        row
                        for row in run["rows"]
                        if row["shape"][2] == length
                        and row["dtype"] == dtype
                        and row["causal"] == causal
                        and row["phase"] == "joint"
                        and row["mode"] == mode
                    )
                    ratios.append(
                        row["median_ms"][reference]["cuda_event_ms"] / row["median_ms"]["public-d256"]["cuda_event_ms"]
                    )
                value = math.exp(sum(math.log(ratio) for ratio in ratios) / len(ratios))
                low, high = min(ratios), max(ratios)
                x = center - 28 + reference_index * 19
                pieces.append(
                    f'<rect x="{x}" y="{y(value):.2f}" width="15" height="{bottom - y(value):.2f}" fill="{COLORS[reference]}"/>'
                )
                pieces.append(
                    f'<path d="M{x + 7.5} {y(low):.2f}V{y(high):.2f}M{x + 3.5} {y(low):.2f}h8M{x + 3.5} {y(high):.2f}h8" stroke="#26354a"/>'
                )
                pieces.append(
                    f'<text x="{x + 7.5}" y="{y(high) - 6:.2f}" text-anchor="middle" font-size="10">{value:.2f}</text>'
                )
            pieces.append(f'<text x="{center}" y="372" text-anchor="middle" font-size="9">{html.escape(label)}</text>')
    for index, reference in enumerate(COLORS):
        x = 460 + index * 145
        pieces.append(f'<rect x="{x}" y="410" width="13" height="13" fill="{COLORS[reference]}"/>')
        pieces.append(f'<text x="{x + 20}" y="421" font-size="12">{LABELS[reference]}</text>')
    pieces += [
        '<text x="660" y="458" text-anchor="middle" font-size="11">Two held-out seeds; whiskers retain their range. Dense and causal rows include actual public selection.</text>',
        '<text x="660" y="478" text-anchor="middle" font-size="11">BF = BF16; FP = FP16; D = dense; C = causal. Synthetic inputs; not complete-model measurements.</text>',
        "</g></svg>",
    ]
    return "\n".join(pieces) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/rtx5090-d256.json"))
    parser.add_argument("--output", type=Path, default=Path("results/rtx5090-d256.svg"))
    args = parser.parse_args()
    args.output.write_text(render(json.loads(args.input.read_text())))


if __name__ == "__main__":
    main()
