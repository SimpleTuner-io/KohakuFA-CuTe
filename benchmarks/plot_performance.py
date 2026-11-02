"""Render the current per-target catalog without pooling measurement cohorts."""

import argparse
import html
import json
import math
from pathlib import Path

TARGETS = (
    "a100",
    "a40",
    "rtx3090",
    "rtx3090ti",
    "rtx4090",
    "rtx5090",
    "l40s",
    "gb10",
    "h100",
    "b200",
    "b300",
)
DIMENSIONS = (16, 64, 128, 256, 512)
COLUMNS = (("forward", "training", "F"), ("forward", "inference", "I"), ("joint", "training", "F+B"))


def row_key(row):
    return (*row["shape"][:3], row["dtype"], row["visibility"])


def graph_cells(report):
    """Use the first retained independent run, never average different studies."""
    cells = {}
    for row in report["measurements"]:
        if row["boundary"] != "graph" or row["phase"] not in ("forward", "joint"):
            continue
        reference = "sdpa" if "sdpa" in row["median_ms"] else "torch_sdpa"
        if reference not in row["median_ms"]:
            continue
        selected, baseline = row["median_ms"][row["backend"]], row["median_ms"][reference]
        if not all(isinstance(v, (float, int)) and math.isfinite(v) and v > 0 for v in (selected, baseline)):
            raise ValueError("Chart cells require finite positive successful timings.")
        key = (*row_key(row), row["shape"][3], row["phase"], row["workload"])
        if key in cells:
            if cells[key][1]["study"] != row["study"]:
                raise ValueError("Overlapping studies must be resolved in the catalog.")
            continue
        cells[key] = baseline / selected, row
    return cells


def color(ratio):
    position = max(-1.0, min(2.0, math.log2(ratio)))
    white = (248, 249, 247)
    extreme = (132, 82, 34) if position < 0 else (16, 116, 105)
    weight = -position if position < 0 else position / 2
    return "#" + "".join(f"{round(a + (b - a) * weight):02x}" for a, b in zip(white, extreme))


def render(reports):
    cell_width, row_height, panel_width = 37, 23, 766
    panels = []
    for report in reports:
        cells = graph_cells(report)
        keys = sorted({key[:5] for key in cells}, key=lambda k: (k[0], k[1], k[2], k[3] != "bf16", k[4] != "dense"))
        panels.append((report, cells, keys))
    bands = [panels[i : i + 3] for i in range(0, len(panels), 3)]
    heights = [105 + row_height * max(len(panel[2]) for panel in band) for band in bands]
    width, height = panel_width * 3 + 30, sum(heights) + 245
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<title>Attention performance across eleven GPU targets</title>",
        "<desc>Default SDPA time divided by selected CuTe time. Each cell uses a same-process comparison from its retained study. F is training forward, I is NoGrad forward, F+B includes forward and all gradients. Gray cells have no retained comparison. Values are graph replay, not eager or model speedups.</desc>",
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        "<style>text{font-family:DejaVu Sans,sans-serif;fill:#243347} .cell{font-size:11px} .axis{font-size:11px} .small{font-size:12px}</style>",
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-size="25" font-weight="bold">Attention performance · default SDPA time / CuTe time</text>',
        f'<text x="{width / 2}" y="61" text-anchor="middle" class="small">Above 1×: CuTe faster (teal) · below 1×: SDPA faster (brown) · gray: no retained comparison</text>',
        f'<text x="{width / 2}" y="84" text-anchor="middle" class="small">F = training forward · I = NoGrad forward · F+B = forward + all gradients · BF = BF16 · FP = FP16 · D/C = dense/causal</text>',
        f'<text x="{width / 2}" y="107" text-anchor="middle" class="small">Warmed CUDA-event graph replay, milliseconds per call; compilation excluded. Hover cells for timings and study/run provenance.</text>',
    ]
    top = 145
    for band, band_height in zip(bands, heights):
        for column, (report, cells, keys) in enumerate(band):
            left = 18 + column * panel_width
            parts.append(
                f'<a href="{html.escape(report["target"])}.json"><text x="{left}" y="{top}" font-size="20" font-weight="bold">{html.escape(report["label"])}</text></a>'
            )
            count = len({value[1]["study"] for value in cells.values()})
            parts.append(
                f'<text x="{left}" y="{top + 21}" class="small">{count} source cohort(s) · exact hardware/stack per study in JSON</text>'
            )
            grid_left, grid_top = left + 171, top + 69
            for dimension_index, dimension in enumerate(DIMENSIONS):
                x = grid_left + dimension_index * cell_width * 3
                parts.append(
                    f'<text x="{x + cell_width * 1.5}" y="{grid_top - 29}" text-anchor="middle" font-size="13">D{dimension}</text>'
                )
                for phase_index, (_, _, label) in enumerate(COLUMNS):
                    parts.append(
                        f'<text x="{x + (phase_index + 0.5) * cell_width}" y="{grid_top - 10}" text-anchor="middle" class="axis">{label}</text>'
                    )
            for row_index, key in enumerate(keys):
                batch, heads, sequence, dtype, visibility = key
                y = grid_top + row_index * row_height
                short_dtype = "BF" if dtype == "bf16" else "FP"
                label = f"B{batch} H{heads} S{sequence} {short_dtype} {visibility[0].upper()}"
                parts.append(f'<text x="{grid_left - 8}" y="{y + 16}" text-anchor="end" class="axis">{label}</text>')
                for dimension_index, dimension in enumerate(DIMENSIONS):
                    for phase_index, (phase, workload, _) in enumerate(COLUMNS):
                        x = grid_left + (dimension_index * 3 + phase_index) * cell_width
                        value = cells.get((*key, dimension, phase, workload))
                        fill, label, tooltip = "#e9edf1", "–", "No retained comparison for this workload."
                        if value:
                            ratio, row = value
                            reference = "sdpa" if "sdpa" in row["median_ms"] else "torch_sdpa"
                            fill, label = color(ratio), f"{ratio:.2f}"
                            tooltip = (
                                f'{report["label"]}; {row["study"]}; {row["run"]}; '
                                f'{phase}, {workload}; CuTe {row["median_ms"][row["backend"]]:.9g} ms; '
                                f'SDPA {row["median_ms"][reference]:.9g} ms. '
                                "Hardware, source, protocol, samples and qualification are in the target JSON."
                            )
                        ink = "#fff" if value and (value[0] < 0.65 or value[0] > 2.5) else "#243347"
                        parts.append(
                            f'<g><title>{html.escape(tooltip)}</title><rect x="{x}" y="{y}" width="{cell_width - 1}" height="{row_height - 1}" fill="{fill}"/><text x="{x + cell_width / 2}" y="{y + 16}" text-anchor="middle" class="cell" style="fill:{ink}">{label}</text></g>'
                        )
            parts.append(f'<path d="M{grid_left} {grid_top - 4}H{grid_left + 15 * cell_width}" stroke="#bcc6d0"/>')
        top += band_height
    for offset, line in enumerate(
        (
            "Each cell compares backends within one source cohort. Devices, power limits, software, shape coverage and forward gradient modes differ.",
            "Latest measured study per workload; older unaffected cells retain their dated source. First independent run shown where multiple runs exist; all runs stay in JSON.",
            "These are attention-call measurements, not complete-model or eager speedups. H200 remains experimental and is excluded. See results/README.md for qualification limits.",
        )
    ):
        parts.append(
            f'<text x="{width / 2}" y="{height - 65 + offset * 22}" text-anchor="middle" class="small">{html.escape(line)}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, help="Defaults to RESULTS/performance.svg.")
    args = parser.parse_args()
    reports = [json.loads((args.results / f"{target}.json").read_text()) for target in TARGETS]
    output = args.output or args.results / "performance.svg"
    output.write_text(render(reports))


if __name__ == "__main__":
    main()
