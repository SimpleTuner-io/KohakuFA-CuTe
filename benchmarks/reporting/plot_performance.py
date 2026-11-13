"""Render the current per-target catalog without pooling measurement cohorts."""

import argparse
import html
import json
import math
from pathlib import Path
from textwrap import wrap

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
    "h200",
    "b200",
    "b300",
)


def row_key(row):
    return (*row["shape"][:3], row["dtype"], row["visibility"])


REFERENCE_LABELS = {
    "sdpa": "default SDPA",
    "fa2": "FA2",
    "fa3": "FA3",
    "gluon": "Gluon",
    "preceding_control": "preceding CuTe",
}


def comparison_reference(row):
    for name in (row.get("reference_backend"), "sdpa", "torch_sdpa", "fa3", "fa2", "gluon", "preceding_control"):
        if name and name in row["median_ms"] and name != row["backend"]:
            return name
    return None


def graph_cells(report, *, series="current", reference="sdpa"):
    """Use the first retained independent run, never average different studies."""
    cells = {}
    for row in report["measurements"]:
        if row["boundary"] != "graph" or row["phase"] not in ("forward", "joint"):
            continue
        if row.get("series", "current") != series:
            continue
        selected_reference = comparison_reference(row)
        canonical_reference = "sdpa" if selected_reference == "torch_sdpa" else selected_reference
        if selected_reference is None or (reference is not None and canonical_reference != reference):
            continue
        selected, baseline = row["median_ms"][row["backend"]], row["median_ms"][selected_reference]
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


def panel_groups(cells, workload):
    """Group rows by their measured dimension/phase columns; never pad missing cells."""
    columns_by_row = {}
    for key in cells:
        if key[-1] == workload:
            columns_by_row.setdefault(key[:5], set()).add((key[5], key[6]))
    groups = {}
    keys = sorted(columns_by_row, key=lambda k: (k[0], k[1], k[2], k[3] != "bf16", k[4] != "dense"))
    for key in keys:
        columns = tuple(sorted(columns_by_row[key], key=lambda c: (c[0], c[1] != "forward")))
        groups.setdefault(columns, []).append(key)
    return list(groups.items())


SVG_WIDTH, MARGIN, LABEL_WIDTH = 960, 24, 250
CELL_WIDTH, ROW_HEIGHT = 62, 28


def text_block(parts, text, left, top, width, *, size=14, bold=False, centered=False):
    """Wrap headings and notes inside the image's readable content width."""
    lines = wrap(text, width=max(1, int(width / (size * 0.62))))
    for line in lines:
        x = left + width / 2 if centered else left
        anchor = ' text-anchor="middle"' if centered else ""
        weight = ' font-weight="bold"' if bold else ""
        parts.append(f'<text x="{x}" y="{top + size}" font-size="{size}"{anchor}{weight}>{html.escape(line)}</text>')
        top += size + 6
    return top


def render_panel(report, cells, groups, workload, left, top, series="current", reference="sdpa"):
    label = next(iter(cells.values()))[1].get("series_label", report["label"])
    reference_label = REFERENCE_LABELS.get(reference, reference)
    parts = [
        f'<g class="gpu-panel" data-target="{report["target"]}" data-workload="{workload}" data-series="{series}" data-reference="{reference}">'
    ]
    parts.append(f'<a href="{html.escape(report["target"])}.json">')
    top = text_block(parts, f"{label} · vs {reference_label}", left, top, SVG_WIDTH - 2 * MARGIN, size=22, bold=True)
    parts.append("</a>")
    selected = [row for _, row in cells.values() if row["workload"] == workload]
    count = len({row["study"] for row in selected})
    top = text_block(
        parts,
        f"{len(selected)} measurements · {count} source cohort(s) · hardware/stack in JSON",
        left,
        top + 4,
        SVG_WIDTH - 2 * MARGIN,
    )
    grid_left, table_top = left + LABEL_WIDTH, top + 12
    for columns, keys in groups:
        grid_top = table_top + 52
        for dimension in dict.fromkeys(d for d, _ in columns):
            positions = [index for index, (d, _) in enumerate(columns) if d == dimension]
            center = grid_left + (sum(positions) / len(positions) + 0.5) * CELL_WIDTH
            parts.append(f'<text x="{center}" y="{grid_top - 30}" text-anchor="middle" font-size="15">D{dimension}</text>')
        for column, (_, phase) in enumerate(columns):
            label = "F+B" if phase == "joint" else "I" if workload == "inference" else "F"
            x = grid_left + (column + 0.5) * CELL_WIDTH
            parts.append(f'<text x="{x}" y="{grid_top - 8}" text-anchor="middle" class="axis">{label}</text>')
        for row_index, key in enumerate(keys):
            batch, heads, sequence, dtype, visibility = key
            y = grid_top + row_index * ROW_HEIGHT
            short_dtype = "BF" if dtype == "bf16" else "FP"
            label = f"B{batch} H{heads} S{sequence} {short_dtype} {visibility[0].upper()}"
            representative = cells[(*key, *columns[0], workload)][1]
            if "kv_shape" in representative:
                kv = representative["kv_shape"]
                label = f"B{batch} H{heads}/{kv[1]} Q{sequence} K{kv[2]} {short_dtype} {visibility[0].upper()}"
            parts.append(f'<text x="{grid_left - 8}" y="{y + 20}" text-anchor="end" class="axis">{label}</text>')
            for column, (dimension, phase) in enumerate(columns):
                cell_key = (*key, dimension, phase, workload)
                ratio, row = cells[cell_key]
                x = grid_left + column * CELL_WIDTH
                selected_reference = comparison_reference(row)
                tooltip = (
                    f'{report["label"]}; {row["study"]}; {row["run"]}; '
                    f'{phase}, {workload}; CuTe {row["median_ms"][row["backend"]]:.9g} ms; '
                    f'{reference_label} {row["median_ms"][selected_reference]:.9g} ms. '
                    "Hardware, source, protocol, samples and qualification are in the target JSON."
                )
                ink = "#fff" if ratio < 0.65 or ratio > 2.5 else "#243347"
                cell_id = html.escape(json.dumps(cell_key, separators=(",", ":")))
                parts.append(
                    f'<g class="measurement" data-key="{cell_id}"><title>{html.escape(tooltip)}</title>'
                    f'<rect x="{x}" y="{y}" width="{CELL_WIDTH - 1}" height="{ROW_HEIGHT - 1}" fill="{color(ratio)}"/>'
                    f'<text x="{x + CELL_WIDTH / 2}" y="{y + 20}" text-anchor="middle" class="cell" style="fill:{ink}">{ratio:.2f}</text></g>'
                )
        parts.append(f'<path d="M{grid_left} {grid_top - 2}H{grid_left + len(columns) * CELL_WIDTH}" stroke="#bcc6d0"/>')
        table_top = grid_top + ROW_HEIGHT * len(keys) + 20
    parts.append("</g>")
    return parts, table_top + 16


def render(reports):
    parts = []
    comparison_count = 0
    width = SVG_WIDTH - 2 * MARGIN
    top = text_block(parts, "Attention performance", MARGIN, MARGIN, width, size=28, bold=True, centered=True)
    for note in (
        "Named reference time / CuTe time · above 1×: CuTe faster (teal) · below 1×: reference faster (brown).",
        "F = training forward · F+B = forward + all gradients · I = NoGrad forward · BF/FP = BF16/FP16 · D/C = dense/causal.",
        "Warmed CUDA-event graph replay; compilation excluded. Hover values for timing and source provenance.",
        "Tables show measured combinations only. Current, historical and experimental results retain their own scope.",
    ):
        top = text_block(parts, note, MARGIN, top + 8, width, centered=True)
    section_specs = (
        ("current", "training", "Current arithmetic · training"),
        ("current", "inference", "Current arithmetic · forward only / NoGrad"),
        ("experimental", "training", "Experimental H200 · unmodified initcheck unresolved"),
        ("captured", "inference", "Captured model inputs · historical forward-only attention, not model throughput"),
        ("historical", "training", "Historical arithmetic / hardware · not current-dispatch qualification"),
        ("historical-mig", "training", "Historical partition · separate hardware and arithmetic"),
    )
    for series, workload, title in section_specs:
        panels = []
        for report in reports:
            references = {
                "sdpa" if comparison_reference(r) == "torch_sdpa" else comparison_reference(r)
                for r in report["measurements"]
                if r.get("series", "current") == series
            }
            order = {name: index for index, name in enumerate(REFERENCE_LABELS)}
            for reference in sorted((r for r in references if r is not None), key=lambda r: (order.get(r, len(order)), r)):
                cells = graph_cells(report, series=series, reference=reference)
                groups = panel_groups(cells, workload)
                if groups:
                    panels.append((report, cells, groups, reference))
        if not panels:
            continue
        top = text_block(parts, title, MARGIN, top + 28, width, size=24, bold=True) + 16
        for report, cells, groups, reference in panels:
            comparison_count += sum(len(columns) * len(keys) for columns, keys in groups)
            panel, top = render_panel(report, cells, groups, workload, MARGIN, top, series, reference)
            parts.extend(panel)
    top += 16
    for note in (
        "Each value compares backends within one source cohort. Devices, power limits, software and coverage differ; exact scope is in JSON.",
        f"The first independent run is shown where several exist. All {comparison_count:,} chart comparisons remain visible; all timing runs stay in JSON.",
        "These are attention-call measurements. H200 is experimental; historical rows do not requalify current arithmetic. See results/README.md for limits.",
        "Missing current timing evidence: A100/A40 D512; RTX 3090 D256/D512; several synthetic NoGrad grids. These gaps need fresh GPU measurements.",
    ):
        top = text_block(parts, note, MARGIN, top + 8, width)
    height = top + MARGIN
    header = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{height}" viewBox="0 0 {SVG_WIDTH} {height}">',
        "<title>Attention performance across twelve GPU targets</title>",
        "<desc>Named reference time divided by CuTe time. GPUs follow a stable order in a single readable column. Training, NoGrad, historical and experimental results are labeled separately. Tables contain measured dimension/phase combinations.</desc>",
        f'<rect width="{SVG_WIDTH}" height="{height}" fill="#fff"/>',
        "<style>text{font-family:DejaVu Sans,sans-serif;fill:#243347} .cell{font-size:16px} .axis{font-size:14px}</style>",
    ]
    return "\n".join([*header, *parts, "</svg>"]) + "\n"


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
