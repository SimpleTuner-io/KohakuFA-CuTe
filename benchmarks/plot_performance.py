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


CELL_WIDTH, ROW_HEIGHT, PANEL_WIDTH = 54, 25, 766


def panel_height(groups):
    return 58 + sum(52 + ROW_HEIGHT * len(keys) for _, keys in groups)


def render_panel(report, cells, groups, workload, left, top, series="current", reference="sdpa"):
    label = next(iter(cells.values()))[1].get("series_label", report["label"])
    reference_label = REFERENCE_LABELS.get(reference, reference)
    parts = [
        f'<g class="gpu-panel" data-target="{report["target"]}" data-workload="{workload}" data-series="{series}" data-reference="{reference}">'
    ]
    parts.append(
        f'<a href="{html.escape(report["target"])}.json"><text x="{left}" y="{top + 23}" font-size="20" font-weight="bold">{html.escape(label)} · vs {html.escape(reference_label)}</text></a>'
    )
    selected = [row for _, row in cells.values() if row["workload"] == workload]
    count = len({row["study"] for row in selected})
    parts.append(
        f'<text x="{left}" y="{top + 44}" class="small">{len(selected)} measurements · {count} source cohort(s) · hardware/stack in JSON</text>'
    )
    grid_left, table_top = left + 171, top + 58
    for columns, keys in groups:
        grid_top = table_top + 39
        for dimension in dict.fromkeys(d for d, _ in columns):
            positions = [index for index, (d, _) in enumerate(columns) if d == dimension]
            center = grid_left + (sum(positions) / len(positions) + 0.5) * CELL_WIDTH
            parts.append(f'<text x="{center}" y="{grid_top - 24}" text-anchor="middle" font-size="13">D{dimension}</text>')
        for column, (_, phase) in enumerate(columns):
            label = "F+B" if phase == "joint" else "I" if workload == "inference" else "F"
            x = grid_left + (column + 0.5) * CELL_WIDTH
            parts.append(f'<text x="{x}" y="{grid_top - 6}" text-anchor="middle" class="axis">{label}</text>')
        for row_index, key in enumerate(keys):
            batch, heads, sequence, dtype, visibility = key
            y = grid_top + row_index * ROW_HEIGHT
            short_dtype = "BF" if dtype == "bf16" else "FP"
            label = f"B{batch} H{heads} S{sequence} {short_dtype} {visibility[0].upper()}"
            representative = cells[(*key, *columns[0], workload)][1]
            if "kv_shape" in representative:
                kv = representative["kv_shape"]
                label = f"B{batch} H{heads}/{kv[1]} Q{sequence} K{kv[2]} {short_dtype} {visibility[0].upper()}"
            parts.append(f'<text x="{grid_left - 8}" y="{y + 17}" text-anchor="end" class="axis">{label}</text>')
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
                    f'<text x="{x + CELL_WIDTH / 2}" y="{y + 17}" text-anchor="middle" class="cell" style="fill:{ink}">{ratio:.2f}</text></g>'
                )
        parts.append(f'<path d="M{grid_left} {grid_top - 2}H{grid_left + len(columns) * CELL_WIDTH}" stroke="#bcc6d0"/>')
        table_top += 52 + ROW_HEIGHT * len(keys)
    parts.append("</g>")
    return parts


def render(reports):
    sections = []
    top = 150
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
            for reference in sorted(r for r in references if r is not None):
                cells = graph_cells(report, series=series, reference=reference)
                groups = panel_groups(cells, workload)
                if groups:
                    panels.append((report, cells, groups, reference))
        if not panels:
            continue
        positions = []
        # Pack each workload section independently so a long focused study does
        # not leave empty space under every shorter neighboring GPU table.
        offsets = [0, 0, 0]
        for index in sorted(range(len(panels)), key=lambda i: -panel_height(panels[i][2])):
            column = min(range(3), key=lambda c: offsets[c])
            positions.append((index, 18 + column * PANEL_WIDTH, top + 32 + offsets[column]))
            offsets[column] += panel_height(panels[index][2]) + 24
        sections.append((series, workload, title, top, panels, sorted(positions)))
        top += 32 + max(offsets) + 24
    width, height = PANEL_WIDTH * 3 + 30, top + 127
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<title>Attention performance across twelve GPU targets</title>",
        "<desc>The named reference backend time divided by selected CuTe time. Each cell uses a same-process comparison from its retained study. Current training, NoGrad forward, historical and experimental results have separate sections. Tables show only measured dimension/phase columns, grouping rows by coverage. Values are graph replay, not eager or model speedups.</desc>",
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        "<style>text{font-family:DejaVu Sans,sans-serif;fill:#243347} .cell{font-size:12px} .axis{font-size:11px} .small{font-size:12px}</style>",
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-size="25" font-weight="bold">Attention performance · named reference time / CuTe time</text>',
        f'<text x="{width / 2}" y="61" text-anchor="middle" class="small">Above 1×: CuTe faster (teal) · below 1×: reference faster (brown) · tables show retained measurements only</text>',
        f'<text x="{width / 2}" y="84" text-anchor="middle" class="small">F = training forward · F+B = forward + all gradients · I = NoGrad forward · BF/FP = BF16/FP16 · D/C = dense/causal</text>',
        f'<text x="{width / 2}" y="107" text-anchor="middle" class="small">Warmed CUDA-event graph replay, milliseconds per call; compilation excluded. Hover cells for timings and study/run provenance.</text>',
        f'<text x="{width / 2}" y="130" text-anchor="middle" class="small">Dimensions and phase columns follow each table’s measured coverage; omitted combinations have no retained comparison.</text>',
    ]
    for series, workload, title, section_top, panels, positions in sections:
        parts.append(f'<text x="18" y="{section_top + 19}" font-size="22" font-weight="bold">{title}</text>')
        for index, left, panel_top in positions:
            report, cells, groups, reference = panels[index]
            parts.extend(render_panel(report, cells, groups, workload, left, panel_top, series, reference))
    for offset, line in enumerate(
        (
            "Each cell compares backends within one source cohort. Devices, power limits, software and workload coverage differ; exact scope is retained in JSON.",
            "Latest measured study per workload; first independent run shown where multiple runs exist. Every retained chart measurement appears once; all runs remain in JSON.",
            "These are attention-call measurements, not complete-model or eager speedups. H200 is experimental; historical rows are not requalification of current arithmetic. See results/README.md for qualification limits.",
            "Missing current timing evidence: A100/A40 D512; RTX 3090 D256/D512; several synthetic NoGrad grids. These gaps need fresh GPU measurements.",
        )
    ):
        parts.append(
            f'<text x="{width / 2}" y="{height - 87 + offset * 22}" text-anchor="middle" class="small">{html.escape(line)}</text>'
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
