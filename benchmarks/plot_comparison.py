"""Render matched attention speed and FP64 precision comparisons."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "kohakufa-cute-comparison"})
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"native": "#d97706", "gluon": "#2563eb", "gluon_before": "#60a5fa", "cute": "#059669", "before": "#64748b"}


def save(figure, path):
    figure.tight_layout(rect=(0, 0.07, 1, 0.93))
    figure.savefig(path, metadata={"Creator": "KohakuFA-CuTe benchmarks.plot_comparison", "Date": None})
    plt.close(figure)
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def speed_figure(report, baseline, output):
    sequences = sorted({row["shape"][2] for row in report["rows"]})
    figure, axes = plt.subplots(2, len(sequences), figsize=(6 * len(sequences), 7), squeeze=False)
    has_initial = all("cute-initial" in row["medians"] for row in report["rows"])
    series = [
        ("sdpa", "Native SDPA", "native"),
        ("kohakufa", "Gluon tuned" if report.get("upstream_tuning") else "Gluon defaults", "gluon"),
        ("cute", "CuTe tuned" if has_initial or baseline else "CuTe", "cute"),
    ]
    if all("kohakufa-default" in row["medians"] for row in report["rows"]):
        series.insert(1, ("kohakufa-default", "Gluon defaults", "gluon_before"))
    if has_initial:
        series.insert(-1, ("cute-initial", "CuTe initial configuration", "before"))
    elif baseline:
        series.insert(2, ("before", "CuTe initial port", "before"))
    # Paired pipeline reports may intentionally contain only CuTe and Gluon.
    series = [entry for entry in series if entry[0] == "before" or all(entry[0] in r["medians"] for r in report["rows"])]
    old = {tuple(row["shape"]): row for row in baseline["rows"]} if baseline else {}
    for column, sequence in enumerate(sequences):
        rows = sorted((r for r in report["rows"] if r["shape"][2] == sequence), key=lambda r: r["shape"][3])
        x = np.arange(len(rows))
        width = 0.8 / len(series)
        for row_index, field in enumerate(("forward_ms", "forward_backward_ms")):
            axis = axes[row_index, column]
            for index, (backend, label, color) in enumerate(series):
                values = [
                    old[tuple(r["shape"])]["medians"]["cute"][field] if backend == "before" else r["medians"][backend][field]
                    for r in rows
                ]
                axis.bar(x + (index - (len(series) - 1) / 2) * width, values, width, label=label, color=COLORS[color])
            axis.set_xticks(x, [str(r["shape"][3]) for r in rows])
            axis.set_yscale("log")
            axis.set_title(f"S={sequence} · {'Forward' if row_index == 0 else 'Forward + backward'}")
            axis.set_ylabel("Milliseconds per replay (log scale)")
            axis.set_xlabel("Head dimension")
            axis.grid(axis="y", alpha=0.2)
            axis.set_axisbelow(True)
            if row_index == 0 and column == 0:
                axis.legend(fontsize=8)
    batch, heads = report["rows"][0]["shape"][:2]
    figure.suptitle(f"{report['device']} · {report['dtype'].upper()} attention · B={batch}, H={heads}", fontsize=13)
    tuning = "supplied device tuning tables" if report.get("upstream_tuning") else "shipped defaults"
    figure.text(
        0.5,
        0.018,
        f"Median of {report['repeats']} × {report['iterations']} CUDA-graph replays. Gluon uses {tuning}.\n"
        "Forward + backward includes forward; first-use compilation is excluded.",
        ha="center",
        fontsize=9,
    )
    save(figure, output)


def precision_figure(report, output):
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    series = [("auto", "Native SDPA", "native"), ("kohakufa", "Gluon", "gluon"), ("kohakufa-cute", "CuTe", "cute")]
    for axis, gradient in zip(axes, ("dq", "dk", "dv")):
        for backend, label, color in series:
            rows = sorted(
                (r for r in report["rows"] if r["backend"] == backend and r["status"] == "ok"),
                key=lambda r: r["max_absolute_logit"],
            )
            values = [(r["max_absolute_logit"], r["metrics"][gradient]["relative_error"]) for r in rows]
            values = [(x, y) for x, y in values if y is not None and y > 0]
            if values:
                x, y = zip(*values)
                axis.loglog(x, y, marker="o", color=COLORS[color], label=label)
        axis.set_title(gradient)
        axis.set_xlabel("Maximum absolute logit")
        axis.grid(which="major", alpha=0.2)
    axes[0].set_ylabel("Relative L2 error against centered FP64")
    axes[0].legend(fontsize=8)
    endpoints = [r for r in report["rows"] if r["status"] == "ok" and r["backend"] == "kohakufa-cute"]
    endpoint = max(endpoints, key=lambda r: r["max_absolute_logit"])
    shape = ",".join(str(n) for n in endpoint["shape"])
    figure.suptitle(f"{report['device']} · {endpoint['dtype']} · [{shape}]", fontsize=13)
    figure.text(
        0.5,
        0.018,
        f"Identical rounded inputs. At the endpoint, FP64 dQ norm = {endpoint['metrics']['dq']['reference_norm']:.3g}.\n"
        "See the source/patch provenance in the report. JSON retains absolute errors and gradient cosines.",
        ha="center",
        fontsize=9,
    )
    save(figure, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed", type=Path, required=True)
    parser.add_argument("--precision", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    speed_figure(
        json.loads(args.speed.read_text()),
        json.loads(args.baseline.read_text()) if args.baseline else None,
        Path(str(args.output_prefix) + "-speed.svg"),
    )
    precision_figure(json.loads(args.precision.read_text()), Path(str(args.output_prefix) + "-precision.svg"))


if __name__ == "__main__":
    main()
