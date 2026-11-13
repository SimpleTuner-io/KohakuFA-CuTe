"""Plot two predeclared BF16 workloads from the sanitized performance summary.

Only qualified graph timings enter the figure. Missing measurements are omitted,
never drawn as zero. This local postprocessor does not alter campaign sources.
"""

import argparse
import json
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BACKENDS = ("kohakufa", "flex", "natten")
LABELS = ("KohakuFA-CuTe", "FlexAttention AUTO", "NATTEN automatic/default presets")
COLORS = ("#4169D1", "#E79A33", "#32A77B")
GPU_ORDER = ("H100", "B200", "B300", "4090", "5090", "L40S")
CASES = (("64sq-w7-bf16", "7 × 7 window"), ("64sq-w13-bf16", "13 × 13 window"))
PHASES = (("forward", "Forward"), ("forward_backward", "Forward + backward"))


def gpu_identity(device):
    name = device["name"].upper()
    for family in GPU_ORDER:
        if family in name:
            if family == "H100":
                if "PCIE" in name:
                    return family, "H100 PCIe"
                if "HBM3" in name and device["sm_count"] == 132:
                    return family, "H100 SXM"
                return family, f"H100 ({device['sm_count']} SMs)"
            return family, "RTX " + family if family in ("4090", "5090") else family
    raise ValueError("Unexpected GPU family in the declared campaign.")


def selected_measurements(summary):
    measurements, labels = {}, {}
    for run in summary["runs"]:
        family, label = gpu_identity(run["device"])
        power_limit = run["device"].get("power_limit_watts")
        if type(power_limit) in (int, float) and math.isfinite(power_limit) and power_limit > 0:
            label += f"\n{power_limit:g} W"
        else:
            label += "\npower limit unavailable"
        if family in labels and labels[family] != label:
            raise ValueError("Different GPU variants cannot share one plot group.")
        labels[family] = label
        configuration = summary["configurations"][run["configuration_sha256"]]
        protocol = configuration["timing"]
        if protocol["samples"] != 7 or protocol["capture_batch"] != 8:
            raise ValueError("This figure requires seven rounds and eight complete calls per graph replay.")
        for case in run["cases"]:
            workload = case["workload"]
            if workload["name"] not in {name for name, _ in CASES}:
                continue
            if (
                workload["batch"] != 1
                or workload["heads"] != 8
                or workload["dtype"] != "bf16"
                or workload["spatial"] != [64, 64]
                or workload["scale"] != 0.125
            ):
                raise ValueError("Predeclared representative workload contract differs.")
            for phase, _ in PHASES:
                key = phase + "/graph"
                for backend in BACKENDS:
                    timing = case["backends"][backend]["timings"].get(key, {})
                    if timing.get("status") != "pass" or timing.get("eligible_for_ratio") is not True:
                        continue
                    values = timing["device_us"]
                    if (
                        timing["sample_count"] != 7
                        or not all(
                            math.isfinite(values[name]) and values[name] > 0 for name in ("median", "minimum", "maximum")
                        )
                        or not values["minimum"] <= values["median"] <= values["maximum"]
                    ):
                        raise ValueError("Invalid retained distribution.")
                    index = family, workload["name"], phase, backend
                    if index in measurements:
                        raise ValueError("Duplicate qualified GPU/workload measurements; select one immutable run.")
                    measurements[index] = values
    if not measurements:
        raise ValueError("No qualified representative graph measurements are available.")
    return measurements, labels


def plot(summary, output, preview=None):
    measurements, labels = selected_measurements(summary)
    families = [family for family in GPU_ORDER if any(index[0] == family for index in measurements)]
    omitted_families = [family for family in GPU_ORDER if family not in families]
    omitted_measurements = []
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": "kohakufa-neighborhood-performance",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(14, 8.6), squeeze=False)
    width = 0.24
    for row, (case_name, case_label) in enumerate(CASES):
        for column, (phase, phase_label) in enumerate(PHASES):
            axis = axes[row][column]
            maximum = 0
            for gpu_index, family in enumerate(families):
                for backend_index, backend in enumerate(BACKENDS):
                    values = measurements.get((family, case_name, phase, backend))
                    if values is None:
                        omitted_measurements.append(f"{labels[family]} {case_label} {phase_label} {LABELS[backend_index]}")
                        continue
                    center = gpu_index + (backend_index - 1) * width
                    median = values["median"]
                    errors = [[median - values["minimum"]], [values["maximum"] - median]]
                    axis.bar(
                        center,
                        median,
                        width=width * 0.91,
                        color=COLORS[backend_index],
                        yerr=errors,
                        capsize=2.5,
                        error_kw={"elinewidth": 1, "capthick": 1, "ecolor": "#27313d"},
                        zorder=3,
                    )
                    if len(families) <= 3 or backend == "kohakufa":
                        axis.annotate(
                            f"{median:.1f}",
                            (center, values["maximum"]),
                            xytext=(0, 4),
                            textcoords="offset points",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            color="#27313d",
                        )
                    maximum = max(maximum, values["maximum"])
            axis.set_title(case_label + " · " + phase_label, loc="left", fontweight="bold")
            axis.set_xticks(range(len(families)), [labels[family] for family in families])
            axis.set_ylabel("Median device interval (µs)")
            axis.set_ylim(0, maximum * 1.23 if maximum else 1)
            axis.grid(axis="y", color="#dfe5eb", linewidth=0.7, zorder=0)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
            axis.spines[["bottom", "left"]].set_color("#c4ccd5")
            axis.tick_params(axis="both", length=0, pad=6)
    figure.suptitle(
        "2D neighborhood attention · BF16 · 64 × 64 tokens · B1 H8 D64",
        x=0.075,
        y=0.988,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color="#172331",
    )
    figure.text(
        0.075,
        0.947,
        "Complete contiguous BHSD calls, including required tensor-layout and gradient copies",
        ha="left",
        fontsize=10.5,
        color="#4b5968",
    )
    figure.legend(
        handles=[Patch(facecolor=color, label=label) for color, label in zip(COLORS, LABELS)],
        loc="upper left",
        bbox_to_anchor=(0.068, 0.928),
        ncols=3,
        frameon=False,
        fontsize=10,
    )
    notes = [
        "CUDA graph: 8 complete calls per replay. Bars: median of 7 paired rounds; whiskers: sample minimum–maximum.",
        "GPU labels show measured power limits. Geometry is cached; preparation is reported separately. Precision policies differ. No complete-model claim.",
    ]
    if len(families) > 3:
        notes.append(
            "Numeric labels mark candidate medians; every backend median and sample range is retained in the full tables."
        )
    if omitted_families:
        notes.append("GPU families without qualified representative data, omitted: " + ", ".join(omitted_families) + ".")
    if omitted_measurements:
        notes.append(
            "Unavailable or unqualified backend/panel measurements are omitted (no zero bars): "
            + "; ".join(omitted_measurements)
            + "."
        )
    wrapped = [line for note in notes for line in textwrap.wrap(note, width=157)]
    bottom = 0.035 + 0.019 * len(wrapped)
    figure.subplots_adjust(left=0.075, right=0.985, top=0.85, bottom=bottom + 0.035, hspace=0.43, wspace=0.24)
    figure.text(0.075, bottom - 0.012, "\n".join(wrapped), ha="left", va="top", fontsize=9, linespacing=1.5, color="#4b5968")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="svg", metadata={"Creator": "KohakuFA-CuTe benchmark plot", "Date": None})
    if preview:
        preview.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(preview, format="png", dpi=120, metadata={"Software": "KohakuFA-CuTe benchmark plot"})
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Sanitized performance summary JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Standalone SVG destination.")
    parser.add_argument("--preview-png", type=Path, help="Optional off-repository raster preview.")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".svg":
        parser.error("The publication artifact must be SVG.")
    root = Path(__file__).resolve().parents[2]
    if args.preview_png and args.preview_png.resolve().is_relative_to(root):
        parser.error("Raster previews must remain outside the repository.")
    plot(json.loads(args.input.read_text()), args.output, args.preview_png)


if __name__ == "__main__":
    main()
