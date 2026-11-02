# Performance results

[Performance across eleven GPU targets](performance.svg) reads the reports below.
There is one JSON file per target and one common figure. Measurements were retained
from existing qualified runs; this cleanup did not run new GPU benchmarks.

![Default SDPA time divided by CuTe time](performance.svg)

## GPU reports

| Target | Retained coverage | Entry / limits | Data |
| --- | --- | --- | --- |
| A100 | D64/D128 long and medium; bounded D256; graph/eager | [Pinned automatic schedules](../docs/backends/a100.md) | [a100.json](a100.json) |
| A40 | D64/D128 long; D256 short/long; graph/eager | [Explicit opt-in](../docs/backends/a40.md) | [a40.json](a40.json) |
| RTX 3090 | D16/64/128 short; D64/D128 long; graph/eager | [Explicit opt-in](../docs/backends/rtx3090.md) | [rtx3090.json](rtx3090.json) |
| RTX 3090 Ti | 60-case default graph grid | [Full 84-SM device](../docs/backends/rtx3090ti.md) | [rtx3090ti.json](rtx3090ti.json) |
| RTX 4090 | 60-case grid plus current short/S1024 studies | [Pinned automatic schedules](../docs/backends/rtx4090.md) | [rtx4090.json](rtx4090.json) |
| RTX 5090 | 60-case grid plus current D64/D128/D256 studies | [Pinned automatic schedules](../docs/backends/rtx5090.md) | [rtx5090.json](rtx5090.json) |
| L40S | 60-case grid plus current short/long/medium training | [Pinned automatic schedules](../docs/backends/l40s.md) | [l40s.json](l40s.json) |
| GB10 | Current-policy D256 owner; shared training and inference | [Opt-in and default scopes](../docs/backends/gb10.md) | [gb10.json](gb10.json) |
| H100 PCIe | 16 BF16 graph cases on full 114-SM device | [Native path](../docs/backends/hopper-blackwell.md) | [h100.json](h100.json) |
| B200 | BF16 D512 graph comparison, including preceding-owner controls | [Bounded wide backward](../docs/backends/hopper-blackwell.md#b200-wide-backward) | [b200.json](b200.json) |
| B300 | 16 BF16 graph cases on full 148-SM device | [Native path](../docs/backends/hopper-blackwell.md) | [b300.json](b300.json) |

The 60-case grids use B1/H16, S128/S1024/S4096, D16/64/128/256/512,
FP16/BF16, dense/causal. Other studies have their own bounds. Qualification applies
to the measured source and hardware, rather than every supported shape or GPU.
H200 is [experimental](../docs/experimental/h200.md) and excluded from this catalog.

## Read the chart

Values are **default SDPA time / selected CuTe time**: above one favors CuTe.
Cells retain their same-process backend comparison; no timings are combined
across hosts or studies. Gray cells have no retained comparison.

Each head dimension has three columns: **F** is forward with autograd, **I** is
NoGrad forward, and **F+B** includes forward plus all three gradients. Rows identify
batch/head counts, length, dtype, and visibility. The chart shows CUDA-event graph
replay; eager wall time remains separately available in the JSON.

Hover a cell for its exact times, study ID, and evaluation run. For studies with
multiple independent runs, the first retained run is shown consistently; every
run remains in the report. This is a coverage chart, not an aggregate GPU ranking.

Latest measured studies replace overlapping workload cells. Unchanged shapes
retain their older cohort; data are not relabeled as remeasured on the latest
source. Some original grids therefore remain partly represented. Hardware,
power, runtime, preparation, and sample counts can differ even within one target.
For example, the RTX 5090 grid used 575 W and later tuned studies used 600 W.

Compilation, allocation warmup, and capture construction precede timing. Short
replay loops can contain submission gaps. Separate phase medians need not sum to
joint medians. Graph gains do not imply eager, model, or serving gains.
Default SDPA kernel selection is retained in each study; standalone FA2 is a
separate comparison and does not support D512.

## Report format

Reports use schema version 2:

- `target`, `label`, and `precision_contract` identify the target and arithmetic.
- `measurements` contains the selected shape, dtype, visibility, phase, workload,
  graph/eager boundary, metric, backend label, run, and unmodified median values.
- Each measurement references a `study` and a `record` index into that study.
- `studies` contains original metadata and retained raw records, including samples,
  numerical checks, kernel resources, source/library hashes, and qualification.
- Study provenance records the original repository-relative path, Git revision,
  and SHA256. Those old paths identify versioned inputs, not current files.

A later study supersedes the same shape/dtype/visibility/phase/boundary/workload,
while its independent evaluation runs remain separate. Unsuccessful backends stay
explicit in raw records and do not acquire timing cells. Some compact GB10 studies
published only sample counts and ranges; missing raw samples are not reconstructed.

Regenerate the chart with `python -m benchmarks.plot_performance`.
[Benchmarking](../docs/benchmarking.md) documents new measurements. The one-time
migration is reproducible with `python tools/consolidate_results.py --output DIR`;
it reads the frozen pre-cleanup Git tree without changing the checkout.

## Accuracy and diagnostics

All current catalog studies use the [default precision contract](../docs/precision.md).
Independent representative-matrix limits are 1% BF16 and 0.5% FP16 per head,
separately for output and each gradient. Directed rounded-equation checks and
known broader-suite failures retain their original scope. See
[validation](../docs/validation.md) for the distinction.

Unresolved initialization findings remain in [diagnostics](../benchmarks/diagnostics/README.md).
Previous precision policies, negative tuning controls, and model compatibility
studies live with [explicit experiments](../benchmarks/experiments/README.md).
