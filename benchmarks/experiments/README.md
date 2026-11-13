# Explicit experiments

Executable studies are grouped under `ampere/`, `ada/`, `hopper/`,
`blackwell/`, `gb10/` and `shared/`. The [tool index](../README.md) lists current
commands. Recorded evidence below retains historical source paths and hashes.

These controls are outside default dispatch. They preserve evidence needed to
interpret accuracy changes, rejected schedules, and unresolved model compatibility.
Current supported-target performance belongs in the [GPU catalog](../../results/README.md).

| Evidence | Scope |
| --- | --- |
| [controls.json](results/controls.json) | Compact negative results, failing diagnostics, and model-shape audit from superseded reports; original source paths/hashes and Git revision retained |
| [precision.json](results/precision.json) | Previous arithmetic diagnostics, mathematical failures, H100 control, and B300 racecheck |
| [rtx5090-pipeline.json](results/rtx5090-pipeline.json) | dO stage and maximum-staging negative controls, complete-call costs and loaded-code hashes |
| [l40s-controls.json](results/l40s-controls.json) | Pipeline/layout and bounded-length experiments |
| [l40s-allocation.json](results/l40s-allocation.json) | Separate allocation-order diagnostic; mixed standard results remain in the main catalog |
| [h200.json](../../results/h200.json) | Explicit H200 WGMMA backend, eager/graph comparisons and unresolved initialization |
| [clef.json](results/clef.json) | Historical captured-attention and model compatibility; whole-model logit gate failed |

Historical notes describe their recorded revisions and arithmetic, not current
selection. Repository-relative source paths inside these records are resolved at
the recorded Git revision. Full prior reports remain recoverable from that tree;
they are not duplicated as an archive of user documentation.

## Source catalogs

Every retained Python module in the device-study directories is linked in these
catalogs, with its invocation role and source-defined purpose. A helper or frozen
snapshot is not a standalone command. Evidence links identify the associated
family; absence of a per-module result is not proof that the source is unused.

| Study family | Questions and controls |
| --- | --- |
| [Ampere](ampere/README.md) | A100/A40/RTX3090 schedules, host preparation and old/new selectors |
| [Ada](ada/README.md) | L40S/RTX4090 short/medium shapes, host guards and schedules |
| [Hopper](hopper/README.md) | Explicit H200 pipeline, persistent scheduling and initialization |
| [Blackwell](blackwell/README.md) | B200/B300/RTX5090 pipelines, readout, code generation and negative controls |
| [GB10](gb10/README.md) | Geometry, cache/metadata overhead, causal ordering and frozen selectors |
| [Shared](shared/README.md) | Joint backward and captured application experiments |

Run CLI entries on the configured remote host from the repository root as
`python -m benchmarks.experiments.FAMILY.MODULE --help`. Some retained scripts
parse arguments or execute work at module scope; do not import them merely to
inspect their purpose. “CLI (top-level execution)” marks that pattern. Imported
helpers and snapshots are not standalone commands. Factories can be selected by
`benchmarks.experiments.FAMILY.MODULE:FACTORY` in the comparison driver; inspect
their required arguments first. Patch-installing controls belong in isolated
experiment processes. Current production selection is documented in the
[source map](../../docs/internals/source-map.md), regardless of historical
experiment names. Preserve the frozen campaign source for historical replay.

Cross-family factories retained at the benchmark root:

| Source | Role and status |
| --- | --- |
| [configured_host_experiments.py](../configured_host_experiments.py) | `make_attention`: explicit native runner configuration; bypasses the normal compatibility selector during the experiment |
| [selected_validation_controls.py](../selected_validation_controls.py) | `make_attention`: previous/current API bodies with isolated cold/warm generic caches; records proof when configured |
| [experimental_precision_control.py](../experimental_precision_control.py) | Previous compensated arithmetic, separate from ordinary default qualification |
| [tune_kohakufa.py](../tune_kohakufa.py) | CLI tuning the installed upstream Gluon comparison on BF16 BHSD inputs |
| [cpu_reference_unittest.py](../validation/cpu_reference_unittest.py) | CLI moving reference work to CPU while GPU kernels remain instrumented; `--agreement` checks reference agreement outside sanitizers |

[`compare_attention_variants`](../compare_attention_variants.py) resolves factory
strings with `importlib.import_module`. Snapshot loaders also read Python source
bytes and execute them in a package namespace. An import-only reachability scan
therefore misses valid callers. See [frozen controls](../../controls/README.md)
for loader ownership and duplicate-snapshot policy.

## Accuracy controls

For ordinary attention, `benchmarks.experimental_precision_control` retains the former compensated policy.
Verify it against the preceding checkout with `benchmarks.validation.verify_previous_precision`
before timing it. Residual products and corrections are a stronger accuracy
experiment; their coordinate gates do not redefine [the default computation](../../docs/precision.md).

Retain failing operands, original acceptance criteria, numerical results, and full
call cost when extending these experiments. An execution failure, unsupported
configuration, or instrumentation timeout is not a successful timing result.

## RTX 5090 pipeline controls

Run from the repository root on the [qualified runtime](../../docs/installation.md#qualified-tuned-runtime):

```sh
python -m benchmarks.compare_attention_variants \
  --operations '[{"label":"do1","factory":"benchmarks.experiments.blackwell.sm120_dout_experiments:make_attention","kwargs":{"dout_stages":1}},{"label":"do2","factory":"benchmarks.experiments.blackwell.sm120_dout_experiments:make_attention","kwargs":{"dout_stages":2}}]' \
  --seq 128 1024 4096 --dim 64 --heads 16 --dtype bf16 fp16 \
  --visibility dense causal --phases backward joint --warmup-ms 1000 \
  --window-ms 100 --repeats 7 --output "$CAPTURE_DIR/dout.json"
```

The retained [maximum-staging patch](sm120_maximum_staging.patch) applies to
`ca0e795982155528e92674813c24b88214d348e2` in an isolated worktree. Its factory is
`benchmarks.sm120_maximum_experiments:make_attention`, with `stage_maximum` false/true.
The patch retains the experiment without rewriting production classes at runtime.

Keep raw captures outside the checkout. Compilation, allocation, and warmup
precede timing. [H200 usage and limits](../../docs/experimental/h200.md) are separate.
