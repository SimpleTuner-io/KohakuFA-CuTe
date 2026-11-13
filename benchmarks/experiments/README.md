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

## Accuracy controls

`benchmarks.experimental_precision_control` retains the former compensated policy.
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
