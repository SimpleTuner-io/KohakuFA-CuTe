# RTX 5090 pipeline controls

These explicit controls preserve the declared rounded attention computation. They are negative experiments, not automatic dispatch presets. Compact timings, source revisions, resource sizes and exact loaded-code hashes are in [the pipeline results](../../results/rtx5090-pipeline-controls.json). Use the qualified optional runtime described in [the RTX 5090 guide](../../docs/rtx5090-configured-host.md).

The output-gradient control uses the existing configured runner and changes only D64's stage count. Run it from the repository root:

```sh
PYTHONPATH=src python3 -m benchmarks.compare_attention_variants \
  --operations '[{"label":"do1","factory":"benchmarks.sm120_dout_experiments:make_attention","kwargs":{"dout_stages":1}},{"label":"do2","factory":"benchmarks.sm120_dout_experiments:make_attention","kwargs":{"dout_stages":2}}]' \
  --seq 128 1024 4096 --dim 64 --heads 16 --dtype bf16 fp16 \
  --visibility dense causal --phases backward joint --warmup-ms 1000 \
  --window-ms 100 --repeats 7 --output dout-pipeline.json
```

The maximum-staging prototype needs an isolated checkout of its qualified base. Its retained patch adds an explicit experimental runner family; it does not modify the production checkout. The patch contains the shared backward methods and its factory, rather than a runtime rewrite of the kernel class.

```sh
git worktree add -b experiment/sm120-maximum .publication/sm120-maximum \
  ca0e795982155528e92674813c24b88214d348e2
git -C .publication/sm120-maximum apply \
  ../../benchmarks/experiments/sm120_maximum_staging.patch
cd .publication/sm120-maximum
PYTHONPATH=src python3 -m benchmarks.compare_attention_variants \
  --operations '[{"label":"control","factory":"benchmarks.sm120_maximum_experiments:make_attention","kwargs":{"stage_maximum":false}},{"label":"maximum","factory":"benchmarks.sm120_maximum_experiments:make_attention","kwargs":{"stage_maximum":true}}]' \
  --seq 128 1024 4096 --dim 64 128 --heads 16 --dtype bf16 fp16 \
  --visibility dense causal --phases backward joint --warmup-ms 1000 \
  --window-ms 100 --repeats 7 --output maximum-pipeline.json
```

Compilation, allocation and warmup precede timing. Eager windows include host submission gaps; captured replay measures a different boundary. These commands reproduce the isolated controls, not a complete-model speedup.
