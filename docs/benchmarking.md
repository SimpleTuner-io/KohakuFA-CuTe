# Benchmarking

Install the [CUDA runtime](installation.md) and set `CAPTURE_DIR` to storage
outside the checkout. Report the GPU, source revision, loaded libraries, input
shapes/layouts, arithmetic, and timing boundary with every result.

## Timing and precision

```bash
python -m benchmarks.speed --dtype bf16 --seq 128 1024 4096 \
  --dim 16 64 128 256 512 --backends sdpa cute \
  --iterations 100 --repeats 6 --output "$CAPTURE_DIR/speed.json"
python -m benchmarks.precision --dtype bf16 --seq 256 --dim 128 \
  --backends auto kohakufa-cute --output "$CAPTURE_DIR/precision.json"
```

Repeat with FP16 and `--causal`. The speed driver uses SDPA as its numerical
control; independent FP64 checks and [directed tests](validation.md) provide
separate validation. Default SDPA is a dispatcher, not a specific backend.

The retained grid driver records samples, graph symbols/resources, and independent
output/gradient errors:

```bash
python -m benchmarks.qualify_baseline_precision \
  --source-revision "$(git rev-parse HEAD)" --repeats 5 --window-ms 100 \
  --output "$CAPTURE_DIR/qualified.json" --export-code "$CAPTURE_DIR/code"
```

These commands benchmark installed source; they do not reproduce a historical
host or every selection study. Published data keep their original protocol and
source identity. Previous-policy controls require separate verification.

## Neighborhood attention

Use `python -m benchmarks.neighborhood.benchmark` for the default API comparison
and `python -m benchmarks.neighborhood.test_key_precision` for the explicit
dQ+dK experiment. Run `--help` on the configured remote host before a campaign.
The [neighborhood guide](neighborhood/README.md) defines inputs, arithmetic,
measured stacks and timing scope. Its [data index](../results/neighborhood/README.md)
separates the original baseline, H100 tuning and later accuracy qualifications.
The [tool index](../benchmarks/README.md) lists all current module locations.

## Standalone FlashAttention

```bash
python -m benchmarks.compare_flashattention --backends cute fa2 sdpa \
  --seq 128 1024 4096 --dim 16 64 128 256 512 --dtype bf16 fp16 \
  --repeats 5 --window-ms 100 --output "$CAPTURE_DIR/comparison.json" \
  --export-code "$CAPTURE_DIR/code"
```

Install the chosen FA2, FA3, or FA4 backend separately; supply `--flash-revision`
when its source revision has been verified. FA2 does not support D512. The driver
uses square causal attention to avoid rectangular-alignment differences.
Failures remain explicit and never silently select another implementation.
Standalone backends have their own arithmetic; these compare complete implementations.

## Timing boundaries

- Forward may retain autograd or run under NoGrad; record which.
- Backward reuses a retained forward. Joint includes fresh forward and all gradients.
- Graph replay uses CUDA events. Short replay loops can contain CPU submission gaps.
- Complete eager wall time includes host preparation and launch; it differs from
  resident device time and single-request serving latency.

Warm compilation, allocation, caches, and capture before timing. Rotate backend
order, retain repeated samples and variability, and keep profiling separate.
Do not combine timings across different devices, power limits, sources, or
protocols into an isolated speedup.

## Profiling and figures

```bash
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
  --trace=cuda,nvtx --cuda-graph-trace=node -o "$CAPTURE_DIR/cute" \
  python -m benchmarks.profiling.profile_nsight --backend cute --phase joint \
  --seq 4096 --dim 128 --heads 16
python -m benchmarks.reporting.plot_performance
```

Use `--eager` to inspect ordinary launches. Identify actual kernels and required
preparations in both modes. Keep raw captures, PTX/SASS, binaries, and telemetry
outside the tracked tree. Counter permission failures remain limitations;
static instruction counts do not establish runtime stall percentages.
The figure reads the [per-target catalog](../results/README.md#report-format) directly.
