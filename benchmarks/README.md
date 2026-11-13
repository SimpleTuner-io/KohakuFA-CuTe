# Benchmark tools

Run these source-checkout tools as `python -m benchmarks.…` from the repository
root, with Python 3.13 and the runtime qualified for the study. GPU benchmarks
and numerical references belong on the remote test host. Store raw captures
outside the checkout. See [benchmarking](../docs/benchmarking.md) for timing
boundaries and [validation](../docs/validation.md) for acceptance criteria.

## Common entry points

| Task | Module |
| --- | --- |
| Ordinary attention timings | `benchmarks.speed` |
| Independent precision diagnostics | `benchmarks.precision` |
| Default rounded-policy matrix | `benchmarks.qualify_baseline_precision` |
| Standalone FlashAttention comparison | `benchmarks.compare_flashattention` |
| Neighborhood versus FlexAttention and NATTEN | `benchmarks.neighborhood.benchmark` |
| Neighborhood reference qualification | `benchmarks.neighborhood.qualify` |
| Directed neighborhood mathematical diagnostics | `benchmarks.neighborhood.qualify_accuracy` |
| Explicit dQ and dQ+dK studies | `benchmarks.neighborhood.test_precision`, `benchmarks.neighborhood.test_key_precision` |
| Neighborhood scheduling experiments | `benchmarks.neighborhood.tune` |
| Ordinary attention performance figure | `benchmarks.reporting.plot_performance` |
| CUDA/NVTX profiling | `benchmarks.profiling.profile_nsight` |

Use `--help` for runner arguments on the configured host. Some retained
experiment modules are imported controls rather than command-line programs.
The `test_*` neighborhood runners launch experiments; unit test families are
listed in the [test map](../tests/README.md).

## Layout

| Directory | Contents |
| --- | --- |
| [neighborhood/](neighborhood) | Backends, independent references, qualification, tuning and report generators for 2D/3D neighborhood attention |
| [experiments/ampere/](experiments/ampere) | A100, A40, RTX 3090 and shared SM80/SM86 studies |
| [experiments/ada/](experiments/ada) | RTX 4090 and L40S controls and validation |
| [experiments/hopper/](experiments/hopper) | Hopper scheduling and experimental backend studies |
| [experiments/blackwell/](experiments/blackwell) | B200/B300 and SM120 scheduling, instrumentation and negative controls |
| [experiments/gb10/](experiments/gb10) | GB10 geometry, host, cache and schedule studies |
| [experiments/shared/](experiments/shared) | Cross-device and application experiments |
| [validation/](validation) | Adversarial, cuDNN, upstream and historical-policy diagnostics |
| [profiling/](profiling) | Generic attention and Nsight drivers |
| [reporting/](reporting) | Common charts and report rendering |
| [diagnostics/](diagnostics/README.md) | Standalone TMA and allocator reproductions |

The root retains shared references, adapters, artifact capture and common
drivers. Each device study keeps its controls beside its validation scripts.
No compatibility wrappers or hidden import redirection are installed for old
benchmark module names. Use the current modules above for new runs and a frozen
source snapshot for historical commands. The [experiment catalog](experiments/README.md)
maps retained runners and factories to their purpose, evidence and status.

## Reports and reproducibility

Start with the [neighborhood guide](../docs/neighborhood/README.md) or the
[ordinary-attention GPU catalog](../results/README.md).
Neighborhood `summarize_*` modules render saved campaign evidence without
running kernels. Different studies retain distinct schemas and acceptance
criteria; their summaries must not be merged as if they were one experiment.

The [experiment evidence index](experiments/README.md) retains rejected
approaches and known failures. Historical JSON, patch files, and external
campaign archives preserve original paths and hashes. Replay an exact old
campaign from its frozen source snapshot; use the new module paths for new
runs. Source capture in the relocated runners records their new paths.
