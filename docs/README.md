# Documentation

Start with [installation](installation.md) and the [API](api.md).
[Precision](precision.md) defines the computation; [determinism](determinism.md)
defines repeatability. [Validation](validation.md) covers accuracy and known
limitations; [benchmarking](benchmarking.md) covers timing and profiling.

## Neighborhood attention

The [neighborhood guide](neighborhood/README.md) is the starting point for the
explicit 2D D64 API. It explains the corrected dQ+dK default, six-GPU
qualification, and the remaining performance gaps
against FlexAttention and NATTEN. Its evidence table links the scoped reports;
[recorded data](../results/neighborhood/README.md) retain the original sources,
metrics and failures.

## GPU guides

Architecture support permits execution. Tuned selection has narrower device,
shape, layout, gradient, and runtime requirements.
The following entries concern ordinary attention; neighborhood qualification
and scheduling are documented separately above.

| GPU | Tuned entry | Guide |
| --- | --- | --- |
| A100 | Standard APIs on pinned runtime | [A100](backends/a100.md) |
| A40 | Explicit opt-in | [A40](backends/a40.md) |
| RTX 3090 | `rtx3090_attention` opt-in | [RTX 3090](backends/rtx3090.md) |
| RTX 3090 Ti | Standard APIs | [RTX 3090 Ti](backends/rtx3090ti.md) |
| RTX 4090 | Standard APIs on pinned runtime | [RTX 4090](backends/rtx4090.md) |
| RTX 5090 | Standard APIs; additional pinned schedules | [RTX 5090](backends/rtx5090.md) |
| L40S | Standard APIs on pinned runtime | [L40S](backends/l40s.md) |
| GB10 | Standard APIs; shared schedules opt-in | [GB10](backends/gb10.md) |
| H200 | Explicit experimental entry | [H200](experimental/h200.md) |
| H100, B200, B300 | Standard APIs | [Hopper and Blackwell](backends/hopper-blackwell.md) |

[Results](../results/README.md) contain one report per GPU target and the common
performance chart. Each measurement retains its own source and protocol;
there is no single all-device qualification run.

## Implementation and diagnostics

- [Native host ownership](internals/native-host.md)
- [Schedule controls](internals/scheduling.md)
- [Experimental H200 backend](experimental/h200.md)
- [Repository layout and evidence organization](internals/repository.md)
- [Benchmark entry points and experiment directories](../benchmarks/README.md)
- [Blackwell allocator initialization diagnostic](diagnostics/blackwell-allocator.md)
- [Standalone TMA initialization diagnostic](../benchmarks/diagnostics/README.md)
- [Retained experiment controls](../benchmarks/experiments/README.md)
