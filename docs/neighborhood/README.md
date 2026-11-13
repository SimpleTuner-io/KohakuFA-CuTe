# 2D neighborhood attention

`neighborhood_attention` is an explicit experimental API for noncausal 2D
attention with shifted boundary windows. Operator qualification covers H100,
B200, B300, RTX 4090, RTX 5090 and L40S. It is not selected by the automatic
SDPA wrapper, and no complete HDiT model speedup or quality gain is established.

## Use and supported inputs

```python
import torch
from kohakufa_cute import neighborhood_attention

height, width = 64, 64
q = torch.randn(1, 8, height * width, 64, device="cuda",
                dtype=torch.bfloat16, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)
out = neighborhood_attention(q, k, v, spatial_shape=(height, width), window_size=7)
out.square().mean().backward()
```

Q/K/V must have equal, nonempty `[batch, heads, height*width, 64]` shapes,
FP16 or BF16 dtype, and a shared supported CUDA device. Spatial flattening is
row-major. Output shape and dtype match Q. Strided inputs and first-order
autograd are supported. Geometry is explicit: no dense mask is prepared or
stored. Saved statistics and gradient workspace grow linearly with sequence
length.

`window_size` is an integer or `(height, width)` pair, with positive sizes no
larger than the image. Stride and dilation are one. The default scale is
`1/sqrt(64)`; finite signed and zero FP32 scales are supported. GQA, masks,
dropout, causal windows, relative positional bias, other head dimensions and
higher-order gradients are outside this API. Unsupported inputs raise errors.

For query `(y, x)`, the window starts at
`(clamp(y - window_h//2, 0, height - window_h),
clamp(x - window_w//2, 0, width - window_w))`.
Every query sees exactly `window_h * window_w` keys. Windows shift inward at
image boundaries; they are not truncated or padded. Even sizes use the
`window//2` anchor. The relation is asymmetric: dK/dV enumerate the queries
that attend each key, rather than applying the forward window around the key.
The [implementation and initial qualification](reports/initial.md) describe the
independent enumeration, tails, views and graph checks.

## Default precision

The public call enables normalized dQ and reconstructed dK corrections on
every supported GPU, using `column_staged` statistics and separate correction
storage. No opt-in flag is needed. The measured complete-call cost is accepted;
dispatch does not revert to weaker arithmetic on slower GPUs or workloads.
The [precision contract](../precision.md#neighborhood-default) owns the equations
and rounding rules. Ordinary attention retains its separate contract.

Independent references use **native FP64 on identical rounded inputs** and
separately prescribed rounded equations with actual rounded O. They do not
emulate tensor-core reduction order. FlexAttention and NATTEN are comparison
implementations, not numerical oracles.

The corrected method passed 29 saved-input qualifications on H100 and was
requalified on B200, B300, RTX 4090, RTX 5090 and L40S. In that five-device
replay, the worst observed per-head component norm errors were **0.04289% FP16**
and **0.28530% BF16**, within the declared 0.5% and 1% limits. All four observed
component gates passed on all 29 fixtures; the eight zero/constant-dP controls
retain their original dQ formula gates, with mathematical dQ passes reported
separately. These are defined-workload results, not universal error bounds.
See the [multi-GPU report](reports/multigpu-precision.md) for the full scope.

A separate [H100 default-promotion receipt](../../results/neighborhood/default-promotion.json)
records byte-identical public-default versus explicitly selected results on
all 29 retained inputs, reuse of compiled kernels, the 16-method GPU suite and
four sanitizer passes. This supplements the earlier method qualification;
it is not a fresh six-device performance campaign.

Historical reports called the now-default method `weighted_dq_reconstructed_dk`.
Their labels, measurements and source hashes are preserved in the
[evidence catalog](../../results/neighborhood/README.md). To reproduce the
former arithmetic explicitly, use the serialized benchmark controls
`benchmarks.neighborhood.precision.variant("baseline")` and
`key_statistics("direct")`. Exiting them restores the corrected default.
These controls are source-checkout research tools, not package configuration.

## Performance and device coverage

Representative resident forward-plus-backward CUDA graph medians, microseconds,
for B1/H8, 64×64, window 7, D64, BF16:

| GPU | Former baseline | Corrected method (now default) | FlexAttention | NATTEN |
| --- | ---: | ---: | ---: | ---: |
| B200 | 148.60 | 151.28 | 140.77 | 207.83 |
| B300 | 141.51 | 140.85 | 141.46 | 201.21 |
| RTX 4090 | 155.98 | 169.40 | 189.27 | 337.87 |
| RTX 5090 | 128.02 | 142.72 | 156.26 | 376.49 |
| L40S | 144.42 | 150.89 | 161.95 | 323.31 |

These values come from the [five-device requalification](reports/multigpu-precision.md),
with seven timing samples per row and required layout copies included.
Compilation and immutable preparation are excluded. Power limits and selected
backends are recorded there; the RTX 4090 and RTX 5090 ran at 350 W and 400 W.
These recorded timings used explicit selection of the now-default method.
Different precision policies make these complete-implementation comparisons.
Window 13 and some tail cases still favor FlexAttention or NATTEN.

H100 was measured separately: the [dK study](reports/dk-precision.md) records
136.09 µs for this stronger BF16 case versus 131.19 µs for the prior dQ method.
Its held-out 48×64/window-9 BF16 case added 9.91%, exceeding the 5% incremental
cost aim. Do not combine timings from distinct campaigns into paired speedups.

## Reproduce and inspect

The measured neighborhood stack is Python 3.13.18, PyTorch 2.13.0+cu130,
CUTLASS DSL 4.8.1, TVM FFI 0.1.15.post0, CUDA bindings 13.0.3, Triton 3.7.1,
and NATTEN 0.21.7+torch2130cu130. This pinned comparison stack differs from
the optional ordinary-attention stack in [installation](../installation.md).
Qualification does not transfer automatically to newer framework releases.

Run numerical tests, compilation and reference calculations on a suitable
remote host. From the checkout, use Python 3.13 and set `CAPTURE_DIR` to
persistent storage outside the repository. Freeze source and launch commands
before an ephemeral deployment; retain outputs before cleanup.

```sh
python -m benchmarks.neighborhood.benchmark --help
python -m benchmarks.neighborhood.qualify --help
python -m benchmarks.neighborhood.test_key_precision --help
python -m unittest tests.test_neighborhood tests.test_neighborhood_precision \
  tests.test_neighborhood_key_precision -v
```

[Benchmark tools](../../benchmarks/README.md) list current entry points.
[Report data](../../results/neighborhood/README.md) map each study to its
unchanged machine-readable evidence. Historical source paths and hashes inside
that evidence identify the frozen run; the later file reorganization does not
retroactively qualify new source identities. Full raw operands, binaries,
traces and receipts remain in the off-host campaign archives.

## Evidence by question

| Question | Report |
| --- | --- |
| How do windows, backward traversal and original gates work? | [Initial implementation](reports/initial.md) |
| How did the original default compare on five GPUs? | [Initial performance](reports/performance.md) |
| What does the bounded H100 schedule select? | [Scheduling](reports/tuning.md) |
| Why can default rounding differ from mathematical FP64? | [Baseline accuracy](reports/accuracy.md) |
| What improved dQ? | [dQ precision](reports/dq-precision.md) |
| What improved dK, and at what complete-call cost? | [dK precision](reports/dk-precision.md) |
| Does the stronger method pass on the other architectures? | [Multi-GPU precision](reports/multigpu-precision.md) |
