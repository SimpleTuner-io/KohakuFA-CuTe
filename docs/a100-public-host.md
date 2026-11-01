# A100 cached public host entry

The eligible A100 public attention path now calls a cached native
`AttentionRunner` with the already selected schedule. C++ allocates owned
outputs and saved statistics, constructs tensor views, and prepares the flat
launch arguments. Each invocation still receives its current tensors, scale,
device and stream. The GPU methods and arithmetic are unchanged.

Eligibility remains the qualified full A100 with 108 SMs, batch 1,
8–16 heads, sequence lengths 2048–4096, D64/D128, aligned contiguous FP16/BF16,
dense or causal attention and a finite FP32-normal positive scale. The existing
medium H8/D128 causal exclusion remains. Masks, GQA, dropout, unsupported scales,
views and compiled calls retain their qualified native CuTe dispatch.
Installation still uses the pinned `sm80-fa4` extra.

## Complete-call controls

[Retained controls](../results/a100-public-host-control.json) compare the explicit
Python wrapper and native runner in the same process on an A100-SXM4-40GB with
108 SMs and a 350 W limit, using Python 3.13.18, Torch 2.15 CUDA 13.2,
CUTLASS DSL 4.8.1 and FA2 2.8.4. Seven rotating windows per backend followed
compilation and a one-second warmup. Each window spans at least 50 ms.
Forward is measured under `no_grad`; joint includes forward and all gradients.

| Boundary | Eligible rows | Native runner faster | Geometric speedup over Python wrapper |
| --- | ---: | ---: | ---: |
| Forward | 30 | 18 | 1.44× |
| Forward + backward | 30 | 22 | 1.64× |

This is a measured host-preparation control, not a new GPU pipeline speedup.
The individual ratios range from 0.94× to 3.58× for forward and 0.97× to 2.52×
for joint. Some long calls are neutral or slightly slower. The retained result
labels these explicit controls separately from the then-current public API;
it does not rename those old public timings as candidate results. FA2 and
SDPA launched their actual FlashAttention kernels, recorded per row.
The graph timing evidence in the earlier A100 performance document remains a
separate boundary and is not used to claim this eager gain.

## Correctness and ownership

Six actual-GPU public methods cover repeated API calls, selected boundaries,
native fallbacks, tiny scales, unaligned inputs/output gradients, changing
scales, literal saved maxima/log normalization and poisoned graph replay.
The full 74-case directed suite passed. Trace checks identify the intended
forward, preparation, joint main and dQ conversion; compiled calls retain their
native CuTe fallback. The final runtime source matches those qualified runs.

Memory, race, synchronization and initialization checks all passed. The original
reference-heavy racecheck was interrupted with status 143 after its diagnostic
FP64 GEMMs dominated instrumentation. The previously passed public, directed
and memcheck stages were retained. Both the original CUDA reference gates and
bit-preserving CPU copies passed outside sanitizers on the same actual outputs
and inputs; only the missing three tools then used CPU diagnostic GEMMs.
No arithmetic or acceptance threshold changed, including the delta from actual
rounded O. Reproduce this bounded sanitizer boundary with:

```sh
python -m benchmarks.cpu_reference_unittest \
  --target tests.test_fa4_sm80.Sm80KernelTests.test_cached_public_scales_statistics_and_changed_graph \
  --agreement --output artifacts/reference-agreement.json
compute-sanitizer --tool racecheck --report-api-errors no --error-exitcode 86 \
  python -m benchmarks.cpu_reference_unittest \
  --target tests.test_fa4_sm80.Sm80KernelTests.test_cached_public_scales_statistics_and_changed_graph \
  --output artifacts/racecheck.json
```

The cache forwards all 13 current schedule fields. Additional scheduling
controls require explicit runner support and their own qualification; silently
dropping an unknown control is unsupported. See the [host ABI](sm80-host-abi.md).
