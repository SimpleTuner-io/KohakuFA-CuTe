# A100 support

SM80 is accepted by the strict and automatic APIs and the native runtime. The
generic schedule preserves the same [arithmetic](precision.md) as the other
devices. Compiled autograd retains the actual rounded forward output for the
FP32 output/dO delta reduction.

This enables A100 execution without installing FlashAttention. It does not select
the RTX 3090 Ti presets or change the precision policy.

## Qualification

Qualification used a full A100-SXM4-40GB with 108 SMs, a 300 W power limit, Python
3.13.18, PyTorch 2.15.0+cu132, CUTLASS DSL 4.8.1, TVM FFI 0.1.15.post0, and Triton
3.9.0. This evidence applies to that full device rather than a partition.

- GPU unittest selection: 23 tests, with one CPU-platform-specific skip. The
  executed tests cover the maintained rounding fixtures, negative statistics,
  partial tiles, masks, changed graph inputs, and compiled autograd.
- Independent timing-matrix correctness: all 60 FP16/BF16 cases passed for
  sequences 128/1024/4096, dimensions 16/64/128/256/512, and dense/causal visibility.
  Each output and gradient retains its independent per-head FP64 norm-error gate
  of 1% for BF16 or 0.5% for FP16.
- CPU unittest discovery: 210 tests with 136 GPU-dependent skips. Publication
  guard tests: 20 passed.

## Baseline performance

Five timing rounds with 100 ms measurement windows compared complete forward,
backward, and joint calls in the same process. Compilation, allocation, and graph
capture were warmed separately. Default SDPA used its selected backend; the
standalone FlashAttention-2 comparison used version 2.8.4 from source revision
`4616f5116d33e3ba3cbb71d582caffcfcb49649c`.

| Generic SM80 call | Faster than default SDPA | Faster than FlashAttention-2 |
| --- | ---: | ---: |
| Forward | 9/60 | 9/48 |
| Backward | 22/60 | 11/48 |
| Forward + backward | 22/60 | 4/48 |

FlashAttention-2 does not support D512; those 12 rows are explicitly unsupported
and excluded from its denominator. These are complete implementation comparisons
under each backend's arithmetic, rather than an isolated scheduling comparison.
The generic SM80 path often trails FlashAttention-2 on D64/D128. Architecture
support alone does not establish a performance advantage for a workload.
