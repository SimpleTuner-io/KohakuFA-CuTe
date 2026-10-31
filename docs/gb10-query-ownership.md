# GB10 D256 query-gradient ownership

The SM121 D256 query-gradient owner shares QK and dP across all 256 output
features. The preceding owner computed those products separately for two
128-feature output slices. On the measured full 48-SM GB10, the full-width owner
gives a 1.06–1.35× backward speedup; forward plus backward improves by 1.07–1.29× in the
B1/H16 matrix. These are attention-call comparisons, not model throughput.

The [compact measurement report](../results/gb10-query-ownership.json) retains
all five timing samples, source identities and worst per-head norm errors.
The same-process comparison starts from `4cc9e3b` and selects the new owner
through an explicit query-configuration override. The production selector returns
that same configuration; both configurations are specified below.

## Arithmetic and resource ownership

The change selects unmasked, aligned FP16/BF16 D256 calls with equal query and
KV head counts. Explicit masks, GQA and other layout classes keep their
preceding selection. Sequence lengths remain runtime values in cached code;
there is no length-dependent selector hidden from the native cache.

| Owner | Query rows | Key columns | Output features | Warps | Operand staging |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous | 64 | 32 | 128, two slices | 4 | 96 KiB |
| Selected | 32 | 64 | 256, one slice | 8 | 96 KiB |

Four column warps give the selected M32/N64 geometry. Both owners fit the
101,376-byte opt-in block limit. Existing staging reuse, predicates and
publication barriers are unchanged. The KV owner and forward owner are unchanged.
A four-warp full-width experiment also passed the initial matrix but lost most
of its advantage at longer lengths; it is not selected.

The arithmetic follows [precision.md](precision.md): saved unscaled statistics,
shift before scaling, one input-dtype P conversion before dS, one dS conversion,
actual rounded-output delta, FP32 accumulation and late dQ/dK scaling. No
compensation or additional accuracy products are enabled.

## Measurements

Full 48-SM GB10, Linux ARM64, Python 3.13.18, PyTorch 2.15.0+cu132, CUDA 13.2,
CuTe DSL 4.8.1, TVM FFI 0.1.15.post0 and Triton 3.9.0. The DSL target is
`sm_121a`. Direct FA2 and FA4 use official revision
`4616f5116d33e3ba3cbb71d582caffcfcb49649c`; FA2 was built explicitly for SM121.

B1/H16, contiguous BHSD, default scale; BF16 examples below are **microseconds**.
Backward retains the completed forward; the complete column includes forward.
Five warmed CUDA-graph event rounds rotate implementation order, targeting
100 ms per phase. Compilation, graph capture and profiler overhead are excluded.
Standalone FA2 retains its own arithmetic policy; its comparison is a complete
implementation comparison.

| Length | Visibility | Previous backward | Selected backward | Previous complete | Selected complete | FA2 complete |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 128 | Dense | 47.53 | 39.67 | 55.03 | 47.09 | 57.93 |
| 128 | Causal | 38.05 | 33.01 | 45.09 | 40.03 | 45.80 |
| 512 | Dense | 528.00 | 467.15 | 610.24 | 545.05 | 462.43 |
| 512 | Causal | 312.67 | 262.46 | 372.35 | 329.60 | 348.70 |
| 1024 | Dense | 2077.74 | 1535.59 | 2462.77 | 1935.50 | 1437.56 |
| 1024 | Causal | 1019.08 | 821.73 | 1233.53 | 1018.62 | 968.43 |
| 4096 | Dense | 23674.28 | 18206.13 | 28600.28 | 23099.12 | 18682.63 |
| 4096 | Causal | 13440.95 | 10905.66 | 15799.60 | 13303.14 | 9423.85 |

The report includes the corresponding FP16 cases. A separate B1/H4 matrix at
S128/512/1024 passed 12 cases and showed complete-call speedups of 1.15–1.22×.
The long D256 FA2 gap remains. The default direct FA4 D256 call requested
131,072 bytes of shared memory and failed the device's block limit; it is
recorded as an execution failure, never as a timing or a fallback result.

## Qualification and limitations

All 16 H16 cases and all 12 H4 cases passed independent per-head FP64 norm
limits of 1% for BF16 and 0.5% for FP16, separately for output and each gradient.
Actual-output rounded backward diagnostics are retained independently. Focused
checks cover rectangular tails, projection strides, explicit masks and hidden
rows through their retained fallback, signed and zero scales, changing graph
inputs and poisoned output storage. The final production selector passed the
new runtime-cache and changed-input graph test. Memory, race and synchronization
checks passed for both the experimental configuration and final production source.

The unchanged six-suite run completed 80 tests with 23 skips and 17 failing
subtests, so this is not a claim that the whole GPU suite passes. A sharp FP16
causal D64 fixture has the same outputs and gradients with and without the new
D256 owner; direct FA2, FA4 and SDPA also miss its existing cosine gate. Twelve
D256/D512 causal tail subtests likewise return byte-identical baseline and
candidate tensors. Their first query has one visible key: the kernel gives zero
dQ while a different FP32 dP/delta reduction order produces a reference norm
around 2e-5. Whole-tensor norms pass. These diagnostics do not change the original
gates or add compensation. Four wide-head sharp-cancellation subtests also fail
at logit scale 1e6. Their outputs and all three gradients are byte-identical with
the previous query owner, including the unchanged D512 control. Their accuracy
failure remains unresolved in the baseline; this change does not claim to fix it.

Nsight Systems identifies separate query and KV owners as the next concentration
of backward time. Performance counters are denied on this system. Explicit joint
backward experiments share five tile products instead of seven, but remain
outside default dispatch and are not part of this D256 change.
