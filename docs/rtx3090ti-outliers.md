# RTX 3090 Ti attention outliers

The measurements use a full 84-SM, 24 GB RTX 3090 Ti at its configured 450 W
limit. The canonical grid covers B1/H16, lengths 128/1024/4096, head dimensions
16/64/128/256/512, FP16/BF16, and dense/causal visibility. Forward retains
autograd; backward includes all three gradients; complete means forward plus
backward through the advertised API. Warmed single-call CUDA graph replays
include any replay/launch scheduling floor. Compilation and input preparation
are outside those intervals.

The previous report used another board and driver. This investigation measures
all controls again on one machine and runtime. The final grid uses source
`812aaeb` and records all 24 runtime file hashes; earlier source-qualified
profiling and sustained controls are labelled separately below. The measured
checkout is retained at tag `measurement-source/gpu-outliers-20261031`.

The qualified stack uses Python 3.13.18, PyTorch 2.15.0+cu132,
CuTe DSL 4.8.1, NVVM 13.4.59, and PTX ISA 9.4. The report records
the actual loaded library basenames and runtime source hashes; an installed package
version alone does not identify the deployed compiler or native backend.

## Separate precision repairs from scheduling gains

The final comparison retains four implementations:

| Label | Computation |
| --- | --- |
| SDPA | Actual default PyTorch SDPA on the same rounded operands. |
| Released | Schedules and arithmetic of main `3e17149`, compiled by the current stack. |
| Previous | Shared precision repairs with the preceding Ti schedules. |
| CuTe | Shared precision repairs and the selected Ti schedules. |

Released is a computation control, not a byte-identical historical cubin.
Every output, dQ, dK and dV is checked independently against centered FP64
on identical rounded inputs. Norm-relative limits are 1% BF16 and 0.5% FP16;
each element also passes `atol=rtol=0.02` BF16 or `0.003` FP16. Exact-zero
references require exact zero. Captured forward/backward/complete phases and
changed-input, poisoned joint replay receive the same checks. Failed released
or native comparisons remain diagnostics and do not become qualified wins.

The [precision notes](precision.md) describe the independent low-component
probability and softmax-derivative products. These are mathematical repairs
shared with other qualified devices. Their cost belongs in the
released/previous comparison; it is separate from the previous/CuTe schedule
comparison. Strict cancellation fixtures additionally check sign, nonzero
values and 2% error at the affected coordinate. The FP16 probability-residual
underflow fixture requires 0.5% coordinate error with zero absolute tolerance.

The [complete grid](../results/rtx3090ti.json) retains seven rotating
windows per phase/backend, targeting 200 ms with at least 16 replays.
Long calls can exceed that target to satisfy the minimum replay count.
All four implementations passed the ordinary grid's eager and
captured FP64 gates. Observed native median wins are 32/60 forward,
26/60 backward and 34/60 complete. Counts describe these samples;
near ties and unchanged schedules remain controls rather than tuning claims.

Across both dtypes and visibility modes, the selected long-D16 schedule
reduces complete latency 36.0–50.4% against the precision-matched previous
schedule. Short D128/D256/D512 calls improve 6.5–34.7% complete and
22.9–65.3% forward; 12/12 short complete medians beat native.
The long D256 query owner improves backward 2.4–3.4%
and complete 2.0–2.7% on top of the shared arithmetic repair. The BF16
D64 fixed owner reduces complete latency 0.6–7.1% within its selected range.
At S4096, D512 complete calls remain 1.53–1.79× faster than native;
its Ti scheduling is unchanged.

The S4096 causal D16 calls remain near parity: their SDPA/CuTe complete
ratios are 0.977 BF16 and 0.995 FP16. Long D64/D128/D256 retain
important native gaps. The following BF16 complete-call medians are in ms;
FP16 and every individual sample are in the grid.

| S / D | Visibility | SDPA | Released | Previous | CuTe |
| --- | --- | ---: | ---: | ---: | ---: |
| 128 / 128 | dense | 0.0436 | 0.0463 | 0.0429 | 0.0308 |
| 128 / 128 | causal | 0.0377 | 0.0464 | 0.0429 | 0.0284 |
| 128 / 256 | dense | 0.0607 | 0.0734 | 0.0767 | 0.0561 |
| 128 / 256 | causal | 0.0595 | 0.0716 | 0.0758 | 0.0548 |
| 128 / 512 | dense | 0.2670 | 0.1289 | 0.1327 | 0.1241 |
| 128 / 512 | causal | 0.2659 | 0.1057 | 0.1092 | 0.1001 |
| 1024 / 64 | dense | 0.3311 | 0.4828 | 0.5432 | 0.5043 |
| 1024 / 64 | causal | 0.2653 | 0.3235 | 0.3686 | 0.3594 |
| 4096 / 16 | dense | 2.2689 | 3.3143 | 4.1385 | 2.1283 |
| 4096 / 16 | causal | 1.2524 | 1.7685 | 2.0156 | 1.2817 |
| 4096 / 64 | dense | 4.0569 | 5.4472 | 6.5863 | 6.3484 |
| 4096 / 64 | causal | 2.3113 | 3.1236 | 3.6528 | 3.6304 |
| 4096 / 128 | dense | 7.5991 | 10.5836 | 12.7827 | 12.7946 |
| 4096 / 128 | causal | 3.9408 | 5.9307 | 6.9672 | 6.9746 |
| 4096 / 256 | dense | 15.8864 | 26.2575 | 28.6429 | 27.9144 |
| 4096 / 256 | causal | 8.3396 | 13.2649 | 14.8283 | 14.4279 |
| 4096 / 512 | dense | 109.7288 | 70.1661 | 71.3803 | 71.3252 |
| 4096 / 512 | causal | 63.7847 | 34.6236 | 35.9325 | 35.9949 |

## Selected Ti schedules

The device selector requires the full Ti, aligned unmasked FP16/BF16,
B1/H16 and equal head counts. Each compiled program keeps runtime length
fallbacks because native cache entries are reused across lengths.

- Long D16 uses a physical width of 32 with logical width 16, a 64-key tile,
  and fused query centering. Both sequence axes must be at least 1024.
- BF16 D64 key/value gradients use the fixed two-column-warp, 32-key-row
  owner only for square lengths 1024 through 4096. FP16 and unsupported
  masks, batches, GQA, strides, rectangles and lengths retain the repaired
  original owner. Query gradients and the prepass are unchanged.
- Short forward for D128/D256/D512 uses 16 query rows, a 128-key tile,
  and 64-feature chunks/output slices. Both axes must be at most 128.
- Short D128 backward uses four warps, four column warps, 16 query/key rows
  and a 32-column tile. It requires a square sequence at most 128.
- Long D256 query gradients use Q64/K32, score ownership `(4,2)` and gradient
  ownership `(2,4)` only for square lengths at least 4096. Other query calls
  retain their preceding owner. Row correction values are remapped by their
  actual coordinates when score and gradient MMA layouts differ.

The shared D256 precision owner uses Q64/K32, eight gradient warps with
ownership `(1,8)`, and 96 KiB of shared memory. Four packed DS/P high/low
planes exactly occupy the released 16 KiB V staging slot. V finishes its
read phase before those planes are published, and all gradient readers
release them before the next copies. dK and dV retain separate FP32 low
accumulators until the complete query sweep ends. V is reloaded for each
query tile because the weight planes occupy its staging slot; the measured
K32 tradeoff beat the resident-V K16 controls. The qualified Ti path uses
unscaled DS residuals; FP16 P residuals are scaled by `2^24` before packing
and rescaled at the final sum to preserve representable gradients.

The shared D512 precision owner uses 16 key rows, 32 query columns and
99 KiB. The Ti uses scalar BF16 weight publication. Its original query
owner and prepass remain independent of this key/value repair.

Broader equal-head correctness qualification includes B2/H4, rectangular and
partial tiles, last-dimension stride two, signed/zero scales, negative maxima,
sharp attention with nonzero small gradients, and changed/poisoned replay.
The performance selector's narrower scope is separate from that precision
qualification.

## What the native machine code shows

The final-source D64 trace separately verifies dense and causal S1024 calls:
four CuTe kernels and six native kernels per complete call, with exact driver
symbols repeated in five eager and twenty graph calls per mode/backend.
Eager and replayed graph outputs each pass FP64; the traced ranges contain
no CUDA memcpy/memset activity. The selected N2 owner uses 168 registers,
zero local bytes and 24 KiB shared memory, versus 255 registers, zero local
bytes and 72 KiB for the loaded native joint backward function.

That native D64 symbol uses
`Flash_bwd_kernel_traits<64,64,128,8,2,4,4,true,false,...>`: Q64/K128,
eight warps, score ownership `(2,4)`, and dK/dV and dQ ownership `(4,2)`.
The pinned traits enable register-held V and Q double buffering for this
shape. The selected CuTe key/value owner uses Q32/K32 and four warps, while
its query owner remains separate. These are different tile and work-sharing
designs; resource bounds alone do not measure achieved overlap. The dense
compiler cubin equals the actual loaded N2 image
`aec6cbb2548d6be73987343d7119f20cc2a74fb0940ba759274ad78cc41724f8`;
the causal image is
`9e527888f52c4629e976155ca33bf0c6d97f136d8dfb442f5b04b8978f81a0aa`.
Both compiler PTX exports target SM86 with ISA 9.4, and selected loaded SASS
has no local-load/store sites. Packaged PTX is absent from the selected
native ELF. The source interpretation is separate from those loaded images.

The retained node-level Systems trace on checkpoint `dd27a9f` verifies the
graph and eager paths on BF16 S4096/D256. The later D64-only selector change
preserves these query/key-value owner sources and their D256 dispatch. CuTe launches four kernels per complete call:
forward, row statistics, query gradients, and key/value gradients. Their
instrumented means are 4.930, 0.076, 9.154 and 13.729 ms. Native launches six
kernels, including two elementwise copy kernels required by its API path;
its main joint backward kernel takes 11.124 ms. These durations are
profiling diagnostics, separate from the canonical medians. No harness
input copies occur inside the ranges; no CUDA memcpy/memset activity was
recorded. All eight exact sanitizer fixtures also executed under named
trace ranges and passed their independent FP64 checks afterward.

Driver resources of the matched backward functions are:

| Owner | Registers/thread | Local bytes/thread | Dynamic shared | Resident CTAs/SM |
| --- | ---: | ---: | ---: | ---: |
| CuTe query | 255 | 0 | 97 KiB | 1 |
| CuTe key/value | 250 | 0 | 96 KiB | 1 |
| Native main | 255 | 24 | 96 KiB | 1 |

These are instantiated driver attributes and occupancy-API results, not
estimates of active utilization.

The deployed native backward symbol has
`Flash_bwd_kernel_traits<256,64,32,8,4,1,2,true,true,...>`: Q64/K32,
eight warps, score ownership `(4,2)`, dK/dV ownership `(1,8)`, dQ ownership
`(2,4)`. The pinned source retains V in registers and deliberately disables
Q double buffering. Matching a pipeline depth alone would not reproduce
its layout, iterator or ownership.

The original captured native cubin SHA-256 is
`f258224fb92f89a324319eab5ecc37393502b468d77d9b6ff8a9394ffcc9e0da`.
It uses 255 registers and a 24-byte stack; 12 whole-function local-load/store
sites hold small scalar/control state. Packaged PTX was absent from that
selected image. Source reconstruction is explanatory evidence, separate
from the loaded cubin and its SASS.

The native owner shares QK and dP across three gradients. Separate CuTe
query and key/value owners recompute those products, and the centered query
correction retains a P-weighted K product. Matched loaded SASS confirms
128 query and 192 key/value HMMA sites per equal dense Q64/K32 cell,
versus 160 in the native inner query loop. All three use eight warps. The
native outer key-tile scheduler backedge is separate; its 128-wide grid
covers the 128 key tiles once in this case.

For these current designs, the matrix-product accounting per dense cell is:

| Product | CuTe query + key/value | Native joint |
| --- | ---: | ---: |
| QK score | 2 | 1 |
| dO/V score derivative | 2 | 1 |
| dQ derivative product | 1 | 1 |
| P-weighted query correction | 1 | 0 |
| dK high/low products | 2 | 1 |
| dV high/low products | 2 | 1 |
| Total | 10 | 5 |

Dividing that known matrix-product work
by the canonical backward medians gives about 60 versus 59 TFLOP/s. This
is a work-normalized rate, not measured HFU, a stall breakdown, or a
mathematical lower bound. Sharing scores or changing saved state could
remove duplication. The CuTe compiler cubin matches the driver-loaded
image hash `d2bc178c4232781572ef693361d9154ce6dfa0c47444f2d8deec7582fd2d7126`;
its separate compiler PTX export has `.version 9.4` and `.target sm_86`.
The driver `PTX_VERSION` attribute names the virtual architecture, not
that ISA version. The
[backward ownership audit](rtx4090-backward-ownership.md) explains the
complete-row dependency and why a five-product native kernel cannot simply
replace this precision contract.

Nsight Compute returned `ERR_NVGPUCTRPERM`. The investigation therefore uses
node timelines, exact driver symbols/resources, exported and driver-loaded
code, and controlled interventions. Static instruction counts are normalized
by query/key work and warp ownership; they are not stall percentages.

## Controls that changed the design

These are selection experiments, separate from the final grid:

| Intervention | Observation |
| --- | --- |
| Nine compact/split producer layouts on S1024/D256 | All lost complete latency; fewer warps or smaller tiles did not establish a better schedule. |
| High-only resident-V Q64/K32 owner | Improved the old long call, but failed the later true dK/dV cancellation fixtures and was not retained. The preceding owner already retained V in shared memory; this experiment retained its matrix B fragment in registers. |
| K16 high/low paired products | Reduced registers but was slower than some separate-product controls. Lower allocation did not establish a win. |
| K16 next-dO or next-Q/dO register prefetch | Actual SASS moved the eight or sixteen vector global loads between 32 and 64 remaining HMMA sites; complete latency improved about 2.9–3.3% over its own corrected K16 control. K32 was still faster. |
| K32 nonresident V, both low products | Passed strict key/value/underflow fixtures without spills and beat the corrected K16 controls. The retained policy uses paired dK and dV high/low products. |
| FP16 DS residual scaling on D256 | Rejected by the unchanged 2% cancellation-coordinate gate on the Ti. Captured FP32 dS showed upstream probability/reduction bias that residual packing alone cannot remove. |
| Direct denominator statistics | A matched private forward/query/key-value prototype improved the scaled key fixture. It changes the saved-statistics contract and remains experimental. |

The four-component correction has a real cost compared with failing
high-only arithmetic. One controlled BF16 S4096/D256 pilot measured about
26.22 ms for released arithmetic and 27.95 ms for corrected K32, versus
29.75 ms for corrected K16. Those are pilot complete-call medians, not the
final published schedule gain.

## Reproduce the canonical boundary

```sh
python -m benchmarks.compare_sm86 \
  --expect-device 'NVIDIA GeForce RTX 3090 Ti' \
  --repeats 7 --window-ms 200 --minimum-replays 16 --seed 1913 \
  --output results/rtx3090ti.json \
  --profiles ../attention-captures/rtx3090ti
```

The harness rotates four backends, checks captured phases against FP64,
records free/allocated/reserved memory and per-case peaks, clears retained
compiled functions between rows, and verifies source hashes at completion.
Activity profiling belongs in another process: initializing Kineto and then
closing its context increased a separate short-call forward control from
roughly 9.4–9.9 µs to 18.1 µs. Ordered-child graph controls identified a
separate replay/launch floor, so their per-call times are not substituted
for the ordinary single-replay grid.

## Qualification and instrumentation limits

One uninterrupted GPU discovery run on `812aaeb` passed 203 methods in
3591.295 seconds: 129 active passes and 74 hardware/backend skips. It includes
all ten common key/value precision methods, all eight Ti schedule methods,
and tracker, training, graph and compiled-call regressions. Separate actual
long-D256 public-dispatch proofs with the selected query owner placed the
key coordinate within 0.70% BF16 and 1.76% FP16 of the closed form, below the
unchanged 2% gate. Broader B2/H4 qualification separately retained 56
rectangular, strided, signed-scale and changed-data checks.

On `dd27a9f`, Compute Sanitizer 2026.1.0.0 (build 37182542) completed
full public memcheck
on eight FP64-checked cases, including S4097/D256 dense and causal, with
zero memory errors when API diagnostics were disabled. The unfiltered
attempt exited 86 with fifteen API diagnostics: seven runtime/driver
version probes, seven `cuGetProcAddress_v2` errors and one TileIR
compiler-loading diagnostic. Its traceback starts at `cudaGetDeviceCount`
and CUDA lazy initialization, before attention compilation or launch.
That unfiltered run is not a successful sanitizer result.

The full S4097 racecheck was stopped at the 15-minute checkpoint before its
first case completed. Focused race instrumentation instead forced the
unchanged query owner on small multi-tile, partial and hidden-key inputs;
this bypassed its host scheduling guard and does not qualify a new production
dispatch. Six direct-owner cases and six unchanged public cases passed
unchanged FP64 checks and racecheck with zero hazards. Captured kernel
symbols confirmed `QueryMajorQueries` and `MediumKeyValue` executed in that
instrumented process. No kernel-name filter was used. Separate invalid
global-write and shared-race controls were detected and exited 86.

The final BF16 D64 selector additionally passed three public GPU methods
covering both dtypes, dense/causal threshold tails, native cache reuse,
compiled calls, real strided views, masks/GQA/batch fallbacks, changed inputs,
and poisoned replay. Selected square strong-key fixtures include both dense
and rolled causal visibility; scaled square value and underflow fixtures
check the intended selected BF16 and FP16 fallback symbols. Four further
selected BF16 negative/zero-scale graph calls passed FP64. On final source,
full public N2 S1025 dense/causal, FP16 N1 S1025, and a vector-one rectangular
fallback passed both memcheck and racecheck with zero kernel findings after
API-diagnostic filtering. The same unfiltered fifteen initialization
diagnostics were retained separately; both fault controls were detected again.
The native D64 cache retains separate maximum-length classes below and above
1024; crossing that threshold compiles both forward/backward pairs, while
repeated lengths reuse their existing programs.

A separate sustained D256 control on `dd27a9f` alternated
previous/selected/selected/previous for four 20-second passes, discarding
the first five seconds. Previous
complete-call medians were 28.648/28.652 ms; selected medians were
27.916/27.916 ms. The retained event-group ranges did not overlap:
selected maximum 27.932 ms was below previous minimum 28.619 ms.
All four passes sampled 1800 MHz. Median board power was 255.86/259.46 W
for previous and 249.90/251.22 W for selected, with temperatures 55–59°C
and a 450 W limit. No sampled power/thermal limiting reason was active.
Whole-board samples are not individual-kernel cycle measurements; these
controls do not replace the single-replay canonical grid.

## Bounded fixed-owner and pipeline controls

Two final controls used frozen private sources, separate from canonical timing.
Every timed backend and phase passed its own centered FP64 checks. Directed
key/value cancellation retained the 2% coordinate gate; the FP16 probability
underflow coordinate retained its stricter 0.5% zero-atol gate. Source hashes
were recorded before importing the controls and checked again at completion.

A fixed D64 N2 owner kept the original query owner, prepass, cache policy and
four independent high/low products. On BF16 it reduced complete-call pilots
from 0.5400 to 0.5034 ms at S1024 dense, 0.3691 to 0.3598 ms at S1024 causal,
6.5997 to 6.3663 ms at S4096 dense, and 3.6585 to 3.6397 ms at S4096 causal.
The last improvement is only about 0.5%; it is not a broad hardware limit.
The key/value owner changed from 255 registers with 40/8 local bytes to
168 registers with zero local bytes at the same 24 KiB shared allocation.
The occupancy API increased its resource bound from two to three CTAs/SM.

FP16 demonstrates why that resource change alone does not select a preset.
Its S1024 complete pilots improved 4.2% dense and 0.65% causal, but S4096
dense tied and causal regressed 0.74%. FP16 therefore retains the repaired
N1 owner. The BF16 range requires its own public-dispatch, boundary, fallback,
compiled-call, replay and sanitizer qualification; private forced-owner
numbers are selection evidence rather than the final comparison grid.

The D256 pipeline control separated geometry from overlap: Q32/K32 with one
Q/dO stage, the same geometry with two stages, and a two-stage variant that
retained the untouched upper half of V. The four Q32 weight planes use only
8 KiB of the 16 KiB V slot; instantiated offset assertions proved the lower
32-by-128 tile contiguous before reloading only that half. The prologue
still refreshed all V on every changed-input replay. Weight readers released
the aliased storage before a new V copy, and next Q/dO copies were issued
before current gradient products.

All 38 strict qualification records passed. Actual resources were 244
registers and 64 KiB for one stage, and 249 registers and 96 KiB for both
two-stage variants, with zero local bytes. Five rotating 100 ms complete
medians were 27.915 ms for the accepted Q64 owner, 29.320 ms for Q32 one
stage, 29.002 ms for Q32 two stages, and 28.894 ms for two stages retaining
upper V. Earlier copies and reduced V reload traffic helped the Q32 control,
but every accepted-Q64 sample beat every variant. All three controls were
rejected; they did not alter the published runtime.
