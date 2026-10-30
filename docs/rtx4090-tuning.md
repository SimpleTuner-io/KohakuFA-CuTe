# RTX 4090 scheduling and the remaining SDPA gap

This comparison measures attention forward, backward, and complete calls on a
full 128-SM RTX 4090 at its 450 W board limit. It compares default PyTorch SDPA,
the preceding CuTe schedule, and the selected CuTe schedule in the same process,
with identical rounded inputs and the same software stack. The preceding control
disables only the RTX 4090 scheduling selector. These are attention timings,
not complete-model or training-iteration speedups.

All **20 of 20 short complete-call cases beat SDPA** in this grid, including
both D128 visibility modes and dtypes. D16/D64 short cases are unchanged controls.
The new long D16 schedule beats SDPA in **8 of eight cases** at S1024/S4096.
Long D64/D128 improve against the preceding CuTe control but still trail SDPA.
Across all rows and each output/gradient independently, the largest CuTe norm-relative
error against FP64 is 0.030% for FP16 and 0.238% for BF16.

## Measured calls

All values below are **microseconds**, BF16, batch one, 16 equal Q/K/V heads,
contiguous B,H,S,D inputs. Forward retains autograd. Complete includes forward
and all three gradients. Smaller is better. The [comparison data](../results/rtx4090.json)
also contains FP16, S1024, backward-only timings, all samples, accuracy metrics,
and source hashes.

| Length | D | Visibility | Forward previous / CuTe / SDPA | Complete previous / CuTe / SDPA |
| ---: | ---: | --- | ---: | ---: |
| 128 | 16 | dense | 4.33 / 4.28 / 4.71 | 16.30 / 16.25 / 18.53 |
| 128 | 16 | causal | 4.40 / 4.40 / 4.80 | 17.33 / 17.26 / 19.15 |
| 128 | 64 | dense | 5.10 / 5.10 / 5.96 | 21.18 / 21.29 / 23.69 |
| 128 | 64 | causal | 5.20 / 5.20 / 6.16 | 19.54 / 19.56 / 24.34 |
| 128 | 128 | dense | 8.02 / 4.05 / 9.97 | 31.25 / 15.54 / 29.33 |
| 128 | 128 | causal | 8.17 / 4.06 / 5.95 | 31.01 / 15.56 / 25.31 |
| 128 | 256 | dense | 20.86 / 5.95 / 11.67 | 46.07 / 30.81 / 36.64 |
| 128 | 256 | causal | 21.42 / 5.99 / 11.21 | 46.59 / 30.89 / 35.38 |
| 128 | 512 | dense | 17.13 / 9.58 / 15.11 | 56.66 / 49.11 / 178.26 |
| 128 | 512 | causal | 17.07 / 9.58 / 14.81 | 52.21 / 44.55 / 178.24 |
| 4096 | 16 | dense | 261.91 / 180.12 / 248.22 | 1306.70 / 718.19 / 876.91 |
| 4096 | 16 | causal | 182.44 / 142.66 / 189.07 | 897.40 / 558.27 / 604.66 |
| 4096 | 64 | dense | 484.52 / 462.88 / 434.80 | 2297.30 / 2144.41 / 1570.63 |
| 4096 | 64 | causal | 317.70 / 288.21 / 322.50 | 1484.81 / 1387.97 / 1096.13 |
| 4096 | 128 | dense | 974.90 / 868.14 / 865.54 | 4401.99 / 4161.69 / 3208.84 |
| 4096 | 128 | causal | 613.92 / 533.73 / 535.98 | 2801.19 / 2645.81 / 1869.70 |
| 4096 | 256 | dense | 1952.33 / 1770.79 / 1792.45 | 11380.25 / 11190.75 / 7979.90 |
| 4096 | 256 | causal | 1305.41 / 1225.09 / 1062.68 | 6224.06 / 6144.74 / 4364.38 |
| 4096 | 512 | dense | 6429.17 / 6718.32 / 7011.91 | 32274.75 / 32044.61 / 67136.32 |
| 4096 | 512 | causal | 3338.91 / 3433.61 / 3550.17 | 16188.22 / 16225.41 / 39595.58 |

Long D64/D128/D256 complete calls remain slower than SDPA in this grid. D512
retains its stronger complete-call results. Its long-forward difference from
the preceding control ranges from +0.2% to +5.0% latency, while
complete-call differences stay within 0.7%. The cause of the forward
difference is not isolated. These comparisons do not claim that the remaining
long backward gap is closed.

## Selected changes

The selector is restricted to the measured RTX 4090 model, FP16/BF16, batch one,
16 equal heads, aligned vector copies, and no explicit mask. Both sequence axes
must be at most 128 for short schedules, or at least 1024 for long schedules.
Other shapes retain their existing schedules. Masks, GQA, other batch/head
geometries, and other Ada models retain their existing selection.

### Short D128 backward

The four-warp backward kernels distribute work across four column warps, each
owning fewer query/key rows. The matrix N permutation uses eight elements per
column warp, and score operands use `ldmatrix.x2`; the product operand loader
retains `ldmatrix.x4`. These are changes to geometry and operand layout, not a
claim that halving the load-instruction count halves transferred bytes.

Packed pairs store rounded probabilities and score derivatives. Shared staging
is reused only after its last reader. The query kernel computes the centered
correction itself, preserving the feature-wise MMA accumulation order and
removing the separate centering launch. The short grid exposes more independent
CTAs and reduces per-thread score storage. The selected schedules passed tails,
strides, sharp-attention tiny gradients, signed and zero scales, and changed-input
graph tests. The constructors reject an undersized shared-memory alias.

### Long D16

The existing physical head layout pads D16 to D32. The selected implementation
retains that layout for cooperative copies, alignment, and zero filling, but
executes only the valid K16 score features and N16 gradient/output features.
This removes matrix work on known zero padding in both directions. It does not
reduce the precision of valid features.

Both backward owners use 64 key/query columns rather than 32. A fused query
centering calculation removes its separate launch. The forward path also skips
padded matrix work. Changing the tile width or padding alone was less effective
than the combined selection. Two- and eight-warp forward alternatives were
slower and were rejected.

### Dense complete tiles

Dense backward D16/D64/D128 calls with both length axes divisible by 64 use a
variant that omits redundant row-bound predicates in copies, visibility, and
statistics. Feature predicates still zero-fill D16's padded storage. The query
variant also fuses centering.

The divisibility checks execute inside the cached launch program. Checked and
complete variants coexist in the same cache entry: a later partial or rectangular
shape cannot inherit an unsafe bounds policy. Tests cross each axis separately
and return to complete tiles through eager and compiled calls. Other causal paths
retain their checked policy; removing bounds alone was not a consistent win.

### Long causal D128 statistics

Complete long causal D128 tiles use a contiguous four-FP32 record per query:
unscaled maximum, log2 normalization, centered correction, and centering base.
The query owner writes one 128-bit record. The KV owner loads that record once
per local query column and reuses it across its score rows. All four values
retain FP32 precision. This combines field packing, column reuse, fused query
centering, and complete-tile bounds elision; the gain belongs to that bundle.

The native backward allocates the expanded workspace only for the matching
RTX 4090 shape and runtime length guard. It adds 12 bytes per query, or 0.75 MiB
at B1/H16/S4096, to reduce independently addressed scalar loads. Checked tails
keep the preceding workspace and backward program. The native allocation and
cached launch guards check both sequence axes. Compiled-cache tests switch
between these workspace interpretations and back, without adding cache entries.

The independent incremental comparison holds the selected forward and all
other policies fixed, and uses the same native workspace for both CuTe paths.
Values are complete-call microseconds; all output/dQ/dK/dV bits match the
checked backward control in all four rows.

| Length | Dtype | Checked backward control | Packed schedule | Reduction |
| ---: | --- | ---: | ---: | ---: |
| 1024 | BF16 | 295.27 | 286.16 | 3.1% |
| 4096 | BF16 | 2720.28 | 2646.42 | 2.7% |
| 1024 | FP16 | 293.23 | 285.30 | 2.7% |
| 4096 | FP16 | 2688.63 | 2639.92 | 1.8% |

### Earlier schedules retained

D64/D128/D256 long forward staggers V transfers across QK and the next K transfer
across softmax/PV, with wider row copies and reversed causal CTA order. Causal
D64/D128 forward retains Q fragments in registers. Dense D64 backward stages
four FP32 query statistics; causal D64 loads column statistics once for both
rows of its score fragment, and its dQ owner retains Q/dO fragments.

Short D128/D256/D512 forward uses a full-key implementation with 16 queries per
CTA, FP32 scores, and two 64-feature input stages. After QK releases staging,
that storage holds probabilities and two V stages. The smaller output
accumulator streams output features. Shared storage is about 36.5 KiB. The
constructor requires two input/output stages so its initial prefetch stays
within the feature dimension. This schedule changes floating-point order and
is qualified against independent FP64 equations, rather than byte equality.

D512 retains its first-round schedule in this follow-up. Its long forward
performance is reported as measured; no preservation gate is added.

## Profiling and deployed code

Binary and resource comparisons use representative BF16 cases. FP16 is measured
and numerically tested separately.

Systems timelines identify SDPA's FlashAttention-2 forward/backward at D64/D128.
At D512, default SDPA selects CUTLASS memory-efficient attention and a copy;
those comparisons are against that selected implementation, not FA2.

Separate CUPTI module-load captures collect the actual driver-loaded native
cubins without performance-counter access. Full kernel symbols from the profiled
graphs are matched against those ELF symbol tables, architecture records, and
resource metadata. The earlier D128 capture also matched two complete ELF
SHA-256 hashes to the corresponding extracted wheel images. Unlaunched entries
in a captured image are retained as code metadata and excluded from executed-work
claims.
SM86 native images can execute on SM89 under NVIDIA's
[Ada compatibility rules](https://docs.nvidia.com/cuda/ada-compatibility-guide/index.html).

CuTe PTX and cubins are exported from its compiled calls and disassembled.
The native library has no packaged PTX for the inspected attention functions.
PTX rebuilt from pinned FA2/CUTLASS sources is a source reconstruction, not
dumped native PTX or a byte-identical wheel rebuild. Driver `PTX_VERSION`
attributes identify the virtual target; they are not the PTX text ISA version.

The loaded D128/S4096 dense native backward graph contains two copies,
a preparation kernel, the joint gradient kernel, and dQ conversion. The joint
function allocates 255 registers/thread, 80 KiB dynamic shared memory, and
16 bytes/thread driver-reported local storage. Its resource-derived residency
bound is one CTA per SM. Preceding CuTe query and KV owners each allocate
255 registers and 48 KiB dynamic shared memory, with respective driver local
sizes of zero and 24 bytes; their resource bound is two CTAs per SM. These are
resource limits, not achieved occupancy or measured stall percentages.

The final BF16 captures identify the kernels below. Shared memory and
resident-CTA figures describe resource bounds, not achieved occupancy.

| Case | Stage | CTAs | Registers/thread | Dynamic shared KiB | Driver local bytes/thread | CTA/SM bound |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| S128 D128 dense | forward | 128 | 52 | 36.5 | 0 | 2 |
| S128 D128 dense | dQ | 128 | 128 | 24.5625 | 0 | 3 |
| S128 D128 dense | dK/dV | 128 | 100 | 26 | 0 | 3 |
| S4096 D16 dense | forward | 1024 | 80 | 12 | 0 | 6 |
| S4096 D16 dense | dQ | 1024 | 120 | 16.25 | 0 | 4 |
| S4096 D16 dense | dK/dV | 1024 | 168 | 16 | 0 | 3 |
| S4096 D64 dense | forward | 1024 | 161 | 24 | 0 | 3 |
| S4096 D64 dense | dQ | 1024 | 168 | 24.25 | 0 | 3 |
| S4096 D64 dense | dK/dV | 1024 | 168 | 24.5 | 0 | 3 |
| S4096 D128 dense | forward | 1024 | 196 | 48 | 0 | 2 |
| S4096 D128 dense | dQ | 1024 | 255 | 48.25 | 0 | 2 |
| S4096 D128 dense | dK/dV | 1024 | 255 | 48 | 0 | 2 |
| S4096 D128 causal | forward | 1024 | 217 | 48 | 0 | 2 |
| S4096 D128 causal | dQ | 1024 | 255 | 48.25 | 0 | 2 |
| S4096 D128 causal | dK/dV | 1024 | 255 | 48 | 32 | 2 |

### Remaining arithmetic and pipeline differences

| Backward path | Matrix products per attention pair |
| --- | ---: |
| CuTe dQ: QK, dO·V, dS·K, P·K correction | 4 |
| CuTe dK/dV: QK, dO·V, dSᵀ·Q, Pᵀ·dO | 4 |
| Native joint backward: shared QK/dO·V and three gradients | 5 |

The two CuTe owners recompute QK and dP. Centered dQ adds another product.
For D64, the inspected steady loops have 64 static HMMA sites in each
four-warp, 64-by-32 CuTe owner, versus 160 in the eight-warp, 64-by-128 native
joint owner. Normalizing equal attention work gives the same 8:5 matrix-work
ratio. Prologues, drains, and predicated padding are excluded. This is static
work evidence, not a runtime percentage or proof of an irreducible ceiling.

The native source holds V in registers, double-buffers Q, reuses one dO buffer,
and issues transfers around the last operand readers. Matching its stage count
alone does not reproduce that schedule. The centered correction remains required
by this project's sharp-attention/tiny-gradient contract. The original Gluon
repository's current SM100/SM103 backward uses a five-product formulation with
centering derived from rounded output; it is useful source evidence, not a
qualified RTX 4090 implementation or numerical oracle.

The subsequent [ownership and value-lifetime audit](rtx4090-backward-ownership.md)
checks the complete-row centering dependency, native accumulator representation,
shared-buffer lifetimes and measured joint-kernel interventions. It separates
required numerical work from avoidable communication and scheduling costs.

### Experiments rejected

Measured interventions also included paired score instructions, smaller N
fragments on long calls, V-register caching, dO-storage reuse, double-buffered Q,
earlier softmax/operand transfers, eight-warp owners, reverse query order,
minimum-residency hints, streamed product operands, and separate QK/dP warp
owners. None provided a reliable complete-call win over the selected policies
for the remaining long D64/D128 gap. Some reduced an isolated phase while
increasing other call costs. Register residency alone was not a consistent gain. Invalid-capacity and setup
failures are recorded separately from correct candidates with slower timings.

The dV-before-dP variants did not beat the selected complete call. Packing
four FP32 statistics produced the retained D128 causal bundle; direct column
loads were preferable there to staging another shared statistics plane.

Dedicated-loader prototypes used one or two producer warps, two or three
shared-memory slots, and two, three, or four compute warps. The first BF16 layout
showed a roughly 1.3% causal pilot gain, while its dense call was slower.
Subsequent complete-tile and three-compute-warp configurations did not give a
consistent gain across FP16/BF16. Their sampled outputs and gradients passed
FP64 checks; the rejected prototypes did not receive release sanitizer
qualification. The ring uses the SM80-compatible test-wait mechanism described
in the [PTX ISA](https://docs.nvidia.com/cuda/pdf/ptx_isa_9.2.pdf).

A controlled launch-bound change raised the dense dedicated-loader kernel
from 136 to 214 registers and reduced its resource residency bound from two
CTAs to one. Removing that hint reduced complete latency from about 2.35 to
2.16 ms, still behind the selected path. The three-compute-warp candidate
admitted more CTAs by resources yet remained slower. These observations support
measuring the complete computation rather than selecting by occupancy alone.

The requested DSL carveout hint was not established in captured runtime
attributes. A separate test explicitly set graph-node preferences, verified
those values, and reinstantiated each graph. Requesting a shared memory preference of 100% on
query, KV, or both owners produced no material improvement in the selected
D64/D128 calls. This is a measured negative preference test, not a claim about
actual cache allocation or a hardware ceiling.

## Qualification and measurement

The CPU suite passes 150 tests with 96 GPU-dependent skips. The GPU suite
passes 150 tests with 33 architecture/optional-dependency skips, including all
10 Ada tests. Two packed-path preflight methods also pass independently.
Compute Sanitizer memcheck and racecheck each pass 31 changed-input graph cases
with zero errors. These cover short D128 full/partial tiles, long D16 in both
dtypes and visibility modes, dense D16/D64/D128 guard boundaries, and packed
causal D128 full, rectangular and partial shapes. The public-text guard passes
20 tests; Black 24.8 and isort 5.13 checks pass.

Every comparative row checks output, dQ, dK, and dV independently against
full-length centered FP64 equations, evaluated in independent two-head chunks.
Each norm-relative error must be below 1% for BF16 or 0.5% for FP16, with exact
zero required when the reference norm is zero. SDPA is a performance baseline,
not the accuracy oracle. Additional tests cover negative maxima/LSE, saturated
softmax and tiny gradients, signed/zero scales, partial and rectangular tiles,
aligned strided inputs, cache reuse, changed graph data, and dispatch thresholds.

Timing uses seven rotating rounds targeting 200 ms per backend and phase,
with at least 16 graph replays. Selection probes use seeds 221, 955, and 959. The final grid
uses separate seed 961; the incremental packed comparison uses seed 967. Compilation, allocation, capture, profiling, and
telemetry are excluded. Backward retains an existing forward; complete computes
forward and all three gradients without accumulation. Separately timed phase
medians need not sum to the complete median. Graph checks compare eager results
and change Q/K/V for forward and complete replay. The sanitizer also changes dO.
Short replay timings can include host submission gaps and do not claim minimum
isolated kernel latency. Samples and replay counts are retained.

The qualified stack is Python 3.13.18, PyTorch 2.15.0+cu132, CUDA 13.2,
CuTe DSL 4.8.1, TVM FFI 0.1.15.post0, Triton 3.9.0, and driver 615.71.09.
Nsight Compute reported `ERR_NVGPUCTRPERM`, including for LaunchStats.
No measured stall, cache-hit, tensor-utilization, bandwidth, or HFU percentages
are inferred. Raw traces and machine-code captures are retained outside the
tracked tree.

## Sustained clocks, power and latency

A separate run sampled whole-board telemetry every 200 ms through four
20-second loops per case, alternating SDPA/CuTe/CuTe/SDPA at BF16
B1/H16/S4096, including forward and all gradients. CUDA events time groups
of 100 graph replays. Each pass excludes its first five seconds of latency
samples and first 25 telemetry samples (about five seconds) from the
summaries below. No profiler runs concurrently.

| D | Visibility | Backend | Pass | Median latency µs | Sample range µs | Median power W | Median SM clock MHz | Temperature °C |
| ---: | --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| 16 | dense | sdpa | 1 | 872.89 | 872.06–875.18 | 352.2 | 2715 | 58–64 |
| 16 | dense | cute | 2 | 730.07 | 725.71–734.12 | 449.5 | 2520 | 71–75 |
| 16 | dense | cute | 3 | 734.11 | 729.69–738.10 | 449.5 | 2505 | 76–78 |
| 16 | dense | sdpa | 4 | 877.59 | 877.28–877.88 | 367.2 | 2700 | 74–74 |
| 16 | causal | sdpa | 1 | 604.68 | 603.17–606.07 | 301.4 | 2700 | 69–70 |
| 16 | causal | cute | 2 | 561.61 | 559.06–564.13 | 420.9 | 2685 | 76–78 |
| 16 | causal | cute | 3 | 561.86 | 559.34–565.48 | 426.7 | 2685 | 79–80 |
| 16 | causal | sdpa | 4 | 607.77 | 606.37–609.05 | 300.6 | 2685 | 72–73 |
| 128 | causal | sdpa | 1 | 1866.86 | 1863.65–1875.34 | 436.3 | 2700 | 74–77 |
| 128 | causal | cute | 2 | 2655.52 | 2650.53–2662.21 | 398.8 | 2685 | 77–77 |
| 128 | causal | cute | 3 | 2655.16 | 2647.95–2661.45 | 400.4 | 2685 | 78–78 |
| 128 | causal | sdpa | 4 | 1871.08 | 1868.10–1873.07 | 442.5 | 2685 | 79–80 |

The 450 W board limit remained fixed. Whole-board samples are separate from
individual event groups; their clock values do not provide precise cycle
normalization or application HFU. Raw event groups, variability, and any
available clock-event reason samples are retained in the comparison data.
These observations do not establish an irreducible hardware ceiling.

S4096 D16 dense: the median of the two sustained pass medians is 732.09 µs for CuTe and 875.24 µs for SDPA (1.20× SDPA/CuTe).

S4096 D16 causal: the median of the two sustained pass medians is 561.73 µs for CuTe and 606.23 µs for SDPA (1.08× SDPA/CuTe).

S4096 D128 causal: the median of the two sustained pass medians is 2655.34 µs for CuTe and 1868.97 µs for SDPA (0.70× SDPA/CuTe).

Both dense D16 CuTe passes report the software power-cap reason in every
retained telemetry sample, with lower clocks than SDPA; its latency advantage
survives those sustained conditions. Causal D16 and D128 CuTe passes report
no active software power-cap reason. The D128 gap persists at comparable
sampled clocks, so these observations do not support attributing it to
power limiting. No retained sample reports a hardware or software thermal
slowdown reason.

## Reproduce

```bash
python -m benchmarks.compare_ada --seq 128 1024 4096 \
  --dim 16 64 128 256 512 --dtype bf16 fp16 --modes dense causal \
  --seed 961 --repeats 7 --window-ms 200 --output ../rtx4090.json
python -m benchmarks.compare_ada --control unpacked --seq 1024 4096 \
  --dim 128 --dtype bf16 fp16 --modes causal --seed 967 \
  --repeats 7 --window-ms 200 --output ../rtx4090-incremental.json
python -m unittest tests.test_ada -v
CUDA_VISIBLE_DEVICES='' python -m unittest discover -v -f
```

The pinned upstream sources are
[PyTorch](https://github.com/pytorch/pytorch/tree/d0f2af8b66b20da4baa50644c0cd3f887a2eae2e),
[FlashAttention](https://github.com/Dao-AILab/flash-attention/blob/14c377950125c70b7a9dabf9c561fca53715ac7d/csrc/flash_attn/src/flash_bwd_kernel.h),
[CUTLASS](https://github.com/NVIDIA/cutlass/tree/098de2a652cf8f00fd70b2df54051c7eccbb855a),
and [original Gluon backward](https://github.com/KohakuBlueleaf/KohakuFA/blob/0c5ede6859307bedc0c6089523c16b1ca51b7da9/src/kohakufa/sm100/bwd.py).
The native template definitions were checked against
[PyTorch's attention build configuration](https://github.com/pytorch/pytorch/blob/d0f2af8b66b20da4baa50644c0cd3f887a2eae2e/aten/src/ATen/CMakeLists.txt).
