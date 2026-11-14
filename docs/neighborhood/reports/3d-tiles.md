# Compact 3D tiles on Ampere and Ada

A 4×8 spatial query patch makes large `11³` windows 1.90–2.19× faster than
LTX on A100 PCIe 40 GB, RTX 3090, RTX 4090 and L40S. It changes traversal and ownership;
the two-term probability product, normalized dQ, reconstructed dK and
column-staged statistics remain the default. The [receipt](../../../results/neighborhood/3d-tiles.json)
separates tile selection, held-out evaluation and numerical checks.

## Measured improvement

The formerly slow large-window cases are **1.90–2.19× faster than LTX** and
**1.92–2.32× faster than our generic owner**, across both dtypes and four GPUs.
Ours has the lower median in all 72 comparisons: six original LTX shapes plus
three held-out/fallback shapes, each in FP16 and BF16 on each GPU.

BF16 complete-call medians; both shapes use four heads and window `11³`:

| GPU; measured power limit | 17×128×128 ours / LTX (ms) | 49×128×192 ours / LTX (ms) |
| --- | ---: | ---: |
| A100 PCIe 40 GB; 250 W | 31.22 / 63.83 | 139.31 / 297.97 |
| RTX 3090; 320 W | 47.73 / 90.92 | 212.22 / 445.30 |
| RTX 4090; 330 W | 20.98 / 42.26 | 93.32 / 197.88 |
| L40S; 350 W | 18.32 / 37.57 | 80.70 / 176.18 |

Nine alternating-order samples target 200 ms each after compilation/allocation
warmup. Synchronized wall time includes attention preparation, copies, output
layout and Python work, with resident inputs. B1/D64 operands use synthetic
normalized Q/K, Q pre-scaled by 0.125, and API scale 1.0. No projection,
RMS/RoPE, backward, graph replay or complete-decoder latency is timed.
LTX selected PyTorch's memory-efficient SDPA in every measured row.

Held-out `13×64×96/H2` and `33×96×128/H1` cases used seed 99173, separate
from selection seed 6173. They were 2.00–2.27× faster than the generic owner.
The `13×64×65/H2` tail correctly retained that owner. Full FP16 results,
samples, variability, host limits and coarse pre/post-call telemetry are in
the receipt; the power limits above are measured settings, not nominal ratings.

## Accuracy and checks

Every timing row passed its independent per-head FP64 output limit: 0.5% FP16
or 1% BF16, at boundary queries plus 128 independently drawn positions. All
complete outputs were finite. Worst sampled forward errors were
**0.0214% / 0.1700%** for ours versus **0.0287% / 0.2287%** for LTX (FP16/BF16).

A separate full-coordinate comparison used seed 99173, two heads, random
rounded operands, scale 0.125 and window `11³` on `11×12×16` and `13×16×16`.
These reference-sized shapes explicitly select the compact owner outside
its production size gate. Worst per-head native CPU FP64 norm errors across
all four GPUs, percent:

| Implementation | Dtype | O | dQ | dK | dV |
| --- | --- | ---: | ---: | ---: | ---: |
| Compact default arithmetic | FP16 | 0.0208 | 0.0294 | 0.0297 | 0.0298 |
| LTX helper | FP16 | 0.0281 | 0.0370 | 0.0362 | 0.0297 |
| Compact default arithmetic | BF16 | 0.1661 | 0.2359 | 0.2358 | 0.2385 |
| LTX helper | BF16 | 0.2252 | 0.2956 | 0.2894 | 0.2385 |

The generic and compact owners retain essentially the same errors. This is a
scheduling gain with the existing precision advantage preserved; dV remains
near the same rounding floor. Neither implementation is an oracle, and these
operator fixtures do not establish decoder output quality or universal bounds.

Each GPU passed 47 focused tests, with five Hopper-only skips. The checks cover
independent forward/backward references, sharp/common-offset logits, signed
and zero scales, offset views, poisoned outputs, changed graph inputs and
actual public dispatch. Another 28 cases per GPU passed using both CPU and CUDA
references. Memcheck, racecheck, initcheck and synccheck each reported zero
errors/hazards on every GPU. CPU discovery ran 555 tests on A100 and L40S
(310 passed, 245 CUDA skips). Final schedule/public-graph checks also passed
after adding the L40S gate. Complete captures were archived and hash-verified
before releasing the devices.

## Selection and dispatch

The old owner groups 64 adjacent width positions. Its bounding box includes
many keys hidden from most queries. The compact owner groups 32 queries into
a 4×8 patch and loads/stores them directly, without an index table or packing
pass. It reuses the existing warp-MMA implementation and backward machinery.

Five tiles were screened on `17×128×128`, four heads, BF16, seed 6173:
`1×8×8`, `2×4×8`, `4×4×4`, `1×4×8` and `2×4×4`. Each used five alternating
150 ms samples, with independent FP64 checks. `1×4×8` had the lowest median
on every device. The closest alternatives on RTX 3090 and L40S were near
sample variability; all were substantially faster than the generic owner.
The other candidates remain explicit experiments, not dispatch alternatives.

The default gate requires the exact qualified SKU and SM count, D64,
FP16/BF16, eight-element vector copies, window `11³`, depth at least 11,
and height/width at least 64 and divisible by eight. Qualified SM counts
are 108, 82, 128 and 142 respectively. MIG, other SKUs, smaller windows and
unaligned shapes retain the generic owner with the same precision policy.
H100 PCIe retains its separately qualified warpgroup schedule. This is a
measured workload family, not a speed guarantee for every eligible shape.

## Code inspection and limits

Exact traversal counts for the two large shapes schedule 2.38× and 2.61× fewer
score pairs than the generic owner, including masked pairs inside executed
tiles. These are analytical work counts, not counter-derived runtime percentages.
The BF16 anchor launches one kernel for ours; LTX launches 18 attention kernels
among 1,086 CUDA activities, including 162 copies.

Timed-process PTX/cubins and traces identify the actual compact path. The BF16
anchor changes 128-thread CTAs, 166 registers/thread and 16 KiB shared memory
to 64 threads, 157 registers/thread and 12 KiB. Both use
`HMMA.16816.F32.BF16`, with no stack/local allocation. Their register-derived
ceiling is the same 12 active warps per SM; this is not measured occupancy.

The exact LTX symbol's packaged SM80 SASS was recovered from the mapped library;
its runtime fatbinary image selection and matching packaged PTX were not
recovered. An initial unmangled-symbol filter returned empty output despite a
zero exit status; that attempt remains in the evidence. Nsight Systems was
unavailable; PyTorch CPU/CUDA timelines are retained. Nsight Compute counters
were unavailable because of permissions or a busy driver profiling resource.
No stall, cache-hit or utilization claims are inferred from those limitations.

## Reproduce

Use Python 3.13.18, Torch 2.13.0+cu130, CUTLASS DSL 4.8.1,
TVM FFI 0.1.15.post0, CUDA bindings 13.0.3 and Triton 3.7.1, with a frozen
copy of LTX's `na3d` helper. This retains the baseline campaign's qualified
stack and its [official NATTEN wheel pairing](https://natten.org/install/),
holding framework and backend versions fixed for the scheduling comparison.
[PyTorch 2.15.0](https://pypi.org/project/torch/2.15.0/) was available at
qualification time; these measurements do not qualify that newer runtime.
All commands below run on a remote CUDA host; keep captures outside the checkout.
`--include-generic` compares the previous traversal with the same current
arithmetic. `--include-previous` is a separate, weaker historical precision
control and is not used for this scheduling comparison.

```sh
python -m benchmarks.neighborhood.benchmark_ltx \
  --ltx-helper "$LTX_HELPER" --output "$CAPTURE_DIR/3d-tiles" \
  --source-revision "$(git rev-parse HEAD)" --include-generic \
  --seed 1729 --reference-queries 128 --samples 9 --target-ms 200
python -m benchmarks.neighborhood.benchmark_ltx \
  --ltx-helper "$LTX_HELPER" --output "$CAPTURE_DIR/3d-heldout" \
  --source-revision "$(git rev-parse HEAD)" --include-generic \
  --cases heldout-13f heldout-33f fallback-tail \
  --seed 99173 --reference-queries 128 --samples 9 --target-ms 200
python -m unittest tests.test_neighborhood_3d tests.test_neighborhood \
  tests.test_neighborhood_volume_forward tests.test_neighborhood_schedule -v
```

The campaign isolated each case and dtype in a fresh process; use
`--cases NAME --dtypes DTYPE` to reproduce that isolation.
For selection only, add `--warp-volume-tile 1 4 8`. Final evaluation exercises
public dispatch. Sanitizer checks target
`tests.test_neighborhood_volume_forward.WarpCompactVolumeGPUTests.test_sanitizer_compact`;
reference arithmetic runs on the CPU outside GPU instrumentation.
