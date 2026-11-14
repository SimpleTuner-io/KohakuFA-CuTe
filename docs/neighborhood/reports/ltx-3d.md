# 3D attention versus the LTX helper

The optimized 3D default is faster than LTX’s helper on all six measured shapes in both dtypes, with lower forward error against native CPU FP64. It combines compact spatial ownership for large H100 PCIe windows with a two-term probability product. Normalized dQ and reconstructed dK remain enabled. The [receipt](../../../results/neighborhood/3d-ltx-performance.json) retains samples, source hashes and qualification scope.

## Complete-call timings

Full H100 PCIe (114 SMs, 80 GB, 350 W), AMD EPYC 7713 host, Python 3.13.18, Torch 2.13.0+cu130, CUTLASS DSL 4.8.1. The existing qualified stack was retained for matched comparisons. LTX selected cuDNN SDPA in every row; its `na3d` helper is Python tile/mask/stack preparation around SDPA, rather than a separate fused LTX CUDA kernel.

Nine alternating-order samples per row, approximately 200 ms per sample after compilation/allocation warmup. Synchronized wall time includes required preparation, copies and output layout; inputs are resident. These are synthetic normalized Q/K operands with Q pre-scaled by 0.125 and `scale=1.0`, B1/D64. No projection, RMS/RoPE, upsampling, graph capture, backward or full-decoder timing is included.

| Spatial T×H×W; heads; window | FP16 ours / LTX (ms) | BF16 ours / LTX (ms) |
| --- | ---: | ---: |
| 5×16×16; 32; 3×7×7 | 0.119 / 0.593 | 0.123 / 0.743 |
| 5×32×32; 16; 3×7×7 | 0.288 / 1.468 | 0.297 / 1.387 |
| 9×32×32; 8; 3×5×5 | 0.211 / 2.139 | 0.215 / 2.151 |
| 17×64×64; 8; 3×5×5 | 2.010 / 12.352 | 2.130 / 12.367 |
| 17×128×128; 4; 11×11×11 | 21.706 / 30.214 | 21.986 / 30.261 |
| 49×128×192; 4; 11×11×11 | 98.825 / 112.941 | 101.914 / 119.492 |

The first five shapes follow the decoder’s 17-frame 512×512 configuration, including latent padding, temporal cropping and block counts `(4,6,4,2,8)`. Weighting the separate medians by those counts gives **183 ms versus 287 ms in BF16** (36% lower), and **181 ms versus 286 ms in FP16** (37% lower). These are attention-only estimates, not complete-decoder measurements.

Compared with our original 3D forward in the same process, the new large-window cases are 2.2–2.4× faster. Some early stages pay roughly 7–11% for the extra forward precision and still substantially beat LTX. The timing control preserves the original C32/single-term forward; it is an explicit benchmark control, not an automatic fallback.

## Accuracy and validation

Every timing row passed its independent per-head sampled FP64 output limit (0.5% FP16, 1% BF16), including boundary queries plus 128 independently drawn query positions. All complete outputs were finite. The comparison uses identical rounded operands. Across this matrix, worst per-head forward errors were **0.0212% FP16 / 0.1710% BF16** for ours and **0.0287% / 0.2272%** for LTX. The new output errors are near the final input-dtype rounding floor.

A separate held-out, full-coordinate comparison covers three window families, two dtypes and two heads per case. Worst per-head norm errors, percent:

| Implementation | Dtype | O | dQ | dK | dV |
| --- | --- | ---: | ---: | ---: | ---: |
| KohakuFA | FP16 | 0.0210% | 0.0293% | 0.0294% | 0.0299% |
| LTX helper | FP16 | 0.0278% | 0.0302% | 0.0297% | 0.0300% |
| KohakuFA | BF16 | 0.1676% | 0.2359% | 0.2362% | 0.2394% |
| LTX helper | BF16 | 0.2234% | 0.2460% | 0.2407% | 0.2394% |

The default and explicitly selected compact owner both passed all component gates and the independent corrected backward equations using actual rounded O. dV is essentially unchanged; tiny head-level differences are not a universal accuracy win. The stronger forward does not imply that every dQ coordinate improves over the prior method. The [precision contract](../../precision.md#neighborhood-default) and independent two-term forward diagnostic distinguish mathematical error from operand rounding and reduction order.

The focused suite ran 37 tests: 36 passed and one architecture-specific case was skipped;
the added rounding/dispatch checks and three ordinary-attention regressions passed.
Final CPU discovery completed 546 tests (238 CUDA skips). Memcheck, racecheck,
initcheck and synccheck reported zero errors/hazards on the compact FP16/BF16
fixture. A final 12-case public-dispatch check covered signed/zero scales and
unaligned inputs: layouts outside the measured eight-element-copy family use
the stronger generic fallback. All source, samples, traces, failures and code
captures were archived and hash-verified off-host before releasing the GPU.

## What changed and what did not help

The previous owner grouped 64 adjacent width positions, spending substantial work on masked pairs. The selected owner groups an 8×8 spatial patch, maps loads/stores directly without a packing pass or index table, and reuses the existing softmax and gradient machinery. It uses Hopper warpgroup products and omits load predicates only when tile divisibility proves every accessed row exists. The two probability terms share V loads/warpgroup synchronization.

Compact scheduling is bounded to H100 PCIe with 114 SMs, D64, FP16/BF16, eight-element vector copies, `11³` windows, depth at least 11, and height/width at least 64 and divisible by eight. This is a measured family, not a guarantee for every geometry. Generic fallback schedules retain the same stronger 3D arithmetic on every GPU; 2D arithmetic and scheduling are unchanged. The later [cross-architecture qualification](3d-multigpu.md) measures the generic path on eight additional GPUs; those results are separate from this H100 study.

The retained screens found that larger flattened key tiles and a double-buffered prefetch candidate regressed. Direct spatial mask coordinates and proven full-row copies helped; extra pipeline stages did not establish a gain. A 128-column Hopper experiment produced zero outputs despite finite score statistics and remains unqualified, with a small failing operand capture retained. An early compiler-abort experiment was also preserved. None is selected by default.

Nsight Compute counters were denied. Exact launched kernels are retained in the traces. Our generated PTX/SASS and loaded-library hashes come from separate untimed matching replays. The cuDNN runtime-generated attention symbol was not recovered from its packaged libraries, so no substitute kernel is presented as its SASS. The BF16 final-stage owner changed from warp `HMMA.16816` to warpgroup `HGMMA.64x64x16`: registers were 128 versus 126, with no stack/local allocation, and dynamic shared memory was 16 versus 24 KiB. Both have a resource-derived limit of four CTAs per SM; this is not measured occupancy or a stall breakdown.

## Reproduce

Use the pinned runtime above on a remote H100 PCIe and freeze the exact LTX `na_kernels.py` helper at the revision/hash in the receipt. Keep outputs outside the checkout.

```sh
python -m benchmarks.neighborhood.benchmark_ltx \
  --ltx-helper "$LTX_HELPER" --output "$CAPTURE_DIR/ltx-3d" \
  --source-revision "$(git rev-parse HEAD)" \
  --seed 1729 --reference-queries 128 --include-previous
python -m unittest tests.test_neighborhood_3d \
  tests.test_neighborhood_volume_forward -v
```
