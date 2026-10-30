# Performance guide

[rtx5090.json](rtx5090.json), [rtx4090.json](rtx4090.json), [rtx3090ti.json](rtx3090ti.json), and [l40s.json](l40s.json) retain the latest four GPU grids, timing samples, source hashes, independent FP64 checks, and software/hardware metadata. The chart reads these reports directly. The October 22 grids in [performance.json](performance.json) are superseded; that file also retains the explicit-mask, B300 MIG, H100 and model comparisons. [gb10.json](gb10.json) retains the GB10 comparison separately.

The four plotted GPUs now cover the same 60 cases, including D16 and S128. Each panel labels its date, PyTorch version and measured power limit. The [RTX 4090 analysis](../docs/rtx4090-tuning.md) details its scheduling changes and remaining backward gap; the new three-device runs measure their released dispatch without enabling the 4090-specific policies.

Times are median milliseconds per attention call unless a table explicitly specifies microseconds. **Forward + backward includes forward.** Timings use CUDA events around CUDA-graph replay, exclude compilation, and rotate backend order. Profiling measurements are excluded from these tables. Compare backends within a row: hosts, software, and replay counts differ between datasets. Full-model throughput and first-use compilation cost require separate measurements.

## GB10 (DGX / RTX Spark)

GB10 supports FP16/BF16 forward and centered backward on Linux ARM64. Use the [GB10 installation instructions](../README.md#install); `CUTE_DSL_ARCH=sm_121a` and the CUDA 13 DSL libraries were used for validation. Automatic selection checks compatibility rather than predicted speed, so benchmark the intended shapes.

### Short-sequence comparisons

B=1, H=4, S=128, contiguous BHSD self-attention, default scale. **These short-sequence tables use microseconds**, unlike the millisecond tables elsewhere. Dense means noncausal without an explicit mask; causal means implicit token-causal attention. All five head dimensions are shown, including controls that receive no new short-forward kernel. Ratios above one favor CuTe; D16/D64 forward differences are within measurement noise and are treated as ties.

Forward retains gradients. BF16 forward, microseconds:

| Head dim | Dense SDPA | Dense CuTe | SDPA / CuTe | Causal SDPA | Causal CuTe | SDPA / CuTe |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 6.15 | 6.15 | 1.00× | 6.15 | 6.15 | 1.00× |
| 64 | 6.15 | 6.15 | 1.00× | 6.15 | 6.15 | 1.00× |
| 128 | 10.24 | 8.20 | 1.25× | 10.24 | 8.20 | 1.25× |
| 256 | 12.07 | 6.15 | 1.96× | 11.32 | 6.15 | 1.84× |
| 512 | 14.34 | 8.20 | 1.75× | 14.34 | 8.20 | 1.75× |

FP16 forward, microseconds:

| Head dim | Dense SDPA | Dense CuTe | SDPA / CuTe | Causal SDPA | Causal CuTe | SDPA / CuTe |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 6.15 | 6.15 | 1.00× | 6.15 | 6.15 | 1.00× |
| 64 | 6.15 | 6.15 | 1.00× | 6.15 | 6.15 | 1.00× |
| 128 | 10.25 | 8.20 | 1.25× | 10.24 | 8.20 | 1.25× |
| 256 | 12.02 | 6.15 | 1.96× | 11.33 | 6.15 | 1.84× |
| 512 | 14.34 | 8.20 | 1.75× | 14.34 | 8.20 | 1.75× |

BF16 forward + backward, microseconds; these times include forward:

| Head dim | Dense SDPA | Dense CuTe | SDPA / CuTe | Causal SDPA | Causal CuTe | SDPA / CuTe |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 18.43 | 16.38 | 1.13× | 18.44 | 15.03 | 1.23× |
| 64 | 20.49 | 18.44 | 1.11× | 21.30 | 18.44 | 1.15× |
| 128 | 26.65 | 26.65 | 1.00× | 27.57 | 25.83 | 1.07× |
| 256 | 36.04 | 28.67 | 1.26× | 35.70 | 29.57 | 1.21× |
| 512 | 198.56 | 38.39 | 5.17× | 197.51 | 36.87 | 5.36× |

FP16 forward + backward, microseconds:

| Head dim | Dense SDPA | Dense CuTe | SDPA / CuTe | Causal SDPA | Causal CuTe | SDPA / CuTe |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 17.38 | 16.04 | 1.08× | 18.44 | 14.74 | 1.25× |
| 64 | 20.50 | 18.46 | 1.11× | 21.49 | 18.44 | 1.17× |
| 128 | 27.80 | 26.92 | 1.03× | 27.90 | 26.63 | 1.05× |
| 256 | 34.85 | 28.82 | 1.21× | 35.61 | 28.76 | 1.24× |
| 512 | 116.50 | 38.91 | 2.99× | 116.69 | 37.30 | 3.13× |

The large D512 complete-call advantage includes the existing centered backward implementation; it is not a speedup attributable solely to the new forward kernel. At S128/B1/H4, the new BF16 D256 forward drops from 14.30 to 6.15 µs and D512 from 16.39 to 8.20 µs against the paired preceding CuTe implementation. D16/D64/D128 retain their existing dispatch.

### Longer sequences and scope

BF16, B=1/H=4, dense attention; milliseconds:

| Tokens | Head dim | SDPA forward | CuTe forward | SDPA F+B | CuTe F+B |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 256 | 0.0328 | 0.0337 | 0.1444 | 0.1712 |
| 512 | 512 | 0.1081 | 0.0876 | 1.0782 | 0.3699 |
| 1024 | 256 | 0.1118 | 0.1118 | 0.3758 | 0.5150 |
| 1024 | 512 | 0.3485 | 0.2391 | 5.4171 | 1.5651 |

D512 benefits at longer lengths from sharing QK and backward score work across output features. D256 full-call latency still trails SDPA at S512/S1024; short-sequence wins do not establish a general GB10 advantage. These are attention-call measurements, not whole-model or generation throughput. Explicit masks use the existing fallback and are outside this timing matrix.

The short-forward path is restricted to SM121, FP16/BF16, eligible aligned D256/D512 layouts, no explicit boolean mask, and both query/key lengths at most 128. Other calls use the existing attention program. A runtime length check preserves compiled-cache reuse across either 128/129 boundary. Query/output tiles depend on batch × heads; performance should be rechecked for the intended geometry.

### Kernel design and profiling

- D512 uses four-warp full-width tiles to share QK/dP work across output columns. This avoids repeated score work and removes the observed dQ stack traffic from the initial GB10 layout.
- Short forward streams Q/K feature chunks through two shared stages while retaining the full active key score tile in FP32. S128 needs one softmax update rather than four 32-key updates.
- Once QK finishes, shared storage is reused for probabilities and double-buffered V stripes. V transfers overlap softmax; one FP32 reciprocal per row is reused across output stripes. Eligible async copies use the L1-bypassing `cp.async.cg` policy.
- Warp-local reductions avoid cross-warp scratch when each warp owns complete rows. The selected 32-head causal D256 layout omits unreachable keys for the first query block.

Nsight Systems and the actual selected PyTorch kernels informed these choices. At D512, default SDPA selects memory-efficient CUTLASS; D256 selects PyTorch FlashAttention, sometimes with split-KV/combine. The selected D512 wheel SASS was inspected. Its matching forward PTX was absent from the wheel, so the exact template was rebuilt from the installed PyTorch headers and matching CUTLASS revision for SM121. That PTX is source-matched reconstruction, not captured driver output or a byte-identical wheel rebuild. PyTorch source revision: `70d99e998b4955e0049d13a98d77ae1b14db1f45`; CUTLASS revision: `0d2b201e8c1c4a03efa6e9c468161916e2334725`.

The B1/H4 D512 short kernel uses 68 registers/thread and 68.5 KiB dynamic shared memory, versus 254 registers and 96.5 KiB for the preceding wide CuTe path. CUDA's occupancy API permits one CTA/SM for that layout; larger-head D512 uses 40.5 KiB and permits two. A narrow D256 layout uses 32 KiB and permits three. These are resource bounds, not measured achieved utilization. Nsight Compute counters were blocked by `ERR_NVGPUCTRPERM`; no measured stall, cache, tensor-utilization or bandwidth percentages are claimed.

### Measurement and validation

Measured October 28, 2026 UTC on a 48-SM GB10 with 128 GB unified memory, Linux ARM64, driver 610.57.04, Python 3.13.18, PyTorch 2.11.0+cu130, CUDA toolkit 13.0.1 and CUTLASS DSL 4.8.1. CuTe's bundled NVVM 13.4.59 emits PTX 9.4; source-matched SDPA PTX reconstruction used NVCC 13.0. Nsight Systems/Compute were 2026.3.2/2026.3.1.

Each backend sample warms and measures 300 single-operation CUDA graph replays using CUDA events. Nine repetitions rotate SDPA, preceding CuTe and new CuTe; compilation is excluded and clocks are not locked. A separate grouped-graph experiment reduces launch overhead, but its numbers are not mixed into these tables. The preceding CuTe control uses the `5014578` forward algorithm and the same centered backward as the new path. The final kernel snapshot is `8940028`; [gb10.json](gb10.json) records its source hashes, medians, all timing samples and output/Q/K/V FP64 errors, including the full 52-case S128 matrix and four longer-sequence controls. The additional S128 geometries are B1/H1, B1/H16, B1/H32 and B4/H16 at D256/D512, both dtypes and modes. D16 was a separate follow-up on the same device/source with the same protocol.

All 52 S128 cases pass independent FP64 checks for output and each gradient, using norm-relative limits of 1% for BF16 and 0.5% for FP16. The timing driver checks graph outputs/gradients against eager execution. Focused GPU tests additionally cover changed-input graph replay, both cache length boundaries, GQA, projection strides, saved statistics, partial tiles, negative maxima, signed/zero scales and tiny gradients. The 92-case precision regression matrix has zero CuTe failures, the six-test adversarial suite passes, and Compute Sanitizer racecheck reports zero hazards for the final causal-prefix and narrow-output layouts. Larger-length comparisons preserve the previous forward path within run-to-run variation.

### Reproduce on GB10

After installing the ARM64 CUDA stack, compare all five dimensions in both modes:

```bash
python -m benchmarks.speed --heads 4 --dtype bf16 --seq 128 \
  --dim 16 64 128 256 512 --backends sdpa cute \
  --iterations 300 --repeats 9 --output ../gb10-dense.json
python -m benchmarks.speed --heads 4 --dtype bf16 --seq 128 --causal \
  --dim 16 64 128 256 512 --backends sdpa cute \
  --iterations 300 --repeats 9 --output ../gb10-causal.json
# Repeat with --dtype fp16. Add --seq 512 1024 for longer-length controls.
python -m unittest tests.test_kernels -v
python -m benchmarks.cudnn_regressions --backends kohakufa-cute math \
  --output ../gb10-precision.json
# Capture warmed ranges separately from timing, keeping reports outside the tree.
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
  --trace=cuda,nvtx --cuda-graph-trace=node -o ../gb10-cute-forward \
  python -m benchmarks.profile_nsight --backend cute --phase forward --seq 128
```

The public speed command compares installed CuTe with default SDPA; it does not recreate the internal previous-CuTe control or exactly the random-input sequence of the retained paired run. It uses SDPA for its numerical control; the FP64 regression command and GPU tests provide independent accuracy checks. Repeat the profiler command with `--backend sdpa` to inspect actual dispatch; use `--phase joint` for forward plus backward or `--phase backward` for a retained-forward backward range.

## RTX 5090, RTX 4090 and RTX 3090 Ti

All four plotted grids use B=1, H=16, contiguous BHSD, S128/S1024/S4096, D16/64/128/256/512, FP16/BF16, and dense/implicit causal visibility. Default SDPA uses FlashAttention-2 for the smaller heads and CUTLASS memory-efficient attention at D512 in the captured runs. The new reports retain actual eager and graph CUDA activities for every row, including required copies and memsets. The standalone `flash-attn` package was not measured.

![RTX and L40S performance across the latest 60-case grids](performance.svg)

Ratios above one favor CuTe. Counts below compare the observed medians; values close to one can be ties within variability. All four GPUs beat default SDPA in the eight long D512 complete-call cases. The new schedules remain selected specifically on RTX 4090, which also wins all 20 short cases and all eight long D16 cases. Its long D64/D128/D256 complete calls still trail SDPA.

| GPU | Forward wins | Backward wins | F+B wins | Long D512 SDPA / CuTe F+B |
| --- | ---: | ---: | ---: | ---: |
| RTX 5090 | 31/60 | 25/60 | 24/60 | 1.90–4.42× |
| RTX 4090 | 46/60 | 31/60 | 36/60 | 2.09–4.32× |
| RTX 3090 Ti | 18/60 | 24/60 | 20/60 | 1.61–2.71× |
| L40S | 19/60 | 24/60 | 20/60 | 2.70–4.68× |

Representative BF16 S4096 forward + backward, milliseconds:

| GPU | Head dim | Visibility | CuTe F+B | Default SDPA F+B | SDPA / CuTe |
| --- | ---: | --- | ---: | ---: | ---: |
| RTX 5090 | 512 | Dense | 23.408 | 50.095 | 2.14× |
| RTX 5090 | 512 | Causal | 11.511 | 29.991 | 2.61× |
| RTX 4090 | 512 | Dense | 32.045 | 67.136 | 2.10× |
| RTX 4090 | 512 | Causal | 16.225 | 39.596 | 2.44× |
| RTX 3090 Ti | 512 | Dense | 63.654 | 102.242 | 1.61× |
| RTX 3090 Ti | 512 | Causal | 30.965 | 57.699 | 1.86× |
| L40S | 512 | Dense | 26.247 | 72.344 | 2.76× |
| L40S | 512 | Causal | 14.159 | 40.870 | 2.89× |

RTX 4090 BF16 S128/D128 complete latency is **15.54 µs dense and 15.56 µs causal**, versus SDPA's 29.33 and 25.31 µs. Long D16 also wins on that selected schedule; the other three released paths still trail SDPA on long D16. Short D256 remains a complete-call loss on those three cards, despite their matching shared-memory capacities.

### Hardware, software and shared memory

These are separate full devices. The new RTX 5090, RTX 3090 Ti and L40S runs use source revision `d4700f5`, Python 3.13.18, PyTorch 2.15.0+cu132, PyTorch CUDA runtime/toolkit 13.2, CuTe DSL 4.8.1, TVM FFI 0.1.15.post0 and Triton 3.9.0. The retained RTX 4090 run uses those framework/runtime versions and records its own source hashes; its compiler inventory is separate. All four measurements are dated October 30, 2026 UTC.

| GPU | SMs / nominal VRAM | Power limit | Host driver | Active CUDA driver library |
| --- | --- | ---: | --- | --- |
| RTX 5090 | 170 / 32 GB | 600 W | 595.99.02 | 595.99.02 |
| RTX 4090 | 128 / 24 GB | 450 W | 615.71.09 | 615.71.09 |
| RTX 3090 Ti | 84 / 24 GB | 450 W | 595.58.03 | 595.58.03 |
| L40S | 142 / 48 GB | 350 W | 580.65.06 | 595.112.07 |

The L40S host has an R580 kernel driver. NVIDIA's supported `cuda-compat-13-2` package supplies the active 595.112.07 CUDA/PTX-JIT libraries and driver API 13.2, allowing the same PyTorch 2.15+cu132 binary and toolkit as the other new runs. See [NVIDIA's forward-compatibility matrix](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html). The actual loaded library basenames are retained in the report.

PyTorch’s CUDA version is separate from the DSL code generator. All three new devices report the NVIDIA NVVM compiler release **13.4.59 / NVVM 23.0.0**, but the installed DSL library extras and exact emitted PTX differ:

| GPU | DSL 4.8.1 library extras | Representative emitted PTX ISA | Target |
| --- | --- | --- | --- |
| RTX 5090 | base/core/cu12; no cu13 distribution | 8.8 | `sm_120a` |
| RTX 3090 Ti | base/core/cu12/cu13; cu13 IR loaded | 9.4 | `sm_86` |
| L40S | base/core/cu12/cu13; cu13 IR loaded | 9.4 | `sm_89` |

The reports retain distribution versions, loaded library basenames, exact PTX headers and hashes. The 3090 Ti/L40S exports are representative released forward calls collected after the baseline campaign. The 5090 export is the released backward control from a separate same-stack comparison, rather than an export from the primary timing process. The retained 4090 report records emitted CuTe PTX 8.8 / `sm_89` with NVVM 13.4.59; its PTX 9.2 record belongs to the source-matched native SDPA reconstruction, not captured CuTe code. Thus these are qualified per-device stacks, not a homogeneous DSL compiler comparison. CUDA function virtual-target attributes are not interpreted as PTX ISA versions.

The RTX 3090 Ti now runs at 450 W, whereas the older retained run used 285 W. Framework, driver, power, and host changes prevent attributing cross-run differences solely to kernel changes. Compare backends within a measured row.

Verified resource limits; these are capacities, not achieved occupancy:

| GPU | Shared / SM | Opt-in shared / block | 32-bit registers / SM | Max threads / SM |
| --- | ---: | ---: | ---: | ---: |
| RTX 5090 | 100 KiB | 99 KiB | 65,536 | 1,536 |
| RTX 4090 | 100 KiB | 99 KiB | 65,536 | 1,536 |
| RTX 3090 Ti | 100 KiB | 99 KiB | 65,536 | 1,536 |
| L40S | 100 KiB | 99 KiB | 65,536 | 1,536 |

Matching shared memory does not produce matching work distribution. In the recorded BF16 S128/D256 dense forward, released CuTe launches **16 CTAs on RTX 3090 Ti and L40S**, and **32 CTAs on RTX 5090**. Default FA2 launches 64 split-KV forward CTAs plus a combine kernel. The compact launch-resource examples retain actual grid/block dimensions, registers and shared bytes. Profiler occupancy estimates and fractional grid-size ratios are excluded from occupancy claims.

### Measurement and qualification

The three new grids use seven alternating SDPA/CuTe rounds targeting 200 ms per backend and phase, with at least 16 graph replays. The RTX 4090 grid rotates SDPA, its preceding CuTe control and the selected schedule under the same window settings. Compilation, allocation, profiling and telemetry are excluded from those timing windows. Short graph measurements can include host submission gaps. Separately timed forward and backward medians need not sum exactly to complete-call medians.

Every new row checks output, dQ, dK and dV against independent full-length centered FP64 on identical rounded inputs, evaluated in two-head chunks. Each component must stay below 1% norm-relative error for BF16 or 0.5% for FP16, with exact zero when its reference norm is zero. All **180 new grid cases pass**, including changed-input forward/complete graph checks. SDPA is a performance baseline, rather than the numerical oracle.

| GPU | GPU-enabled discovery | Skips | Result |
| --- | ---: | ---: | --- |
| RTX 5090 | 150 | 50 | Pass |
| RTX 4090 | 150 | 33 | Pass |
| RTX 3090 Ti | 150 | 50 | Pass |
| L40S | 150 | 43 | One rectangular BF16 dV failure; see below |

Separate sustained BF16 B1/H16/S4096 checks run D128 dense and D512 causal through four 20-second SDPA/CuTe/CuTe/SDPA passes on each new GPU. Event groups target 200 ms and cap at 100 replays; the first five seconds of latency samples and first 25 board samples are excluded from summaries. The reports retain latency variability, board power, clocks, temperatures and available clock-event reasons. These whole-board observations do not provide precise cycle normalization or HFU. The 4090 report retains its own sustained D16 and D128 checks.

## L40S

The new full 142-SM/48-GB L40S comparison measures current released dispatch at its 350 W board limit, with the same 60-case grid and precision gates as the RTX refresh. Medians improve on SDPA most consistently at D512. Smaller long heads remain slower for complete calls.

| Head dim | S128 SDPA / CuTe F+B range | S1024/S4096 SDPA / CuTe F+B range |
| ---: | ---: | ---: |
| 16 | 1.08–1.11× | 0.60–0.68× |
| 64 | 1.07–1.19× | 0.62–0.73× |
| 128 | 0.87–0.97× | 0.73–0.77× |
| 256 | 0.87–0.91× | 0.81–0.85× |
| 512 | 3.33–3.57× | 2.70–4.68× |

The performance grid passes its independent FP64 and graph gates. Full GPU discovery has one unresolved failure in `test_dynamic_lengths_preserve_forward_configuration_thresholds`: BF16 **B2/H16/Sq1023/Sk17/D128 dV** exceeds the unchanged pointwise `atol=rtol=0.02` gate for one of 69,632 elements. The failing suite element has absolute error 0.0255118 and relative error 0.138132.

A standalone rerun preserves exact operands, outputs and gradients with source hashes. Its dV norm-relative error against independent centered FP64 is **0.2360%**, with one pointwise violation; the other three components pass that pointwise gate. Torch FP64 MATH and the independent centered FP64 dV reference agree to approximately 8e-16 norm-relative error. This is a current-main numerical failure outside the square timing grid. No tolerance was changed, and the failed rectangular call has no successful timing claim in these tables.

## RTX 5090 explicit boolean masks

BF16, B=1/H=16. Mask preparation is included. The causal triangle is supplied as a boolean mask with `is_causal=False`; this is a different backend selection from implicit `is_causal=True`.

| Tokens | Head dim | Boolean mask | CuTe F+B | SDPA F+B | SDPA / CuTe |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1024 | 128 | Causal triangle | 0.294 | 0.843 | 2.87× |
| 4096 | 128 | All true | 4.041 | 10.067 | 2.49× |
| 4096 | 128 | Causal triangle | 2.318 | 10.050 | 4.33× |
| 1024 | 256 | Causal triangle | 0.545 | 2.575 | 4.72× |
| 4096 | 256 | Causal triangle | 6.052 | 24.294 | 4.01× |
| 1024 | 512 | Causal triangle | 2.306 | 5.095 | 2.21× |
| 4096 | 512 | Causal triangle | 32.895 | 48.754 | 1.48× |

All seven tested explicit-mask cases favor CuTe. The triangle lets CuTe skip fully hidden tiles; the all-true control also favors CuTe. These results do not establish performance for arbitrary mask densities or layouts. Retained October 21 measurements use PyTorch 2.11.0+cu130, CUDA 13.0, CuTe DSL 4.8.1 and driver 580.178.04, with five rotated rounds of 30 replays. The current unmasked optimizations retain these mask paths.

## B300 MIG versus KohakuFA Gluon

**B300 MIG 1g.34gb, 18 SMs**, BF16 D512, B=1/H=16. Gluon is upstream KohakuFA `0c5ede6859307bedc0c6089523c16b1ca51b7da9`, unmodified and supplied with the measured device tuning tables. CuTe retains its centered FP32 correction. This compares complete attention calls on the same partition.

| Tokens | Visibility | CuTe F+B | Gluon F+B |
| ---: | --- | ---: | ---: |
| 512 | Causal | 0.486 | 0.420 |
| 1024 | Causal | 1.293 | 1.147 |
| 2048 | Causal | 4.020 | 3.682 |
| 4096 | Causal | 13.649 | 13.703 |
| 8192 | Causal | 50.766 | 51.276 |
| 512 | Dense | 0.612 | 0.512 |
| 4096 | Dense | 24.963 | 24.941 |

The largest causal cases are approximately at parity (within 1% at S4096/S8192); smaller shapes still favor Gluon. The full JSON includes 18 cases, including FP16, D256, and different batch/head counts. It retains the published medians; individual replay samples for this dataset are in the external profiling evidence, rather than duplicated here.

Measured October 21, 2026 with PyTorch 2.12.0+cu130, CUDA 13.0, and CuTe DSL 4.8.1. Four rotated rounds of 20 CUDA-graph replays, with output/gradient comparisons and eager/graph checks. Source hashes identify the final vector-staging implementation. These are partition measurements. Full-device B200/B300 performance after the later shared-kernel changes remains unmeasured.

## H100

### Clef 27B text inference

The non-Flash [Clef 27B](https://huggingface.co/Cloudflare/clef) snapshot `a30162b99d68a481d8bd75d38df069a0cf83df40` uses causal D256 GQA24/4 in 16 backbone layers. These forward comparisons use its actual BF16 projection tensors and strides, captured from synthetic three/64-question records with 256 state words. Its 48 linear-attention layers and native MHA fastpaths retain their implementations. The schema head uses dense D64 attention with 16 query/KV heads.

Median milliseconds per attention call on a 132-SM/80-GB H100 at its 700 W limit. Each row compares all backends on the same GPU in one process. “Prior CuTe” uses the pre-Hopper device kernels. Default SDPA selects cuDNN's generated Hopper WGMMA kernel; forced FlashAttention uses PyTorch's FlashAttention-2 backend.

| Backbone tokens | Prior CuTe | Tuned CuTe | Default SDPA | Forced FlashAttention | SDPA / tuned |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 787 | 0.068326 | 0.027910 | 0.034655 | 0.051895 | 1.242× |
| 9835 | 3.183050 | 1.890106 | 1.945790 | 4.015904 | 1.029× |

Tuned CuTe beats cuDNN on **both captured shapes in all 15 paired rounds**. Latency falls **59.2% and 40.6% versus prior CuTe**, and **19.5% and 2.9% versus cuDNN**. A separate H100 reproduces both wins over 12 balanced rounds. This covers BF16, unmasked causal D256 GQA24/4 forward attention; it does not establish a general Hopper or backward speedup.

A broader ten-case grid covers B1/S128–16384 and two rectangular B1/B2 cases. It improves prior CuTe throughout and beats default SDPA in **7/10** cases. **S128/S256/S512 still favor default SDPA**, with SDPA/CuTe ratios of 0.883/0.922/0.933. An existing serial M64/N128 alternative was measured and rejected on all three. CuTe beats forced FlashAttention on all eight square cases; PyTorch's forced FlashAttention backend rejects the two rectangular causal shapes. The JSON retains each grid case's samples and protocol.

| Schema query × key tokens | Prior CuTe | Tuned CuTe | Default SDPA | Forced FlashAttention |
| --- | ---: | ---: | ---: | ---: |
| 12 × 787 | 0.016118 | 0.006596 | 0.013772 | 0.010478 |
| 3 × 787 | 0.016379 | 0.006185 | 0.013694 | 0.010389 |
| 256 × 9835 | 0.114041 | 0.061661 | 0.099247 | 0.046007 |
| 64 × 9835 | 0.113677 | 0.023108 | 0.098930 | 0.024376 |

The unchanged split-forward schema path beats default cuDNN by **1.61–4.28×**, including its FP32 partial-output merge. Forced FlashAttention wins the 256-query case. Hopper uses 64-row/128-column tiles above 32 queries. These schema measurements use seven rotating rounds with 100-ms backend windows; they have a separate protocol from the backbone table.

#### What the cuDNN comparison changed

Nsight Systems 2025.3.2 identifies actual loaded kernels; Nsight Compute 2025.3.1 supplies instruction counts, resources and sampled stalls. Native loaded SASS and CuTe PTX/cubins are retained; native PTX was unavailable. Profiling runs separately from timing, with unfixed clocks. These observations led to four changes:

- **128-key tiles:** two 64-query WGMMA consumer groups share independent 64-KiB K and V buffers. Separate producer warps refill K after QK and V after PV, instead of coupling their reuse. This keeps the payload within the earlier 192-KiB shared-memory budget.
- **Softmax overlap:** synchronization orders the groups' softmax sections so one can perform scalar work while the other issues tensor operations. Every consumer warp contributes to the phase token; memory and race checks caught and eliminated a faster but unsafe single-warp handoff prototype.
- **Less scalar work:** positive/default-scale selection moves outside the tile loop, and denominator reductions move to the epilogue. The default scale folds two FP32 multiplies into one before `exp2`; an exhaustive device check found zero probability differences over all 2,139,095,040 finite FP32 bit patterns with their sign bit set, including negative zero and subnormals. Other scales retain the general expression. The compensated FP32 reciprocal epilogue preserves infinity/NaN propagation.
- **Earlier long blocks:** the grid schedules heads before reversed causal query tiles, bringing the longest blocks into the first scheduling wave. Device timestamps confirmed that this reduces the short input's execution tail. Persistence alone and several lower-instruction-count variants regressed and were rejected.

Long captured call, dynamic warp instruction totals unless stated otherwise:

| Measurement | Tuned CuTe | cuDNN |
| --- | ---: | ---: |
| HGMMA | 13,837,824 | 13,837,824 |
| FMUL | 113,748,096 | 113,749,152 |
| SHFL | 2,365,440 | 2,385,240 |
| Compute (SM) throughput | 78.77% | 75.48% |
| Launch registers per thread | 168 | 168 |
| Dynamic shared memory per block | 197,632 B | 232,448 B |
| Spilling requests | 0 | 0 |

CuTe still executes more handoff/polling instructions: 18.94 million SYNCS versus 3.42 million, and 7.74 million NANOSLEEP versus 0.36 million. It launches 1,848 blocks versus cuDNN's 132 persistent blocks. These are remaining design differences, **not percentages of time**. Extra producer warps parked at the final barrier also inflate active-warp counters, so those counters alone cannot establish productive occupancy.

#### Correctness and model limits

All **65,261,568 actual backbone output values** pass the unchanged BF16 `.02/.02` absolute/relative bounds against a chunked FP64 oracle. Captured graphs match eager results and respond to changed inputs. Four Hopper tests pass: 21 serial-control output/gradient/statistics configurations, strict FP64 outputs and all gradients over six signed/zero/tiny scales, changed Q/K/V graphs, and infinity/NaN propagation. Denominator reassociation can change BF16 rounding ties; the new kernel is not bitwise equivalent to prior CuTe.

Compute Sanitizer memcheck and racecheck both pass 15 signed/zero-scale tail configurations with zero errors, hazards or warnings. CPU discovery passes 116 tests with 65 device skips. Full GPU discovery retains one extreme-logit FP16 dQ/dK tolerance failure, reproduced bitwise on prior CuTe; 20 setup errors from missing Git and test support files were resolved before those tests were rerun successfully. An expanded FP64 stress check also finds one BF16 dK tolerance violation at Sq257/Sk129/scale −0.125 in both the new and serial64/serial128 controls. These failures and exact inputs remain recorded; bounds were not relaxed.

**Whole-model compatibility has not passed.** Against prior CuTe, short/long-record maximum logit differences are 0.02734/0.42578, and the long record changes one decision. Both fail the unchanged model logit gate, so no new whole-model timing or speedup is claimed. Small attention rounding differences accumulate through the BF16 backbone.

A separate reference replaces all 22 explicit SDPA calls per record with FP64 QK/softmax/PV, rounding attention outputs back to BF16. Weights, linear attention and native MHA remain unchanged. On the long record, native SDPA and prior CuTe each disagree with three reference decisions, versus two for tuned CuTe; tuned CuTe also has the larger maximum logit error (0.49219 versus 0.26172/0.27734). These mixed results on unlabeled records do not establish better task accuracy. Schema-only replacement retains its earlier six-record validation: all attention/logit bounds pass, with one changed decision among 201 on a native BF16 tie; prior and tuned schema logits agree in that case.

PyTorch 2.11.0+cu130, CUDA 13.0, CuTe DSL 4.8.1, TVM FFI 0.1.15.post0, driver 595.71.05. Backbone timing uses 15 rotating rounds, balanced across five backend positions, 32 calls per CUDA graph, at least 500-ms backend windows, and at least five seconds of preheating after correctness checks. Compilation, profilers and task-owned telemetry are excluded from timed windows. Source identifiers, every timing round, numerical gates and external evidence hashes are retained in the JSON.

### Retained training comparison

Last published dense BF16 results, B=1/H=16/S=4096; the JSON also covers S1024. These training cases favor native SDPA; CuTe provides the stronger backward calculation described in the [precision guide](../docs/precision.md).

| GPU | Head dim | CuTe F+B | Default SDPA F+B |
| --- | ---: | ---: | ---: |
| H100 | 64 | 0.866 | 0.574 |
| H100 | 128 | 1.300 | 0.927 |
| H100 | 256 | 4.710 | 2.469 |

Six rotating rounds of 100 graph replays, PyTorch 2.11.0+cu128. The original reports did not identify the default SDPA backend, so no FlashAttention-specific claim is made for these rows.

## Reproduce

Install the [CUDA dependencies](../README.md#install), then run on the target GPU:

```bash
# Matched default SDPA and CuTe, including separate forward and F+B times.
python -m benchmarks.speed --dtype bf16 --seq 1024 4096 --dim 64 128 256 512 \
  --backends sdpa cute --iterations 100 --repeats 6 --output speed.json
# Repeat with --dtype fp16 and/or --causal for those input families.
python -m benchmarks.speed --dtype bf16 --causal --seq 1024 4096 --dim 512 \
  --backends sdpa cute --iterations 100 --repeats 6 --output causal-speed.json
# With upstream KohakuFA installed on a supported device:
python -m benchmarks.speed --dtype bf16 --seq 1024 4096 --dim 512 \
  --backends sdpa cute kohakufa --output upstream-speed.json
python -m benchmarks.precision --dtype bf16 --seq 256 --dim 128 \
  --backends auto kohakufa-cute --output precision.json
# Regenerate the figure from the four per-device JSON reports in results/.
python -m pip install '.[plots]'
python -m benchmarks.plot_performance
```

These commands rerun the general timing/precision protocol on the installed stack; they do not recreate every historical host or the specialized explicit-mask driver. Check actual SDPA dispatch with `benchmarks.profile_attention` when interpreting a new result. GPU captures, PTX/SASS, rental records, and exploratory reports belong outside the package; the repository keeps the usage comparisons above.

### Reproduce the new RTX and L40S grids

Run from a source checkout with the CUDA extras installed. The defaults cover all 60 cases and use seven alternating 200-ms rounds. Profiling passes run after each row's uninstrumented timing and write raw traces outside the tracked tree.

```bash
python -m benchmarks.compare_devices --expect-device 'NVIDIA GeForce RTX 5090' \
  --output ../rtx5090.json --profiles ../rtx5090-profiles
python -m benchmarks.compare_devices --expect-device 'NVIDIA GeForce RTX 3090 Ti' \
  --output ../rtx3090ti.json --profiles ../rtx3090ti-profiles
python -m benchmarks.compare_devices --expect-device 'NVIDIA L40S' \
  --output ../l40s.json --profiles ../l40s-profiles
```

The retained reports add the separate suite, launch-resource, environment and sustained evidence. Their runtime and benchmark source hashes identify the qualified code. The [4090 analysis](../docs/rtx4090-tuning.md#reproduce) retains its preceding-control reproduction command.

### Clef 27B captures and replay

Download the pinned model snapshot outside the source tree. Its model loader additionally requires Transformers 5.12.1, flash-linear-attention 0.5.2, causal-conv1d 1.7.0, Accelerate, Pillow, Hugging Face Hub and Safetensors. The CUDA extensions must support the installed PyTorch/CUDA stack.

Use the pinned Clef 27B snapshot and explicit model metadata. Unmodified BF16 weights occupy about 54 GB before activations; the measured device has 80 GB.

```bash
hf download Cloudflare/clef \
  --revision a30162b99d68a481d8bd75d38df069a0cf83df40 \
  --local-dir ../clef-27b-weights
python -m benchmarks.clef_forward --model ../clef-27b-weights \
  --model-id Cloudflare/clef --revision a30162b99d68a481d8bd75d38df069a0cf83df40 \
  --output ../clef-27b-captures --state-words 256 --questions 3 64 \
  --backends native --capture-tensors --rounds 1 --repeat 1
python -m benchmarks.clef_forward --captures ../clef-27b-captures \
  --output ../clef-27b-head --attention-dim 64 --rounds 7 \
  --calls-per-graph 32 --window-ms 100
python -m benchmarks.clef_forward --captures ../clef-27b-captures \
  --output ../clef-27b-backbone --attention-dim 256 --rounds 15 \
  --calls-per-graph 32 --window-ms 500 --preheat-seconds 5
python -m benchmarks.clef_forward --model ../clef-27b-weights \
  --model-id Cloudflare/clef --revision a30162b99d68a481d8bd75d38df069a0cf83df40 \
  --output ../clef-27b-model-head --state-words 16 256 2048 --questions 3 64 \
  --backends native cute-head --rounds 3 --repeat 5
```

Use fresh output directories. Replay restores strides and storage offsets, validates every output against a chunked FP64 oracle, checks that captured graphs respond to changed values, and rotates native SDPA/CuTe timing windows. Add `--dtype fp16` for casts of the captures; this does not measure an FP16 model. Replay compares the installed CuTe version with SDPA; reproducing the old-main control requires a separate checkout of the recorded source. The model harness records numerical failures and excludes failed variants from timing rather than changing its correctness bounds.

The public replay uses default SDPA. To check its stronger FlashAttention alternative separately, run the SDPA call inside `torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION)` and verify the actual loaded kernel. The retained H100 comparisons include both backends.
