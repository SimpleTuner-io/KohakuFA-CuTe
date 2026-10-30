# Performance guide

[performance.json](performance.json) retains the RTX and L40S comparisons, software versions, source identifiers, and timing samples where available. [gb10.json](gb10.json) retains the GB10 comparison separately.

The [RTX 4090 scheduling follow-up](../docs/rtx4090-tuning.md) and [comparison data](rtx4090.json) add same-process PyTorch 2.15 comparisons, all five head dimensions at S128/S1024/S4096, and the remaining backward arithmetic gap. The RTX grid and figure below retain their October 22 measurement conditions; use the follow-up for the newer 4090 schedules.

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

Each GPU has a complete 32-case unmasked grid: B=1, H=16, matching contiguous BHSD inputs, S1024/S4096, D64/128/256/512, FP16/BF16, dense/implicit causal. Actual default PyTorch dispatch is **FlashAttention-2 at D64/128/256** and **CUTLASS efficient attention at D512**. “SDPA” means that actual default. Forced PyTorch FlashAttention-2 controls on the Ti select the same smaller-head kernels; the standalone `flash-attn` package was not measured.

![Current RTX and L40S performance across the complete unmasked grids](performance.svg)

Ratios above 1 favor CuTe. All three GPUs beat default SDPA for complete D512 calls; **smaller unmasked heads still favor FlashAttention-2 for the complete call**.

| GPU | Forward wins | F+B wins | D512 SDPA / CuTe F+B |
| --- | ---: | ---: | ---: |
| RTX 5090 | 18/32 | 8/32 | 1.84–4.41× |
| RTX 4090 | 10/32 | 8/32 | 1.91–3.94× |
| RTX 3090 Ti | 10/32 | 8/32 | 1.67–2.53× |

Representative BF16 S4096 complete calls:

| GPU | Head dim | Visibility | CuTe F+B | SDPA F+B | SDPA / CuTe |
| --- | ---: | --- | ---: | ---: | ---: |
| RTX 5090 | 512 | Dense | 23.175 | 48.215 | 2.08× |
| RTX 5090 | 512 | Causal | 11.449 | 29.683 | 2.59× |
| RTX 4090 | 512 | Dense | 31.541 | 60.833 | 1.93× |
| RTX 4090 | 512 | Causal | 16.011 | 35.765 | 2.23× |
| RTX 3090 Ti | 512 | Dense | 68.507 | 114.648 | 1.67× |
| RTX 3090 Ti | 512 | Causal | 33.843 | 63.758 | 1.88× |

The selected full-width D512 tiles share score work across output columns. Backward overlaps full-order QK and dO·V in separate warp groups, retains stationary V and packs DS/P shared stores. Wider operand copies are selected on SM86/SM89; measured cache policy and a causal D128 query layout are selected separately on SM120. These choices retain the centered FP32 corrections described in the [precision guide](../docs/precision.md).

Against paired previous-main controls, RTX 4090 F+B latency falls **75.6–78.1% at D512** and **13.1–23.1% at D256**. RTX 5090 D512 backward falls **15.9–20.2%**, and F+B falls **12.8–16.8%**; D128 F+B falls **0.2–15.9%**. The 5090 D256 cache-policy change is small and mixed: **0.36% slower to 2.12% faster** for F+B. D64 retains its existing 5090 selection. SM86 is newly supported, so Ti performance is compared with default SDPA rather than claiming a speedup over unsupported main.

### Measurement conditions

Measured October 22, 2026 UTC with PyTorch 2.11.0+cu130, CUDA 13.0, CuTe DSL 4.8.1 and TVM FFI 0.1.15.post0:

| GPU | SMs / memory | Power limit | Driver | Five-round timing windows |
| --- | --- | ---: | --- | --- |
| RTX 5090 | 170 / 32 GB | 600 W | 595.71.05 | 0.5 s forward; 1 s backward/F+B; minimum 32 replays |
| RTX 4090 | 128 / 24 GB | 450 W | 580.173.02 | 0.5 s forward; 1 s backward/F+B; minimum 32 replays |
| RTX 3090 Ti | 84 / 24 GB | **285 W** | 595.91.07 | 0.75 s per phase; minimum eight replays |

No profiler or telemetry collector runs in timed windows. Graph outputs/gradients are checked against eager execution. Ti backward-only retains its forward graph; separately timed phases need not sum exactly to a complete-call median. Source hashes identify the architecture-specific snapshots used for timing. On the three RTX cards, the integrated source passed the full GPU suite; all 128 device functions across each 32-case grid match the timed snapshot in encoded instructions and resource records. The JSON retains the integrated source and comparison hashes. These device results do not establish performance on other cards sharing the same architecture. In particular, the Ti result describes its recorded 285 W limit.

## L40S

Current main (`c0786dd`) is compared with the pre-Ada-tuning kernels (`b33ee8f`) and default SDPA on the **same 142-SM/48-GB L40S at its stock 350 W limit**. The 32-case grid uses B=1/H=16, contiguous BHSD, S1024/S4096, D64/128/256/512, FP16/BF16 and dense/implicit causal attention. Actual default SDPA selects FlashAttention-2 at D64/128/256 and CUTLASS efficient attention at D512.

Complete-call ratios across all eight cases per head dimension; values above 1 favor current CuTe:

| Head dim | Pre-tuning / current CuTe F+B | SDPA / CuTe F+B |
| ---: | ---: | ---: |
| 64 | 1.006–1.022× | 0.59–0.72× |
| 128 | 1.000–1.082× | 0.71–0.75× |
| 256 | 1.124–1.199× | 0.76–0.85× |
| 512 | 3.084–3.622× | 2.50–4.20× |

BF16 S4096 dense, milliseconds:

| Head dim | Pre-tuning CuTe | Current CuTe | Default SDPA | SDPA / CuTe |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 2.610 | 2.560 | 1.506 | 0.59× |
| 128 | 4.906 | 4.648 | 3.363 | 0.72× |
| 256 | 12.198 | 10.176 | 7.901 | 0.78× |
| 512 | 93.902 | 26.523 | 67.186 | 2.53× |

Forward wins in **8/32** cases; backward alone wins in **8/32**; forward + backward wins in **8/32**. The complete grid and separate backward samples are retained in the JSON; the figure above shows forward and complete-call comparisons. These timings describe individual attention calls.

Measured October 22, 2026 UTC with PyTorch 2.11.0+cu130, CUDA 13.0, CuTe DSL 4.8.1, TVM FFI 0.1.15.post0 and driver 595.91.07. Five rotated CUDA-graph rounds target 250-ms forward and 500-ms backward/F+B windows, with at least 32 replays. Compilation, profiling and task-owned telemetry are excluded from timed windows. Graph outputs/gradients match eager execution. All 224 FP64 cases and 16 changed-input graph checks passed. The unmodified 100-test suite completed with 19 skips and one existing BF16 dV tolerance failure at B2/H16/Sq1023/Sk17/D128. All outputs and gradients in that failing case are bitwise identical on pre-tuning kernels and fresh current main. Exact inputs and diagnostics are retained with the evidence; no accuracy tolerance was changed.

### Clef-Flash text inference

Forward-only comparisons on 142-SM L40S devices at their stock 350 W limit, using actual Q/K/V tensors captured from [Clef-Flash](https://huggingface.co/Cloudflare/clef-flash) revision `17f0b0ad64efb65d273590632833508766b2aae6`. Backbone, schema and model investigations use separate devices; each comparison within a row uses the same GPU. Two synthetic records contain three or 64 four-choice questions and 256 state words, producing 787 or 9,835 input tokens. The eight full-attention backbone layers use causal D256 GQA16/4. The schema head uses dense D64 attention with short queries into that input sequence. Projection strides are preserved. The model's 24 linear-attention blocks and native MHA self-attention fastpath keep their existing implementations.

BF16, B=1; median milliseconds per forward attention call, including the split-forward merge. Projections, tokenization, and the rest of the model are excluded from this table:

| Query × key tokens | Head dim | Q/KV heads | Main CuTe | Tuned CuTe | Default SDPA | SDPA / tuned |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 787 × 787 | 256 | 16/4 | 0.083867 | 0.064955 | 0.073119 | 1.13× |
| 9835 × 9835 | 256 | 16/4 | 4.697606 | 4.189800 | 4.197516 | 1.00× |
| 12 × 787 | 64 | 16/16 | 0.017716 | 0.005214 | 0.007848 | 1.51× |
| 3 × 787 | 64 | 16/16 | 0.017919 | 0.004951 | 0.007858 | 1.59× |
| 256 × 9835 | 64 | 16/16 | 0.168197 | 0.061462 | 0.064517 | 1.05× |
| 64 × 9835 | 64 | 16/16 | 0.157460 | 0.021609 | 0.025747 | 1.19× |

Default SDPA selects FlashAttention-2. All four schema shapes beat it in all seven paired rounds, for both BF16 and FP16 casts of the captured inputs. The JSON retains all eight schema rows. The D256 short case wins all five rounds; the long case is near parity, with round ratios on both sides of one. Against main CuTe, D256 latency falls 22.5% and 10.8% on these two shapes. A separate 13-case B1/B2, S128–16384 grid improves main throughout; several power-of-two cases still favor SDPA. These results do not establish gains for every shape on SM89 or for the vision encoder.

The schema kernel partitions the key sequence across thread blocks, retains FP32 partial outputs, and merges backward-compatible maxima and sums. Queries above 64 use 128-row tiles. The larger schema tiles and D256 path stagger V loads across QK compute and next-K loads across softmax/PV, while complete tiles bypass per-element boundary checks. D256 also uses coalesced copies and schedules longer causal query blocks first. Its values, softmax statistics, and gradients retain main's per-row arithmetic order and match the serial control bitwise in the regression checks.

PyTorch 2.11.0+cu130, CUDA 13.0, CuTe DSL 4.8.1, TVM FFI 0.1.15.post0, driver 595.71.05. Schema timings use seven rotating rounds, 32 calls per CUDA graph and at least 100 ms per backend window. Old-main schema controls run separately on the same GPU and inputs under that protocol, rotating with native SDPA. D256 uses one call per graph and five rotating main/candidate/SDPA rounds targeting 250 ms, with at least 32 replays. Compilation and task-owned profiling/telemetry are excluded from all timing windows. Nsight Systems, actual loaded native kernel names, PTX/SASS resources, and controlled layout experiments informed the changes. Nsight Compute counters were unavailable because the driver resource was busy.

Validation retains the existing FP16 `.003/.003` and BF16 `.02/.02` absolute/relative bounds. Schema checks cover 96 parameter configurations, 192 changed-input graph replays, 22 adversarial cases and 23 cache boundaries. D256 checks cover exact serial equivalence across 18 configurations, changed-input graphs, a signed-scale FP64 oracle, and all 43,507,712 output values in the two actual backbone captures.

Full-model measurements and final integrated-suite results are recorded with the consolidated dataset. Whole-model gains are much smaller than the attention ratios: GEMMs and linear-attention work dominate these records, while softmax attention occupies roughly 1–3% of the warmed Nsight ranges. Model timing includes the encoded batch through device logits and excludes tokenization and collation. Full-model timing uses normal eager execution rather than the attention-only CUDA-graph protocol.

Whole-model BF16 medians in milliseconds, three rotating rounds of five forwards, with native MHA fastpaths retained:

| Questions / tokens | Native SDPA | Main CuTe | Tuned CuTe |
| --- | ---: | ---: | ---: |
| 3 / 787 | 81.248 | 81.185 | 81.175 |
| 64 / 9835 | 1017.795 | 1021.774 | 1020.018 |

Timing distributions overlap. These samples establish neither a reliable end-to-end SDPA win nor a large model speedup over main.

Head-only replacement passes all six records' unchanged attention/logit gates and preserves all 201 decisions, including four held-out records. Backbone replacement preserves existing CuTe results bitwise, but existing CuTe and native SDPA can exceed those model-level bounds after BF16 rounding accumulates through the backbone. Both implementations pass the attention-level FP64 checks. Zero observed decision changes do not establish full-model numerical equivalence to native SDPA; the failed comparisons remain in the evidence.

Local Python 3.13 CPU discovery completes 111 tests with 62 device skips and no failures. Full L40S discovery runs 109 tests with 19 skips, one baseline-identical D128 dV tolerance failure and no setup errors; its exit status remains a failure. After fixing stale native-build reuse, four affected GPU tests, four CPU tests, all six BF16 attention replays, the head-only native model gates, and the combined model comparison against main pass with the source-keyed binary. The replay verifies full FP64 outputs and changed-input graph behavior. Its two-round timing samples are retained separately from the table's qualification runs; the long backbone remains variable and has no established SDPA advantage. No correctness tolerance was changed.

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
# Regenerate the comparison figure from the retained timing data.
python -m pip install '.[plots]'
python -m benchmarks.plot_performance
```

These commands rerun the general timing/precision protocol on the installed stack; they do not recreate every historical host or the specialized explicit-mask driver. Check actual SDPA dispatch with `benchmarks.profile_attention` when interpreting a new result. GPU captures, PTX/SASS, rental records, and exploratory reports belong outside the package; the repository keeps the usage comparisons above.

### Clef captures and replay

Download the pinned model snapshot outside the source tree. Its model loader additionally requires Transformers 5.12.1, flash-linear-attention 0.5.2, causal-conv1d 1.7.0, Accelerate, Pillow, Hugging Face Hub and Safetensors. The CUDA extensions must support the installed PyTorch/CUDA stack. The measured model uses unmodified BF16 weights and occupies about 19 GB before activations.

```bash
hf download Cloudflare/clef-flash \
  --revision 17f0b0ad64efb65d273590632833508766b2aae6 \
  --local-dir ../clef-flash-weights
# Capture the six distinct SDPA signatures and their actual projection layouts.
python -m benchmarks.clef_forward --model ../clef-flash-weights \
  --output ../clef-captures --state-words 256 --questions 3 64 \
  --backends native --capture-tensors --rounds 1 --repeat 1
# Replay schema attention without loading the model weights.
python -m benchmarks.clef_forward --captures ../clef-captures \
  --output ../clef-head --attention-dim 64 --rounds 7 \
  --calls-per-graph 32 --window-ms 100
# Replay the two causal GQA backbone shapes.
python -m benchmarks.clef_forward --captures ../clef-captures \
  --output ../clef-backbone --attention-dim 256 --rounds 5 \
  --calls-per-graph 1 --window-ms 250
# Measure model-level head replacement separately from backbone replacement.
python -m benchmarks.clef_forward --model ../clef-flash-weights \
  --output ../clef-model-head --state-words 256 --questions 3 64 \
  --backends native cute-head --rounds 3 --repeat 5
```

Use fresh output directories. Replay restores strides and storage offsets, validates every output against a chunked FP64 oracle, checks that captured graphs respond to changed values, and rotates native SDPA/CuTe timing windows. Add `--dtype fp16` for casts of the captures; this does not measure an FP16 model. Replay compares the installed CuTe version with SDPA; reproducing the old-main control requires a separate checkout of the recorded source. The model harness records numerical failures and excludes failed variants from timing rather than changing its correctness bounds.

For the non-Flash 27B model, use its own pinned snapshot and explicit model metadata. Unmodified BF16 weights occupy about 54 GB before activations; the measured device has 80 GB.

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

The public replay uses default SDPA. To check its stronger FlashAttention alternative separately, run the SDPA call inside `torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION)` and verify the actual loaded kernel. The retained H100 comparisons include both backends.
