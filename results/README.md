# Performance guide

[performance.json](performance.json) holds the latest retained comparisons, software versions, source identifiers, and timing samples where available. The current consumer GPU grid replaces the earlier unmasked RTX 5090 runs.

All times below are median milliseconds per attention call. **Forward + backward includes forward.** Timings use CUDA events around CUDA-graph replay, exclude compilation, and rotate backend order. Profiling measurements are excluded from these tables. Compare backends within a row: hosts, software, and replay counts differ between datasets. Full-model throughput and first-use compilation cost require separate measurements.

## RTX 5090, RTX 4090 and RTX 3090 Ti

Each GPU has a complete 32-case unmasked grid: B=1, H=16, matching contiguous BHSD inputs, S1024/S4096, D64/128/256/512, FP16/BF16, dense/implicit causal. Actual default PyTorch dispatch is **FlashAttention-2 at D64/128/256** and **CUTLASS efficient attention at D512**. “SDPA” means that actual default. Forced PyTorch FlashAttention-2 controls on the Ti select the same smaller-head kernels; the standalone `flash-attn` package was not measured.

![Current consumer GPU performance across the complete unmasked grids](performance.svg)

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

No profiler or telemetry collector runs in timed windows. Graph outputs/gradients are checked against eager execution. Ti backward-only retains its forward graph; separately timed phases need not sum exactly to a complete-call median. Source hashes identify the architecture-specific snapshots used for timing. The integrated source passes the full GPU suite on each card; all 128 device functions across each 32-case grid match the timed snapshot in encoded instructions and resource records. The JSON retains the integrated source and comparison hashes. These device results do not establish performance on other cards sharing the same architecture. In particular, the Ti result describes its recorded 285 W limit.

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

## H100 and L40S

Last published dense BF16 results, B=1/H=16/S=4096. L40S has not been remeasured after the current SM89 changes; its rows retain their recorded source identifier. The full JSON also covers S1024. These training cases favor native SDPA; CuTe provides the stronger backward calculation described in the [precision guide](../docs/precision.md).

| GPU | Head dim | CuTe F+B | Default SDPA F+B |
| --- | ---: | ---: | ---: |
| H100 | 64 | 0.866 | 0.574 |
| H100 | 128 | 1.300 | 0.927 |
| H100 | 256 | 4.710 | 2.469 |
| L40S | 64 | 1.911 | 1.242 |
| L40S | 128 | 3.547 | 2.721 |
| L40S | 256 | 10.517 | 7.888 |

Six rotating rounds of 100 graph replays. H100 uses PyTorch 2.11.0+cu128; L40S uses the Torch version recorded in the JSON. The original reports did not identify the default SDPA backend, so no FlashAttention-specific claim is made for these rows.

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
