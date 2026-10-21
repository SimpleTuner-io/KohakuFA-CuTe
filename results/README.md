# Performance guide

These are the latest retained measurements for each input family, consolidated from completed runs. This documentation update does not rerun GPU benchmarks. [performance.json](performance.json) holds all **69 comparisons**, software versions, source identifiers, and timing samples where available.

All times below are median milliseconds per attention call. **Forward + backward includes forward.** Timings use CUDA events around CUDA-graph replay, exclude compilation, and rotate backend order. Profiling measurements are excluded from these tables. Compare backends within a row: hosts, software, and replay counts differ between datasets. Full-model throughput and first-use compilation cost require separate measurements.

## RTX 5090

B=1, H=16, matching contiguous BHSD inputs. The unmasked grid covers S1024/S4096, D64/128/256/512, FP16/BF16, dense/implicit causal. Actual default PyTorch dispatch is **FlashAttention-2 at D64/128/256** and **CUTLASS efficient attention at D512**. Explicit boolean masks in the tested cases select efficient attention. “SDPA” means that actual default, rather than a forced slower backend.

![RTX 5090 performance across the complete unmasked grid](performance.svg)

Ratios above 1 favor CuTe. Forward alone wins in **19/32** tested unmasked cases; forward + backward wins in **8/32**, all D512. Smaller unmasked heads still favor FlashAttention-2 for the complete call. The explicit-mask comparisons below form a separate dataset.

### D512: faster forward and backward

| Tokens | Dtype | Visibility | CuTe F+B | SDPA F+B | SDPA / CuTe |
| ---: | --- | --- | ---: | ---: | ---: |
| 1024 | BF16 | Dense | 1.877 | 5.068 | 2.70× |
| 1024 | BF16 | Causal | 1.028 | 3.971 | 3.86× |
| 1024 | FP16 | Dense | 1.907 | 3.232 | 1.69× |
| 1024 | FP16 | Causal | 1.038 | 1.895 | 1.82× |
| 4096 | BF16 | Dense | 27.802 | 48.211 | 1.73× |
| 4096 | BF16 | Causal | 13.194 | 29.634 | 2.25× |
| 4096 | FP16 | Dense | 28.414 | 47.780 | 1.68× |
| 4096 | FP16 | Causal | 13.314 | 21.296 | 1.60× |

CuTe uses **37–74% less latency** than SDPA in these eight cases. The selected full-width tiles share score work across all 512 output columns. Boolean masks use a separate path, measured below.

D512 measurements: October 21, 2026; 170-SM/32-GB RTX 5090, 600 W limit, driver 595.71.05, PyTorch 2.11.0+cu130, CUDA 13.0, CuTe DSL 4.8.1, Triton 3.6.0, TVM FFI 0.1.15.post0. Five rotated rounds target 1-second F+B and 0.5-second forward windows, with at least 32 replays. All samples are retained; graph outputs/gradients are checked against eager execution. No profiler or telemetry collector runs in the timed windows.

D64/128/256 measurements use the same Torch/CUDA/DSL versions, driver 580.178.04, seven rotated rounds of 30 measured replays after 30 warmups. Their selected paths are unchanged by the later D512 optimization. This is a separate host run; its absolute times are not paired with the D512 run.

### Explicit boolean masks

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

All seven tested explicit-mask cases favor CuTe. The triangle lets CuTe skip fully hidden tiles; the all-true control also favors CuTe. These results do not establish performance for arbitrary mask densities or layouts. Five rotated rounds of 30 replays use the smaller-head software/driver stack above. Subsequent unmasked optimizations retain these mask paths.

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

Last published dense BF16 results, B=1/H=16/S=4096. The full JSON also covers S1024. These training cases favor native SDPA; CuTe provides the stronger backward calculation described in the [precision guide](../docs/precision.md).

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
