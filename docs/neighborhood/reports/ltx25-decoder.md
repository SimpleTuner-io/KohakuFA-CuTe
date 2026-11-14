# LTX 2.5 complete VAE decoding on RTX 5090

The [ComfyUI integration refresh](https://github.com/SimpleTuner-io/ComfyUI-CuTe-Attention/blob/f185a2dc2d857ae0037a40196887fda2f77bdf96/results/ltx25-na3d-2026-11-14.md)
measured **1.29–1.34× faster complete VAE decoding** with the neighborhood-union
traversal and native launcher merged in PR #110. This is 22.5–25.6% lower warm
decode latency for one BF16, 384×256, 33-frame video on RTX 5090.
Diffusion sampling and video encoding are outside this measurement.

The [imported receipt](../../../results/neighborhood/ltx25-decoder.json) retains
the decode samples, pixel comparisons, source hashes and environment from the
published integration run. This documentation adds no new GPU measurements.
The separate [four-GPU operator study](comfy-kitchen-3d.md) measures attention
calls; its speedups and power limits must not be substituted for these decoder
results.

## Complete decode results

| Comfy Kitchen | CuTe warm median | Kitchen warm median | Speedup | Lower latency |
| --- | ---: | ---: | ---: | ---: |
| 0.2.40 | 126.91 ms | 170.61 ms | 1.34× | 25.6% |
| 0.2.41 | 131.99 ms | 170.30 ms | 1.29× | 22.5% |

Each row uses seven CUDA-synchronized wall-clock samples per backend, alternating
backend order after excluding one first call. Both backends use identical saved
latents and BF16 weights, resident during warm measurements, on the same device.
The timed boundary is the complete decode node: projections, RMS normalization,
RoPE, convolutions, residuals, attention, node wrapping and its output handling.
Post-call pixel comparisons, image saving, HTTP queue overhead and video encoding
are excluded.

The input latent is `[1,128,5,8,12]`, saved by an eight-step, seed-1729 workflow.
The refresh replays its decoder through the real ComfyUI HTTP queue; diffusion
sampling was not repeated. All 24 neighborhood blocks call CuTe in strict mode,
retaining BF16 `[B,T,H,W,heads,64]` operands and `scale=1.0` for pre-scaled Q.
Temporary block forwards are checked for restoration after each decode.

The previous generic CuTe implementation measured 252.57 ms in a separate
same-device control, versus 126.91 ms for the update: a 1.99× speedup between
these sequential CuTe runs. Its paired Kitchen median was 186.95 ms, with samples
spanning 168.29–188.70 ms. These sequential controls have visible variability; use each
paired Kitchen comparison above for the reported backend speedup.

The
[November 13 report](https://github.com/SimpleTuner-io/ComfyUI-CuTe-Attention/blob/f185a2dc2d857ae0037a40196887fda2f77bdf96/results/ltx25-na3d-2026-11-13.md)
used the older generic implementation on a 400 W RTX 5090 and reported a slowdown
(270.3 ms CuTe versus 166.0 ms Kitchen). The November 14 refresh supersedes that
performance result. Its 575 W device and seven-sample protocol are different;
the two dates are not a paired before/after comparison.

## Pixel agreement and first-use costs

Both Kitchen versions produced the same reported agreement metrics with the
updated CuTe decoder, measured before encoding in ComfyUI's `[0,1]` IMAGE range:

| Decode mode | CuTe calls | Mean absolute difference | PSNR | Maximum difference |
| --- | ---: | ---: | ---: | ---: |
| Regular | 24 | 0.000495 | 61.98 dB | 0.00684 |
| Tiled | 96 | 0.000496 | 62.13 dB | 0.00781 |

Both modes pass the predeclared MAE ≤ 2/255 and PSNR ≥ 35 dB gates. Tiled decoding
uses spatial size 256/overlap 64 and temporal size 24/overlap 8. Only one tiled
execution per backend was measured, so there is no warm tiled speedup claim.
Regular peak Torch allocation was approximately 2.38 GiB for both backends.
These pixel comparisons cover one prompt and seed; Kitchen is a comparison
implementation, and pixel agreement does not establish FP64 model accuracy or
general perceptual quality.

First regular calls took 75.51 s for CuTe versus 1.01 s for Kitchen 0.2.40,
and 44.33 s versus 1.03 s in the Kitchen 0.2.41 run. They include model preparation
and/or compilation, with partially populated on-disk caches from preceding tests.
They are not comparable cold-start benchmarks. The warm speedup excludes this
substantial first-use cost.

## Provenance and validation

- Full RTX 5090: SM 12.0, 170 SMs, 32607 MiB, 575 W power limit,
  driver 610.57.04. Power and clock settings were not changed during the run.
- Python 3.13.18, Torch 2.13.0+cu130, CUDA 13.0, Triton 3.7.1,
  CUTLASS DSL 4.8.1 and TVM FFI 0.1.15.post0.
- KohakuFA-CuTe `70d5f0bff4c6c22b044611b6c44312e6bb1a016f`, now merged;
  previous control `cf97ce0e204f0ba970b04dcd5af5ad7d5c6e560d`.
  The integration preserves the upstream arithmetic and scheduling, with its
  documented namespace/import/native-launcher adaptations.
- ComfyUI `35060f0156181ae78a86b10f098f67ce8e83697f` and Lightricks/LTX-2.5
  model revision `49b093a39c19169a8a81721dcc541e07e945232f`.
  The receipt includes the verified VAE file and input-latent hashes.

The integration reports four passing 3D CUDA tests, including eight FP16/BF16
output/gradient cases against independent native FP64, Kitchen comparison,
FP32 fallback, native/Python launch equivalence and changed-input graph replay.
Its sampled operator comparisons also passed per-head FP64 limits of 0.5% FP16
and 1% BF16. Those are operator checks, separate from the pixel gates above.
Five dense CUDA regressions and three 2D regressions passed; four RTX 4090 checks
were skipped on the 5090. All 71 integration CPU tests and package builds passed.

The source report records a hash-verified 193-file capture archive and release
of the rented GPU. This import checked the published receipt against the supplied
file, verified the tested integration source hashes, and recomputed sample
counts, medians, call counts and pixel gates. It did not repeat the GPU campaign
or revalidate the raw archive. Larger videos, other GPUs, full-model FP16/HDR
and compiled decoders remain outside this integration qualification.

## Reproduce

Use the pinned [integration snapshot](https://github.com/SimpleTuner-io/ComfyUI-CuTe-Attention/tree/f185a2dc2d857ae0037a40196887fda2f77bdf96)
and its installation/model instructions. Start ComfyUI with `--bf16-vae`, save a
latent using the supplied workflow, and run from the extension directory in the
ComfyUI Python environment:

```sh
python -m tests.ltx_decode_smoke \
  --comfyui-root "$COMFYUI_ROOT" \
  --latent "$CAPTURE_DIR/latents/ltx25-cute-na3d_00001_.latent" \
  --output "$CAPTURE_DIR/decode" --repeats 7
```

Repeat separately with Comfy Kitchen 0.2.40 and 0.2.41. Keep captures outside
the checkout and retain all first-call and warm samples. The pinned
[measurement file](https://github.com/SimpleTuner-io/ComfyUI-CuTe-Attention/blob/f185a2dc2d857ae0037a40196887fda2f77bdf96/results/ltx25-na3d-2026-11-14.json)
also contains the separate synthetic operator comparisons.
