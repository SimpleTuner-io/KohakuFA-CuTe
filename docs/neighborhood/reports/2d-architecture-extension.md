# 2D qualification on Ampere and GB10

This campaign fills the missing A100, RTX 3090 and GB10 coverage for the public corrected neighborhood default. The existing [H100 and five-device evidence](../README.md#performance-and-device-coverage) remains unchanged. NATTEN is an implementation comparison; independent native FP64 is the mathematical reference.

The [receipt](../../../results/neighborhood/2d-architecture-extension.json) retains paired timing samples, per-head errors, changed-input graph checks, source hashes and runtime metadata. No kernel or dispatch changed.

| GPU | Capability / SMs | CUDA-visible memory (GiB) | Power limit (W) |
| --- | --- | ---: | ---: |
| A100 PCIe | 8.0 / 108 | 39.5 | 250 |
| RTX 3090 | 8.6 / 82 | 23.6 | 300 |
| GB10 | 12.1 / 48 | 121.7 | Unavailable |

## Complete-call performance

Five geometries in both FP16/BF16: 32²/window-7, 64²/window-7, 64²/window-13, 128²/window-7, and a 33×49/window-7 tail. D64; B1/H8 except B2/H4 for the tail. Seven rotating/alternating paired samples per timing mode after warmup, approximately 100 ms per sample. Required layout conversions stay inside calls; immutable geometry preparation is measured separately and reused. Compilation and graph capture are excluded.

Representative BF16 64²/window-7 medians in microseconds, **ours / NATTEN**:

| GPU | Graph forward | Graph forward + backward | Eager forward + backward | NATTEN backend |
| --- | ---: | ---: | ---: | --- |
| A100 PCIe | 74.23 / 115.72 | 251.11 / 620.39 | 1366.44 / 2410.65 | cutlass-fna |
| RTX 3090 | 102.05 / 143.63 | 451.83 / 918.81 | 1387.74 / 1735.63 | cutlass-fna |
| GB10 | 90.26 / 227.32 | 433.24 / 1306.65 | 496.79 / 1363.98 | cutlass-fna |

Across all ten cases, the range of NATTEN/ours median latency ratios is below. Values above one favor ours; near-one ratios need the retained sample variability for interpretation.

| GPU | Graph forward | Graph forward + backward |
| --- | ---: | ---: |
| A100 PCIe | 1.09–2.41× | 1.55–3.95× |
| RTX 3090 | 0.97–2.46× | 1.25–3.18× |
| GB10 | 1.55–2.58× | 2.06–5.97× |

The receipt contains all shapes, dtypes and both eager/graph modes. Graphs replay eight complete calls per batch and are checked after changing Q/K/V/dO. Eager timings include host/autograd dispatch and depend on the recorded host CPU. Different precision policies make these complete-implementation comparisons, not isolated scheduling improvements or HDiT model speedups.

## FP64 accuracy

Worst per-head mathematical norm errors across the ten timing cases, percent, **ours / NATTEN**:

| GPU / dtype | O | dQ | dK | dV |
| --- | ---: | ---: | ---: | ---: |
| A100 PCIe / FP16 | 0.0275 / 0.0283 | 0.0313 / 0.0437 | 0.0351 / 0.0459 | 0.0334 / 0.0334 |
| A100 PCIe / BF16 | 0.2219 / 0.2248 | 0.2520 / 0.3683 | 0.2574 / 0.3897 | 0.2672 / 0.2672 |
| RTX 3090 / FP16 | 0.0272 / 0.0278 | 0.0329 / 0.0478 | 0.0333 / 0.0448 | 0.0335 / 0.0336 |
| RTX 3090 / BF16 | 0.2174 / 0.2186 | 0.2517 / 0.3768 | 0.2786 / 0.3538 | 0.2580 / 0.2571 |
| GB10 / FP16 | 0.0281 / 0.0280 | 0.0329 / 0.0441 | 0.0345 / 0.0424 | 0.0354 / 0.0354 |
| GB10 / BF16 | 0.2168 / 0.2238 | 0.2639 / 0.4398 | 0.2651 / 0.3594 | 0.2636 / 0.2636 |

Each implementation is checked on identical rounded inputs against independent native CPU FP64. Sparse output/query probes cover boundaries and interiors; sampled key gradients include every contributing query. Limits are independently 0.5% FP16 and 1% BF16 for O/dQ/dK/dV. Our separate directed qualification also checks the corrected equations with actual rounded O and agrees with device-reference gates. NATTEN accuracy here covers the timing matrix, not every directed cancellation fixture.

The corrected gradients reduce the worst observed dQ/dK errors in this matrix; output and dV differences may favor either implementation. Historical single-rounded-P/dS diagnostics are retained in the raw archive but do not define the current corrected method or its acceptance criteria.

## Scope and reproduction

The measured source is `8be8fc2a2e20fc52d5801268399983b23817a9f2`; its tracked contents match the merged 3D implementation, with unchanged 2D arithmetic. The [pinned stack](../README.md#reproduce-and-inspect) is retained for comparability, including Python 3.13.18, Torch 2.13.0+cu130 and NATTEN 0.21.7+torch2130cu130. GB10 runs ARM64 wheels; package versions, CUDA-visible memory, SM count, power limits and host quotas are recorded separately for each device.

The 2D/3D regression suite passed on each device. The benchmark traces record actual eager and graph backend execution. This report does not borrow the 3D sanitizer fixture as 2D sanitizer coverage. Complete source, samples, traces and code captures were archived and verified; raw machine captures remain outside the repository.

Run on a remote GPU using the pinned stack:

```sh
python -m benchmarks.neighborhood.qualify \
  --source-revision "$(git rev-parse HEAD)" --output "$CAPTURE_DIR/2d.json" \
  --reference-agreement --profile
python -m benchmarks.neighborhood.benchmark \
  --source-revision "$(git rev-parse HEAD)" --output "$CAPTURE_DIR/2d-performance.json" \
  --backends kohakufa natten --samples 7 --target-ms 100 --profile
```
