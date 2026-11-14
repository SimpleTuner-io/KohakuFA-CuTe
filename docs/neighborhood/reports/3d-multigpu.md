# 3D qualification across GPU architectures

The public 3D default passed output and gradient qualification on 8 additional GPUs. It retains two-term forward probabilities and the normalized dQ/reconstructed dK corrections. Performance varies by GPU and workload; larger windows expose remaining gaps on some devices. This campaign changes no kernels or dispatch.

The [receipt](../../../results/neighborhood/3d-multigpu.json) retains per-head errors, timing samples, source hashes, hardware/runtime metadata and validation receipts. The earlier [H100 PCIe comparison](ltx-3d.md) is separate.

| GPU | Capability / SMs | CUDA-visible memory (GiB) | Power limit (W) |
| --- | --- | ---: | ---: |
| A100 PCIe | 8.0 / 108 | 39.5 | 250 |
| RTX 3090 | 8.6 / 82 | 23.6 | 300 |
| L40S | 8.9 / 142 | 44.4 | 350 |
| RTX 4090 | 8.9 / 128 | 23.5 | 350 |
| RTX 5090 | 12.0 / 170 | 31.4 | 500 |
| B200 | 10.0 / 148 | 178.4 | 1000 |
| B300 | 10.3 / 148 | 267.7 | 1100 |
| GB10 | 12.1 / 48 | 121.7 | Unavailable |

GB10 uses shared system/GPU memory; its reported power limit is unavailable. These are measured devices, not qualifications of every product sharing their compute capability. Host CPU models and quotas are retained in the receipt.

## Complete attention forward calls

Six LTX decoder geometries, FP16/BF16, B1/D64, normalized synthetic Q/K with Q pre-scaled by 0.125 and API scale 1.0. Nine alternating-order samples per pair, approximately 200 ms per sample, after separate compilation/allocation warmup. Synchronized wall time includes required preparation, copies, Python dispatch and output layout. Inputs are resident; projections, normalization/RoPE, upsampling, backward and whole-decoder latency are excluded.

BF16 medians in milliseconds, **ours / LTX**:

| GPU | 17×128×128, 11³ | 49×128×192, 11³ | LTX/ours range, all cases | LTX SDPA backend |
| --- | ---: | ---: | ---: | --- |
| A100 PCIe | 65.59 / 63.54 | 311.00 / 291.77 | 0.94–15.78× | PyTorch efficient |
| RTX 3090 | 105.71 / 97.49 | 514.21 / 434.22 | 0.84–13.44× | PyTorch efficient |
| L40S | 35.96 / 36.99 | 174.99 / 173.25 | 0.99–17.18× | PyTorch efficient |
| RTX 4090 | 41.82 / 41.17 | 204.68 / 181.39 | 0.89–18.42× | PyTorch efficient |
| RTX 5090 | 28.78 / 35.41 | 143.41 / 161.65 | 1.13–18.48× | PyTorch efficient |
| B200 | 34.12 / 41.71 | 161.24 / 185.41 | 1.15–10.64× | cuDNN |
| B300 | 32.19 / 38.86 | 155.35 / 185.45 | 1.19–11.00× | cuDNN |
| GB10 | 89.24 / 113.59 | 430.90 / 543.82 | 1.26–19.18× | PyTorch efficient |

The ratio range spans all six geometries and both dtypes; values above one favor ours. The receipt includes every timing sample. Small median differences with overlapping samples should be treated as ties. The largest-window slowdowns are retained as optimization targets, not excluded from the matrix. Comparisons within each pair share a process and software stack; different host CPUs, power limits and selected backends prevent interpreting this table as a controlled GPU ranking.

## Independent accuracy

Each timing case checks boundary queries plus 128 independently drawn query positions against native CPU FP64 on identical rounded operands, with separate per-head gates and finite checks over the full output. A held-out comparison checks every output and gradient coordinate for three window families, two dtypes and two heads. Our gradients are also checked against the declared corrected equations using actual rounded O.

Worst per-head norm errors across the held-out BF16 cases, percent, **ours / LTX**:

| GPU | O | dQ | dK | dV |
| --- | ---: | ---: | ---: | ---: |
| A100 PCIe | 0.1676 / 0.2247 | 0.2360 / 0.3121 | 0.2362 / 0.3023 | 0.2394 / 0.2394 |
| RTX 3090 | 0.1676 / 0.2251 | 0.2360 / 0.3121 | 0.2362 / 0.3023 | 0.2394 / 0.2394 |
| L40S | 0.1676 / 0.2271 | 0.2360 / 0.3121 | 0.2362 / 0.3023 | 0.2394 / 0.2394 |
| RTX 4090 | 0.1676 / 0.2248 | 0.2360 / 0.3121 | 0.2362 / 0.3023 | 0.2394 / 0.2394 |
| RTX 5090 | 0.1676 / 0.2245 | 0.2359 / 0.3121 | 0.2362 / 0.3022 | 0.2394 / 0.2394 |
| B200 | 0.1676 / 0.2353 | 0.2359 / 0.2463 | 0.2362 / 0.2398 | 0.2394 / 0.2393 |
| B300 | 0.1676 / 0.2352 | 0.2359 / 0.2466 | 0.2362 / 0.2403 | 0.2394 / 0.2394 |
| GB10 | 0.1676 / 0.2240 | 0.2359 / 0.3121 | 0.2362 / 0.3022 | 0.2394 / 0.2394 |

All recorded FP16/BF16 component gates passed: 0.5% FP16 and 1% BF16 independently for output, dQ, dK and dV. Directed cancellation/sharp fixtures retain prescribed-policy gates rather than universal pointwise FP64 equality. Forward error is near the final dtype rounding floor; dV is essentially unchanged, and small component differences can favor either implementation. These synthetic checks do not establish full-decoder output quality.

## Qualification and reproduction

Each device passed the 33-test 2D/3D suite (one unrelated H100-specific skip), 28 3D qualification cases with independent CPU and CUDA reference checks, and the held-out LTX-layout comparison. Changed-input graph replay, poisoned outputs, tails, shifted boundaries, signed/zero scales and strided views are covered. Memcheck, racecheck, initcheck and synccheck passed the FP16/BF16 temporal-seam fixture. Sanitizer coverage is scoped to that fixture, not every timing shape.

The source is `8be8fc2a2e20fc52d5801268399983b23817a9f2`, whose tracked contents match the merge of the 3D implementation. The previously qualified Python 3.13.18 / Torch 2.13.0+cu130 / CUTLASS DSL 4.8.1 stack was retained to keep the comparison consistent; the compared package and driver versions are in the receipt. The external LTX helper is frozen at the source revision/hash recorded by the [H100 study](ltx-3d.md).

Traces confirm actual backend execution. PTX/SASS and loaded binaries are retained from separate untimed matching replays; they are not extracted from the timed process. Raw captures, complete source and launch scripts were archived and hash-verified before resource removal. The data omits machine identifiers and storage locations.

Use the [3D guide](../3d.md) for the API and the [precision contract](../../precision.md#neighborhood-default) for the arithmetic. On a remote GPU using the pinned stack:

```sh
python -m unittest tests.test_neighborhood_3d tests.test_neighborhood -v
python -m benchmarks.neighborhood.qualify --dimensions 3 \
  --source-revision "$(git rev-parse HEAD)" --output "$CAPTURE_DIR/3d.json" \
  --reference-agreement --profile
python -m benchmarks.neighborhood.benchmark_ltx \
  --source-revision "$(git rev-parse HEAD)" --ltx-helper "$LTX_HELPER" \
  --output "$CAPTURE_DIR/ltx-3d" --seed 1729 --reference-queries 128
```
