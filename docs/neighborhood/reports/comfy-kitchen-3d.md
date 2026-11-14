# 3D attention versus Comfy Kitchen

Direct neighborhood-union traversal reduces forward work while retaining the
existing two-term probability product, separate unscaled statistics and corrected
backward. This study compares the public LTX-layout API with Comfy Kitchen's CUDA
`na3d` on H100 PCIe, RTX 3090, RTX 4090 and RTX 5090. It measures synthetic attention
operators with representative decoder shapes, not a complete LTX decoder.

The [receipt](../../../results/neighborhood/comfy-kitchen-3d.json) retains paired
samples, per-head errors, hardware and source identities, validation receipts and
archive hashes. Earlier [LTX helper timings](ltx-3d.md) used a different comparator.

A separate [LTX 2.5 integration study](ltx25-decoder.md) measures the complete
VAE decode node on RTX 5090, with 1.29–1.34× warm speedups and pixel-agreement
checks. It has a different device power limit and timing boundary from this
operator study.

## What changed

A query patch now gathers a fixed union of its shifted neighborhoods directly.
The shared attention implementation still owns score reconstruction, softmax,
probability conversion, accumulation and output/statistics stores. Every query
applies its own visibility predicate, including shifted boundary windows. The
last key tile is predicated and zero-filled. Spatial edge query tiles are padded:
invalid coordinates load zeros and cannot publish output or saved statistics.
In particular, a padded column cannot alias the next physical image row.
Backward keeps the existing reverse neighborhood traversal and normalized
dQ/reconstructed dK corrections.

For an interior 4×8 query patch and an 11³ window, the previous warp owner visits
11×20×24 = 5,280 key positions after rounding spatial bounds to image-aligned
tiles. The new owner visits 11×14×18 = 2,772 positions, padded to 2,784 for its
32-column MMA tiles: 87 tiles instead of 165. This comparison holds query ownership
and arithmetic constant; it is not a count of Comfy Kitchen's work or of the old
H100 8×8 owner.

Default union dispatch requires D64, FP16/BF16, vector-eight loads and the exact
full-device identities tested here: H100 PCIe/114 SMs, RTX 3090/82, RTX 4090/128,
and RTX 5090/170. `(3,5,5)` and `(3,7,7)` windows use 1×4×4 query tiles when height
and width are at least 16 and divisible by four. An 11³ window uses 1×4×8 tiles
when height and width are at least 64 and divisible by eight; otherwise it uses
1×4×4 tiles for spatial dimensions at least 64. The smaller tile won the width-65
screen and is separately evaluated on odd heights and other odd widths. All
windows must fit. Other inputs retain their existing owner. H100 SXM, MIG and
other GPU models are not qualified by these gates.

Host work also matters for short calls. The operator uses direct CUDA dispatcher
registration with the original symbolic geometry schema, cached unused arguments,
native output/statistics allocation and a program cache keyed by exact geometry,
dtype, device, shape, stride and alignment. The cache stores programs, never tensor addresses; each invocation
passes its current operands and stream. Explicit scheduling experiments continue
through the Python launch path. Autograd and fake implementations remain registered.

## Performance

Complete eager forward medians below use separate processes for each case and
dtype; profiling follows the uninstrumented measurements.

BF16 milliseconds; each cell is **KohakuFA / Comfy Kitchen (speedup)**.

| Volume / heads | H100 PCIe | RTX 3090 | RTX 4090 | RTX 5090 |
| --- | ---: | ---: | ---: | ---: |
| 5×16×16 / H32 | 0.067 / 0.099 (1.48×) | 0.136 / 0.222 (1.63×) | 0.078 / 0.100 (1.28×) | 0.082 / 0.082 (1.00×) |
| 5×32×32 / H16 | 0.138 / 0.191 (1.38×) | 0.258 / 0.336 (1.31×) | 0.128 / 0.171 (1.34×) | 0.092 / 0.133 (1.44×) |
| 9×32×32 / H8 | 0.065 / 0.130 (1.99×) | 0.140 / 0.221 (1.58×) | 0.079 / 0.110 (1.40×) | 0.081 / 0.079 (0.97×) |
| 17×64×64 / H8 | 0.447 / 1.057 (2.37×) | 0.921 / 2.461 (2.67×) | 0.614 / 1.280 (2.08×) | 0.353 / 0.859 (2.44×) |
| 17×128×128 / H4 | 11.987 / 16.354 (1.36×) | 23.995 / 39.021 (1.63×) | 11.144 / 18.798 (1.69×) | 7.291 / 13.803 (1.89×) |
| 49×128×192 / H4 | 52.213 / 71.092 (1.36×) | 101.781 / 170.386 (1.67×) | 54.427 / 82.648 (1.52×) | 31.665 / 60.353 (1.91×) |
| 13×64×96 / H2 | 1.652 / 2.372 (1.44×) | 3.443 / 5.662 (1.64×) | 1.620 / 2.697 (1.66×) | 1.055 / 2.001 (1.90×) |
| 33×96×128 / H1 | 4.255 / 6.037 (1.42×) | 8.723 / 14.309 (1.64×) | 4.080 / 6.854 (1.68×) | 2.677 / 5.067 (1.89×) |
| 13×64×65 / H2 | 1.175 / 1.945 (1.65×) | 2.391 / 4.548 (1.90×) | 1.219 / 2.147 (1.76×) | 0.771 / 1.568 (2.03×) |
| 13×65×67 / H2 | 1.281 / 1.984 (1.55×) | 2.540 / 4.648 (1.83×) | 1.287 / 2.193 (1.70×) | 0.826 / 1.598 (1.94×) |
| 17×67×73 / H1 | 0.957 / 1.389 (1.45×) | 1.874 / 3.193 (1.70×) | 0.957 / 1.524 (1.59×) | 0.618 / 1.119 (1.81×) |

For the width-65 case, FP16 medians are:

| GPU | KohakuFA | Comfy Kitchen | Speedup |
| --- | ---: | ---: | ---: |
| H100 PCIe | 1.199 ms | 1.970 ms | 1.64× |
| RTX 3090 | 2.355 ms | 4.602 ms | 1.95× |
| RTX 4090 | 1.235 ms | 2.204 ms | 1.78× |
| RTX 5090 | 0.787 ms | 1.591 ms | 2.02× |

Small differences are not treated as robust wins. The receipt retains all nine
samples and variability; cases within 5% or slower are described below.

On RTX 5090, the shortest stage1 eager calls are effectively tied; stage3 favors
Comfy Kitchen by 2–3% in both dtypes. The separately measured BF16 resident graph
medians are 0.048 / 0.084 ms for stage1 and 0.047 / 0.082 ms for stage3
(KohakuFA / Comfy Kitchen). Removing repeated host calls changes the comparison,
so graph gains are not substituted for complete eager latency.

The six development workloads cover spatial shapes 5×16×16/H32, 5×32×32/H16,
9×32×32/H8, 17×64×64/H8, 17×128×128/H4 and 49×128×192/H4. The first two use
`(3,7,7)`, the next two `(3,5,5)`, and the last two 11³. Additional geometries are
13×64×96/H2 and 33×96×128/H1. The formerly slow 13×64×65/H2 fallback became a
tuning workload; 13×65×67/H2 and 17×67×73/H1 independently evaluate spatial tails.
These additional geometries all use 11³ windows.
Batch is one and head dimension is 64 throughout.

Each eager comparison uses identical rounded inputs in the same process and
software stack, nine alternating-order samples, and nominal 200 ms samples after
compilation, allocation and cache warmup. Synchronized wall time includes the
public adapter, required layout work, allocation and kernel execution. CUDA-event
spans are retained but also include host launch gaps. Compilation and references
are outside the timed region. Graph replay is reported separately, with all API
GPU work inside capture and host validation/capture excluded.

The full devices ran at these power limits; each device's two implementations
shared the same host and stack. Host differences matter for eager latency.

| GPU | SMs | Power limit | Host CPU |
| --- | ---: | ---: | --- |
| H100 PCIe | 114 | 350 W | Xeon Gold 6438N |
| RTX 3090 | 82 | 320 W | Xeon E5-2673 v4 |
| RTX 4090 | 128 | 330 W | EPYC 7K62 |
| RTX 5090 | 170 | 500 W | EPYC 7642 |

## Accuracy and qualification

References use native CPU FP64 on identical rounded operands. Every eager row
checks output finiteness and independent per-head norm error at image boundaries
and 128 seeded sample queries. BF16 graph checks use 64 seeded queries after
changing inputs and poisoning outputs. FP16 and BF16 limits remain 0.5% and 1%.

Final full-coordinate checks cover two smaller volumes with spatial tails in
both dtypes: 5×17×17 with `(3,7,7)` windows and 13×17×25 with 11³ windows. These
checks explicitly select the union owner; larger public-API fixtures separately
check production dispatch. They evaluate O, dQ, dK and dV against mathematical
FP64 and retain the corrected actual-output equations. Across the four GPUs,
the worst observed norm errors
were:

| Dtype | KohakuFA O | Comfy Kitchen O | KohakuFA dQ | KohakuFA dK | KohakuFA dV |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP16 | 0.02076% | 0.02781% | 0.02941% | 0.02935% | 0.02964% |
| BF16 | 0.16701% | 0.22232% | 0.23496% | 0.23576% | 0.23670% |

Forward error is close to the final output-rounding floor on these inputs,
roughly 25% below Comfy Kitchen's error. This preserves the existing two-term
forward advantage; the scheduling change does not introduce a new precision
policy. Comfy Kitchen is compared for output only. These results establish
neither a universal error bound nor decoder-level quality.

Independent enumeration checks shifted windows and tile unions. GPU fixtures
cover strided/offset operands, partial tiles, even windows, signed/zero scales,
sharp logits, cancellation, saved statistics, changed-input graph replay and
compiled symbolic geometry. Native-cache tests compare identical kernels through
both host launch paths while changing geometry, alignment, scale and operands.
Four sanitizer tools qualify the traversal, including the selected 1×4×4 C32
spatial-tail owner with large windows and strided operands. The final GPU suite
also reruns the aligned owners and native host path. Earlier qualification and
ordinary-attention precision results remain in the retained campaign archives.

## Evidence and limits

Torch traces identify the actual union kernel and Comfy Kitchen's CUDA
`na3d_kernel`, with no SDPA fallback in these forward calls. Retained PTX/cubins,
matching deployed SASS and launch metadata distinguish the 32-column warp-MMA
paths from the earlier H100 WGMMA owner. For example, H100 stage5 uses 187
registers/thread and 12 KiB total launch shared memory for KohakuFA, versus 145
and 18 KiB for Comfy Kitchen. KohakuFA's cubin declares no stack or local-memory
allocation. These resource counts do not establish occupancy, utilization or
causes of elapsed-time differences.

Comfy Kitchen's module bytes are captured with the CUPTI module-load callback
outside timing. This also identifies the compatible binary actually loaded on
RTX 3090, whose compute capability differs from the packaged binary target.

Simple prefetch, larger aligned query tiles and one-/two-buffer H100 TMA
experiments did not close the large-window gap; their source snapshots and
negative timings are retained. Direct union traversal did. Hardware performance
counters were denied on the measured hosts; traces, deployed code and controlled
timings supply the evidence. No stall, cache-hit or hardware-utilization ceiling
is inferred. Early harness import failures and an alignment-proof compilation
failure are retained as failed stages, not successful measurements.

The comparison uses Comfy Kitchen 0.2.41's official CUDA wheel. Its
[release source](https://github.com/Comfy-Org/comfy-kitchen/tree/cbc44a5b7cc3b8f910ef76ae5251c94971eb05d1)
was inspected and matched the CUDA source on the inspected main revision. The
[pinned neighborhood stack](../README.md#reproduce-and-inspect) uses Python
3.13.18, Torch 2.13.0+cu130 and CUTLASS DSL 4.8.1. Torch 2.15.0 was current at
review; the prior qualified stack was retained to isolate this scheduling study.
Results do not automatically transfer to another framework, power limit or GPU
partition. CPU models and quotas are recorded because eager host overhead varies
between the machines.

## Reproduce

Run on a remote CUDA machine with the pinned stack and Comfy Kitchen 0.2.41.
Use persistent capture storage outside the checkout:

```sh
python -m unittest tests.test_neighborhood_union tests.test_neighborhood_3d \
  tests.test_neighborhood_volume_forward tests.test_neighborhood_schedule -v
for dtype in float16 bfloat16; do
  for case in stage1 stage2 stage3 stage4 stage5 stage5-49f-768x512 \
      heldout-13f heldout-33f fallback-tail \
      heldout-spatial-tails heldout-spatial-tails-wide; do
    python -m benchmarks.neighborhood.benchmark_ltx --comfy-kitchen \
      --source-revision "$(git rev-parse HEAD)" \
      --output "$CAPTURE_DIR/comfy-3d/$case-$dtype" \
      --cases "$case" --dtypes "$dtype" --seed 99173 \
      --reference-queries 128 --samples 9 --target-ms 200
  done
done
```

Freeze the complete source and commands before deployment. Retained off-host
archives include experimental source, reference results, failed-stage receipts,
raw timing samples, traces and loaded binaries. The compact receipt identifies
those archives without embedding raw captures in the repository.
