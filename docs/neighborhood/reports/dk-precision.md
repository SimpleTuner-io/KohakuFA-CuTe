# Experimental neighborhood key-gradient precision

This H100 report defines the combined method. The [multi-GPU report](multigpu-precision.md) replays its frozen inputs on B200, B300, RTX 4090, RTX 5090 and L40S. See the [current guide](../README.md) for the API and explicit-selection example.

The default arithmetic is unchanged. This explicit follow-up reconstructs the softmax correction during the existing query pass to reduce the sharp-attention dK error. Results below concern defined neighborhood operators and already-rounded operands, rather than model quality or universal pointwise accuracy.

The main timing matrix meets its predefined incremental-cost goal, but independent held-out testing retains performance exceptions: `heldout-48x64-w9-bf16` adds 9.91% versus the prior dQ method. These misses of the 5% aim remain explicit; no tuning followed the verified held-out evaluation.

Measured device: NVIDIA H100 80GB HBM3, 132 SMs. Selected KV statistics schedule: `column_staged`. Exact stack, device, source hashes, all per-head component errors, raw timing samples and negative results are retained in [the compact JSON](../../../results/neighborhood/dk-precision.json).

## Arithmetic and accuracy contract

`t = sum(P32 * (dP32 - delta_O)) / sum(P32)` is computed with FP32 row accumulation during the existing query traversal. A separate FP32 row buffer keeps `delta_O` from the actual rounded forward output. The KV owner forms `dS_key = round_input(P32 * ((dP32 - delta_O) - t))`; its products accumulate FP32 and the attention scale is applied afterward. No additional GEMM, neighborhood traversal or GPU launch is required.

The selected `weighted_dq_reconstructed_dk` method retains the prior normalized weighted-query correction. O, dQ and dV are checked bytewise against the previous method; dV still uses input-rounded P. The unprojected `reconstructed_dk` variant is a dK control and does not claim to repair its known dQ stress errors.

Native FP64 references evaluate identical already-rounded inputs. Mathematical accuracy and independently prescribed rounded equations are tested separately. The predefined per-head component limits remain 0.5% FP16 and 1% BF16. New exact zero/constant-dP controls target dK and preserve the previous dQ policy.

Use the explicit experiment around the complete forward/backward call, including warmup and graph capture. These process-wide selections are serialized benchmark controls, with cache keys for the chosen arithmetic and scheduling. Q/K/V use the existing BHSD, head-dimension-64 FP16/BF16 neighborhood API.

```python
import torch
from benchmarks.neighborhood.precision import key_statistics, variant
from kohakufa_cute import neighborhood_attention

with variant("weighted_dq_reconstructed_dk"), key_statistics("column_staged"):
    out = neighborhood_attention(q, k, v, spatial_shape=(height, width), window_size=window)
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
```

## Sharp attention: matched dK heads

Relative L2 errors against mathematical FP64, in percent. These are full small-tensor comparisons on the exact retained seed-178 operands, with both signed scales. Every head is shown; reference norms and maximum absolute discrepancies keep cancellation-sensitive errors interpretable.

| dtype | scale | head | FP64 dK L2 | original baseline | prior dQ correction | new dQ+dK correction | new max absolute error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| float16 | 0.125 | 0 | 22.073502 | 0.20570% | 0.20570% | 0.03617% | 0.00222827 |
| float16 | 0.125 | 1 | 29.974898 | 0.10034% | 0.09558% | 0.04084% | 0.00413805 |
| float16 | -0.125 | 0 | 10.261216 | 0.30373% | 0.30373% | 0.02537% | 0.00065120 |
| float16 | -0.125 | 1 | 0.908967 | 9.85962% | 9.81425% | 0.02245% | 0.00005993 |
| bfloat16 | 0.125 | 0 | 10.713372 | 2.17412% | 2.14846% | 0.28530% | 0.01099186 |
| bfloat16 | 0.125 | 1 | 21.961911 | 0.31288% | 0.26626% | 0.21864% | 0.01087637 |
| bfloat16 | -0.125 | 0 | 11.760870 | 2.81554% | 2.83325% | 0.22404% | 0.00571107 |
| bfloat16 | -0.125 | 1 | 0.175785 | 98.12111% | 98.12044% | 0.14345% | 0.00006511 |

The selected method passes 29/29 defined fixture qualifications, including 29 required mathematical gates and 29 independent policy gates. All four observed component gates pass on 29 fixtures, including the new controls whose prior dQ policy was preserved. All 8/8 zero/constant-dP controls retain exactly zero dK. Its O/dQ/dV unchanged-component checks pass on 29 fixtures. These counts describe the measured fixtures and do not promise exact zero gradients for every possible cancellation.

Worst full-small-tensor per-head mathematical errors across the targeted fixture matrix; each component maximum may come from a different fixture or head:

| dtype | O | dQ | dK | dV |
| --- | ---: | ---: | ---: | ---: |
| bfloat16 | 0.20378% | 0.28013% | 0.28530% | 0.25569% |
| float16 | 0.02624% | 0.04288% | 0.04084% | 0.03098% |

The original unprojected-query control retains its prior FP16 negative common-offset dQ failure. It is kept as a failed prescribed-reference gate, rather than hidden or converted into a passing result. The prior [dQ precision report](dq-precision.md) records the retained-product diagnosis: a native FP16-input GEMM with FP32 output changes dP enough to cross one dS rounding midpoint. The new weighted-query method preserves the already-qualified dQ behavior.

## Complete resident forward plus backward cost

CUDA graph device-event medians in microseconds, with matched quantized operands and alternating measurement order. Required per-call work is included; compilation, allocation warmup and immutable geometry preparation are outside the resident interval. The predeclared aim is at most 5% incremental median latency versus the prior combined dQ method, with cases above 10% requiring investigation. Shape exceptions remain explicit.

| case | original baseline | prior dQ correction | new dQ+dK correction | incremental vs prior | incremental vs baseline | <=5% goal |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 64sq-w7-fp16 | 117.06 | 131.57 | 133.98 | +1.83% | +14.46% | pass |
| 64sq-w13-fp16 | 204.23 | 231.05 | 228.06 | -1.29% | +11.67% | pass |
| tail-33x49-w7-fp16 | 78.77 | 79.20 | 82.13 | +3.70% | +4.26% | pass |
| 64sq-w7-bf16 | 116.96 | 131.19 | 136.09 | +3.74% | +16.36% | pass |
| 64sq-w13-bf16 | 206.40 | 230.02 | 236.51 | +2.82% | +14.59% | pass |
| tail-33x49-w7-bf16 | 78.45 | 80.12 | 81.61 | +1.86% | +4.03% | pass |

All timing cases retain separate sampled mathematical FP64 errors for each component in JSON. Sparse probes cover recorded boundary/interior query owners and every query contributing to sampled key owners. They do not establish full large-tensor norm bounds. Failed or unsupported calls do not receive successful timing ratios.

Selected-method sampled mathematical FP64 errors, worst head per component:

| case | O | dQ | dK | dV |
| --- | ---: | ---: | ---: | ---: |
| 64sq-w7-fp16 | 0.02813% | 0.02934% | 0.03324% | 0.03215% |
| 64sq-w13-fp16 | 0.02790% | 0.02994% | 0.03300% | 0.03044% |
| tail-33x49-w7-fp16 | 0.02670% | 0.03232% | 0.03076% | 0.03086% |
| 64sq-w7-bf16 | 0.22016% | 0.22936% | 0.23760% | 0.25969% |
| 64sq-w13-bf16 | 0.22649% | 0.23999% | 0.25422% | 0.26866% |
| tail-33x49-w7-bf16 | 0.21157% | 0.24721% | 0.25170% | 0.25075% |

## Held-out shapes after schedule selection

These shapes and operands use seed 10816 and the already-selected `column_staged` schedule. They are evaluated after selection, without further tuning. Both dtype cases, accuracy failures and misses of the original 5% aim remain visible; the JSON retains all component errors and paired samples.

| case | prior us | new us | incremental vs prior | sampled new dK error | <=5% goal |
| --- | ---: | ---: | ---: | ---: | --- |
| heldout-31x32-w5-fp16 | 34.87 | 34.49 | -1.08% | 0.03103% | pass |
| heldout-32x64-w5-fp16 | 63.53 | 62.02 | -2.36% | 0.03099% | pass |
| heldout-48x64-w9-fp16 | 124.35 | 130.17 | +4.68% | 0.03208% | pass |
| heldout-65x33-w7-fp16 | 57.69 | 57.78 | +0.16% | 0.03249% | pass |
| heldout-31x32-w5-bf16 | 41.46 | 36.40 | -12.20% | 0.24485% | pass |
| heldout-32x64-w5-bf16 | 63.48 | 63.63 | +0.25% | 0.25999% | pass |
| heldout-48x64-w9-bf16 | 143.03 | 157.21 | +9.91% | 0.24779% | miss |
| heldout-65x33-w7-bf16 | 56.64 | 57.42 | +1.38% | 0.25542% | pass |

The first attempt at this held-out evaluation had a harness selection failure: a cached Python alias kept the direct statistics path while its report label said `column_staged`. Inspection found both scheduling flags false on all eight compiled KV modules. Its timings remain as a direct-path diagnostic in `negative_sweeps` and are excluded from the selected schedule table. The repaired harness reruns the identical eight shapes and seed-10816 operand hashes, with positive context counters and both actual compiled flags verified on all eight selected KV modules. Kernel source and the selected policy are unchanged; this is a harness repair, not retuning on held-out results.

## Complete eager API cost

These calls include output/statistics/scratch allocation operations, dispatch, autograd and the complete forward/backward API. The caching allocator and compilation are warmed; no CUDA graph hides per-call host work. Host wall medians and CUDA stream elapsed medians are both reported. CUDA elapsed time includes eager launch gaps, rather than measuring summed kernel execution time. Profiling runs are separate from timing.

| case | prior wall us | new wall us | wall increment | prior device us | new device us |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64sq-w7-fp16 | 1066.82 | 1082.87 | +1.50% | 1066.39 | 1082.46 |
| tail-33x49-w7-fp16 | 1028.33 | 1066.35 | +3.70% | 1027.83 | 1066.10 |
| 64sq-w7-bf16 | 1008.60 | 1022.12 | +1.34% | 1008.24 | 1021.81 |
| tail-33x49-w7-bf16 | 991.71 | 1005.80 | +1.42% | 991.34 | 1005.42 |

Paired eager wall-clock ratios vary from -26.97% to +21.72% across the retained rounds. The table's ratios of medians are descriptive and do not establish a precise small host-overhead bound; resident graph timing and eager host timing are separate measurements.

## Executed path and qualification

Untimed profiles retain actual launched-kernel family counts. The selected method has 4 CUDA launches per complete call on the profiled cases. Profiled symbols and retained compiler-returned binaries remain in the raw archive; public metadata omits compiler object addresses. Exported names and resource metadata were matched to the profiles; direct CUPTI CUfunction-to-module binding was not established. Eager profiles also retain CPU empty/allocation operation counts when measured.

Separate instrumented KV intervals on the retained bf16 workload are 49.09 us for the prior method, 62.67 us for the direct-statistics new method, and 50.60 us for the selected schedule. These are kernel intervals from profiling, rather than complete-call medians.

Inspection of the matching retained KV binary finds 128 repeated LDG sites in the direct path versus 0 in the selected path, with 64 LDS sites and 4 asynchronous statistics-copy sites. HMMA counts remain 128 and 128 for the compared work. Static instruction counts do not establish runtime percentages or bytes transferred.

Supplementary qualification receipts and direct-versus-selected byte checks:

| evidence | stage | result | scope |
| --- | --- | --- | --- |
| qualification | cpu-tests | pass | 525 tests, 227 skipped |
| qualification | selected-gpu-tests | pass | 16 methods |
| qualification | guard-tests-explicit | pass | 24 tests |
| qualification | memcheck | pass | 2 tests, 0 errors |
| qualification | racecheck | pass | 2 tests, 0 hazards |
| qualification | initcheck | pass | 2 tests, 0 errors |
| qualification | synccheck | pass | 2 tests, 0 errors |
| loaded-code | selected-code | pass |  |
| loaded-code | dependencies | pass |  |
| bytes-column | complete report | pass | |
| bytes-staged | complete report | pass | |
| bytes-column_staged | complete report | pass | |

Earlier direct-load and alternative scheduling measurements remain in JSON under `negative_sweeps`; their failed goals, failed qualification and shape regressions are retained separately from the selected final timing matrix.

The method is explicit experimental arithmetic. Automatic production dispatch and the default precision contract are unchanged. Measurements qualify this hardware, stack and workload set; they do not establish a universal numerical bound or a speedup on another GPU.

Regenerate from retained completed reports:

```sh
python3.13 -m benchmarks.neighborhood.summarize_key_precision \
  --directed RESULTS/directed/report.json --timing RESULTS/timing/report.json \
  --output results/neighborhood/dk-precision.json \
  --markdown docs/neighborhood/reports/dk-precision.md
```

Optional `--eager`, `--heldout`, `--path-audit`, `--kernel-evidence`, repeated `--validation LABEL=PATH` and `--negative-sweep LABEL=PATH` retain supplementary evidence. Selected held-out reports require context counters and the compiled-path audit.
