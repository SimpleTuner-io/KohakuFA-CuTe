# L40S causal CTA ordering

The bounded L40S training selector uses causal head group 8 for D64/S4096
and group 16 for D128/S1024 or S4096. Dense schedules, including the selected
D128/S4096 N32 forward, keep their existing configuration. All device, training,
shape, alignment, scale and fallback guards in the [training policy](l40s-warp-training.md)
remain unchanged. This change enables an existing CTA mapping; it changes no
GPU method, synchronization or arithmetic policy.

The [November 2 report](../results/l40s-causal-groups.json) and
[current causal chart](../results/l40s-causal-groups.svg) retain the independent
public API comparison. The November 1 [dense N32 and initial training report](../results/l40s-warp-training.json)
and [chart](../results/l40s-warp-training.svg) remain dated controls. The historical
60-case cross-device grid has a different workload/runtime scope.

## Complete-call measurements

Measurements use a full 142-SM, 350 W L40S, Python 3.13.18, Torch 2.15.0/CUDA
13.2, driver 615.71.09, CuTe DSL 4.8.1, TVM FFI 0.1.15.post0 and quack 0.6.5.
The loaded standalone FA2 binary is verified by SHA. Inputs retain the declared
single-rounded P/dS computation, actual rounded O for delta, FP32 products and
late dQ/dK scaling. FA2 and SDPA are complete implementation comparisons under
their own arithmetic policies.

The held-out input seed is 9713, separate from the group-selection pilot.
All 96 forward/backward/joint, dense/causal, dtype, length and graph/eager rows
remain in the report, with nine rotating samples, 1000 ms warmup and approximately
100 ms windows. Compilation, allocation and capture precede timing. Public is
the actual `scaled_dot_product_attention` API. **G1 is an explicit native-runner
control with a shorter Python eligibility check, not the preceding complete
public API or the original generic CUDA implementation.**

Median forward + backward CUDA-event time in milliseconds:

| Dtype | S | D | G1 graph | Public graph | G1 eager | Public eager |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 1024 | 128 | 0.14129 | 0.11775 | 0.22508 | 0.23898 |
| FP16 | 1024 | 128 | 0.14397 | 0.11878 | 0.29323 | 0.30726 |
| BF16 | 4096 | 64 | 0.80685 | 0.70679 | 0.81437 | 0.71145 |
| FP16 | 4096 | 64 | 0.83204 | 0.73013 | 0.83354 | 0.73281 |
| BF16 | 4096 | 128 | 1.55187 | 1.46368 | 1.56174 | 1.47821 |
| FP16 | 4096 | 128 | 1.68352 | 1.56124 | 1.67625 | 1.57198 |

Against G1, graph training latency falls 16.7–17.5% at D128/S1024,
12.2–12.4% at D64/S4096 and 5.7–7.3% at D128/S4096. All six selected
public graph and eager training calls beat standalone FA2 in this run.
Short eager public calls cost about 14 microseconds more than the G1 control;
no short eager improvement over the preceding public API is established.
They are slightly slower than SDPA here. This host cost remains visible.

Every retained paired graph sample favors the selected group for these six
families. Long eager sample ratios also favor it; the D128 ratios vary more
than D64. The report includes minima, medians and maxima rather than replacing
mixed samples. Unchanged dense joint graph ratios range from 0.990 to 1.002;
dense and forward differences are not claimed as gains from causal ordering.
The selection pilot created G1 captures first, while final evaluation created
public captures first. This is a separate input and creation-order check, not
a universal allocation-independence claim.

## Loaded code and qualification

All 18 backward bindings (public, G1 and FA2 across both dtypes and the three
selected shapes) retain the exact graph symbol, loaded cubin, matching compiler
artifact where applicable, PTX, SASS and resource records. The 12 public/G1
forward bindings have identical cubin, PTX and SASS hashes within each pair.

At equal query work the selected and G1 backward loops retain 160 HMMA
instructions, the same matrix-load counts and the same barrier counts.
D64 retains 32 nontransposed and 48 transposed LDSM sites with four barriers;
D128 retains 40 of each with five barriers. Ordering adds one or two static
query-loop instructions. Backward allocation remains 255 registers/thread,
74752 shared bytes at D64 and 66560 at D128, limiting capacity to one 256-thread
CTA per SM. D128 BF16 reports eight local bytes with grouping versus zero for
G1; FP16 and D64 report zero. The gain is measured despite this tradeoff.
No counter-derived cache, bank-conflict, stall or utilization explanation is
claimed: Nsight Compute counters were denied.

Qualification passes 96 directed/tail/gradient/changed-graph cases and 60 saved
max/log-normalization checks, all four public GPU test methods, and memcheck,
racecheck, synccheck and initcheck with eight causal probes each. Public tests
exercise all three APIs, selected statistics, fallbacks, changed/poisoned graphs
and unaligned dO. Per-head output/dQ/dK/dV limits remain 1% BF16 and 0.5% FP16;
directed cases retain their original rounded-reference and coordinate checks.
CPU references use the same previously calibrated equations and actual rounded
outputs at four CPU threads, while every implementation GPU operation remains
instrumented. The timing process retains its original thread policy.

Complete source snapshots, exact commands, all timing operands and samples,
loaded binaries/PTX/SASS and stage receipts are verified off-host by physical
file SHA inventory. The compact report records archive and source hashes;
raw captures remain outside the tracked tree. The first release attempt stopped
before GPU work because an import-order preflight failed. Its receipt is retained;
the immutable corrected run passed all 45 stages. No failed or skipped stage is
counted as a pass. After refreshing the publication branch onto its newer
upstream base, all active L40S/API/kernel/native-ABI files remain byte-identical
to qualification. The final CPU suite passes 358 tests with 179 expected GPU
skips; all 24 publication-guard tests pass separately.

Reproduce after installing the qualified runtime:

```bash
# CAPTURE_DIR names a directory outside the checkout.
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m benchmarks.validate_l40s_causal_groups \
  --cpu-reference --output "$CAPTURE_DIR/qualification.json"
python -m unittest tests.test_l40s_static -v
python -m benchmarks.summarize_l40s_causal_groups \
  --captures "$CAPTURE_DIR/release" --verified-checkpoint "$CAPTURE_DIR/checkpoint" \
  --source-tar "$CAPTURE_DIR/frozen-source.tar.gz" \
  --publication-cpu "$CAPTURE_DIR/publication-cpu" \
  --publication-proof "$CAPTURE_DIR/publication-rebase-proof.json"
```

The retained K-residency, query-tile, positive-scale and register-dKV/copy
experiments remain separate rejected controls. Their negative timings and failed
implementation attempts did not change this selector or its accuracy criteria.
