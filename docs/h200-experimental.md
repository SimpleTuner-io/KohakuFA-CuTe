# Experimental H200 WGMMA backend

This adds an explicit Hopper forward and joint backward backend and a native
host adapter. The normal attention APIs keep their existing selection. The
experiments below are intended for reviewing the H200 implementation and its
remaining performance gaps.

Measurements use a full H200 NVL: 132 SMs, 232,448 bytes of opt-in shared memory
per CTA, and approximately 140 GiB of device memory. The comparison matrix is
contiguous square attention, batch 1, 16 equal Q/KV heads, 4,096 tokens, D64 or
D128, BF16 or FP16, dense and causal. Other devices, head dimensions, masks,
rectangular attention and GQA need their own qualification.

## Arithmetic and implementation

All project variants retain the [precision contract](precision.md): separate
unscaled maxima and log2 normalization, shifting before scaling, input-dtype P
before dS, one dS conversion, FP32 accumulation, actual rounded-output delta, and
late dQ/dK scaling. FA3 and FA4 retain their own arithmetic; comparisons with
them are complete implementation comparisons.

The static methods derive from FlashAttention revision
`4616f5116d33e3ba3cbb71d582caffcfcb49649c`. Unchanged helpers come from that pinned
dependency. Dispatch does not generate or rewrite Python source.

The producer stages maxima alongside normalization in the Q pipeline. A compact
statistics layout fits the Hopper shared-memory budget. Joint backward reuses
reconstructed scores/probabilities across the gradient products; dK/dV have a
single KV-tile owner and dQ partials accumulate in an FP32 workspace. Dense D128
uses 64 query rows. Long dense D128 forward uses an independently qualified
persistent scheduler with dedicated output storage, preserving Q/output buffer
ownership.

The causal interleaving experiment adopts FA3's head ordering. For the measured
cases it changes the main grid from `(32,16,1)` to `(512,1,1)`, retaining 384
threads, 168 allocated registers, zero local storage, and the same shared-memory
allocation. Deterministic accumulation stays disabled. It improves the spread
of causal work without changing the products or operand conversions.

## Measured improvements

Warmed graph-device medians below include required preparations. Baseline and
candidate rotate order over seven rounds with 100 ms timing windows. These are
the matched Python control and interleaved variant, in milliseconds.

| Dtype | D | Backward control | Interleaved | Joint control | Interleaved |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 64 | 0.27524 | 0.22379 | 0.37106 | 0.32182 |
| BF16 | 128 | 0.46387 | 0.39091 | 0.57293 | 0.52191 |
| FP16 | 64 | 0.26812 | 0.22298 | 0.36077 | 0.32799 |
| FP16 | 128 | 0.47530 | 0.40471 | 0.59184 | 0.54018 |

Backward decreases 14.9–18.7%; joint time decreases 8.7–13.3%. Dense controls
remain within measurement variation. The native host then removes the Python
dispatch and redundant maximum padding preparations from eligible aligned calls.

Complete eager joint wall medians include allocation, preparation, launches and
per-call synchronization. Each backend rotates first over eleven samples.

| Dtype | D | Visibility | Python interleaved | Native host | FA3 |
| --- | ---: | --- | ---: | ---: | ---: |
| BF16 | 64 | Dense | 0.8101 | 0.6002 | 0.7186 |
| BF16 | 64 | Causal | 0.6402 | 0.3657 | 0.5704 |
| BF16 | 128 | Dense | 1.1118 | 0.9039 | 0.9983 |
| BF16 | 128 | Causal | 0.7667 | 0.5326 | 0.7180 |
| FP16 | 64 | Dense | 0.8062 | 0.5980 | 0.7109 |
| FP16 | 64 | Causal | 0.6363 | 0.3594 | 0.5767 |
| FP16 | 128 | Dense | 1.1406 | 0.9118 | 1.0149 |
| FP16 | 128 | Causal | 0.7846 | 0.5400 | 0.7224 |

The native host wins these eight complete eager comparisons. Graph-device
causal joint time remains roughly 4–8% behind FA3. Host savings therefore do not
establish resident-kernel parity.

A second experiment separates D64's masked diagonal and steady query loops,
following the actual FA3 source and disassembly. It preserves query order and
the key-tail predicate. BF16 backward decreases another 2.27% and joint 1.22%;
FP16 joint is essentially flat. It remains an explicit comparison rather than a
universal selection preset. Whole-function SASS counts include both loop bodies;
their duplication does not mean twice the executed GEMM work.

The [compact results](../results/h200-experimental-summary.json) retain samples,
resource geometry, stack versions, source-report hashes and completion statuses.
Raw machine captures remain outside the repository.

## Validation and initialization tracking

The primary matrix passes independent per-head output and gradient FP64 norm
limits of 1% for BF16 and 0.5% for FP16. Directed checks use the declared rounded
equations with actual O and changing, poisoned graph buffers. The native host
passes its guards, saved-tensor hooks, shape/cache reuse and changed graph tests,
12 directed cases and all eight long matrix cases. Its four memcheck, four
racecheck and four synccheck probes pass. The separate-mask candidate passes its
directed checks and both-dtype memcheck/racecheck/synccheck probes.

Unmodified initialization checking is **not recorded as clean**. Retained reports
include real global-uninitialized-read flags and bounded long-case timeouts:

- Preprocess reads map to the actual O allocation, including PC `0x3b0` and
  16-byte loads. Earlier paired original-FA4, ordinary O-store and NaN-output
  controls isolate the actual 4D TMA output producer's tracking behavior.
- Gradient diagnostics initialize only O before its overwrite, then isolate
  dK/dV. The repaired ordinary-gradient producer passes both-dtype math and
  poisoned replay; whole-output-read BF16 initcheck is zero for D64 and D128. The
  ordinary control uses real global pointers and materialized converted
  fragments, preserving the final input-dtype conversion.
- An independent constant `UTMASTG.3D` producer reports initialization errors
  with both read-completion and full-completion waits; ordinary stores pass.
  It writes the expected constant. This supports that instruction family and
  does not replace the separate actual-attention 4D evidence.

NVIDIA documents unsupported tensormap/tcgen05 global tracking as a possible
source of initcheck false positives in its
[Compute Sanitizer limitations](https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html#known-limitations).
The instruction, allocation and producer controls above supply the specific
evidence. They do not waive unrelated reports or claim an unmodified initcheck
pass. Full-length instrumentation timeouts remain timeouts.

Nsight Systems captures record actual kernels and launch/preparation order.
Strict CUPTI binding passes for the four interleaved CuTe and four FA3 causal
cases: the graph symbol matches the loaded ELF; CuTe also matches its retained
compiler cubin. FA3 packages no PTX for these symbols. CuTe PTX/SASS is retained.
Nsight Compute counters are denied on this device; no stall percentage,
bandwidth ceiling or utilization limit is inferred from that denial.

## Reproduce

Use Python 3.13.18, CUDA-enabled PyTorch `2.15.0+cu132`, then install:

```bash
python -m pip install '.[cuda,hopper-fa4]'
```

The qualified core dependencies are CUTLASS DSL 4.8.1, TVM FFI 0.1.15.post0,
Quack 0.6.5 and torch-c-dlpack-ext 0.1.5. CUDA runtime/toolkit is 13.2; the bundled
CuTe compiler emits PTX ISA 9.4 for `sm_90a` and records its own NVVM version.
The FA3 baseline is a separate official source build of the pinned revision,
including D64/D128, BF16/FP16 and backward. Its binary hash is in the summary.

From a checkout, choose one explicit Hopper runner policy per process:

```python
from benchmarks.configured_hopper import make_attention

attention = make_attention()
out = attention(q, k, v, is_causal=True)
```

This factory fixes the interleaved policy before native compilation. The native
cache keys family, configuration and runtime tensor metadata; changing policy
construction midway through a process is outside this experiment's contract.

```bash
python -m benchmarks.compare_hopper_runner \
  --backends cute fa3 fa4 --seq 4096 --dim 64 128 \
  --dtype bf16 fp16 --modes dense causal --phases forward backward joint \
  --repeats 7 --window-ms 100 \
  --candidate benchmarks.configured_hopper:make_attention \
  --output results-local/h200-runner.json

python -m benchmarks.qualify_hopper --heads 16 --check-heads 0 7 15 \
  --lengths 128 512 4096 --scales default --modes causal \
  --candidate benchmarks.configured_hopper:make_attention \
  --output results-local/h200-directed.json
```

Use `compare_hopper_lpt` with `experimental_hopper_lpt:make_attention` for the
matched scheduler comparison, or `compare_hopper_masking` with
`experimental_hopper_lpt:make_separate_masking_attention` for the additional
D64 masking experiment. Initialization producer controls are diagnostics and
are excluded from the advertised timing boundaries.
