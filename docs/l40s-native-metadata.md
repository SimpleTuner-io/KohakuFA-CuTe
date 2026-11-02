# L40S live metadata checks

The bounded L40S public path checks live tensor metadata in one native call.
Its [training schedules](l40s-warp-training.md), eligibility bounds, arithmetic,
GPU kernels and fallbacks stay unchanged. The change targets repeated Python
metadata work in short complete calls.

The helper reads all three tensors on every invocation: shape, dtype, device,
current CUDA device, contiguity, pointer alignment and gradient requirements.
It requires active grad mode and at least one Q/K/V requiring gradients. It
caches only the function handle and full-device classification: NVIDIA L40S,
SM89, 142 SMs, 101376-byte opt-in shared limit. Eligibility is never cached by
tensor identity, shape or storage address.

Compiler, deterministic-mode, mask, dropout, positive-normal FP32 scale and
qualified-runtime checks remain in Python. Equal-shape MHA still accepts either
value of `enable_gqa`; differing Q/KV shapes keep their preceding handling.
Before the first selected call, the preceding Python predicate rejects unsupported
cold calls without loading an extension merely to reject them. A required helper
initialization error propagates instead of choosing another computation.

## Measured complete calls

The [96-row report](../results/l40s-native-metadata.json) and
[paired-variability chart](../results/l40s-native-metadata.svg) compare the actual
public API with the preceding public Python namespace. The control loads pristine
parent API/policy files at revision `3f6ba0465a2f0887a0992831f0caa904beb939b3`.
It explicitly shares the unchanged native runner, Schedule and GPU modules to
isolate eligibility work and avoid duplicate custom-op registration. It is not
a shorter direct-runner or predicate-bypass proxy.

The full 142-SM L40S runs at a 350 W limit with driver 615.71.09, Python 3.13.18,
Torch 2.15.0/CUDA 13.2, CuTe DSL 4.8.1, TVM FFI 0.1.15.post0 and quack 0.6.5.
The held-out seed is 21331, distinct from the earlier bypass diagnostic. Nine
rotating rounds use a 1000 ms warmup and approximately 100 ms windows. Forward,
backward and joint, graph and complete eager, both dtypes, D64/D128, dense/causal
and S1024/S4096 rows remain intact. Compilation, allocation and capture are warmed
separately. Profiling is separate from these measurements.

S1024 complete eager forward + backward wall time, microseconds:

| Dtype | D | Visibility | Parent median | Native median | Median paired saving |
| --- | ---: | --- | ---: | ---: | ---: |
| BF16 | 64 | dense | 124.27 | 119.41 | 3.36 |
| BF16 | 64 | causal fallback | 124.06 | 120.04 | 0.01 |
| FP16 | 64 | dense | 233.84 | 224.61 | 9.24 |
| FP16 | 64 | causal fallback | 118.84 | 110.17 | 0.87 |
| BF16 | 128 | dense | 173.15 | 173.75 | -0.74 |
| BF16 | 128 | causal | 125.27 | 125.31 | -0.25 |
| FP16 | 128 | dense | 235.54 | 226.96 | 8.24 |
| FP16 | 128 | causal | 238.15 | 227.07 | 10.40 |

Short FP16 dense D64 and causal D128 improve in every paired round, by
5.1–11.3 and 5.6–19.8 microseconds respectively. Their median latencies fall
4.0% and 4.7%; dense D128 falls 3.6%, with one essentially tied paired round.
Other short rows are mixed or tied. In particular, the D64 causal difference of
latency medians exceeds its median paired saving; both statistics and all samples
remain visible. The earlier 12–13 microsecond bypass potential is not a uniform
realized gain.

Long-call differences are mixed. Dense D128 eager joint medians regress about
0.7–0.9%, while causal D128 improves about 0.9–1.0%; paired samples vary.
There is no consistent long-call or CUDA-graph speedup claim. FA2 and SDPA remain
complete implementation comparisons under their own arithmetic policies, with
all rows retained rather than only favorable cases.

## Execution and qualification

All 24 exact loaded forward/backward code bindings form 12 byte-identical
public/parent cubin, PTX and SASS pairs across the three selected causal shape
families and both dtypes. All 12 also match the retained causal-ordering run.
Every public/parent graph row has identical launch symbols, grids, blocks,
registers, local bytes and shared allocations. The actually mapped host C++ bridge
is separately retained by source SHA, library SHA and matched file inode.
No GPU occupancy, stall, cache or bank-conflict explanation is inferred.

The final source passes 12 GPU/API test methods: all three guarded APIs and the
explicit shared entry, cold/warm caches, saved statistics, changing/poisoned graph
storage, unaligned dO, fallback/compile paths and required-loader failures. More
than 400 actual metadata comparisons include both dtypes and GQA flag values,
shape/scale boundaries, parameters, plain tensor subclasses, parametrized modules,
mutable storage offsets and gradient flags. An alternate-current-device branch
requires multiple GPUs; this machine has one.

The CPU suite reports **378 total tests, 187 expected GPU skips**; 24 publication
checks pass separately. All CUDA-method source and memory ownership remain
unchanged, and deployed binary identity matches the earlier qualification, so its
passing memcheck, racecheck, synccheck and initcheck are retained. They are not
reported as newly rerun tools.

Independent per-head output/dQ/dK/dV limits stay at 1% BF16 and 0.5% FP16. The
saved unscaled maximum/log normalization, once-rounded input-dtype P and dS,
FP32 products, actual-rounded-O delta and late dQ/dK scale are unchanged.
Identical previously calibrated CPU reference equations use four threads for
checks; the timing process keeps its original thread policy. No accuracy threshold
was widened and no extra accuracy method was enabled.

All 380 actual artifact files, 553576666 bytes, match their physical off-host SHA
inventory: input tensors, final samples, loaded GPU code, the mapped native bridge
and build command, logs and receipts. Complete frozen source includes both the
candidate and pristine parent namespace. Raw captures remain outside the tree.
The initial upload-only attempt failed before launch; its source and failure are
retained separately from the passing 34-stage run.

Recover the measured parent control and regenerate the report:

```bash
python -m benchmarks.l40s_public_metadata_controls --prepare-parent \
  --revision 3f6ba0465a2f0887a0992831f0caa904beb939b3
# CAPTURE_DIR is outside this checkout and holds the retained release artifacts.
python -m benchmarks.summarize_l40s_native_metadata \
  --captures "$CAPTURE_DIR/release" --checkpoint "$CAPTURE_DIR/checkpoint" \
  --source-tar "$CAPTURE_DIR/frozen-source.tar.gz"
```

The final integration preserves the merged A40 D256 and GB10 helpers. All six
actual GPU/API tests and 16 fresh short-shape joint timing rows completed. Graph
execution remains essentially unchanged. Eager samples retain substantial host
variability; BF16 dense D128 has a 1.4% regression in differences of medians,
while its median paired difference is only -0.21 microseconds. These samples
supplement the original evaluation without replacing it. The final CPU suite
ran 385 tests with 187 skipped and no failures, and all 24 publication checks
passed. All 24 final raw files (103415873 bytes), including the actual mapped
native bridge and its build configuration, match the physical SHA inventory.

To retain the independent integration supplement when regenerating the report,
add `--integration "$CAPTURE_DIR/integration"`,
`--integration-checkpoint "$CAPTURE_DIR/integration-checkpoint"` and
`--integration-source-tar "$CAPTURE_DIR/integration-source.tar.gz"`. The generator
verifies every retained input file and requires completed GPU and CPU stages.
