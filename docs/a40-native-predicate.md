# A40 public eligibility checks

The explicit A40 entry now checks live tensor metadata in one native call.
This reduces complete short causal-call overhead while preserving the
[selected schedules](a40-causal-grouped.md) and [arithmetic](precision.md).
The GPU kernels and the default attention API are unchanged.

## Use and selection

Use Python 3.13, Torch 2.15.0/CUDA 13.2 and `.[cuda,sm80-fa4]`:

```python
from kohakufa_cute.fa4_sm80.a40 import scaled_dot_product_attention

output = scaled_dot_product_attention(q, k, v, is_causal=True)
```

The measured family remains a full NVIDIA A40, SM86/84 SMs/101376-byte shared
capacity; contiguous aligned BHSD FP16/BF16 tensors, B1/H16, square S1024 or
S4096; causal D64/D128 and dense D128. Other calls delegate to ordinary strict
CuTe. Default scaling and finite positive scales from `2**-126` through `2e38`
retain their existing eligibility. Compilation is warmed before timing.

The native helper checks shape, dtype, device equality, the current CUDA device,
contiguity and all three data-pointer alignments on every invocation. It caches
only physical device classification by CUDA device index. Python caches the
native function handle and still checks compiler state, deterministic mode,
mask, dropout, GQA, scale and optional-runtime availability on every call.
Tensor eligibility is never cached by object identity or shape.

## Actual public measurements

The [compact report](../results/a40-native-predicate.json) retains every sample,
all 48 forward/F+B graph/eager rows, independent accuracy metrics, source hashes,
tools and exact code bindings. `Python predicate` is the preceding qualified
public predicate with the same current schedule and native runner; this isolates
the host guard. `Direct runner` omits the public guard and is a diagnostic control.
FA2 and SDPA use their own complete implementations and arithmetic policies.

Measurements alternate six implementations in one process, using nine rotating
rounds, approximately 100 ms windows and a 300 ms warmup. The evaluation input
seed differs from earlier tuning and evaluation captures. Profiling runs are
separate. Compare paired rows within this run: absolute short eager latencies
across different captures are not an isolated change measurement.

Complete eager F+B wall time, milliseconds:

| Dtype | S | D | Python predicate | Native predicate | Direct runner | Ordinary CuTe | SDPA | FA2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 1024 | 64 | 0.345 | 0.320 | 0.312 | 0.334 | 0.382 | 0.752 |
| FP16 | 1024 | 64 | 0.347 | 0.322 | 0.313 | 0.334 | 0.383 | 0.753 |
| BF16 | 1024 | 128 | 0.350 | 0.324 | 0.316 | 0.386 | 0.390 | 0.756 |
| FP16 | 1024 | 128 | 0.346 | 0.322 | 0.314 | 0.390 | 0.385 | 0.752 |
| BF16 | 4096 | 64 | 1.440 | 1.448 | 1.450 | 2.090 | 1.566 | 1.503 |
| FP16 | 4096 | 64 | 1.478 | 1.488 | 1.485 | 2.121 | 1.591 | 1.531 |
| BF16 | 4096 | 128 | 2.797 | 2.813 | 2.802 | 3.832 | 2.933 | 2.792 |
| FP16 | 4096 | 128 | 2.873 | 2.866 | 2.856 | 3.936 | 2.987 | 2.863 |

Graph F+B CUDA-event time, milliseconds:

| Dtype | S | D | Python predicate | Native predicate | Direct runner | Ordinary CuTe | SDPA | FA2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 1024 | 64 | 0.147 | 0.146 | 0.146 | 0.227 | 0.194 | 0.176 |
| FP16 | 1024 | 64 | 0.147 | 0.147 | 0.147 | 0.226 | 0.193 | 0.176 |
| BF16 | 1024 | 128 | 0.272 | 0.272 | 0.273 | 0.383 | 0.316 | 0.282 |
| FP16 | 1024 | 128 | 0.273 | 0.276 | 0.276 | 0.390 | 0.316 | 0.284 |
| BF16 | 4096 | 64 | 1.433 | 1.439 | 1.443 | 2.085 | 1.557 | 1.498 |
| FP16 | 4096 | 64 | 1.465 | 1.484 | 1.476 | 2.117 | 1.582 | 1.518 |
| BF16 | 4096 | 128 | 2.802 | 2.802 | 2.803 | 3.818 | 2.922 | 2.787 |
| FP16 | 4096 | 128 | 2.852 | 2.867 | 2.853 | 3.925 | 2.974 | 2.850 |


At S1024, the native predicate reduces complete causal eager F+B by about
23–25 microseconds, or 6.8–7.2%, across these four cases. Both D64 dtypes now
improve over ordinary CuTe directly in this paired run. About 8–9 microseconds
remain between the public entry and direct runner.

Graph F+B does not consistently improve, and long causal and dense differences
are small and mixed. The compact report retains those rows, including regressions;
there is no uniform speedup claim for long calls. All three captured selected
backward ELFs, their SASS and resource allocations are byte-identical to the
preceding public entry. Every before/native graph row has identical symbols,
grids, blocks, registers, local bytes and shared-memory requirements. This is a
host-dispatch improvement, with no new pipeline or numerical method.

## How the intervention was selected

A metadata-only microbenchmark measured the old predicate at about 7.5
microseconds, with schedule/runner lookup near 1 microsecond. Those controls
perform no attention. Instrumented cProfile and scoped NVTX runs changed the
public/direct ranking, so they were not used to assign the complete-call gap to
individual functions or a thread-handoff mechanism.

A separate uninstrumented same-input experiment bypassed only the predicate
after verifying eligibility outside the timing window. Across four S1024
D64/D128 dtype cases, complete eager latency fell by 36–38 microseconds while
graph resources stayed identical. A schedule-prewired control removed another
3.5–4.3 microseconds. These are diagnostic controls, not production bypasses.
They supported implementing a combined native guard and then remeasuring the
actual guarded API rather than extrapolating the isolated microbenchmark.

The Systems capture includes 160 warmup F+B calls and 64 ranged F+B calls:
224 calls total, each with forward, backward preparation, backward main and
backward postprocessing. Its 896 launches therefore match four kernels per
F+B; dividing by only the ranged calls would overcount launches per call.
Backward work can launch on an autograd engine thread, so an NVTX GPU projection
of the Python caller's range alone does not establish the complete F+B launch
count. Scope/thread correlation and exact kernel names are retained in the raw
analysis. No runtime stall percentage or GIL mechanism is inferred from it.

## Qualification

Actual-object parity tests pass more than 100 comparisons against the preceding
Python policy, including valid families, shape boundaries, both dtypes,
misaligned and noncontiguous views, mixed devices/dtypes, live storage-offset
mutation, scale/options and runtime flags. Native tests also reject CPU and
non-tensor objects. This full A40 has one GPU; an alternate-current-device test
branch runs only when another device exists.

The candidate passes 106 directed arithmetic/statistic cases, 13 public/control
GPU tests, and memcheck, racecheck, synccheck and initcheck. Tool probes use one
actual selected S4096 F+B per BF16 D64/G4 and FP16 D128/G4 family, plus changed
and poisoned S1024 graph replays for BF16 D64/G16 and FP16 D128/G4. Identical
rounded operands and actual outputs go to the unchanged CPU FP64, rounded-policy
and saved-statistic oracles. Only host API negotiation diagnostics use
`--report-api-errors no`; device checks remain enabled. No timeout is counted as
a successful result in this qualification.

Output and dQ/dK/dV retain separate per-head limits of 1% for BF16 and 0.5% for
FP16. Directed fixtures use the declared rounded equations and actual rounded O
for delta, with their existing cancellation/underflow checks. Literal maxima and
shifted log2 normalizers are checked separately. CPU checks pass 284 tests with
162 skips; publication-guard tests are run separately.

Reproduce from the qualified stack, retaining captures outside the checkout:

```sh
python -m unittest tests.test_a40_native_predicate tests.test_a40_shared -v
python -m benchmarks.compare_attention_variants --seq 1024 4096 --dim 64 128 \
  --dtype bf16 fp16 --visibility causal --phases forward joint \
  --warmup-ms 300 --window-ms 100 --repeats 9 --seed 2477 \
  --operations '[{"label":"before","factory":"benchmarks.a40_predicate_candidates:make_attention","kwargs":{"mode":"python"}},{"label":"native","factory":"benchmarks.a40_predicate_candidates:make_attention"},{"label":"sdpa","backend":"sdpa"},{"label":"fa2","backend":"fa2"}]' \
  --output "$CAPTURE_DIR/paired.json"
```
