# RTX 4090 live metadata checks

The optional [RTX 4090 entry](rtx4090-shared-joint.md) now checks Q/K/V metadata
in one native call. The existing schedules, GPU kernels and
[rounding contract](precision.md) are unchanged. This reduces host overhead for
short eager attention calls; it does not make the backward kernel faster.

The native helper checks the actual tensors on every call: shapes, strides,
dtype, device, current CUDA device and pointer alignment. Only immutable device
classification and the function handle are cached. Python retains the live
compiler, determinism, mask, dropout, GQA, scale and optional-runtime guards.
Parameters, Tensor subclasses and parametrized module tensors remain eligible
when their current metadata qualifies. Unsupported calls keep the existing
CuTe fallback. The supported device and shape family are unchanged.

## Complete-call measurements

Milliseconds, lower is better. These are synchronized eager forward plus
backward calls, including all three gradients, normal allocation, dispatch,
preparation and launches. Both policies used the same native Runner and GPU
schedules in one process. The previous policy is retained in
[`rtx4090_predicate_candidates.py`](../benchmarks/rtx4090_predicate_candidates.py).

B1/H16/S1024, full 128-SM RTX 4090, 400 W limit:

| Dtype | Head dim | Visibility | Previous Python check | Combined native check | Saved microseconds |
| --- | ---: | --- | ---: | ---: | ---: |
| bf16 | 64 | dense | 0.3008 | 0.2658 | 35.0 |
| bf16 | 64 | causal | 0.3041 | 0.2675 | 36.6 |
| fp16 | 64 | dense | 0.2608 | 0.2233 | 37.5 |
| fp16 | 64 | causal | 0.3060 | 0.2689 | 37.1 |
| bf16 | 128 | dense | 0.3052 | 0.2698 | 35.4 |
| bf16 | 128 | causal | 0.3089 | 0.2722 | 36.7 |
| fp16 | 128 | dense | 0.3044 | 0.2688 | 35.7 |
| fp16 | 128 | causal | 0.3063 | 0.2699 | 36.5 |

The reduction is 11.6–14.4% in these eight calls. Seven rotated 100 ms sampling
windows followed 500 ms warmup, using independent evaluation seed 9731. The
[compact record](../results/rtx4090-native-metadata.json) retains event and wall
samples separately, 96 training measurements and 32 no-grad measurements across
S1024/S4096, D64/D128, both dtypes and both visibility modes. It also includes
ordinary CuTe and SDPA controls. Graph replay excludes host eligibility checks
and therefore does not establish this host improvement. Longer GPU-bound calls
and standalone backward do not have a general speedup claim.

A separate metadata-only loop measured about 1.77 microseconds for the combined
check versus 12.86 microseconds for the preceding Python check. This diagnostic
has a different timing boundary; its difference is not an additive explanation
of the complete-call saving. Instrumented trace durations are excluded from
the reported medians.

## Qualification and deployed code

Actual GPU tests checked 168 metadata decisions, including supported boundaries,
other head widths, tails, misalignment, noncontiguous views, CPU and FP32 tensors,
changed storage, Parameters, subclasses and parametrized tensors. Compiler,
determinism and runtime guards were checked live. The current-device mismatch
test is conditional on a second GPU; this machine had one. Another 28 public
fallback cases passed. The timing matrix independently checked output and
dQ/dK/dV against the existing per-head FP64 limits of 1% BF16 and 0.5% FP16.

Native and preceding-Python controls had equal captured resource lists. Three
selected backward captures, covering short/long D64 causal and short D128 dense
calls, had byte-identical loaded CUBINs and SASS. Systems captured the same four
GPU passes per eager training call, twelve times for each policy. The actual
new host extension was separately archived and its SHA matched the preflight
binary. The record includes these hashes and the verified archive inventory.
The GPU implementation retains its prior directed and sanitizer qualification;
this change introduces no GPU memory ownership or synchronization changes.

The qualified stack was Python 3.13.18, PyTorch 2.15.0/CUDA 13.2, CuTe DSL 4.8.1
and Triton 3.9. Full source, commands, runtime configuration, reports and actual
loaded code were retained outside Git before and throughout the run. The main
top-level attention APIs retain their existing selection; this remains opt-in.
