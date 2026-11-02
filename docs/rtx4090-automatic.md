# RTX 4090 automatic training selection

The standard attention APIs select the measured shared and short RTX 4090
training schedules on the pinned optional runtime. Install Python 3.13,
PyTorch 2.15.0/CUDA 13.2 and the `cuda,sm80-fa4` extras. Selection requires a
full 128-SM RTX 4090 with 101,376 bytes of opt-in shared memory.

```python
import torch
from kohakufa_cute import scaled_dot_product_attention

out = scaled_dot_product_attention(q, k, v, is_causal=True)
grads = torch.autograd.grad(out, (q, k, v), dout)
```

The selected inputs are contiguous, 16-byte-aligned FP16/BF16 BHSD tensors with
B1/H16 and equal Q/K/V shapes. The measured training families are S128/D16 and
S4096 with D64/D128, dense or implicit causal. The subsequent
[S1024 guarded-call update](rtx4090-s1024-host.md) extends automatic training
selection to S1024/D64/D128. Gradients must be enabled
and at least one input must request a gradient. Equal-head MHA remains eligible
with either value of `enable_gqa`.

Masks, dropout, genuine GQA, tails, rectangular inputs, other dimensions or
batch/head counts, strided or negative views, inference, compiled calls,
deterministic mode and other runtimes keep their preceding paths. An explicit
scale must be a positive normal FP32 value no larger than `2e38`; signed and
zero scales retain the existing implementation. Selected-path errors propagate.
The explicit RTX 4090 entry remains available for S1024 and its separately
qualified long no-grad behavior.

## Dispatch and arithmetic

Only immutable hardware classification and the native function handle are
cached. The native predicate checks the actual tensors, current device,
gradient mode and live gradient flags on every call. It also checks dimensions,
dtypes, shapes, contiguity, alignment and view flags. Python retains the live
mask, dropout, compilation, determinism, scale and optional-runtime checks.
Unsupported cold calls avoid loading a native extension solely for rejection.

A successful predicate calls the cached native Runner directly. The public
route uses the existing [shared schedules](rtx4090-shared-joint.md) and
[short layout/statistics choices](rtx4090-short-tuning.md). It adds no GPU
operation or precision correction. P and dS are each converted once to the input
dtype, products accumulate in FP32, delta uses actual rounded O, and dQ/dK scale
is applied after accumulation.

## Results

All **12/12** selected complete calls improve against the preceding actual public
API in both graph replay and eager execution on independent evaluation inputs.
Short eager latency falls about **9–10%**; S4096 eager speedups range from
**1.25–1.45×**. The full default grid wins **50/60 forward** and **44/60 backward
and combined** comparisons against SDPA. These counts compare measured medians;
the [record](../results/rtx4090-automatic.json) retains samples and variability.

BF16 complete calls, microseconds; lower is better. Graph and eager are separate
boundaries. The GPU is a full RTX 4090 at 400 W. Both implementations use this
same machine and stack; the earlier 450 W grid is not the speedup control.

| S | D | Visibility | Graph previous / automatic | Graph FA2 / SDPA | Eager previous / automatic |
| ---: | ---: | --- | ---: | ---: | ---: |
| 128 | 16 | dense | 14.31 / 9.28 | 14.50 / 17.83 | 280.46 / 256.00 |
| 128 | 16 | causal | 14.07 / 9.10 | 15.01 / 18.41 | 289.49 / 262.60 |
| 4096 | 64 | dense | 1926.55 / 1501.66 | 1489.98 / 1513.53 | 1928.14 / 1513.51 |
| 4096 | 64 | causal | 1219.66 / 857.98 | 1012.21 / 1052.86 | 1225.68 / 861.54 |
| 4096 | 128 | dense | 3810.07 / 3020.31 | 3007.75 / 3112.96 | 3792.75 / 3026.94 |
| 4096 | 128 | causal | 2319.80 / 1718.66 | 1773.68 / 1821.62 | 2328.37 / 1724.87 |

S1024 is retained as a measured negative control. Its initial candidate improved
graph F+B by 1.28–1.76× but made complete eager calls roughly 1–4% slower. That initial candidate stayed
explicit; the subsequent [host update](rtx4090-s1024-host.md) qualified a separate
actual public route. Graph gains alone did not qualify the initial candidate. The final evaluation uses a separate input seed.

Exact selected BF16 backward symbols were bound to compiler-returned and loaded
cubin bytes, with PTX/SASS retained. The mapped native host extension is recorded
by SHA-256. Systems verifies the actual short route and preceding route; the
long initial candidate also has separate S1024 timelines. These captures are
outside the timing windows. Compute counters remain unavailable in this
container; the results do not infer a hardware utilization ceiling.

The original bitwise gradient-subset test failed on existing FP32 atomic dQ
order, and its actual operands were retained. Six identical all-gradient calls
varied 46–55 dQ coordinates while the other components stayed identical; maximum
per-head dQ norm error against FP64 remained 0.2993%. All seven subset patterns
launched the same selected kernel symbols and resources. The final tests check
the established per-component norm contract, without a new bitwise guarantee.

## Validation and measurements

The preceding public Python namespace is retained verbatim for the paired
comparison and shares unchanged generic GPU methods with the candidate. The
new standard public API, preceding public API, explicit tuned API, official
FlashAttention 2 and default PyTorch SDPA receive identical rounded operands in
one process. Graph replay and complete eager calls have separate results;
profiling is outside timing windows. Forward retains autograd, backward uses a
saved forward, and joint includes forward plus all three gradients.

Per-head output and gradient norm limits remain 1% BF16 and 0.5% FP16 against
independent mathematical FP64 equations, with the declared rounded equations
retained as diagnostics. The existing dQ FP32 atomics can change final rounded
coordinates between identical calls. This route does not promise bitwise
identity between those reductions. Gradient-subset checks use the same
component accuracy contract.

The 60-case default-API grid also includes S128/S1024/S4096, D16/64/128/256/512,
both dtypes and both visibility modes. Its D512 timings compare actual default
SDPA; standalone FA2 does not support that width. Separate forward and backward
medians need not add to the complete-call median. These are attention-call
measurements, not complete-model training speedups.

Reproduce the preceding namespace with the recorded complete base revision:

```bash
python -m benchmarks.rtx4090_automatic_controls --prepare-parent --revision REVISION
python -m unittest tests.test_rtx4090_automatic -v
python -m benchmarks.qualify_baseline_precision \
  --source-revision REVISION --label 'RTX 4090 automatic training' \
  --repeats 5 --window-ms 100 --output results/rtx4090.json
python -m benchmarks.plot_performance
```
