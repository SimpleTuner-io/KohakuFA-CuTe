# Explicit warp inference experiment

`AttentionRunner.inference(q, k, v, scale=None, causal=False)` is an experimental
forward-only control. It does not change `AttentionRunner.__call__` or automatic
dispatch. Initial qualification targets A100, square contiguous aligned
FP16/BF16 D64/D128 inputs and the existing positive-normal scalar scale range.

The method requires NoGrad or inputs that do not request gradients. Forward-AD
dual inputs are rejected even under NoGrad. The inference branch independently
checks contiguity, alignment and pending logical conjugation/negation, so a later
training-layout extension cannot silently enable unqualified inference views.

The existing sixteen configuration fields remain unchanged. Inference has a
separate compile mode and cache discriminator. Its initial ABI retains fourteen
tensor arguments: unused slots alias the owned output, with matching real and
fake dtype/rank. Auxiliary fast-division setup reads Q/K dimensions, while the
maximum-buffer access is inside the absent-LSE store branch.

The experiment omits maximum/log-normalization allocations, their stores and
autograd context creation. A default-true softmax-finalization flag permits
omitting the unused final log2 operation. Internal maxima, shifted exponentials,
FP32 sums, the normalization reciprocal and output rounding stay unchanged.
Generated PTX and captured resources must establish which operations the
compiler actually removes; source conditions alone do not establish a gain.

Fourteen rank-four descriptors plus seventeen configuration values and seven
header values require 164 cache-key integers. The inference path reserves 176;
the previous 160 reservation remains for ordinary calls. Eager measurements
include this host behavior and the fourteen-descriptor ABI. Device-event graph
measurements isolate the captured device path. A four-tensor inference ABI is a
separate possible follow-up, not part of this control.

Acceptance requires independent per-head FP64 output errors below 1% BF16 or
0.5% FP16, tails, scales, changed/poisoned graphs, current-stream execution,
training/inference cache alternation and all four sanitizer tools. Comparisons
include the ordinary runner under NoGrad, the public API, standalone FA2 and
SDPA on identical inputs. No default activation is implied by those results.

## Qualified A100 results

The actual A100-SXM4-40GB run passed all three GPU test methods, all four
sanitizer tools, and 24 eager plus 24 graph rows. It used the pinned Python
3.13.18, Torch 2.15 CUDA 13.2 and CUTLASS DSL 4.8.1 stack on the full 108-SM,
350 W device. Qualification includes batch 1, H2/H16, D64/D128, both dtypes,
dense/causal visibility, short and partial tiles, changing scales, poisoned
replay, current streams and alternating training/inference cache use.
Other targets require their own qualification. This method remains an explicit
experiment and is not selected by the ordinary public API.

The [retained measurements](../results/a100-explicit-inference.json) use five
rotating windows targeting 50 ms, after compilation and one-second warmup.
Both the ordinary runner and the explicit inference method run under NoGrad in
the same experimental runtime. Eager includes required preparation and launch;
graph measurements isolate the captured device path.

| Sequence | Eager geometric speedup | Eager wins | Graph geometric speedup | Graph wins |
| --- | ---: | ---: | ---: | ---: |
| 128 | 1.663× | 8/8 | 1.026× | 8/8 |
| 1024 | 1.009× | 2/8 | 1.004× | 4/8 |
| 4096 | 0.989× | 2/8 | 0.989× | 2/8 |

Each group covers both head dimensions, dtypes and visibility modes. Long calls
have no demonstrated benefit; this evidence supports short explicit inference
use, not a broad automatic replacement.

For example, H16/S128/D128 BF16 latencies in microseconds were:

| Boundary / visibility | Ordinary runner | Inference | FA2 | SDPA |
| --- | ---: | ---: | ---: | ---: |
| Complete eager / dense | 41.02 | 25.69 | 153.63 | 54.32 |
| Device graph / dense | 9.76 | 9.53 | 10.21 | 10.68 |
| Complete eager / causal | 40.09 | 23.96 | 155.24 | 54.65 |
| Device graph / causal | 6.62 | 6.41 | 12.05 | 13.13 |

These are complete backend comparisons under their respective implementations,
not isolated scheduling comparisons. The FA2 eager wrapper has substantial
host overhead on this stack; its graph results show the separate device cost.

Actual traces retain a single forward kernel in each path. The ordinary runner
records two `aten::empty` statistic buffers; inference records neither, while
both retain the owned output allocation (`empty_like`/`empty_strided`). This is
evidence of removed host preparation. It does not assign a runtime percentage
to any device instruction. D128 causal resources change from 220 to 232
registers without local storage; the longer causal call is slower. That
association alone does not establish the cause of the regression.

The original compiler-returned PTX/CUBIN properties pointed to unavailable
files. Those are recorded as unavailable, not successful disassembly. Source
conditions express the intent to omit final log2 and statistic stores; a
separate CUPTI loaded-image capture is retained as the next diagnostic before
claiming exact device instruction removal. Timing and arithmetic qualification
above do not depend on that artifact export.

```python
import torch
from dataclasses import asdict
from kohakufa_cute.fa4_sm80._schedules import a100_schedule
from kohakufa_cute.native import extension

runner = extension().AttentionRunner("warp", asdict(a100_schedule(128, 128, False)))
with torch.no_grad():
    output = runner.inference(q, k, v)
```

The method rejects requested reverse-mode gradients, forward-AD dual inputs,
unsupported scales and unqualified storage views. Its ordinary `__call__`
continues to produce saved statistics and gradients.
