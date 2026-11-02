# RTX 4090 short full-row backward

The existing optional RTX 4090 entry now selects an exclusively owned backward
kernel for contiguous, aligned square **B1 H16 S128 D16 FP16/BF16 training** on
a full 128-SM RTX 4090. Dense and causal calls are qualified. The D64/D128
schedules and ordinary CuTe fallbacks remain available. Install the
[pinned optional runtime](rtx4090-shared-joint.md).

```python
from kohakufa_cute.fa4_sm80.rtx4090 import scaled_dot_product_attention

out = scaled_dot_product_attention(q, k, v, is_causal=True)
grads = torch.autograd.grad(out, (q, k, v), dout)
```

The short specialization requires active gradient recording and at least one
input requesting gradients. No-grad calls retain ordinary CuTe. Masks, dropout,
GQA, unsupported scales, other shapes/devices and misaligned or noncontiguous
inputs retain their existing selection. Gradient flags and mutable tensor
metadata are checked live. Selected-kernel errors propagate. Default top-level
attention dispatch is unchanged; this entry remains opt-in.

## Ownership and code

This reuses the [short full-row design](rtx3090-short.md) with independent
qualification on the actual RTX 4090. One 256-thread CTA per head owns all
128 query/key rows and writes dQ/dK/dV directly. QK and dP are reconstructed
once. Delta is computed inside that CTA from the actual rounded O and dO.
The kernel needs no global dQ scratch, gradient atomics, separate delta
preprocessing or dQ conversion pass. The unchanged forward plus one backward
kernel replaces the preceding entry's four-pass training graph.

The [declared arithmetic](precision.md) is unchanged: separate unscaled saved
statistics, input-dtype P before dS, one dS conversion, FP32 products, and late
dQ/dK scaling. Shared memory remains 99,840 bytes: four padded input tiles,
separate P/dS tiles and three FP32 row-statistic arrays. No staging buffers are
aliased. All gradient coordinates have one writer.

The measured dtype/visibility policies use register prefetch of immutable row
statistics, a separate gradient-product MMA layout, or both. The
[tuning evidence](rtx4090-short-tuning.md) explains the independent controls and
why they are selected differently. Their arithmetic and shared buffers are
unchanged.

Actual selected backward code uses 128–174 registers/thread and zero reported
local bytes across the four cases. Matched FA2 uses 255 registers/thread,
98,304 shared bytes and 48–80 local bytes. Its selected sequence-parallel
backward retains global dQ accumulation and separate preprocessing/conversion.
These resources and static instruction sites do not establish spill costs,
stall fractions or utilization. The specialized code has no query/key tile
loop; FA2's inner query and outer key loops each cover one tile here. Static
counts include prologue, loop and drain code and are not execution percentages.

## Measurements

Microseconds, lower is better. Every row uses B1/H16/S128/D16. Graph timings
include all GPU preparation and gradient work; compilation, capture and host
allocation are excluded. Standalone FA2 and SDPA retain their own arithmetic
and preparation, so their columns compare complete implementations.

| Dtype | Visibility | Shared baseline F+B | FA2 F+B | SDPA F+B | Selected F+B | Selected backward |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| bf16 | dense | 10.133 | 14.465 | 17.801 | 9.256 | 5.412 |
| bf16 | causal | 9.249 | 14.991 | 18.469 | 9.132 | 5.186 |
| fp16 | dense | 9.600 | 14.459 | 17.814 | 9.038 | 5.144 |
| fp16 | causal | 9.247 | 15.008 | 18.480 | 9.117 | 5.170 |

These policies reduce graph latency 1.3–8.7% against the shared class with both
controls disabled, and are 1.56–1.65× faster than standalone FA2 in these calls.
The initial four-pass-to-two-pass change is retained in the
[original dataset](../results/rtx4090-short.json). This run selected
PyTorch's Flash backend, including two input-copy kernels in its six-pass
training graph. All preparation is included. Systems independently confirms
two GPU passes per eager call for both shared-kernel policies.

The [current record](../results/rtx4090-short-optimized.json) includes 24 graph/eager
forward, backward and combined measurements plus eight no-grad controls. Nine
rotated 100-ms sampling windows followed one second of warmup. Evaluation seed
607931 differs from selection seed 98271. Same-process eager F+B is effectively
tied; this follow-up claims resident graph gains. Both capture orders with
shared input storage preserve those gains. A separate persistent-process
comparison uses the exact initial class through the matched native host,
including eight original compilation units that match the initial release's
loaded code. Its timings have a different process boundary and are retained
separately. No model or serving gain is inferred. No-grad forward preserves
its previous fallback.

## Qualification and reproduction

Eighteen finite directed cases retain the existing per-head output/dQ/dK/dV
limits: 1% BF16 and 0.5% FP16 against independent FP64 diagnostics, plus the
qualified declared-rounding checks. Cases include negative score offsets,
sharp attention, cancellation and operand underflow, saved statistics,
changed inputs with poisoned graph outputs and negative tensor views. A
separate causal FP16 fixture intentionally overflows the final dV cast; its
infinity mask/sign and representable coordinates are checked separately. A
distinct scaled-dO fixture checks all four finite components. Existing per-head
and directed acceptance criteria are retained.

Memcheck, racecheck, synccheck and initcheck each passed twelve training calls
with CPU reference calculations outside GPU instrumentation. Actual public
tests also cover each independently requested Parameter gradient and no-grad
fallback. Exact graph symbols were matched to four loaded specialized cubins
and four actual FA2 cubins, with compiler PTX, selected PTX entries and nonempty
SASS retained outside Git. The qualified stack is Python 3.13.18,
PyTorch 2.15.0/CUDA 13.2, CuTe DSL 4.8.1 and Triton 3.9.0. Official FA2 revision
`4616f5116d33e3ba3cbb71d582caffcfcb49649c` uses its verified SM80 binary on this
SM89 device. The GPU's configured power limit was 400 W.

The final selected CUBIN/SASS pairs match the qualified per-case experiment
byte for byte. The basic shared-class defaults were also requalified on
RTX 3090 and L40S. Some default-false code generation changes, so compatibility
is established by actual checks and timings rather than a byte-identity claim.
The actual host binary and final evidence inventories are separately recorded
and hash-verified.

```sh
capture_dir=$(mktemp -d)
python -m benchmarks.validate_rtx4090_short --output "$capture_dir/qualification.json"
python -m benchmarks.compare_attention_variants --operations '[{"label":"selected","factory":"benchmarks.rtx4090_short_optimized_controls:make_attention"},{"label":"previous","factory":"benchmarks.rtx4090_short_optimized_controls:make_attention","kwargs":{"policy":"previous"}},{"label":"fa2","backend":"fa2"},{"label":"sdpa","backend":"sdpa"}]' --seq 128 --dim 16 --heads 16 --dtype bf16 fp16 --visibility dense causal --phases forward backward joint --warmup-ms 1000 --window-ms 100 --repeats 9 --seed 607931 --source-revision "$(git rev-parse HEAD)" --output "$capture_dir/comparison/report.json"
```

Complete source, exact commands and runtime configuration were saved before
launch. Actual qualification inputs were saved before each call. Reports, loaded code and Systems captures
were copied off-host and checked against their complete inventories and hashes.
