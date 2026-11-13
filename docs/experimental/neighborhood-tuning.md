# H100 neighborhood scheduling

The scheduling change keeps 32-column forward and dQ tiles, uses 64-column
dK/dV tiles in a bounded H100 workload family, and removes two unused
statistic-gradient fills. At image width 64, forward and dK/dV enumerate their
contiguous active tile interval directly. dQ keeps its original traversal.
The [precision contract](../precision.md), operand conversions, unscaled
statistics and FP32 accumulation remain the same.

On the representative B1/H8/D64 BF16 64 x 64/window-7 case, final resident
training latency fell from **151.82 to 116.61 us**, a 23.2% reduction.
FlexAttention measured **121.51 us** in the same process. This closes that
case's gap; other geometries remain slower than FlexAttention.

Final training CUDA graph medians, in microseconds:

| dtype | image/window | original | production | FlexAttention |
| --- | --- | ---: | ---: | ---: |
| FP16 | 32 x 32 / 7 | 54.47 | 41.29 | 27.70 |
| FP16 | 64 x 64 / 7 | 167.26 | 116.71 | 122.27 |
| FP16 | 64 x 64 / 13 | 290.69 | 204.33 | 172.25 |
| FP16 | 128 x 128 / 7 | 698.74 | 688.57 | 597.86 |
| FP16 | 33 x 49 / 7 | 83.48 | 79.41 | 79.88 |
| BF16 | 32 x 32 / 7 | 49.12 | 41.92 | 27.72 |
| BF16 | 64 x 64 / 7 | 151.82 | 116.61 | 121.51 |
| BF16 | 64 x 64 / 13 | 269.54 | 205.61 | 171.66 |
| BF16 | 128 x 128 / 7 | 790.11 | 786.40 | 596.82 |
| BF16 | 33 x 49 / 7 | 85.12 | 78.38 | 79.78 |

The tail case uses B2/H4; the others B1/H8. Seven of eight fresh evaluation
training medians improved over the original control. FP16 52 x 64/window 5
was 1.7% slower (95.65 versus 94.03 us). That residual regression is retained,
without retuning to the final evaluation. The dispatch remains a bounded
heuristic rather than an all-shape speed guarantee.

Selection measurements and final production evaluation are separate cohorts.
The [complete tables](neighborhood-tuning-tables.md) and
[machine-readable evidence](neighborhood-tuning.json) retain paired samples,
ranges, every component's FP64 diagnostics, actual kernel counts, source hashes,
and resolved production schedules. The
[validation record](neighborhood-tuning-validation.json) records actual stage
outcomes and archive verification.

## Default dispatch

The wider dK/dV schedule applies to contiguous FP16/BF16 D64 inputs on a full
132-SM H100 80GB HBM3 with capability 9.0. It requires at least 1,024 tokens,
at least 128 owner CTAs, an image width from 32 through 64, and a square odd
window from 5 through 13. Owner CTAs are
`batch * heads * ceil(height * width / 64)`.
The interval `264 < owner CTAs <= 396` uses C32 as well: this is the first
range requiring an additional residency batch at the wider kernel's measured
two-CTA resource limit, while C32 has a three-CTA limit in the failing FP16
case. Its profiled dK/dV duration rose from 18.27 to 25.65 us. BF16 at that
geometry retains a two-CTA limit and improves; the shared exclusion deliberately
forgoes that gain. This is a conservative dispatch exclusion supported by a
measured regression, not an occupancy or hardware-ceiling measurement.

Only image width 64 enables direct interval traversal, for forward and dK/dV.
Other widths use the spatial row traversal. Calls outside this family use the
original C32 schedule. Removing unused statistic gradients applies to the
autograd interface without selecting a different arithmetic policy by device.
The bounded scheduling heuristic does not promise a speedup for every shape
inside its range. Numerical support for other API shapes is unchanged.

## Measured interventions

The first screen changed one owner at a time. On the BF16 64 x 64, window-7
case, widening forward increased its profiled duration from 37.4 to 53.9 us;
widening dQ increased 29.7 to 37.7 us. Widening only dK/dV reduced 79.3 to
54.9 us. These instrumented owner durations diagnose the choice; complete-call
uninstrumented samples establish performance.

Compiler resources explain a relevant scheduling boundary without measuring
occupancy or stalls. The C32-to-C64 change raised forward registers from
128 to 168, dQ from 128 to 214, and dK/dV from 168 to 248. Resource-derived
CTA limits changed from 4 to 3, 4 to 2, and 3 to 2 respectively. There were
512 CTAs on 132 SMs. Higher occupancy did not explain the winning dK/dV change.

For equal 64-column active work, the inspected dK/dV traversal loop contained
1,806 versus 1,204 static instructions, with equal HMMA counts, four versus two
barriers, and 80 versus 72 LDSM sites. These are compiler instruction counts
with predicates and branches, not executed-instruction percentages or measured
stall attribution. Compiler-returned artifacts and cache captures remain
distinct from a CUPTI loaded-module binding.

Direct traversal preserves the complete ordered list of active tiles. CPU
enumeration checks both forward and inverse visibility, shifted boundaries,
and partial owners. GPU experiments require every O/dQ/dK/dV byte, including
signed zero, to match the same-column control. Removing statistic fills has
its own full-tensor byte gate against the original C32 control.

Direct traversal for dQ was a negative result: its register allocation rose
from 128 to 130 and its resource-derived CTA limit fell from 4 to 3. Restoring
its original traversal improved the complete call. C128 dK/dV experiments
also regressed. At image width 128, C64 dK/dV slowed the retained FP16 case,
so production keeps C32 there. Width-32 direct traversal was slower than its
same-column row-traversal control, so it also remains disabled.

## Measurement and qualification

All final compilation, formatting, CPU checks, GPU checks, numerical reference
calculations and sanitizer work run on the rented H100. The production source
and exact launch scripts are frozen off-host before execution. Raw operands,
traces, compiled code, captures and stage receipts remain outside the tracked
tree; complete final inventories and hashes must verify before resource removal.

The broader candidate's ten-case matrix uses both eager and CUDA graph
forward/training intervals. Its first six held-out geometries passed numerical
gates but exposed a roughly 10% FP16 latency regression at 96 x 32/window 5,
with 384 owner CTAs. The narrower final rule excludes that work-count interval.
The first evaluation is retained as refinement evidence; it is not held-out
validation for the revised rule.

Final default timings cover the original ten cases and four fresh geometries,
in both dtypes, using resident CUDA graph forward and training calls. Each
final timing group has seven paired
rounds targeting 100 ms, with eight complete calls per graph replay. Backend
order rotates and alternates. Compilation, allocation, warmup, graph capture,
changed-input validation and profiling are outside the reported samples.
Layout, inputs and output contracts match within each process.

The first evaluation geometries were 48 x 64/window 9, 96 x 32/window 5,
45 x 51/window 9, 64 x 47/window 11, and 64 x 64/window 7 with one or sixteen
heads. The first two use B1/H8, the third B2/H4, and the fourth B1/H8.
The fresh final evaluation uses B1/H8 at 64 x 32/window 5,
40 x 64/window 5, 52 x 64/window 5, and 54 x 49/window 11. These exercise
256- and 416-CTA fast paths and 320- and 336-CTA fallbacks. The revised
dispatch predicate and this new evaluation set are declared before measurement.

Independent sparse FP64 references check sampled boundary and interior tokens
for O and every gradient, including all contributing queries for sampled keys.
Per-head mathematical norm-error limits remain 0.5% FP16 and 1% BF16. Rounded
diagnostics retain actual rounded O. Every captured graph must read changed
Q/K/V/dO. The additional production GPU method asserts that the default resolver
selects KV64, then checks offset-one gradients, poisoned graph outputs and two
changed-data replays on both an exact-traversal shape and a partial-tail shape.
Focused sanitizers include that method so fallback-only tests cannot qualify
the new fast path.
An intermediate version of this fixture first initialized autograd state on
the default stream and then captured on a side stream. PyTorch 2.13 rejected
that capture. Those qualification and sanitizer stages remain failed in the
receipts, despite zero reported memory/race diagnostics. The repaired fixture
uses the same side stream from its first autograd call and is requalified under
a new immutable source identity.

The final source passed 513 CPU tests with 221 skips, 10 neighborhood GPU
methods with no skips, and memcheck, racecheck, initcheck and synccheck with
three focused methods each and no reported errors. The standalone repaired
production method also passed before the final benchmark. These checks do not
assert that the entire upstream GPU suite passes; its separately documented
baseline failures remain outside this neighborhood qualification.

A separate untimed deployment captured the selected FlexAttention launcher
configurations and their module-load binary associations for BF16 window 7,
alongside CuTe compiler objects and actual eager/graph kernel traces. It kept
the static launcher enabled and did not install Triton launch hooks. The opaque
cached C++ runner association uses its original static function and matching
cache/configuration; this is explicitly not independent CUPTI handle capture
or retroactive binding of the earlier timing process. Selected cubin/PTX and
bounded SASS/resource dumps are retained. The optional disassembly of every
autotune cache candidate was intentionally interrupted to reserve final archive
time; its failed/interrupted receipt remains unchanged. Complete compiled
binaries and PTX remain retained.

These are synthetic operator measurements, including the advertised BHSD API
costs. They do not establish an HDiT model speedup. The shared qualified runtime
uses PyTorch 2.13.0+cu130 and NATTEN 0.21.7+torch2130cu130, with CUTLASS DSL
4.8.1 and CUDA bindings 13.0.3. This retains the newest prebuilt pairing in
[NATTEN's installation documentation](https://natten.org/install/) and the
earlier matched baseline. Current stable PyTorch documentation describes
[FlexAttention 2.15](https://docs.pytorch.org/docs/2.15/nn.attention.flex_attention.html);
these results do not compare against that newer release. FlexAttention uses
AUTO selection and `max-autotune-no-cudagraphs`, without an exhaustive backend
tuning claim.

The [FP64 accuracy report](neighborhood-accuracy.md) separates mathematical
accuracy from agreement with the declared rounded computation. Performance
tuning does not establish a general FP64 accuracy advantage over FlexAttention
or NATTEN.
