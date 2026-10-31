# RTX 5090 attention schedules and precision

The RTX 5090 changes reduce padded D16 work and improve short-sequence work
ownership. The comparison also includes new low-component backward products:
those repair cancellation errors in the preceding implementation and can cost
latency. A faster result with the old arithmetic is not an accuracy-equivalent
baseline.

The [measured matrix](../results/rtx5090.json) records a full 170-SM, 32-GB,
600-W RTX 5090, Python 3.13.18, PyTorch 2.15.0+cu132 and CuTe DSL 4.8.1.
The DSL installation uses base, core and cu12 components. Its actual loaded
compiler DSO is the cu12 variant, emitting PTX ISA 8.8 for `sm_120a`. This retains
the previously qualified compiler combination with the checked current framework
and DSL releases; another installation's cu13 compiler is a different stack.

## Comparison boundary

Four implementations run on identical rounded inputs in each fresh process:

| Label | Computation |
| --- | --- |
| SDPA | The actual default PyTorch backend and launched functions |
| Released | The computation at `3e17149`, with subsequent device and precision selectors disabled, rebuilt with the current compiler |
| Previous | The preceding RTX 5090 schedule with the shared numerical repairs enabled |
| CuTe | The selected RTX 5090 schedule with the same repairs |

Released is a computation control, not a claim of historical binary identity.
Previous versus CuTe isolates the D16 schedules, D64 ranges and short schedules
in `sm120_schedule.py`. Both retain the current shared precision owners, their
qualified packed-store policy and the compensated normalizer. Released versus
CuTe includes scheduling, ownership and arithmetic changes. A failed numerical row
remains diagnostic and does not establish a qualified speedup.

The 60 cases use B1/H16, S128/1024/4096, D16/64/128/256/512, FP16/BF16 and
both dense and causal attention. Forward, retained-forward backward, and complete
forward/backward are timed separately. Compilation, allocation and graph capture
are excluded. Device-event intervals include graph-entry and dependent-node
scheduling, including submission stalls if present; separately measured medians
need not add up to the complete median.

Each backend gets seven rotating rounds targeting 200 ms per phase, with actual
replay counts retained. No activity profiler, CUPTI subscriber or telemetry loop
is initialized in the timing process. Actual graph function names and allocated
resources are read through driver APIs after timing. Eager and every captured
phase are independently checked against centered FP64, component by component:
less than 1% norm-relative error for BF16 or 0.5% for FP16, plus pointwise
`atol=rtol=0.02` or `0.003`, respectively. A zero reference requires an exact-zero
result. The relevant GPU suite separately checks changing and poisoned storage.

The final 60-case report passes the component gates for all four implementations
on every ordinary matrix row. Median wins against SDPA are:

| Implementation | Forward wins | Backward wins | Complete wins |
| --- | ---: | ---: | ---: |
| Released | 34/60 | 25/60 | 24/60 |
| Previous, precision matched | 34/60 | 21/60 | 23/60 |
| Selected CuTe | 46/60 | 28/60 | 29/60 |

Against the precision-matched previous schedule, complete latency falls by
36.1–52.7% for long D16 and 12.6–42.9% for the D128/D256/D512 short families. No phase
regresses by more than 5% across the matrix. Win counts include near ties; the
report retains all seven samples and actual replay counts. The long D64/D128/
D256 backward gaps remain, and several repaired paths cost more than released
arithmetic. These are measured limitations rather than gains hidden by an
aggregate average.

A separate sustained BF16 B1/H16/S4096/D16 dense complete-call check alternates
previous/selected/selected/previous for four 20-second passes, excluding each
pass's first five seconds. Previous medians are 1.470/1.475 ms and selected medians 0.698/0.700 ms,
with 148–153 retained samples per pass. Coarse samples show selected power near
600 W and SM clocks around 2726–2733 MHz, versus previous power 462–471 W and
clocks 2842–2850 MHz. The gain persists at lower sampled clocks; those samples
are not cycle normalization or HFU. Only one compute context is observed.
Telemetry is confined to this separate diagnostic.

## Selected work ownership

The device-specific schedules require the exact qualified RTX 5090, unmasked
FP16/BF16, B1/H16, equal query/key head counts and the tested vector layout.
Both sequence axes remain runtime predicates inside cached programs.

| Family | Selected work | Boundary behavior |
| --- | --- | --- |
| D16, both lengths at least 1024 | Logical D16 products, a 64-column sweep and fused query centering | Shorter axes retain the preceding schedule; partial tiles retain checked copies |
| D128/D256/D512 forward, both lengths at most 128 | 16-query-row blocks and 64-feature output slices | Either axis reaching 129 returns to the original forward owner |
| D64 backward, square lengths 128–4096 causal or 128–512 dense | M32/C32 with two output-column warps, four independent high/low accumulators and column statistics | Rectangles, smaller/larger lengths and other visibility families retain the corrected original owner |
| D128/D256 backward, equal lengths at most 128 | Four-warp narrow output ownership and fused query centering | Rectangles and longer calls retain their existing query owner |

These policies expose more independent work or avoid padded products. More
blocks alone is not evidence of a gain. Query owners preserve SM120
`reload_center`, cache policy and the existing workspace contract. The port does
not enable the RTX 4090 long D128 packed-statistics policy.

A shared precision policy runs after the architecture selectors and repairs each
qualified backward owner, including runtime fallback leaves. It preserves the
query owner, centering prepass and wrapper boundaries. The detailed arithmetic
contract is described in [precision.md](precision.md).

| Key/value-gradient family | Geometry and storage |
| --- | --- |
| D16/D64/D128 | Generic owner with independently accumulated high/low dS and probability products; narrow short D128 keeps its compact ownership |
| D256 | Query-major Q64/K32 scores, eight gradient warps and 96 KiB shared memory; V staging is released and reused for weight planes, then reloaded on the next query block |
| D512 | M16/C32, eight gradient warps and four score warps (128 threads); 99 KiB shared memory, including three 1-KiB weight planes |

The D512 shared-memory lifetime is explicit: all dS-low readers finish before
that plane is republished as probability-low, with full-CTA release/publication
barriers. Independent FP32 high/low sums remain separate until the epilogue.
FP16 key residuals are packed after multiplying by 1024 and divided by 1024 at
the epilogue; probability residuals use a factor of 2^24. Those powers of two
preserve low values that would otherwise underflow during FP16 operand packing.
The D256 key-residual scaling is enabled only for the qualified RTX 5090 policy.

D512 BF16 causal weight publication uses packed half-pair stores on this device.
The same bytes are written, with fewer store instructions. Dense BF16 and FP16
keep their separately qualified publication choices. Lower register allocation
in the packed control did not consistently predict lower latency.

The exact loaded BF16 causal wide function matches its exported cubin and uses
240 registers, zero local bytes and 101,376 shared bytes. Its SASS contains
64 half-store sites and eight word-store sites; the other 24 vector-store sites
are unchanged. The eight word stores publish the same 4,096 bytes per query
tile as 16 half stores in the scalar control. The remaining half stores belong
to the gradient epilogue. These are static function sites, not executed-work or
runtime percentages. Matching uses the full graph symbol, architecture,
allocated resources and a unique loaded image; ambiguous generic Q/forward
associations are excluded from instruction claims.

## D64 fragments and nearby lengths

The corrected D64 key/value owner initially retained M64/N1 ownership. Its
BF16 dense function allocates 255 registers and 56 local bytes; the M32/N2
control allocates 168 registers, no local bytes and the same 24 KiB shared
memory. The resource bound rises from two to three resident CTAs. BF16 causal
allocation falls from 246 to 130 registers, with no local allocation in either
owner. These are allocated-resource bounds, not measured occupancy.

Both owners preserve scaled low operands and separate FP32 high/low sums.
The N2 owner has smaller gradient fragments and shared weight planes, with
separate high/low products. It changes no query, forward or centering owner.
Seven rotating 100-ms rounds found complete-call gains of 16.8–18.6% at causal
S1024. The same fixed owners also improve causal 1023/1025/1536/2048 and both
dense/causal 128/512 for both dtypes. Dense S1024 is neutral or slightly slower,
so it keeps the original owner. The selected ranges preserve that distinction
and leave unmeasured extremes on the original schedule.

The final 60-case grid launches this owner in all eight intended D64 rows,
with complete-call gains of 1.2–18.6% against the precision-matched previous
schedule. Actual graph symbols confirm the selection and the untouched dense
long-sequence fallback.

The public-path regressions cross both sequence axes, the 128/512/4096 range
boundaries and cached rectangular fallbacks. They also check strided, changed
and poisoned graph inputs, signed/zero scales and negative maxima. A causal
closed-form regression shifts the strong key-cancellation query/dO groups by
1031 rows: all active queries see the same original 1031 valid keys, retaining
the exact small negative dK target while exercising the selected long owner.

A separate node-level Systems capture of BF16 S1024/D64 causal backward
locates the remaining active time in both owners: query 55.3 µs, key/value
54.0 µs and centering preparation 1.49 µs. The native joint-gradient function
averages 63.3 µs, plus its copy, dot-preparation and conversion stages. These
are instrumented active-kernel durations, not canonical event medians. The
trace includes both eager API/autograd calls and graph replays; harness operand
clones occur before profiling. This rules out the small centering launch as the
main remaining concentration in this case, without assigning counter-derived
stall percentages to either gradient owner.

## A forward normalizer error exposed by backward cancellation

A deterministic D256 FP16 example initially passed ordinary norm and pointwise
checks but failed the stricter closed-form dK coordinate check. The target is
`-0.001713324311`. The original normalizer and unscaled low operand produce
`-0.001667022705`, a 2.70% error. Scaling the low dS operand alone still leaves
2.15% error.

The actual FP32 dS operands were captured from the executed key/value owner.
The instrumented and uninstrumented outputs and all gradients were bitwise
identical. An FP64 product of those captured FP32 operands produces
`-0.001666713506`; rounding them into two half operands barely changes that
value. This localizes the dominant error before the half operand product.
Small bias in the saved probability normalization is amplified by cancellation.

A bounded 2-by-2 control gives:

| Row-normalizer sum | Low dS packing | dK target relative error |
| --- | --- | ---: |
| Original | Unscaled | 2.7024% |
| Original | Scaled | 2.1458% |
| Compensated | Unscaled | 0.4760% |
| Compensated | Scaled | 0.0807% |

On the same strong fixture, default SDPA gives dK `-0.177734375` for BF16
and `+0.055694580078` for FP16, against the same negative `-0.001713324311`
reference. Both have 16 dK pointwise failures; their dQ norm errors are 11.69%
and 1.43%, respectively. Actual graph symbols identify the launched native
backend. These are numerical diagnostics, not qualified timing comparisons.

The selected forward owner retains one FP32 correction per row, including the
rounding error from rescaling the running sum when its maximum changes.
Explicit round-to-nearest arithmetic boundaries preserve the compensation.
The final denominator is the sum and correction; max/logsum storage and the
workspace format are unchanged. This repair is centrally selected only for
qualified RTX 5090 FP16 D256 forward leaves, so Previous and CuTe share it.
The one-pass short forward owner remains unchanged. The measured control uses
255 registers, zero local bytes and the same 50,176 shared bytes as before.

## What the machine code and rejected experiments establish

The released long D256 backward was compared with the exact native launched
function using actual loaded cubins, matching symbols, architecture and function
resources. Native traits select Q64/K32, eight warps, score layout (4,2), dK/dV
layout (1,8), dQ layout (2,4), register-resident V and no Q double buffer. In the
final FP16 capture, the driver reports 255 registers and 72 bytes for
`LOCAL_SIZE_BYTES`; the matched ELF reports `STACK:72` and `LOCAL:0`. The
actual SASS contains 21 `LDL` and 21 `STL` sites, establishing local-memory
traffic. Allocation metadata alone is not a spill-traffic measurement, and
static instruction sites are not executed byte counts or runtime percentages.
The preceding separate CuTe owners have no local allocation yet are slower;
resource counts alone do not explain the gap.

A private joint prototype reproduced that geometry, shared scores, kept V in
registers and used coalesced opaque FP32 dQ scratch. It was rejected. Before the
new low-component repairs, three rotating 100-ms rounds at BF16 B1/H16/S4096/
D256 gave complete backward times of 6.956 ms for released CuTe, 7.768 ms for
the prototype and 4.599 ms for default SDPA. Separate node-level Systems captures
located 5.747 ms in the joint main loop, 1.928 ms in the centered prepass,
0.078 ms in conversion and 0.031 ms in scratch clearing. Those instrumented
stage means are diagnostics, not the canonical benchmark samples. The prototype
was not qualified against the later low-component cancellation tests and is not
an accuracy-equivalent performance ceiling for a future joint design.

Other private controls changed eight-warp ownership, stationary V, paired
products and output layouts. None established a qualified complete-call win in
that epoch. A later query-major key/value owner provided the starting point for
the independently qualified D256 precision owner; it is a different score
orientation from merely retaining V in the transposed owner.

A later fixed forward-owner control kept the current arithmetic, cache policy
and FP16 compensated normalizer, replacing the D256 M64/K16 four-warp TMA
owner with generic M128/K32/eight-warps. At S4096 dense, strict FP64 and changed
storage checks passed, but seven rotating 100-ms rounds showed BF16 forward
1.606→1.762 ms and complete 9.692→9.823 ms; FP16 forward 1.603→1.839 ms and
complete 9.577→9.823 ms. It was rejected. Halving the query-CTA count and its logical KV reads did not
outweigh the complete owner's different copy and compute schedule.

Nsight Compute returned `ERR_NVGPUCTRPERM`. Timing, Systems nodes, loaded SASS,
allocated resources and controlled interventions remain usable evidence; they
do not measure stall percentages, cache hits or hardware FLOP utilization.

## Numerical limits

A scalar correction formed by multiplying missing dS mass by an arbitrary
query anchor was tested and rejected. It improves a common-query example but
creates a large new error when that anchor is an unrelated finite outlier hidden
by the mask. The selected low-dS matrix product handles that counterexample.

The dedicated cancellation tests check the sign and relative error of small
coordinates as well as all-tensor norm and pointwise gates. This matters: a
wrong-sign coordinate can pass a norm-only check. Probability-low tests also
cover FP16 underflow: a directly packed low component gives dV zero against a
`+0.67997257` reference; scaled packing recovers `+0.6796875`.

Very small FP16 gradients remain a separate limitation. Final output conversion
alone can exceed the 0.5% norm gate. The retained diagnostics compare actual
error with direct FP64-to-output-dtype rounding and count representable reference
values lost to zero. Low-component arithmetic can reduce avoidable losses but
cannot remove final output rounding. A BF16 diagnostic also exposes probability
differences already lost when FP64 probabilities become FP32; adding another
BF16 operand component cannot recover that information. Neither family is
reported as a successful strict-gate case.

The following separate final-source diagnostic uses seed 1277, B1/H16,
S128/D128 and dO multiplied by `1e-5`. Every one of the four cases fails the
unchanged strict norm gate. A lost nonzero means that the FP64 reference rounds
to a nonzero FP16 value but the computed result is zero. The rounding floor is
direct RNE conversion of FP64, not another attention implementation.

| Visibility / KV heads | Component | Released error (%) | Selected error (%) | FP16 RNE floor (%) | Lost nonzeros: released / selected |
| --- | --- | ---: | ---: | ---: | ---: |
| Dense / 16 | Output | 0.027 | 0.027 | 0.021 | 0 / 0 |
| Dense / 16 | dQ | 1.772 | 1.771 | 1.252 | 1130 / 1128 |
| Dense / 16 | dK | 1.765 | 1.247 | 1.247 | 1164 / 1 |
| Dense / 16 | dV | 1.227 | 1.227 | 1.227 | 16 / 0 |
| Dense / 4 | Output | 0.026 | 0.026 | 0.021 | 0 / 0 |
| Dense / 4 | dQ | 1.797 | 1.797 | 1.272 | 1138 / 1138 |
| Dense / 4 | dK | 1.404 | 0.627 | 0.627 | 260 / 1 |
| Dense / 4 | dV | 0.614 | 0.614 | 0.614 | 7 / 0 |
| Causal / 16 | Output | 0.023 | 0.024 | 0.020 | 0 / 0 |
| Causal / 16 | dQ | 0.972 | 0.972 | 0.791 | 621 / 618 |
| Causal / 16 | dK | 0.957 | 0.780 | 0.780 | 967 / 1 |
| Causal / 16 | dV | 0.615 | 0.614 | 0.614 | 12 / 0 |
| Causal / 4 | Output | 0.024 | 0.024 | 0.020 | 0 / 0 |
| Causal / 4 | dQ | 0.961 | 0.961 | 0.782 | 646 / 646 |
| Causal / 4 | dK | 0.679 | 0.390 | 0.390 | 185 / 0 |
| Causal / 4 | dV | 0.311 | 0.310 | 0.310 | 5 / 0 |

The scaled low product brings dK close to its final FP16 rounding floor. dQ
still loses additional representable values; the table does not attribute that
avoidable loss to output rounding. GQA retains its existing query schedule.

The final combined source passes 200 GPU discovery tests, with 66 hardware or
optional-dependency skips. The D64 owner also passes memcheck, racecheck and
synccheck on both dtypes and visibility modes, including changed/poisoned
strided partial tiles: zero memory/synchronization errors and zero race hazards.
Known framework/driver API-version lookup probes are excluded with
`--report-api-errors no`; memory, race and synchronization checks remain active.
The wide helper's exact source bytes were separately qualified with the same
three tools.

Reproduce the canonical matrix and focused qualification with:

```sh
PYTHONPATH=src python3 -m benchmarks.compare_sm120 --output results/rtx5090.json
PYTHONPATH=src python3 -m unittest tests.test_sm120_outliers tests.test_key_precision tests.test_value_precision -v
```
