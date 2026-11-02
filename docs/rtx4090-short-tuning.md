# Short RTX 4090 layout and statistics tuning

This follow-up reduces resident GPU time for the existing opt-in S128/D16
training path. It preserves the [declared arithmetic](precision.md), all five
tensor products, actual-output delta and the same shared-memory ownership.
The [current measurements](rtx4090-short.md) separate graph and eager behavior.

## Two independent controls

The original gradient products use a score-derived MMA layout whose per-warp
useful output work is M32×N8. A separate eight-by-one warp layout assigns
M16×N16 to each warp. Score reconstruction keeps its original four-by-two
layout. Matrix-copy assignments, output coordinates and all three gradient
products change together. This improves A-fragment reuse across N outputs.

The other control loads the immutable FP32 maximum, logarithmic normalization
and delta values into register fragments before any shared P/dS stores. The
existing producer barrier remains. Separating these reads from shared stores
lets the compiler eliminate repeated row-statistic loads. The values and
operand conversions are identical; the control adds no precision work.

Actual BF16 dense machine-code evidence, normalized to the same head, 256
threads, full 128×128 score tile and all three gradients:

| Control | Registers | Local bytes | HMMA sites | Matrix-load sites | Scalar FP32 shared-load sites |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original layout/statistics | 174 | 0 | 112 | 96 | 96 |
| Product layout only | 195 | 0 | 112 | 72 | 12 |
| Statistics prefetch only | 164 | 0 | 112 | 96 | 12 |
| Both | 195 | 0 | 112 | 72 | 12 |

Every row uses 99,840 shared bytes and retains three CTA barrier sites and no
gradient atomics. Matrix-load instructions retain their four-matrix width;
counts do not by themselves establish transferred-byte totals. HMMA counts
are equal: source-level output padding removal does not establish reduced
tensor work. Static sites are not runtime percentages, cache counters or stall
measurements. Higher register count did not establish a better schedule.

## The controls are coupled

Microseconds of graph forward plus backward in the selection experiment:

| Dtype / visibility | Neither | Product layout | Statistics prefetch | Both |
| --- | ---: | ---: | ---: | ---: |
| BF16 dense | 10.135 | 9.381 | 9.270 | 9.645 |
| BF16 causal | 9.287 | 9.172 | 9.366 | 9.440 |
| FP16 dense | 9.643 | 9.454 | 9.187 | 9.075 |
| FP16 causal | 9.248 | 9.128 | 9.252 | 9.092 |

The selected policy uses statistics prefetch for BF16 dense, both controls for
FP16 dense, and the product layout for causal calls. The very small FP16 causal
difference between product-only and both did not justify another policy branch.
The independent public evaluation and both shared-storage capture orders retain
the selected causal gains. Statistics prefetch alone regresses BF16 causal;
the controls are not enabled globally or assumed additive.

## Scope and evidence

Each new flag combination passed eighteen finite directed cases, the separately
labelled final FP16 cast-overflow diagnostic, changed/poisoned replay and
negative views. Four sanitizer tools each checked twelve calls per combination.
The final public selection passed its own directed and live Parameter/no-grad
checks, 24 independent-seed graph/eager measurements, eight no-grad controls,
and eight shared-storage graph comparisons. Its four selected backward
CUBIN/SASS pairs match the corresponding qualified experiment exactly. Systems
confirms two kernels per eager training call in both policies.

The native optimized family has its own compilation/cache identity and a strict
full RTX 4090 device gate. The tensor ABI and configured-field schema are
unchanged. The basic family keeps both flags false. Its default-false source
was requalified on actual RTX 3090 and L40S hardware; some generated instructions
change, so those checks do not claim identical old/new binaries. RTX 3090 graph
medians remained within about 0.5%; L40S also passed four sanitizer tools.

The [compact record](../results/rtx4090-short-optimized.json) retains all timing
samples, exact code/resources, software/source identities and verified archive
inventories. It separately labels the same-process flag comparison, shared-input
capture-order control and original-kernel comparison in persistent processes
through a matched host library. Startup/compilation are excluded. Process and
host scheduling affect eager results; no general eager or serving gain is claimed.
