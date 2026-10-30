# RTX 4090 backward ownership and value lifetimes

This audit follows the released generic warp-MMA backward at revision
`d4700f5`, the final BF16 S4096/D128 captures in
[rtx4090.json](../results/rtx4090.json), and the pinned
[FA2 backward source](https://github.com/Dao-AILab/flash-attention/blob/14c377950125c70b7a9dabf9c561fca53715ac7d/csrc/flash_attn/src/flash_bwd_kernel.h).
It addresses reuse across tiles and blocks. The existing
[timing and code comparison](rtx4090-tuning.md) covers the scheduling sweeps.

## What the current blocks own

`JointBackward` launches a query-owner kernel followed by a key/value-owner
kernel. Its name describes a host-side program containing those launches; the
three gradients are not computed by a single device kernel.

| Owner | Output kept in FP32 registers | Persistent shared operands | Streamed operands | Published statistics |
| --- | --- | --- | --- | --- |
| Query tile | dQ numerator and P-weighted K | Q, dO | K, V across all visible key tiles | Centering base and full-row residual |
| Key tile | dK and dV | K, V | Q, dO across all visible query tiles | None |

In each query tile, Q and dO survive the key sweep. The score accumulator is
converted to a centered weighted derivative. The dP accumulator is repurposed
as P. Shared V staging can be reused for rounded derivative/probability operands
only after every dP reader finishes. The final P-weighted-K correction is applied
after the last visible key tile.

In each key tile, K and V survive the query sweep. Each query tile produces QK
and dP, reconstructs P, then computes both dK and dV before its score storage is
reused. This already shares scores between dK and dV. The repeated work is
between the query owner and this key owner, which have different output
ownership and loop directions.

Shared memory belongs to one block invocation. Keeping a score in shared memory
at the end of a query block does not make it available to a later key block or a
separate launch. A barrier within a block cannot establish that communication.
The existing launch boundary does establish that all row residuals are available
before any key owner consumes them.

Replacing that boundary with a global spin flag inside an ordinary oversized
grid is unsafe: resident consumers can wait for producers that have not been
scheduled. A cooperative or persistent design would need a compatible resident
grid, explicit work distribution and progress guarantees. It changes the
execution model; it does not remove the complete-row dependency.

## The centering dependency is a complete-row dependency

Let `b` denote the output/dO centering base, `r` the FP32 residual and `R` the
conversion to the input FP16/BF16 type. For one query row, the implemented query
path forms:

```text
x[j] = P[j] * (dP[j] - b)
r = sum_FP32(x[j])
u[j] = R(x[j])
t = sum_FP32(u[j])
A = sum_MMA(u[j] * K[j])
M = sum_MMA(R(P[j]) * K[j])
dQ = scale * (A - t * M)
```

The key owner consumes the separately stored `b` and `r`:

```text
dS_KV[j] = R(P[j] * ((dP[j] - b) - r))
dK = scale * dS_KVᵀ @ Q
dV = R(P)ᵀ @ dO
```

`t` and `r` are different quantities. The query correction uses the sum after
conversion; the key owner consumes the residual before conversion. Sharing one
rounded dS between both owners therefore changes a numerical policy as well as
an ownership policy.

The query tile knows `r` only after its complete visible key sweep. It avoids a
second sweep by keeping both A and M until then. The extra P·K product is the
cost of this particular one-sweep correction. A joint tile cannot consume a
finished `r` merely because it has computed the current tile's QK and dP.

## Saving a forward output moment is not a qualified shortcut

The residual depends on dO, which is unknown during forward. Saving a more
precise forward output vector, or one centered vector moment, does not in itself
preserve every small derivative direction.

A small BF16 example uses D64, default scale, one active query feature of 8,
and three corresponding key features `[0, -4.59375, -50]`. The value vectors
start with `[0, 0]`, `[1, 1]` and `[1, 0]`; dO starts with `[1, -1]`. All other
value/dO features are zero. The latter key has probability approximately
`1.9094e-22`.

Computing dP first gives `[0, 0, 1]`, so the centered residual retains that tiny
probability. A forward value moment instead starts with
`[P_second + P_third, P_second]`. The tiny addition is lost even in one FP64
moment. Projecting that rounded moment onto `[1, -1]` gives zero. The constructed
CPU case consequently loses enough centering information to produce about
70.36% dK norm error for this naive moment route.

This is a finite-precision counterexample to a particular state compression,
not a proof against every richer or multicomponent saved state. It explains why
an O(SD) forward-cache proposal needs a derivative-level test before replacing
the residual sweep. The existing tiny-gradient and shifted-key tests alone do
not cover every such cancellation direction.

## What FA2 actually carries across blocks

The captured D128 native function is
`pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel`, with
`Flash_bwd_kernel_traits<128,64,64,8,4,2,2,true,false,bfloat16,...>`.
Its actual launch is 64 × 1 × 16 blocks of 256 threads. The retained qualification capture's driver-loaded cubin SHA-256 is
`b313f870778d0f6a06ee919051faa493699b81524adcfc31f6b0729fb8d9a572`.
The full mangled symbol, module match, architecture and resource attributes are
retained in the linked comparison data. The new experiment records the matching
full native traits symbol in its own captured graph. Its SASS comparison uses
that retained module; it does not assert a newly captured byte-identical module.
Native packaged PTX was absent for the inspected functions. The pinned source
explains the algorithm; it is not presented as a deployed PTX dump.

That exact function's SASS contains 32 static
`RED.E.ADD.F32.FTZ.RN.STRONG.GPU` sites. These are code sites in a loop, not a
runtime count or a measured percentage. They corroborate the pinned source's
`atomicAdd` path into FP32 dQ accumulation storage.

FA2 gives each block a key tile and retains dK/dV locally while sweeping query
tiles. Within each attention tile it uses QK and dP once for all three gradients.
It sends the resulting partial dQ to the global FP32 accumulator. A subsequent
conversion kernel writes the final low-precision dQ. Its preparation kernel
provides the conventional output/dO softmax correction before the joint sweep.

This reuse comes from gradient ownership and cross-block accumulation. It does
not require retaining the entire attention matrix across blocks. The native
rounded-output correction also differs from the centered residual policy above;
copying the native five-product graph alone does not establish this project's
numerical guarantees.

## The accumulator representation also matters

The native sequence-parallel path does not store each dQ fragment directly at
its final output coordinate. Its `GmemTiledCopydQaccumAtomicAdd` uses a thread
layout of 8 rows by 32 columns and one value per thread. It writes the MMA
fragment into this accumulator representation; conversion reconstructs the
output representation afterward. The source also uses an interleaved-head
accumulator row stride of `heads * rounded_head_dimension`.

These are separate layout decisions. Merely changing the prototype's canonical
scratch from physical B/H/S/D to B/S/H/D made it slower. The useful intervention
was to change where each warp's fragment elements are accumulated. The new
opaque layout maps fragment element `i` to local row
`warp + 8 * (i % 8)` and column `lane + 32 * (i // 8)`. Each reduction instruction
then addresses 32 consecutive FP32 locations across its warp. A subsequent
conversion uses the same MMA coordinate layout to restore dQ and applies the
pivot correction using the true query coordinate.

This is an independently defined invertible fragment representation, inspired
by the native communication design. It does not claim byte identity with every
native layout. Scratch sequence lengths are padded to a complete 64-row tile:
a valid final-output coordinate can map to a different opaque row, so allocating
only the unpadded sequence length would make partial tiles unsafe. Scratch
clearing covers the padding as well.

The representation change preserves the FP32 contribution count and O(SD)
storage. It changes the addresses presented together by one warp instruction.
Equal RED instruction counts and equal logical byte counts therefore do not
establish equivalent communication behavior. The timing intervention, rather
than an invented cache-hit or atomic-throughput counter, establishes its value.

## Storage and communication costs

The following are logical byte counts for B1/H16/S4096/D128, excluding padding,
allocation bookkeeping and cache behavior. They are not measured DRAM traffic
or bandwidth ceilings.

| Retained object | Storage |
| --- | ---: |
| One full FP32 S×S plane | 1 GiB |
| P and centered dP as two FP32 planes | 2 GiB |
| Two full FP16/BF16 planes | 1 GiB |
| One query tile's FP32 64×4096 plane | 1 MiB |
| One final FP32 dQ accumulator | 32 MiB |
| Separate dQ partial plane for every 64-key owner | 2 GiB |

A query tile preserving two FP32 score-shaped planes across its entire key
sweep would need 2 MiB, versus the measured device's 99 KiB opt-in block shared
memory. Preserving two low-precision planes still needs 1 MiB and introduces new
rounding boundaries. Global retention is possible, but writing and reading two
low-precision full planes adds 2 GiB of payload before the gradient products.

The native accumulator uses only 32 MiB of storage, but every dQ element can
receive contributions from 64 key owners. That represents 2 GiB of FP32
contribution payload for dense attention, before read-modify-write effects. The
actual native implementation demonstrates that such communication can be a
useful tradeoff. Neither logical payload nor nominal bandwidth proves the
performance of a different implementation.

The short regime is different: a query tile with 16 rows and 128 keys needs
only 8 KiB for one FP32 score plane. A strategy that fits an entire short row in
shared memory does not automatically extend to S4096.

## Controlled joint experiment

`benchmarks.experimental_joint_backward` implements a separate analysis path,
with no automatic dispatch change:

1. Compute the complete-row centered residual using QK and dP, without dQ
   products, retaining only O(S) row statistics.
2. Run a key-owned joint sweep that retains K/V and dK/dV, shares its QK/dP
   values with dQ, and accumulates dQ contributions into O(SD) FP32 storage.
3. Convert the completed dQ accumulator after all owners finish.

The uncorrected single-component version uses seven matrix products including
the prepass. It still reconstructs scores and probabilities twice overall. The
optional two-component dQ version rounds both the leading dS and its FP32
residual into tensor-core operands, sums the two dQ products in registers and
publishes one contribution. That returns to eight matrix products. Its possible
benefit is the changed resource/operand schedule; it does not eliminate a score
pass or prove that atomic accumulation is free.

A CPU derivative-rounding model already exposes a gate for the uncorrected
seven-product version: with a shared key component of 1024 and a zero query component along
that feature, attention logits do not change, but rounding the final dS before
its single dQ product can create a large spurious gradient along that key
component. This must be tested alongside sharp attention and ordinary random
inputs. CPU matrix arithmetic is feasibility evidence, not GPU instruction
qualification.

A third variant preserves the seven-product count while enforcing the rounded
dS row mass through a pivot correction. The prepass records the row's argmax
key index; joint owners accumulate `m = sum(R(dS))` alongside dQ; conversion
computes `dQ = scale * (dQ_accumulator - m * K_argmax)`. It adds O(S) mass/index
storage and an O(SD) gather/epilogue, rather than another attention-shaped matrix
product. It changes the existing weighted-mean correction to an argmax-key
correction, so its numerical suitability and separately ordered atomic sums
require their own GPU qualification.

## The first joint implementation has avoidable layout costs

For the initial S4096/D128 pivot prototype, a separate Nsight Systems capture
attributes about 3.733 ms to the joint sweep, 1.016 ms to the residual prepass,
0.016 ms to scratch clearing and 0.062 ms to conversion. The same capture's
native FA2 joint kernel takes about 2.101 ms. These are instrumented durations,
separate from uninstrumented benchmark medians. Clearing is a small part of this
prototype's loss; moving it into another launch is not the first intervention.

The exact exported initial prototype has source SHA-256
`38123ab773f9ad66a510fad1cb2a6d7417d0324d670b34fb1c74ecdb3c491ce8`.
Its inner query loop and the matched native function both execute five matrix
products on a 64-query by 64-key tile, D128, eight warps. The static SASS counts
below exclude prologues, drains and the native outer key-grid-stride loop. That
outer loop executes only once per block for the recorded 64-block key grid;
it is not the attention steady loop.

| Inner-loop code sites | Initial joint prototype | Native FA2 joint |
| --- | ---: | ---: |
| HMMA | 160 | 160 |
| LDSM.x4 | 108 | 80 |
| CTA barriers | 5 | 3 |
| Scalar global statistic loads | 32 | 4 |
| Shared stores | 48 × 16 bits | 16 × 32 bits |
| Scalar 16-bit shared loads | 32 | 0 |
| FP32 global reduction sites | 33 | 32 |

The extra reduction site publishes row mass and has different predication from
the dQ sites. Static site counts do not give executed instruction counts or
runtime percentages. The shared stores represent 24 KiB versus 16 KiB of
logical per-tile payload: three times the store instruction count is only
1.5 times these bytes.

The load difference is source-explained. Native gradient products use a (2,4)
warp layout, while the first prototype uses (4,2). In the recorded geometry,
the native three gradient products have 48 LDSM sites versus the prototype's
60. Native V remains in B-operand registers across query tiles, leaving 32
score/dP LDSM sites versus the prototype's 48. Both implementations use x4
matrix loads; the operands, reuse and work distribution still differ.

Statistic issue counts also reflect score orientation. Native scores have
query rows, so each thread loads two rows of two statistics. The first
prototype computes transposed key-by-query scores: each thread touches eight
query columns and loads four separate statistics. Two extra statistics belong
to centered correction, while the remaining duplication comes from this layout.
Packing fields can reduce issue count without reducing bytes; changing the
score orientation can reduce how many copies each thread needs.

The native source double-buffers Q and issues the next dO transfer after its
last reader, overlapping transfers with gradient products. Equal async-copy
site counts do not imply equal placement. The first prototype waits for a new
Q/dO pair before starting each tile. These are concrete layout and pipeline
interventions for the joint loop, rather than evidence that fusion itself
should already be faster.

The FP32 atomic path also changes cross-block summation order and uses the
architecture's flush-to-zero floating-point reduction instruction. The existing
approximately 1e-21 regression exercises normal FP32 contributions. It does not
qualify FP32 subnormal accumulation. The distinct BF16 diagnostic scales the
sharp-attention dO by approximately 1e-18: production dQ has about 5.05% norm
error and the opaque prototype about 6.23%, both rejected by the 1% gate.
Correctly rounding the FP64 dQ into BF16 has approximately 0.295% error in a
separate CPU representability check. Low-precision operand rounding, tensor
instructions and atomic FTZ all require consideration; these measurements do
not isolate their individual contributions. Subnormal accumulation remains
unqualified and is not hidden inside the normal tiny-gradient pass count.

The prototype supports only equal-head D64/D128 at the default positive scale.
It has explicit scratch clearing and separate conversion; both costs belong in
its measurements. It is not a released replacement or an extension of the
D512 schedule.


## Measured interventions

The diagnostic controls below use a full RTX 4090, B1/H16/S4096/D128, BF16,
dense attention, seed 1009, three rotating-order device-event graph samples
with approximately 100 ms per sample. They measure backward including explicit
scratch clearing, residual prepass and conversion. Production CuTe and default
SDPA retained-forward backward run on identical rounded inputs in the same
process. The production forward statistics are retained for both CuTe paths;
SDPA retains its own forward. Instrumented profile durations above are separate.

| Joint experiment | Backward ms |
| --- | ---: |
| Initial canonical accumulator, CA copies | 4.825 |
| CG copies | 5.125 |
| Direct transposed matrix loads, gradient warp layout and packed stores | 4.757 |
| Next-Q/dO overlap with dV → dQ/RED → dK | 4.709 |
| Full query-by-key shared weights and packed stores | 4.706 |
| Also retain V in registers | 4.704 |
| Also retain both K and V in registers | 4.803 |
| Reduce rounded score row mass before publishing shared weights | 4.686 |
| Early probability computation/store | 4.683 |
| Canonical interleaved-head accumulator | 5.069 |
| Opaque contiguous-lane accumulator and inverse conversion | 3.375 |

The opaque accumulator cuts approximately 28% from the 4.686 ms
non-interleaved canonical-layout control. Its same-process production CuTe and SDPA controls
are approximately 3.259 and 2.246 ms. It is consequently a substantial prototype
improvement, still slower than both controls. This is not a released API
speedup. The other small differences require their retained variability; a
fraction of a percent does not establish a stable improvement.

Register allocation for the complete query-by-key layout is 215 registers
without resident K/V, 255 with V, and 255 with K plus V. The latter has a
40-byte stack and actual LDL/STL instructions; V alone has no stack. Removing
matrix loads did not improve the measured complete backward. A lower load
count alone was insufficient.

The current stack reports Python 3.13.18, PyTorch 2.15.0+cu132 and CUDA 13.2,
with CUTLASS DSL 4.8.1 and driver 595.71.05. The DSL export separately identifies
its bundled NVVM compiler as CUDA 13.4.59 and emits PTX ISA 9.4 targeting SM89.
The driver-reported `PTX_VERSION` value 89 identifies a virtual compute target;
it does not mean PTX ISA 8.9.


## Reproduce the selected experimental configuration

Use the qualified Python 3.13 GPU environment and the repository dependencies.
The following keeps raw code exports and results outside the tracked tree:

```sh
PYTHONPATH=src:. python3 benchmarks/probe_joint_backward.py \
  --seq 4096 --dim 128 --seed 1033 --repeats 7 --window-ms 200 \
  --pivot-correction --direct-dq --query-column-warps 4 \
  --gradient-column-warps 4 --packed-weights --overlap-query \
  --query-row-scores --reduce-score-mass --opaque-accumulator \
  --output ../rtx4090-joint/result.json \
  --dump-dir ../rtx4090-joint/code
```

The selected source hashes are:

| File | SHA-256 |
| --- | --- |
| `benchmarks/experimental_joint_backward.py` | `cd0ece1017592ab94aadefdc3cebf6f86e3b6ca8b10840c01834f0c253230495` |
| `benchmarks/probe_joint_backward.py` | `978fde5cbf49e2731ddb5b167dc0d93adfd1e36ebf783e171467fafd57e3f6e0` |
| `src/kohakufa_cute/kernels.py` production control | `3e410c2e5f128395d2591aecce182e74f4c89153450b2a5ade67bc19b7bb0816` |

The harness rejects unsupported shape/layout combinations and supports only
contiguous equal-head square Q/K/V at D64/D128, default positive scale, and
FP16/BF16. Its device gate requires a full 128-SM RTX 4090. The controls are
analysis experiments; qualification of one configuration does not qualify every
combination of diagnostic flags. There is no automatic backend selection or
production dispatch change. D512 remains on its existing path.

For each retained case, the harness compares dQ, dK and dV separately with
centered FP64 equations on identical rounded inputs, checks graph replay with
changed dO, and checks changed Q/K/V with freshly computed forward statistics.
It poisons scratch, row statistics, pivots and output storage before replay.
SDPA's own retained-forward backward is measured separately and is never used
as the accuracy oracle. Its failed accuracy cases are diagnostic, not qualified
speedup comparisons.


## Final qualification scope

The frozen opaque configuration passes 13 focused cases covering D64/D128,
FP16/BF16, dense/causal attention, lengths 1/17/65/129/257 and a 4096-token
shifted-key case. These include partial tiles, negative maxima, sharp attention
and the directional cancellation example above. Every accepted case also passes
changed-dO and changed-Q/K/V replay with fresh forward statistics and poisoned
storage. A separate fourteenth subnormal diagnostic is rejected by both CuTe
paths, as detailed above. Selection used seed 1009, qualification seed 1021,
and the seven-round holdout uses seed 1033.

Racecheck passes three focused configurations: BF16 causal S129/D128 shifted
keys, FP16 dense S65/D64 random inputs, and BF16 dense S17/D64 directional
cancellation. Device-memory checks cover the same configurations. The
unfiltered memcheck runs do not pass API validation: each reports 88 API errors,
including unsupported CUDA-version queries and internal-function lookups from
the cuBLASLt FP64-reference heuristic. The retained backtrace establishes that
library origin; it does not establish its closed-source probe intent.

The separate device-memory runs retain checking and use `--report-api-errors
no`, with `--kernel-name regex=.*benchmarksexperimental_joint_backward.*`.
CUDA bindings are 13.4.4 while the driver exposes CUDA 13.2. Original unfiltered
logs are retained separately. This is a device-memory qualification with an
explicit API-reporting limitation, rather than a clean unfiltered API check.


The exact-filter negative control deliberately writes beyond a directly bounded
four-element allocation. It disables PyTorch allocation caching only for that
control process and emits a symbol matching the same selected-kernel regex.
Memcheck reports one invalid global write and returns the requested error code
86. Positive runs retain the ordinary allocator: their memory checks concern
CUDA allocation bounds, rather than proving every logical tensor boundary
inside a caching-allocator slab. The independent shape guards, padded scratch
layout, FP64 comparisons and poisoned-storage tests remain necessary.

For the selected overlapped pipeline, the buffer ownership is:

| Buffer | Last reader before reuse | Reuse/publish ordering |
| --- | --- | --- |
| Current Q stage | Final dK product | Bottom CTA barrier releases its slot before a later alternate-stage prefetch |
| Single dO stage | dV product | CTA barrier precedes next-dO async copies; next iteration waits and synchronizes before reading |
| P shared plane | dV product | Next score stores follow the bottom CTA barrier |
| dS shared plane | Final dK product | All dQ/dK readers finish before the bottom CTA barrier and next score stores |
| Two row-mass partials | Row-mass publication | Score-store barrier publishes both N-warp partials; bottom barrier precedes their next overwrite |
| K/V staging | Completed joint sweep | Gradient-output staging is used only after sweep readers finish |

The alternate Q slot is prefetched while the current slot remains available to
dK. Both next-Q and next-dO copy groups are waited on before the next tile's
score products. Kernel launch dependencies publish complete row statistics
before joint consumption and complete global atomics before conversion. There
is no inter-block spin barrier or assumption that an oversized grid is resident.


## Holdout timing and remaining concentrations of time

The final seven-sample, 200 ms/window graph measurements use the disjoint seed
1033 and include all backward stages. All three implementations pass the
independent FP64 gates on these random-input cases. Values are milliseconds;
parentheses show the retained minimum–maximum sample range.

| S4096 BF16 | Production CuTe | Opaque joint experiment | Default SDPA |
| --- | ---: | ---: | ---: |
| D64 dense | 1.698 (1.653–1.698) | 1.856 (1.831–1.857) | 1.160 (1.142–1.161) |
| D64 causal | 1.083 (1.081–1.099) | 1.096 (1.094–1.115) | 0.775 (0.774–0.779) |
| D128 dense | 3.311 (3.252–3.311) | 3.413 (3.380–3.422) | 2.280 (2.254–2.305) |
| D128 causal | 2.078 (2.072–2.082) | 2.071 (2.068–2.072) | 1.320 (1.320–1.320) |

This does not select a production replacement. The small D128 causal difference
is not evidence of a stable overall win, and the other cases are slower than
production. The substantial improvement belongs to the experimental joint
implementation's accumulator representation.

A separate node-level Systems capture averages 20 warm S4096/D128 dense calls:

| Stage | Average ms |
| --- | ---: |
| Opaque joint sweep | 2.274 |
| Complete-row residual prepass | 1.015 |
| Scratch clear | 0.016 |
| Opaque conversion and pivot correction | 0.054 |
| Native FA2 joint sweep | 2.101 |
| Native output/dO preparation | 0.035 |
| Native conversion | 0.035 |
| Two native copy calls, combined | 0.075 |

These are instrumented stage durations, not the uninstrumented holdout medians.
The corrected capture explicitly uses `--cuda-graph-trace=node`; the preceding
graph-level capture is retained but cannot establish per-kernel durations.
The opaque joint loop is now close to the native joint loop in this capture.
The residual sweep is a measured major remaining cost of this chosen centered
strategy. This is neither a proof that the sweep is irreducible nor a claim
that precision or shared memory establishes a hardware performance ceiling.

The final D128 joint uses 208 registers, 98,816 dynamic shared-memory bytes and
zero driver-reported local storage. Its converter uses 48 registers and 16 KiB
of shared memory. Exact exported PTX/cubin/SASS and all graph symbols are retained
for the frozen configuration. All four selected graph symbols match the same
sanitizer regex used by the negative control. Compute Sanitizer is version
2026.1.0.0; no hardware-counter stall/cache/utilization percentages are claimed.
