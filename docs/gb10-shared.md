# Explicit GB10 shared attention

`kohakufa_cute.fa4_sm80.gb10.scaled_dot_product_attention` is an opt-in D128
entry for the full 48-SM GB10. It combines the existing shifted forward and
joint backward with the cached native host runner. Ordinary dispatch is unchanged.

```python
from kohakufa_cute.fa4_sm80.gb10 import scaled_dot_product_attention

out = scaled_dot_product_attention(q, k, v, is_causal=False)
```

## Qualified boundary

Use Linux ARM64, Python 3.13.18, Torch 2.15.0 with CUDA 13.2, CuTe DSL 4.8.1,
TVM FFI 0.1.15.post0, Quack 0.6.5, torch-c-dlpack-ext 0.1.5, and the pinned
`sm80-fa4` extra. The FA4 source revision is
`4616f5116d33e3ba3cbb71d582caffcfcb49649c`. These are qualified pins, rather
than a claim about newer releases. Other runtimes retain ordinary CuTe.
Run the qualified target with `CUTE_DSL_ARCH=sm_121a`.

Selection requires aligned, contiguous BHSD FP16/BF16 with equal Q/K/V shapes,
batch one, 4 or 16 heads, and D128. Dense lengths are 512, 1024 and 4096;
causal lengths are 512 and 4096. It also requires the current CUDA device,
the reported `NVIDIA GB10` name, 48 SMs, and 101376 bytes of opt-in shared
memory per block. Masks, dropout, GQA, other shapes, tails, strided/unaligned
inputs, signed/zero/subnormal scales, deterministic mode and compiled calls
retain the ordinary public CuTe API with the original arguments. Selected
kernel failures propagate.

## Selected schedules

Forward uses M64/N64, one stage, 128 threads and shared Q. Backward uses
256 threads and one dO stage. Warp layouts below are score/KV/dQ.

| Family | Backward M / N | Q stages | Warp layouts | V |
| --- | --- | --- | --- | --- |
| Dense S512 | 64 / 64 | 2 | 4 / 2 / 2 | Registers |
| Dense S1024, H4 | 32 / 128 | 1 | 2 / 2 / 2 | Shared |
| Dense S1024, H16 | 64 / 64 | 2 | 4 / 2 / 2 | Registers |
| Dense S4096, BF16 | 64 / 64 | 2 | 4 / 2 / 2 | Registers |
| Dense S4096, FP16 | 32 / 128 | 1 | 2 / 2 / 2 | Shared |
| Causal S4096, H4, FP16 | 64 / 64 | 1 | 4 / 2 / 2 | Shared |
| Other qualified causal families | 64 / 64 | 2 | 4 / 2 / 2 | Registers |

M32 is a coupled tile, warp-layout and storage change. N128 halves key-owned
CTA count, logical full-problem Q/dO rereads and partial-dQ updates relative
to N64. M32 doubles query iterations per CTA; half the atomic sites in one
iteration alone does not explain the full-problem count. At H4/S512, M32
launches only 16 key CTAs versus M64's 32 on the 48-SM device and loses device
time, so it is excluded there.

Controls tested M64/N64 with Q1/register V and Q1/shared V. Neither replaces
Q2/register V for long dense cases. H16/S1024 M32 is tied or order-sensitive,
so M64 stays selected. BF16/S4096 also retains M64; FP16/S4096 favors M32.

## Measurement scope

Identical rounded inputs run in one process and stack. Seven rounds rotate
backend order, with at least 100 ms per window after compilation and warmup.
Graph device time includes full forward/backward preparation, scratch
initialization, joint main and dQ conversion. Eager wall time also includes
API preparation and host dispatch. Backward-only retains the actual forward
result; joint includes both calls. FA2 is a separately loaded native ARM
build at the pinned revision, not SDPA or a numerical oracle. These are
complete implementation comparisons: FA2's sharp-logit rounding policy differs
from the project's contract.

Final actual-public dense joint graph medians follow, in microseconds, from
the final seven-round run. This entry's graph medians were 1.1-19.7% below FA2
across the twelve measured cases. Small long-case differences require the
retained sample ranges in [the compact summary](../results/gb10-shared-summary.json).

| H | S | Dtype | Public us | Ordinary CuTe us | FA2 us |
| --- | --- | --- | ---: | ---: | ---: |
| 4 | 512 | BF16 | 42.00 | 68.04 | 52.29 |
| 4 | 512 | FP16 | 43.24 | 67.66 | 52.03 |
| 16 | 512 | BF16 | 127.48 | 163.49 | 148.97 |
| 16 | 512 | FP16 | 138.87 | 161.58 | 147.10 |
| 4 | 1024 | BF16 | 125.64 | 194.11 | 152.44 |
| 4 | 1024 | FP16 | 125.64 | 193.86 | 151.32 |
| 16 | 1024 | BF16 | 509.02 | 688.40 | 519.30 |
| 16 | 1024 | FP16 | 524.82 | 645.44 | 530.60 |
| 4 | 4096 | BF16 | 1538.49 | 2307.93 | 1561.17 |
| 4 | 4096 | FP16 | 1554.14 | 2237.36 | 1587.04 |
| 16 | 4096 | BF16 | 5991.11 | 8749.08 | 6065.74 |
| 16 | 4096 | FP16 | 6017.46 | 8796.05 | 6202.60 |

H16/S4096 complete public eager medians were 6000.39 versus ordinary CuTe
8781.94 and FA2 6081.69 us for BF16; FP16 measured 6000.99 versus 8794.92
and 6152.66 us. Short eager measurements have substantial host variability
and observed regressions: H4/S512 FP16 public measured 287.32 versus ordinary
CuTe 87.65 us; H16/S512 BF16 measured 428.78 versus 202.72 us. The entry is
explicitly opt-in. A graph win does not imply an eager win.

Corrected causal measurements are retained separately. Long causal BF16 is
approximately tied with FA2; FP16/S4096 can be slower despite improving
ordinary CuTe. H16 FP16 complete eager measured 3528.65 versus ordinary CuTe
4471.84 and FA2 3431.89 us. There is no all-shape or all-execution-policy
claim of beating FA2.

Hardware was full GB10 SM121, CUDA-reported usable memory 130661138432 bytes,
driver 610.57.04, and no concurrent benchmark or compilation in timed windows.
Between-round control samples ranged from 2314 to 2561 MHz, 37 to 62 C, and
26.01 to 100.29 W; the queried interface reports no GPU power limit.
These coarse samples neither normalize cycles nor establish a power ceiling.

## Arithmetic and validation

The [default arithmetic](precision.md) is unchanged: retain unscaled maxima
and shifted log normalization separately, round P once to the input dtype
before dS, round dS once for gradient products, accumulate products in FP32,
and apply dQ/dK scale after accumulation. Delta uses actual rounded forward O.

Native/Python M64 and M32 experiments each passed 63 cases. The original
public selector passed 29 selected cases and fallback tests. The final selector
passed 29 cases at each of H4 and H16, and all four Compute Sanitizer tools
passed their 11-case changed-data/poisoned-storage smokes at both head counts.
Tests independently check output and every gradient, saved statistics,
tails, common offsets, rounded cancellation/underflow and changed graph replay.
The final head/dtype selection was exercised through the actual public entry.

Random cases retain independent per-head FP64 limits of 1% BF16 and 0.5%
FP16. Directed cases use the existing actual-O rounded equations and gates.
The earlier common-offset diagnostic of 1.647329843% FP64 error remains
retained. With current shared code, that input's worst dQ FP64 error is
2.1065403%, versus 0.1673675% against actual-O rounded equations. Expected
operand rounding/cancellation makes these different questions; the FP64
diagnostic is neither erased nor presented as a random-case pass.

## Deployed code

Actual CUPTI `MODULE_LOADED` payloads were captured. All twelve graph nodes
across M64, M32 and FA2 matched uniquely by exact symbol/register resources
against loaded ELF images. Their symbols/resources matched the timing reports.
Two compiler-export hashes also matched captured payloads.

| Backward main | Shared bytes | Registers/thread | Local bytes |
| --- | ---: | ---: | ---: |
| M64 Q2, register V | 82944 | 244 | 0 |
| M64 Q1, register V | 66560 | 255 | 16 |
| M64 Q1, shared V | 82944 | 211 | 0 |
| M32 Q1, shared V | 99328 | 255 | 0 |
| Selected FA2 BF16 D128 | 81920 | 252 | 0 |

Loaded M64 and M32 BF16 main ELF hashes:
`9603d8fe52f7313dccab88aea6eed4ad18de82597e76457eedeae4b61e9df42b`,
`bd4e005426a1996c14af5ca059844ac3f5ca7e8e59846c5f24e5ab4362ac8a16`.
FA2 backward:
`fa7d322a3e3760b974b21aef66379f4f1429b77836bb4c8d6c0d4229f3e14f1e`.
Graph binary/PTX attributes were 121; CuTe used `sm_121a`. FA2 was built
for native SM121 with compute_121f and a compute_121 PTX fallback.

SASS query-loop regions were M64 `0x1670..0x4000`, M32
`0x1900..0x4060`, and FA2 `0x1b30..0x3970`. FA2's outer seqk scheduler
backedge is separate. Each loop region has 160 BF16 FP32-accumulating HMMA
instructions. M64 has 80 LDSM instructions per loop, M32 96; M64 also
preloads V into registers in its prologue. M64 has 32 scalar dQ reduction
sites per loop versus M32's 16. Both query-iteration and key-CTA counts are
needed for whole-problem normalization. M32 has a larger epilogue per CTA.
These are static counts, not runtime stall percentages, transferred bytes,
or counts of executed predicated work.

## Immutable device classification cache

The opt-in predicate now caches only device classification by CUDA ordinal:
capability, name, full SM count and shared-memory limit. Current device,
determinism/compile state, tensor metadata, scale and runtime checks still
execute on every call. The cache owns CPU booleans, not tensors or buffers.

On the same ARM GB10 lease, eleven alternating rounds of 10000 hot calls
measured predicate medians of 5.166 versus 2.624 us, a 2.542 us reduction.
Fifty-two complete API rows retained identical graph resources for cached,
uncached and direct-runner controls and passed the existing FP64 gates.
The [cache summary](../results/gb10-device-cache-summary.json) keeps every row,
including regressions and sample ranges. There is no broad API latency gain
claim: H4/S512 FP16 eager joint improved 60.75 to 58.11 us, while H16/S512
BF16 changed 184.19 to 274.08 us and H4/S1024 BF16 changed 182.38 to 230.36 us.
Long H16/S4096 stayed approximately 5.93 ms.

The run kept Torch intra/inter-op thread counts at 20, with no OMP/MKL override
or affinity/governor changes. The container allows CPUs 0-19 with a 19.2-CPU
quota. After-run snapshots showed performance governors, heterogeneous
2.808/3.9 GHz policies, zero recent pressure and very little cumulative
throttling; the campaign's one-minute load samples ranged 0.07-1.30.
These snapshots do not identify the cause of individual timing outliers.

CPU cProfile's 64-call forward sample attributes 1.856 ms inclusive to the
uncached wrapper and 1.215 ms to the cached wrapper; predicate calls dominate
the difference. Joint samples instead concentrate in autograd's inclusive
run_backward path and vary substantially. This does not identify a C++ descriptor
lookup bottleneck or turn profiling samples into latency measurements.

## Systems timeline and profiler limits

Six repaired Systems captures succeeded for public/ordinary CuTe/FA2, each
in eager and graph mode, on H4/S1024/BF16. The public path launches shifted
forward, delta/scratch preprocessing, joint backward and dQ conversion:
80 kernels for twenty joint calls. Ordinary CuTe also has four launches;
FA2 has five because its selected forward uses split-KV plus a combine pass,
followed by preprocessing, backward main and dQ conversion. No device-copy
or memset activities were recorded in these warmed capture ranges.
Eager submits 80/100 individual launches; graph mode submits twenty graph
launches. Actual captured kernel resources agree with the qualified paths.

Public eager's instrumented kernel span was 9.755 ms with 2.573 ms summed
kernel durations, versus graph's 2.563 ms span and 2.505 ms kernel sum.
The timelines expose launch gaps; instrumented timings are not benchmark
speedup evidence or a performance ceiling. Original profiling attempts
failed on nonleaf CloneBackward stream ownership and are not counted as
successful captures. The repaired harness passes detached operands into the
existing capture helper so it creates independent leaf operands.

Nsight Compute returned ERR_NVGPUCTRPERM: hardware performance counters
were unavailable. No cache-hit, atomic-stall, bandwidth-utilization or
counter-derived ceiling claims follow from these runs.

## Reproduce

Use the qualified `sm80-fa4` installation and a separate native ARM FA2 build.
Run from a Python 3.13 environment. Keep raw captures outside the tracked tree:

```bash
export CUTE_DSL_ARCH=sm_121a
mkdir -p ../gb10-results
python -m benchmarks.qualify_gb10_shared --public-selected --heads 4 \
  --output ../gb10-results/public-h4.json
python -m benchmarks.qualify_gb10_shared --public-selected --heads 16 \
  --output ../gb10-results/public-h16.json
python -m unittest tests.test_gb10_shared tests.test_gb10_benchmark -v
python -m benchmarks.qualify_gb10_shared --public-selected --smoke \
  --cpu-reference --reference-agreement --output ../gb10-results/reference-agreement.json
for tool in memcheck racecheck synccheck initcheck; do
  compute-sanitizer --tool "$tool" --report-api-errors no --error-exitcode 99 \
    python -m benchmarks.qualify_gb10_shared --public-selected --smoke --cpu-reference \
    --output "../gb10-results/$tool.json"
done
python -m benchmarks.paired_gb10_schedules --public --seq 512 1024 4096 \
  --heads 4 16 --repeats 7 --window-ms 100 \
  --output ../gb10-results/paired-dense.json
python -m benchmarks.paired_gb10_schedules --public --causal --seq 512 4096 \
  --heads 4 16 --repeats 7 --window-ms 100 \
  --output ../gb10-results/paired-causal.json
python -m benchmarks.bind_gb10_loaded \
  --output ../gb10-results/loaded/bound.json
```

CPU/CUDA reference agreement passed the same 17 selected fixtures at each of
H4 and H16, including common offsets and rounded key/value cancellation.
Maximum differences between reported error metrics were 7.17e-7 and 1.69e-6,
respectively; both devices passed the unchanged gates.

The optional --cpu-reference harness mode copies the identical rounded operands,
actual outputs/gradients and statistics to CPU and calls the unchanged per-head
assess/statistics gates with four CPU threads. It changes the reference execution
location only. Use --reference-agreement outside sanitizers to check both CUDA
and CPU gates before using CPU references under a sanitizer. This avoids
instrumenting reference FP64 GEMMs while retaining the same acceptance criteria.

The pairing driver retains samples, per-head mathematical/rounded diagnostics,
graph resources, loaded-library names and coarse board/CPU activity samples.
Its CPU stub executes dense/causal eager and graph builders, checking repeated
backward reuse, operand identity and gradients. An early causal eager-joint
benchmark omitted the visibility keyword; those measurements are excluded and
the corrected driver is rerun. Exact frozen source/commands, raw reports,
wheel, hashes, loaded code and failures remain outside the tracked tree.


## Causal head grouping

The common scheduler's final configuration field groups heads when ordering
causal key tiles. It changes CTA order while preserving every tile's ownership,
masking, stages, rounded operands and reductions. The opt-in GB10 selector uses
G16 only for batch one, H16/S512/D128 BF16 causal attention. The long H4 FP16
exception below uses G4; other qualified families retain G1, including FP16 at
the short BF16 shape.

A nine-round same-process G1/2/4/16 study kept the classification cache and
M64/N64 Q2/register-V geometry fixed. At H16/S512 BF16, backward graph medians
were 75.97 us for G1 and 67.96 us for G16; joint graph was 97.45 versus
93.86 us. The retained rounds do not overlap: joint G1 ranged 97.02–98.33 us
and G16 93.33–94.54 us. FP16 at this shape loses with G16. H4/S512 has only
32 key-owned CTAs on 48 SMs and showed no retained improvement.

The same ordering transfer fails for long FP16. H16/S4096 joint graph measured
3511.53 us with G1, 3573.44 us with G4 and 6536.82 us with G16, versus
3408.39 us for FA2. Instantiated FP16 main resources were 255 registers,
82944 shared bytes, and local allocation of 0/48/88/88 bytes for G1/2/4/16.
These resources support a follow-up storage intervention; they do not measure
local-memory traffic or prove a stall percentage. Larger grouping is not
selected for those cases. [The group summary](../results/gb10-causal-groups-summary.json)
retains all measured rows, rounds, negative results and source hashes.

G2/G4/G16 each passed 24 partial-head, batch and tail cases, plus all four
Compute Sanitizer tools over six changed-data/poisoned-graph cases. The
validator checks O, dQ, dK, dV and the literal saved unscaled statistics
independently. CPU references copy the identical rounded operands and actual
O, preserve the existing assessment functions and gates, and use four threads.
Separate CPU/CUDA oracle controls passed both unchanged gates. These checks
add no new public shape or precision policy.

The final actual-public before/selected/FA2 pair used nine rotating rounds,
500 ms warmup and 100 ms windows, with no concurrent compilation. Complete
joint graph measured 98.47/93.88/114.90 us, respectively: the selected entry
was 4.7% below its cached public predecessor. Backward graph measured
71.71/65.06/71.83 us. Complete joint eager wall medians were
127.98/117.90/461.67 us. The unusually large FA2 eager result is retained;
earlier short-eager variability remains unresolved and this row does not
establish a general host-latency gain.

The actual selected public entry passed 17 numerical cases and CPU/CUDA
reference agreement, plus all four sanitizer tools over 11 cases. Together
with the unchanged earlier public qualification, this checks the single new
route without repeating unrelated full-family gates. CPU validation ran 281
tests total: 123 passed and 158 GPU tests skipped; publication guard tests
passed all 24 cases.

CUPTI-captured loaded ELF matched both changed main kernels uniquely by
symbol and graph register resources. G1 used 246 registers, G16 254; both
used 82944 shared bytes, zero local bytes, binary/PTX version 121, and the
same CTA grid. Their whole-function SASS each contains 160 BF16 HMMA,
96 LDSM, 32 FP32 dQ atomic reduction sites, and eight block barriers;
no LDL/STL instructions appear. The remapping adds integer address work,
not fewer attention products or atomic updates. Static counts describe equal
logical attention work and are not runtime stall percentages. The unchanged
pre/post kernels have identical SASS in both containing ELFs, but cannot be
bound uniquely to one container by symbol/register matching; the retained
binding report is explicitly marked incomplete for that ambiguity.

Loaded G1 main ELF SHA256:
`00ed2661a18a93623db17c38e0c322a15981b976b301e1f70942b6d7c16e3190`.
Selected G16 main:
`674bd64d9363b85885ffa20eff91844bd8df6714e38fa26570340f0c409dce45`.
Compute counters remain unavailable under the previously recorded
`ERR_NVGPUCTRPERM` limitation; no cache-hit or stall explanation is claimed.

## Long causal H4 FP16

The opt-in public entry selects Q1/shared V/G4 only for batch one,
H4/S4096/D128 FP16 causal attention. Its tile, eight-warp layout, forward,
rounded operands, output types and arithmetic gates stay fixed. H16, BF16,
other lengths and other visibility families keep their previous schedules.

Two independent actual-public before/selected/FA2 comparisons use thirteen
rotating rounds, 1000 ms warmup and 200 ms windows. Times below are microseconds;
complete calls include forward and all backward preparation.

| Repeat | Boundary | Previous | Selected | FA2 |
| --- | --- | ---: | ---: | ---: |
| First | Backward graph | 645.16 | 611.57 | 605.20 |
| First | Complete graph | 878.64 | 848.64 | 871.99 |
| First | Complete eager wall | 881.30 | 855.49 | 876.26 |
| Second | Backward graph | 649.64 | 617.97 | 612.82 |
| Second | Complete graph | 885.37 | 861.17 | 877.00 |
| Second | Complete eager wall | 893.40 | 858.21 | 892.58 |

Complete graph latency falls by 2.7–3.4% against the previous public entry.
The first previous/selected round ranges are 875.72–891.23 and
846.09–857.98 us; the second ranges are 883.12–899.36 and
856.95–868.34 us. Selected complete calls beat FA2 in both repeats, while
backward alone still trails FA2. These results do not establish an eager gain
for the earlier short families or a long H16 improvement.

The storage/order factorial includes every Q2/register-V, Q1/shared-V and
Q2/shared-V combination at G1/G4/G16. Complete graph FP16 medians follow,
again in microseconds, with the same D128/S4096 computation:

| Q stages / V | Group | H4 | H16 |
| --- | ---: | ---: | ---: |
| 2 / Registers | 1 | 881.95 | 3484.36 |
| 2 / Registers | 4 | 879.46 | 3570.86 |
| 2 / Registers | 16 | 883.82 | 6529.66 |
| 1 / Shared | 1 | 869.72 | 3439.48 |
| 1 / Shared | 4 | 852.09 | 3483.23 |
| 1 / Shared | 16 | 852.82 | 6342.27 |
| 2 / Shared | 1 | 875.64 | 3448.49 |
| 2 / Shared | 4 | 862.12 | 3506.39 |
| 2 / Shared | 16 | 859.58 | 6341.45 |
| FA2 | — | 867.17 | 3387.33 |

Q2/shared V removes local allocation from the large-group H16 control, yet G16
still loses heavily. A spill-only explanation is therefore insufficient.
Counter access remains unavailable; these timings do not establish cache-hit
rates, traffic or a specific stall cause. The [compact long-causal report](../results/gb10-long-causal-summary.json)
retains all factorial controls, both actual-public repeats, samples and resources.

The final actual-public route passed seven numerical/statistics cases with
unchanged CPU and CUDA reference gates: random, common offset, rounded key/value
cancellation, underflow and two poisoned-data graph replays. Memcheck, racecheck,
synccheck and initcheck each passed the same seven cases. The new selector's CPU
boundary test covers both head counts and dtypes; the complete CPU suite passed
283 tests total with 158 GPU skips, and all 24 publication-guard tests passed.

The original noncausal underflow fixture overflows FP16 when reused with causal
prefixes: its rounded reference reaches 400934.78, and both the reference's FP16
cast and the kernel produce 94 positive infinities per head at exactly the same
coordinates. That failed fixture and its operands remain retained; it is not a
successful finite-output case. The causal qualification multiplies dO by the
exact power-of-two factor 1/8, keeping P and its rounding/underflow unchanged
while bounding dV near 50117. A CPU regression preserves the original overflow
positions and signs and checks the representable rescaling. No kernel arithmetic,
reference equation or acceptance tolerance changes.

CUPTI captures bind both changed main kernels uniquely to loaded ELF. Their
symbols and allocated resources match both actual timing repeats. Q2/register-V
G1 uses 255 registers and Q1/shared-V G4 uses 215; both have 82944 shared bytes,
zero local bytes and the same grid/block geometry. Each whole function contains
160 FP16 HMMA and 96 LDSM instructions; static block barriers change from eight
to seven. Those counts include prologue, loop and drain and are not runtime
percentages or an attribution of the measured gain. The unchanged preprocess
and conversion code appears identically in multiple containing ELFs, leaving
those container bindings explicitly incomplete.

Loaded previous main ELF SHA256:
`1bf6f8578195ae1a70a9835867a24464908c01c595172fc7934145b0e38012b7`.
Selected main:
`8e30efe88c18b868617cae25804ce4d18b39308ac56affe1828e441d90416009`.

Reproduce the bounded qualification using the pinned installation above:

```bash
python -m benchmarks.qualify_gb10_shared --public-selected --heads 4 \
  --long-causal-only --cpu-reference --reference-agreement \
  --output ../gb10-results/long-causal-h4.json
```
