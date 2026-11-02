# Qualified L40S training schedules

The shared warp-MMA implementation serves a bounded set of training calls on
a full 142-SM L40S. The [current causal-ordering measurements](l40s-causal-groups.md)
and [November 2 chart](../results/l40s-causal-groups.svg) qualify its causal groups.
The [November 1 dense/initial training measurements](../results/l40s-warp-training.json)
and [dated chart](../results/l40s-warp-training.svg) retain the preceding native
CuTe and standalone FA2 comparison. The historical cross-device chart and its
arithmetic-policy measurements are unchanged.

Automatic selection requires batch 1, 16 heads, aligned contiguous square MHA,
FP16/BF16, D64/D128 and **exactly S1024 or S4096**. D64 causal selects only
S4096. Gradients must be enabled and at least one Q/K/V operand must require
gradients. Inference, short calls, intervening lengths, masks, GQA, other head
counts, D16/D256/D512, compiled calls and unsupported scales retain the preceding
path. Positive normal FP32 scales through `2e38` are supported; signed, zero,
subnormal and nonfinite scales retain their existing handling. Device eligibility
also verifies the L40S name, 142 SMs and 101376-byte opt-in shared-memory limit.

The exact length guard is intentional. M32/N128 halves key CTAs relative to
M64/N64 and doubles query iterations. Their last-wave geometry changes at other
lengths; endpoint timings do not qualify a continuous interval. Additional
1152/1280/1536/2048/4608 experiments remain separate from automatic selection.

## Schedules and arithmetic

| Family | Forward M/N | Backward M/N | Q/dO stages | Score/KV/dQ M warps | V in registers | Causal head group |
| --- | --- | --- | --- | --- | --- | --- |
| D64 | 128/64 | 64/128 | 2/1 | 2/4/4 | yes | 8 at causal S4096; otherwise 1 |
| D128 dense S1024 | 64/64 | 32/128 | 1/1 | 2/2/2 | no | 1 |
| D128 dense S4096 | 128/32 | 32/128 | 1/1 | 2/2/2 | no | 1 |
| D128 causal | 64/64 | 64/64 | 1/1 | 4/2/2 | yes | 16 at S1024/S4096 |

Forward uses 128 threads and backward 256. Schedules and native host runners
are cached without retaining operand pointers. Other SM89 devices do not enter
the L40S route, including through the explicit shared-package entry.

The D128 dense M32/shared-V configuration reuses the existing kernel methods.
The larger forward tile is restricted to dense S4096; its causal control
regressed. Earlier P staging, packed P conversion and publication-barrier
elision did not produce a general improvement and are not enabled here. The
[separate experiment summary](../results/l40s-warp-experiments.json) retains
their comparisons, the earlier host controls and the larger-forward-tile screen.

## Dense D128 forward refinement

A second independent M64/M128 by N32/N64 experiment selected M128/N32 only
for dense D128/S4096. S1024 retains M64/N64 because the N32 controls tied or
regressed. The public selector's training, dtype, device and exact-length
bounds are unchanged; the backward configuration is unchanged.

The pinned [FA2 D128 launcher](https://github.com/Dao-AILab/flash-attention/blob/4616f5116d33e3ba3cbb71d582caffcfcb49649c/csrc/flash_attn/src/flash_fwd_launch_template.h#L245-L280)
also selects M128/N32 with four warps for noncausal SM86/SM89. Its source
motivates the experiment, while the paired measurements decide this preset.
N32 reduces the measured forward shared allocation from 65536 to 49152 bytes.
Both controls still use 255 registers/thread. Reported local bytes/thread fall
from 32 to 16 in BF16 and from 112 to zero in FP16. These are exact launched
function resources. The CUDA occupancy API reports capacity for two 128-thread
CTAs/SM with N32 versus one with N64 on this device's 102400 shared bytes and
65536 registers/SM. That is a resource bound, not measured achieved occupancy.

The deployed steady key loops also show the tradeoff. At equal 64-key work,
two N32 iterations and one N64 iteration each contain 256 HMMA instructions per
warp. N32 has more static instructions, matrix loads and CTA barriers after
that normalization. FP16's N64 hot loop has 24 local loads and 22 local stores;
N32 has none. BF16 changes eight loads/eight stores per N64 iteration to four
loads/no stores per N32 iteration. FA2's separate 64-HMMA inner backedge is not
mistaken for its full 128-HMMA key loop. Static counts include predicates and
are not runtime stall or traffic percentages.

Standard 96-row protocol, milliseconds; ratios above one favor public N32:

| Dtype / mode | N64 forward | N32 forward | N64/N32 | N64 joint | N32 joint | N64/N32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 graph | 0.6576 | 0.6419 | 1.025× | 2.4577 | 2.4673 | 0.996× |
| BF16 eager | 0.6615 | 0.6402 | 1.033× | 2.4786 | 2.4949 | 0.993× |
| FP16 graph | 0.7324 | 0.6604 | 1.109× | 2.6267 | 2.6144 | 1.005× |
| FP16 eager | 0.7334 | 0.6861 | 1.069× | 2.6376 | 2.6326 | 1.002× |

The standard matrix has mixed BF16 joint results despite the forward gain.
A [separate allocation diagnostic](../results/l40s-n32-allocation.json) reuses
identical rounded input storage and balances three graph capture/allocation
orders. Its per-order public N32 joint-graph medians are 0.7–1.9% lower in BF16
and 0.2–1.5% lower in FP16. The N64 and N32 backward loaded cubin, exact-symbol
PTX and SASS hashes are identical within each dtype. Thus the earlier backward
timing difference did not come from different deployed backward instructions.
One duplicate native N32 allocation still lost; allocation sensitivity remains.
The main table and chart use the original standard protocol, including its
mixed result, rather than substituting the favorable diagnostic values.

The diagnostic eager runs reused saved side-stream contexts and emitted
AccumulateGrad stream-mismatch warnings; they are retained but are not the
standard eager release measurements. The initial diagnostic failed before
timing from a saved default-stream node. Its repair creates saved forwards and
warmup on the capture stream and retains all owners; no safety guard was disabled.

The `parent_n64` control calls the explicit N64 native schedule and omits the
public Python selector overhead. The public N32 measurements include that
overhead. Selection pilot inputs, the first independent evaluation, and the
final public matrix use distinct seeds; the final public seed is 1613.
The complete fresh matrix below includes both changed and unchanged cases.

The [default precision contract](precision.md) is unchanged: separate unscaled
maxima and shifted logarithmic normalization, one input-dtype conversion of P
before dS, one input-dtype conversion of dS for the gradient products, FP32
accumulation and actual-rounded-O delta. Attention scale is applied after FP32
dQ/dK accumulation. No residual products or stronger accuracy policy are added.

## Complete-call results and limits

The compact report retains every paired timing sample and labels excluded
selector cases. Forward retains gradients; backward includes preparation and
all dQ/dK/dV products; joint includes forward. Graph replay and complete eager
calls are separate measurements. Eager CUDA-event intervals include host launch
gaps. Compilation, initial allocator warming and profiling are outside the measurements;
per-call allocation and preparation remain included.
The preceding control calls the native extension directly and omits its old
Python selection overhead; that component makes the eager comparison conservative.
Nine rounds rotate backend order, after one second of warming per mode and
minimum 100 ms windows. These are attention-call results, not model throughput.

Complete eager forward + backward, milliseconds; lower is better:

| Case | Preceding | Candidate | FA2 | Preceding/candidate | FA2/candidate |
| --- | ---: | ---: | ---: | ---: | ---: |
| S1024 D64 dense BF16 | 0.1263 | 0.1024 | 0.1638 | 1.23× | 1.60× |
| S1024 D64 dense FP16 | 0.1283 | 0.0975 | 0.1682 | 1.32× | 1.73× |
| S1024 D128 dense BF16 | 0.2067 | 0.1670 | 0.1837 | 1.24× | 1.10× |
| S1024 D128 causal BF16 | 0.1809 | 0.1420 | 0.1687 | 1.27× | 1.19× |
| S1024 D128 dense FP16 | 0.2188 | 0.1676 | 0.1852 | 1.31× | 1.11× |
| S1024 D128 causal FP16 | 0.1857 | 0.1439 | 0.1653 | 1.29× | 1.15× |
| S4096 D64 dense BF16 | 1.8002 | 1.2317 | 1.2185 | 1.46× | 0.99× |
| S4096 D64 causal BF16 | 1.0945 | 0.7992 | 0.7830 | 1.37× | 0.98× |
| S4096 D64 dense FP16 | 1.8857 | 1.2849 | 1.2977 | 1.47× | 1.01× |
| S4096 D64 causal FP16 | 1.1333 | 0.8171 | 0.8024 | 1.39× | 0.98× |
| S4096 D128 dense BF16 | 3.1779 | 2.4949 | 2.6469 | 1.27× | 1.06× |
| S4096 D128 causal BF16 | 1.8907 | 1.5213 | 1.5042 | 1.24× | 0.99× |
| S4096 D128 dense FP16 | 3.4296 | 2.6326 | 2.8363 | 1.30× | 1.08× |
| S4096 D128 causal FP16 | 1.9807 | 1.6085 | 1.5828 | 1.23× | 0.98× |

FA2 is a separately loaded 2.8.4 implementation with its own normalization and
rounding policy, not a numerical oracle or a precision-matched scheduling
control. D64 and causal cases can still trail FA2. D128 forward can also trail
FA2 despite a faster complete training call. Short-call eager regressions and
forward-only losses motivated the training and length gates; a graph-only win
was insufficient for automatic selection. Per-window sustained power and clock
telemetry was not retained, so no power, bandwidth or utilization ceiling is
claimed. The configured board limit was 350 W.

## Qualification and deployed code

The native C++ path passed 96 directed cases with 60 literal maximum/log-norm
checks. Cases include BF16/FP16 random and long tails, sharp scores, negative
common offsets, signed/zero-scale fallbacks, rounded key/value cancellation,
underflow, model head counts and changed poisoned graphs. Each output and each
gradient has an independent per-head gate: 1% BF16 or 0.5% FP16. Directed cases
use the rounded equations with the actual O; expected operand rounding does not
impose universal agreement with unrounded FP64.

The M128/N64 forward tile and its M128/N32 replacement have independent
96-case directed qualifications. N32 also passed a dense BF16/FP16 tail and
changed/poisoned-graph smoke under all four tools, with CPU references and six
cases per tool. The earlier M128/N64 qualification also includes all four
sanitizer tools. The original native
geometry also passed all four tools. Each smoke has six cases; all report zero
device errors or hazards. Host API negotiation diagnostics alone were suppressed
with `--report-api-errors no`. The later smoke runs the same reference equations
on CPU while every CUDA operation remains instrumented.

Final public tests check all three API entries, literal saved statistics,
individual gradients, cache reuse, changed/poisoned graphs, unaligned dO, rejected
short/intervening lengths and inference fallback. CPU, publication-guard and
final public GPU test counts are retained in the compact report. The final host
cache changes do not change the qualified GPU bodies.

Exact graph symbols were matched to CUPTI-captured loaded ELF bytes. Candidate
ELFs also match compiler-returned cubin hashes. Both candidate and FA2 have
retained exact-symbol PTX and deployed SASS; these are not source rebuilds.
CuTe targets SM89 with PTX ISA 9.4 and the loaded FA2 library targets SM80 with
PTX ISA 9.2. The loaded FA2 extension SHA256 is
`4c9dc972340c047907bfd611072c1680a28559d20b8721673e3eb915bea2223c`.

For BF16 S4096 D128 dense backward, the candidate main launches 512 CTAs with
99328 shared bytes, 255 registers/thread and 8 local bytes/thread. FA2 launches
1024 CTAs with 81920 shared bytes, 255 registers and 16 local bytes. Both have
separate preparation and dQ conversion kernels. Static equal-MN/K loop regions
contain 160 HMMA instructions per warp in both implementations: M32xN128 and
M64xN64 each cover 4096 score elements at K128 with eight warps. The candidate
has 56 M88.4 plus 40 transposed matrix loads, versus FA2's 40 plus 40; both use
128-bit async copies. Five versus three static CTA barriers and six versus eight
async-copy instructions describe those regions, including predicates, rather
than runtime percentages. FA2's outer CTA scheduler backedge is excluded.
At the full dense shape, the N128 key owners traverse Q/dO and contribute to dQ
half as many times as N64 owners, while M32 doubles query-loop iterations per
CTA. This changes work distribution and logical input/partial-gradient traffic;
it does not establish transferred DRAM bytes or a measured bandwidth fraction.
Register, storage and instruction counts alone do not explain the timing result.

Nsight Compute returned `ERR_NVGPUCTRPERM`. No stall, cache-hit, occupancy or
utilization counters were inferred from timings or disassembly.

## Reproduction

The measured stack is Python 3.13.18, PyTorch 2.15.0+cu132, CUDA 13.2, driver
595.99.02, CUTLASS DSL 4.8.1, TVM FFI 0.1.15.post0 and quack 0.6.5. The optional
source is pinned to FlashAttention revision
`4616f5116d33e3ba3cbb71d582caffcfcb49649c`. This stack is retained because both
the optional dispatcher and the paired FA2 wheel are qualified on it; other
versions require their own qualification. Source, driver, report and loaded-code
hashes are retained alongside the compact samples. Full frozen source archives,
commands and raw captures remain outside the tracked repository.

```sh
# Choose an output directory outside this checkout.
CAPTURE_DIR=../l40s-captures
python -m benchmarks.validate_l40s_static --output "$CAPTURE_DIR/native-qualification.json"
python -m benchmarks.validate_l40s_static --candidate-kwargs '{"forward_m":128,"forward_n":32}' \
  --output "$CAPTURE_DIR/forward128-n32-qualification.json"
compute-sanitizer --tool racecheck --report-api-errors no --error-exitcode 86 \
  python -m benchmarks.validate_l40s_static --candidate-kwargs '{"forward_m":128,"forward_n":32}' \
  --cpu-reference --smoke --output "$CAPTURE_DIR/forward128-n32-racecheck.json"
python -m benchmarks.compare_attention_variants \
  --operations '[{"label":"public","backend":"cute"},{"label":"parent_n64","factory":"benchmarks.l40s_schedules:make_attention","kwargs":{"forward_n":64}},{"label":"previous","factory":"benchmarks.l40s_schedules:make_previous"},{"label":"fa2","backend":"fa2"}]' \
  --seq 1024 4096 --dim 64 128 --dtype bf16 fp16 --phases forward backward joint \
  --seed 1613 --warmup-ms 1000 --window-ms 100 --repeats 9 --output "$CAPTURE_DIR/timings.json"
```

`benchmarks.l40s_schedules:make_attention` exposes the same configured native
schedule for explicit tests outside automatic bounds; `native=False` selects the
Python host control. An explicit `forward_m` is a tile experiment. A
`forward_m`-only control retains its original scope across families. Providing
`forward_n` limits both forward overrides to dense D128. These controls
do not expand the public selector. The compact-summary tool consumes the retained
campaign capture bundle; its JSON/SVG output uses repository-relative metadata.
