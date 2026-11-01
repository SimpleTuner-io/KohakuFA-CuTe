# Qualified L40S training schedules

This change selects the shared warp-MMA implementation for a bounded set of
training calls on a full 142-SM L40S. The [candidate measurements](../results/l40s-warp-training.json)
and [separate chart](../results/l40s-warp-training.svg) compare the actual public
API with the preceding native CuTe dispatch and standalone FA2. The historical
cross-device chart and its arithmetic-policy measurements are unchanged.

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

| Family | Forward M/N | Backward M/N | Q/dO stages | Score/KV/dQ M warps | V in registers |
| --- | --- | --- | --- | --- | --- |
| D64 | 128/64 | 64/128 | 2/1 | 2/4/4 | yes |
| D128 dense S1024 | 64/64 | 32/128 | 1/1 | 2/2/2 | no |
| D128 dense S4096 | 128/64 | 32/128 | 1/1 | 2/2/2 | no |
| D128 causal | 64/64 | 64/64 | 1/1 | 4/2/2 | yes |

Forward uses 128 threads and backward 256. Schedules and native host runners
are cached without retaining operand pointers. Other SM89 devices do not enter
the L40S route, including through the explicit shared-package entry.

The D128 dense M32/shared-V configuration reuses the existing kernel methods.
The larger forward tile is restricted to dense S4096; its causal control
regressed. Earlier P staging, packed P conversion and publication-barrier
elision did not produce a general improvement and are not enabled here. The
[separate experiment summary](../results/l40s-warp-experiments.json) retains
their comparisons, the earlier host controls and the larger-forward-tile screen.

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
| S1024 D64 dense BF16 | 0.1260 | 0.1056 | 0.1729 | 1.19× | 1.64× |
| S1024 D64 dense FP16 | 0.1289 | 0.0983 | 0.1669 | 1.31× | 1.70× |
| S1024 D128 dense BF16 | 0.2065 | 0.1672 | 0.1880 | 1.23× | 1.12× |
| S1024 D128 causal BF16 | 0.1822 | 0.1422 | 0.1717 | 1.28× | 1.21× |
| S1024 D128 dense FP16 | 0.2187 | 0.1683 | 0.1894 | 1.30× | 1.13× |
| S1024 D128 causal FP16 | 0.1854 | 0.1437 | 0.1789 | 1.29× | 1.25× |
| S4096 D64 dense BF16 | 1.7931 | 1.2343 | 1.2175 | 1.45× | 0.99× |
| S4096 D64 causal BF16 | 1.0964 | 0.7980 | 0.7818 | 1.37× | 0.98× |
| S4096 D64 dense FP16 | 1.8722 | 1.3005 | 1.2924 | 1.44× | 0.99× |
| S4096 D64 causal FP16 | 1.1463 | 0.8203 | 0.8049 | 1.40× | 0.98× |
| S4096 D128 dense BF16 | 3.2250 | 2.4867 | 2.6471 | 1.30× | 1.06× |
| S4096 D128 causal BF16 | 1.8943 | 1.5400 | 1.5229 | 1.23× | 0.99× |
| S4096 D128 dense FP16 | 3.4547 | 2.6663 | 2.7951 | 1.30× | 1.05× |
| S4096 D128 causal FP16 | 1.9931 | 1.6150 | 1.5884 | 1.23× | 0.98× |

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

The newly selected M128 forward tile has its own 96-case directed qualification
and memcheck/racecheck/synccheck/initcheck smoke checks. The original native
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
python -m benchmarks.validate_l40s_static --candidate-kwargs '{"forward_m":128}' \
  --output "$CAPTURE_DIR/forward128-qualification.json"
compute-sanitizer --tool racecheck --report-api-errors no --error-exitcode 86 \
  python -m benchmarks.validate_l40s_static --candidate-kwargs '{"forward_m":128}' \
  --cpu-reference --smoke --output "$CAPTURE_DIR/forward128-racecheck.json"
python -m benchmarks.compare_attention_variants \
  --operations '[{"label":"public","backend":"cute"},{"label":"previous","factory":"benchmarks.l40s_schedules:make_previous"},{"label":"fa2","backend":"fa2"}]' \
  --seq 1024 4096 --dim 64 128 --dtype bf16 fp16 --phases forward backward joint \
  --warmup-ms 1000 --window-ms 100 --repeats 9 --output "$CAPTURE_DIR/timings.json"
```

`benchmarks.l40s_schedules:make_attention` exposes the same configured native
schedule for explicit tests outside automatic bounds; `native=False` selects the
Python host control. An explicit `forward_m` is a tile experiment. These controls
do not expand the public selector. The compact-summary tool consumes the retained
campaign capture bundle; its JSON/SVG output uses repository-relative metadata.
