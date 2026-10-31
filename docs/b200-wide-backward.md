# B200 D512 backward

The B200 selector uses 256-column gradient output tiles for BF16 self-attention
with batch 1 and head dimension 512: 16 equal query/key/value heads at
512–1024 tokens, or 8 heads at 1024 tokens. It covers dense and implicit causal
attention without an explicit mask, using the existing aligned TMA pipeline. Runtime length checks
preserve the preceding owner outside this range, including cached calls that
cross 511/512 or 1024/1025. Other devices and shapes retain their existing
selection; the SM103 selector is unchanged.

## Complete-call measurements

Final measured-source results are retained in
[the compact report](../results/b200-wide-backward.json). Times are milliseconds;
forward + backward includes forward, actual-output delta, and all three
gradients. The comparison rotates warmed CUDA graphs over five timing rounds.
Compilation and profiler runs are excluded. The preceding CuTe owner uses the
same arithmetic contract.

B=1/H=16, BF16, D512; milliseconds.

| Tokens | Visibility | Previous F+B | Wider F+B | Reduction | SDPA F+B |
| ---: | --- | ---: | ---: | ---: | ---: |
| 512 | dense | 0.1508 | 0.0861 | 42.9% | 0.8364 |
| 512 | causal | 0.1425 | 0.1027 | 27.9% | 0.8005 |
| 1024 | dense | 0.4755 | 0.3066 | 35.5% | 3.2123 |
| 1024 | causal | 0.3457 | 0.2801 | 19.0% | 2.5346 |

H8/S1024 reduces complete-call time by 20.7–41.3% in the measured dense/causal pair.

Forcing the wider owner at H8/S512 regressed causal forward + backward by
10.7% in the preceding experiment. That geometry has only 64 wide CTAs,
compared with 128 preceding-owner CTAs; H16/S512 has 128 versus 256. The H8
selector therefore starts at the measured 1024-token case. Its smaller calls
retain the preceding owner.

SDPA selects PyTorch's packaged SM80 memory-efficient kernels for these D512
calls. Its numbers compare complete implementations. The pinned ordinary FA4
backend failed D512 with its score/output TMEM capacity error; these results
make no successful D512 timing claim for that backend.

## Work and ownership

The preceding owner produces four 128-column slices for D512. Each slice
reconstructs the same QK, dP and probabilities. The wider owner produces two
256-column slices, sharing those computations across more output features.
The full gradient products and output-delta pass remain present.

At B1/H16/S1024, the query and KV grids shrink from 32×16 to 16×16 CTAs.
Each CTA uses 512 threads instead of 384 and 232,448 shared bytes instead of
201,728. The retained launch metadata and independently loaded cubins describe
the actual kernels. These resource changes do not by themselves establish a
speedup or an occupancy improvement.

The KV scalar-statistics stage also receives an explicit buffer-lifetime
handoff. Thirty-two producer threads and 128 probability/derivative consumers
participate in two unaligned named barriers with a compiler memory clobber.
One orders reuse of the shared statistics buffer; the other publishes its
new max, log-normalizer and actual-output delta. This repairs the observed
write-after-read reports for those staging rows. The B200 flag leaves the
existing SM103 handshake unchanged.

Systems captures independently establish the delta/query/KV launch order.
SASS was extracted from cubins copied by CUPTI's module-loaded callback and
matched to the exact graph symbols. Instruction counts cover whole functions;
they are not percentages of runtime or normalized inner-loop work. Compute
counters were denied with `ERR_NVGPUCTRPERM`.

## Arithmetic and validation

Both owners retain `baseline_precision=True`. QK and dP accumulate in FP32.
Reconstruction uses the separate unscaled score maximum and logarithmic
normalizer. P is rounded to BF16 before dS, and dS is rounded to BF16 before
the FP32 gradient products. An FP32 storage copy of rounded P retains those
same BF16 values. The existing `OutputDelta` consumes the actual rounded O
and dO. dQ/dK receive the attention scale after accumulation, before their
final output conversion. Compensated correction branches remain disabled.

The directed tests compare rounded equations using actual O. Random cases
retain independent per-head 1% BF16 norm limits for output and each gradient.
Tests check selected wide/fallback kernels across both length boundaries,
rectangular fallback, negative and zero scales, cached compilation, and graph
replay with changed inputs. The timing matrix independently checks every
component against FP64.

Targeted memcheck, racecheck and synccheck cover both repaired backward owners,
including all their eager and changed-graph launches. An unfiltered global-only
initialization check also passes with input, forward and delta producers
included. Consumer-filtered global diagnostics and a bounded readback-control
failure are retained separately; they do not establish a producer-complete
initialization failure. Shared-memory initialization diagnostics
remain explicit: initcheck reports byte `0x54` in compiler-generated allocation
code. An allocator-only control places the application holding slot at `0x4a0`
and records the reserved-region registers as `(0, 0x120, 0x400, 0x40, 0)`.
The exact loaded forward instruction is an atomic OR at `0x40 + 0x14`;
the application's holding-slot store/read use a separate address. The control
reproduces the report with and without a poisoned application slot, while all
threads observe its hardware-written allocation result. This locates the report
in reserved allocator state; it does not establish a clean initcheck result.
Unchanged native-forward race reports are retained separately. These checks
do not establish race-free complete attention or checkpoint-level quality.

## Reproduction

The rotated comparison driver is provided by the preceding benchmark-tools
change. In this report `cute` is the current default and `candidate` is
the explicit preceding-schedule control:

```sh
python -m unittest tests.test_b200_wide -v
mkdir -p ../b200-captures
python -m benchmarks.compare_flashattention --backends cute sdpa \
  --seq 512 1024 --dim 512 --heads 16 --dtype bf16 --modes dense causal \
  --context benchmarks.b200_schedule_control:previous_backward \
  --repeats 5 --window-ms 80 --output ../b200-captures/comparison.json \
  --export-code ../b200-captures/code
```

Integration on the current main base passes both GPU tests and 225 CPU tests
(139 GPU skips). Both actual loaded cubins remain byte-identical to the measured
source captures after the unrelated SM80/GB10 host-dispatch updates.

Raw profiler and sanitizer captures remain outside the tracked tree.
