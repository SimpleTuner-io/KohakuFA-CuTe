# GB10 long D64 forward-only attention

The explicit `gb10_attention` entry selects the existing shared forward-only kernel for full-device B1/H16/S4096/D64 dense and causal FP16/BF16 NoGrad calls. Dense attention uses M128/N64 with two K/V stages; causal attention uses one stage. Both use 128 threads and shared Q. Training, default dispatch, existing D128/D256 policies and short-width selection retain their behavior.

## Complete-call measurements

Full 48-SM GB10, the pinned optional shared-warp runtime, PyTorch 2.15.0+cu132 and standalone FlashAttention 2.8.4. Thirteen rotating 150 ms windows compare the actual preceding public entry, selected public entry, matched raw runner, FA2 and SDPA in one process. Graph replay contains 32 real attention calls; reported device events divide by that count. Eager windows include selection, allocation, launch and a synchronization after the complete enqueue window. They measure sustained complete calls, not isolated request latency. Compilation, references, capture construction and profiling are outside measurement windows.

| Dtype / visibility | Parent graph / selected µs | FA2 graph µs | Parent eager / selected µs | FA2 eager µs | SDPA eager µs |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 / dense | 1107.04 / 761.45 | 741.48 | 1094.28 / 741.56 | 727.51 | 751.87 |
| BF16 / causal | 616.20 / 444.92 | 427.90 | 611.90 / 444.18 | 427.93 | 443.57 |
| FP16 / dense | 1109.47 / 757.94 | 743.77 | 1117.60 / 760.01 | 746.01 | 774.24 |
| FP16 / causal | 619.74 / 457.58 | 434.90 | 630.41 / 465.52 | 441.86 | 459.59 |

Complete eager medians improve by 1.35–1.48× over the preceding public entry. All comparisons and remaining gaps against FA2/SDPA are retained.

[All eight rows and samples](../results/gb10-long-d64.json) include paired comparisons, a 1% descriptive tie band, per-head errors and the original successful timing attempt. The [chart](../results/gb10-long-d64.svg) retains every comparison. FA2 and SDPA keep their own arithmetic policies; these are complete implementation comparisons, not isolated schedule comparisons against those libraries.

## Scheduling and retained alternatives

A bounded private 2×2 experiment compared M64/N64 and M128/N64 with one or two stages at fixed 128 threads and shared Q. It retained all sixteen shape/configuration cases. M128/N64 doubles the queries sharing a CTA's K/V stream and halves the query CTA count. Dense selection uses two stages; causal selection uses one because two was slower in these measurements. These mechanisms motivate the intervention; measured complete-call costs decide the selection.

The selected dense kernel allocates 49,152 shared bytes and 244 BF16 / 248 FP16 registers with no driver-reported local allocation. Causal selection allocates 32,768 shared bytes and 255 registers with no reported local allocation. Actual full-length FA2 selects M128/N128/128 threads, 49,152 shared bytes and 255 registers, with 128 dense / 88 causal local bytes in both dtypes. Exact graph symbols bind to loaded cubins, packaged or compiler-returned PTX, and SASS. Static instruction and resource records are not occupancy, utilization, stall or bandwidth measurements. The remaining FA2 gap is retained, not treated as a hardware limit.

## Qualification and provenance

Independent CPU FP64 checks use identical rounded operands and per-head output norm limits of 1% BF16 and 0.5% FP16. Four public cases cover negative offsets and three changed, poisoned graph replays. Requested gradients use the preceding training implementation and retain bitwise-equal O/dQ/dK/dV; NoGrad and inference mode save no autograd statistics. Parameters, identity subclasses, live gradient flags, fullgraph compiled fallback, explicit forward-AD rejection, signed/zero scales, masks, hidden rows, strides, alignment, tails and GQA are checked. Selected adapter failures propagate. Existing D256, D128/H4 training and short-width gates also pass.

All four ownership tools passed the private factorial's exact full-length base/changed replay cases. Each actual public loaded cubin must match its corresponding checked image byte-for-byte before those passes are reused. GPU arithmetic and kernel bodies remain unchanged. The default precision contract in [precision.md](precision.md) retains shifted scores, one input-dtype probability conversion and FP32 accumulation.

The actual preceding control loads pristine PR86 Python and C++ metadata code through a separate native bridge; it cannot inherit the newly widened predicate. Both actual timed processes retain current and parent mapped libraries, inodes, source hashes and compiler/include configuration. An initial observer failed before reading mappings because a legacy qualifier had already created its output directory. All its successful gates and eight timing rows remain retained. A new immutable timing-only continuation changes only the observer destination, uses the same evaluation seed without retuning, and requires both timed-library proofs. Source and benchmark equivalence are checked before it runs. The final table comes from a subsequent complete run after composing the preceding branch’s RTX 3090/L40S integration. That run rechecks every public gate and both timed libraries; the GB10 arithmetic, schedules and live metadata boundary retain their qualified behavior.

Complete source snapshots, operands, actual loaded code, both timed host libraries and exact build flags are physically SHA-verified against complete offhost inventories. CPU checks ran389 tests total:185 skipped,204 executed, no failures;24 publication-guard tests passed separately.

## Reproduce

Install the pinned optional runtime from [the GB10 setup](gb10-shared.md), retain pristine preceding `native.py`, `native.cpp` and `gb10.py` as `parent_native.py`, `parent_native.cpp` and `parent_gb10.py`, and set `PARENT_SOURCE` and `ARTIFACT_DIR` outside the checkout.

```sh
python -m benchmarks.validate_gb10_long_d64 \
  --parent-source "$PARENT_SOURCE" --output "$ARTIFACT_DIR/qualification.json"
python -m benchmarks.gb10_long_d64 --parent-source "$PARENT_SOURCE" \
  --graph --output "$ARTIFACT_DIR/graph.json"
python -m benchmarks.gb10_long_d64 --parent-source "$PARENT_SOURCE" \
  --output "$ARTIFACT_DIR/eager.json"
python -m benchmarks.plot_gb10_long_d64
```
