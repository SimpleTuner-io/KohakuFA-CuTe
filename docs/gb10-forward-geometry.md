# GB10 long dense forward geometry

The explicit `gb10_attention` entry now selects M128/N32 forward tiles for
B1/H16/S4096/D128 dense FP16/BF16 attention. This measured subset uses the pinned
optional shared-warp runtime. Its existing backward schedule and numerical
contract are retained. The [installation and other measured shapes](gb10-shared.md)
remain the starting point for using the entry.

## Complete-call result

Full 48-SM GB10, PyTorch 2.15.0+cu132 and standalone FlashAttention 2.8.4.
The same process alternates the previous M64/N64 policy, selected public entry,
FA2 and default SDPA. Forward retains gradients; F+B includes forward and all
three gradients, including preparation, scratch initialization and conversion.
Compilation, profiling and capture construction are outside the measurements.

Final evaluation uses an independent seed, 13 rotating rounds, 1,000 ms warmup
and 150 ms measurement windows. The table is CUDA-graph median milliseconds;
[the compact data](../results/gb10-forward-geometry.json) also retains eager
measurements, aligned samples, per-head checks and source/code hashes.

| Dtype | Phase | Previous | Selected | FA2 | SDPA | Previous / selected |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | Forward | 1.5341 | 1.4574 | 1.5536 | 1.6120 | 1.053× |
| BF16 | F+B | 5.9936 | 5.8918 | 6.0206 | 6.4394 | 1.017× |
| FP16 | Forward | 1.5922 | 1.4965 | 1.5851 | 1.6560 | 1.064× |
| FP16 | F+B | 6.0794 | 5.9508 | 6.1858 | 6.5585 | 1.022× |

Eager forward improves by 5.1–6.3%; complete eager F+B improves by about 1%.
The selected graph F+B is 2.2–4.0% faster than FA2 and 9.3–10.2% faster than
SDPA in these cases. These are complete implementation comparisons; the
standalone baselines retain their own arithmetic policies.

## Why this geometry

M64/N64 stores 16 KiB of Q and 32 KiB of K/V staging. M128/N32 stores 32 KiB
of Q and 16 KiB of K/V staging. Both instantiate 49,152 bytes of shared memory.
The wider query tile halves the query CTA count and reuses each CTA's K/V
stream for twice as many output queries. Narrowing N keeps the shared allocation
bounded while doubling the key-loop iteration count.

Exact graph-bound PTX and loaded SASS were retained for both policies and FA2
in both dtypes. At S4096, the old grid is 64 query CTAs × 16 heads; the selected
grid is 32 × 16. Register counts change from 202 BF16 / 165 FP16 to 251 in both
dtypes, with zero reported local memory. The selected FA2 M128/N64 forward uses
65,536 shared bytes, 255 registers and 160 local bytes. These resource records
are not measured occupancy or stall percentages.

For equal steady-state work of 128 queries × 64 keys, compare two old CTAs
with one new CTA executing two N32 iterations. The matched CuTe SASS has:

| Static instruction sites | M64/N64 | M128/N32 |
| --- | ---: | ---: |
| HMMA | 256 | 256 |
| 128-bit LDGSTS | 32 | 16 |
| LDSM, all `.4` forms | 144 | 96 |
| BAR / DEPBAR | 4 / 4 | 4 / 4 |
| Total, BF16 | 1054 | 1152 |

This is a reuse and scheduling tradeoff with more total instruction sites.
The prologue and epilogue are excluded from that normalized loop table.
Fewer load sites do not directly establish physical transaction counts;
performance counters were unavailable in the retained GB10 counter attempt.
FA2 has overlapping branch spans, so its whole-function counts are not treated
as a directly comparable steady loop.

## Qualification and retained alternatives

Random representative cases keep the independent per-head FP64 norm limits of
1% BF16 and 0.5% FP16 for O, dQ, dK and dV. Directed negative-offset fixtures
also check the declared once-rounded equations using actual forward O. On a
retained BF16 S1024 offset fixture, the old and experimental layouts have 1.10%
and 1.08% mathematical dQ error; their respective declared rounding floors
predict almost all of it. Both match that rounded reference within 0.20%.
The mathematical diagnostics and the original failed threshold run are retained.
These checks follow the [declared precision contract](precision.md), including
one input-dtype conversion each for P and dS.

The final source passed eight full output/gradient and changed/poisoned graph
cases, existing fallback checks, and all four Compute Sanitizer tools on the
changed forward ownership. Each tool covers both dtypes, partial S129/S257 and
full S1024 tiles, random/negative offsets, and changed graph data. Four CuTe loaded-code bindings match compiler-returned cubins; the two FA2
bindings identify the mapped installed extension. All six retain exact selected
PTX entries and the loaded ELF/SASS. Raw captures and exact operands are retained outside Git.

S1024 forward also improved in selection runs, but complete F+B was mixed, so
its existing policy is retained. H4 and causal cases retain their existing
policies. Their source, samples and negative experiments remain available for
further tuning; the final seed is separate from policy selection.

## Reproduce

Use the qualified optional runtime and an artifact directory outside this
checkout. The committed validator also supports `--forward-only` for memory
ownership tooling.

```sh
python -m benchmarks.validate_gb10_forward_geometry \
  --output "$ARTIFACT_DIR/qualification.json"

python -m benchmarks.compare_attention_variants \
  --operations '[{"label":"before","factory":"benchmarks.gb10_forward_geometry:make_attention"},{"label":"selected","factory":"benchmarks.gb10_forward_geometry:make_public"},{"label":"fa2","backend":"fa2"},{"label":"sdpa","backend":"sdpa"}]' \
  --seq 4096 --heads 16 --dim 128 --dtype bf16 fp16 \
  --phases forward backward joint --visibility dense --seed 104729 \
  --warmup-ms 1000 --window-ms 150 --repeats 13 \
  --source-revision "$(git rev-parse HEAD)" --output "$ARTIFACT_DIR/paired.json"
```
