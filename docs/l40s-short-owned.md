# L40S short training

This change selects the basic exclusive-gradient owner for a full NVIDIA L40S
(142 SMs, SM89, 101376-byte opt-in shared-memory capacity), square B1 H16 S128
D16 FP16/BF16 training. Dense and causal visibility are qualified. The GPU source
is unchanged from the basic kernel introduced in PR #76; this report qualifies
its public selection on L40S separately.

The native predicate requires active gradient mode and at least one Q/K/V
requiring gradients, equal shapes and dtypes, contiguous aligned storage, the
current CUDA device, no mask/dropout, and a positive normal finite FP32 scale.
Equal-shape MHA still accepts `enable_gqa=True`. Compilation, deterministic
execution, inference, signed/zero/subnormal scales, masks, other sizes, GQA and
unsupported devices retain their preceding routes. The existing L40S long
schedules and live metadata guard remain in place. An explicit native predicate
argument enables the new shape; its default retains the preceding scope so the
parent namespace remains an independent control.

The arithmetic remains the shared contract in [precision.md](precision.md):
separate unscaled maximum/log normalization, one input-dtype P conversion before
dS, one input-dtype dS conversion, FP32 products/accumulation, delta from actual
rounded O, and late dQ/dK scaling. No compensated or stronger-accuracy mode is
selected.

## Held-out complete calls

The [24-row report](../results/l40s-short-owned.json) and
[matching chart](../results/l40s-short-owned.svg) use the actual public SDPA API
and a pristine PR #78 Python namespace in the same process, with matching
unchanged GPU methods and native bridge. They include forward, backward and
combined training, CUDA-graph replay and complete eager calls. The independent
seed is 296659, with compilation/allocation warmup, 1000 ms execution warmup,
nine rotating rounds and 100 ms windows. Profiling is separate.

Combined forward-plus-backward medians, microseconds:

| Input / visibility | Public graph | Parent graph | FA2 graph | SDPA graph | Public eager | Parent eager |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 dense | 10.25 | 14.97 | 13.91 | 18.01 | 206.23 | 224.18 |
| BF16 causal | 9.52 | 14.76 | 14.56 | 18.74 | 210.53 | 228.61 |
| FP16 dense | 9.89 | 15.05 | 13.93 | 18.06 | 205.35 | 222.73 |
| FP16 causal | 9.53 | 14.34 | 14.45 | 18.76 | 206.71 | 224.82 |

Graph complete calls are 1.36–1.53 times faster than standalone FA2 and
1.46–1.55 times faster than the preceding public route. Eager complete calls
improve by about 8%; their absolute host costs differ from the earlier pilot.
All samples and paired differences are retained. Standalone FA2 and SDPA are
complete implementation comparisons, with their own arithmetic and adapters.
Forward graph timings remain essentially unchanged; the benefit comes from
backward ownership, not a new forward algorithm.

Eight no-grad controls retain the generic path. Eight BF16 D128 long controls
retain the preceding schedules: graph differences are below 0.2%; eager results
include a 0.16% dense S4096 regression. These controls establish compatibility
and expose variability, rather than claiming long-shape gains. A separately
seeded eight-row combined-call confirmation is supplemental; the original
24 rows remain the primary data and chart.

## Execution and correctness

All 13 public/API and metadata test methods pass without skips. They cover the
four entrypoints, each live Q/K/V gradient flag, parameters/subclasses,
parametrized modules, changed storage, cold/warm caches, required-loader errors,
compiled calls and fallback policies. More than 400 live metadata comparisons
retain the prior scope plus the new bounded shape. CPU checks ran 394 tests,
with 190 expected GPU skips; all 24 publication tests pass.

The actual public route passes 18 finite directed cases, saved-statistic checks,
changed poisoned graphs and negative views. A separate FP16 causal fixture
retains the expected final-cast infinity diagnostic and checks finite components
and infinity signs. Per-head output/dQ/dK/dV limits remain 1% BF16 and 0.5% FP16.
The preceding L40S pilot passed all four sanitizer tools, 12 cases per tool;
those passes are reused only because GPU source and memory ownership are
byte-identical. No new sanitizer pass is implied by a host-only route change.

Sixteen strict bindings retain the exact public and FA2 forward/backward symbols,
loaded cubins, PTX and SASS for both dtypes and visibility modes. The selected
backward uses 174 registers, 99840 bytes shared memory and no local storage.
Its grid has 16 CTAs, each owning every row of one head and all three gradients.
That reduces complete graph calls from four kernels to two. Systems verifies
five consecutive eager calls per case through runtime/kernel correlation IDs;
the CUDA graph API independently reports two versus four nodes. These are
selection/count proofs, not profiler-derived speedup measurements or occupancy
claims.

The first public run's Systems statistics command failed because its SQLite
export was older than the retained capture. The capture and failed export remain
preserved; an immutable continuation re-exports to a new filename and checks
launch sequences offline. Another small verification executes all four public
cases and captures its actually mapped native library, inode, build command and
source hash. That new mapping proof does not reconstruct the earlier timed
process. Full raw artifacts and source snapshots are physically SHA-verified
outside the repository before publication. The earlier payload-permission failure
and the first offline-statistics retry are also retained with their nonzero
stage receipts and physical inventories; neither ran new GPU math.

## Reproduction

Install the pinned `sm80-fa4` extra; this study retains Python 3.13.18,
PyTorch 2.15.0+cu132, CuTe DSL 4.8.1, TVM FFI 0.1.15.post0 and quack 0.6.5.
The actually loaded FA2 extension hash is recorded with the bindings.
Keep raw outputs outside the checkout:

```bash
python -m benchmarks.l40s_short_controls --prepare-parent
python -m benchmarks.l40s_public_metadata_controls --prepare-parent
python -m benchmarks.validate_l40s_short_public --output "$CAPTURE_DIR/qualification.json"
python -m unittest tests.test_l40s_short tests.test_l40s_native_predicate -v
python -m benchmarks.compare_attention_variants \
  --operations '[{"label":"public","factory":"benchmarks.l40s_short_controls:make_attention","kwargs":{"mode":"public"}},{"label":"parent","factory":"benchmarks.l40s_short_controls:make_attention","kwargs":{"mode":"parent"}},{"label":"fa2","backend":"fa2"},{"label":"sdpa","backend":"sdpa"}]' \
  --seq 128 --dim 16 --heads 16 --dtype bf16 fp16 --visibility dense causal \
  --phases forward backward joint --seed 296659 --warmup-ms 1000 \
  --window-ms 100 --repeats 9 --output "$CAPTURE_DIR/paired/report.json"
```

The frozen launch retains the two pristine parent namespaces, exact commands,
compiler configuration and recovery scripts. The report generator checks
completed stage receipts, physical inventories and code bindings before writing
the compact JSON and SVG. The historical four-device grid and older dated L40S
charts retain their original measurement scope.
