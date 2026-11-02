# L40S short training: product layout and saved statistics

This independently qualified public policy is limited to full 142-SM NVIDIA
L40S, B1 H16 S128 D16, FP16/BF16, dense or causal training. It preserves the
preceding live gradient/metadata guards, basic-device whitelist and fallback
policies. No-grad/inference, masks, GQA, other sizes, unsupported scales and
compilation retain their preceding routes. Equal-shape MHA continues to accept
`enable_gqa=True`.

The shared GPU class remains the exact frozen class qualified in the preceding
study. L40S has its own dtype/visibility policy, so future RTX 4090 tuning cannot
silently change it:

| Family | Narrow product tile | Prefetch saved statistics |
| --- | --- | --- |
| BF16 dense | No | Yes |
| FP16 dense | Yes | Yes |
| BF16 / FP16 causal | Yes | No |

The product layout changes the warp ownership of the logical D16 gradient
products. Prefetching caches the saved FP32 maximum, log normalization and delta
in registers before probability/dS reconstruction. These changes retain
[the same arithmetic](precision.md): separate unscaled maximum/log normalization,
P rounded once before dS, dS rounded once, FP32 accumulation, delta from actual
rounded O, and late dQ/dK scaling. There is no stronger-accuracy mode.

## Held-out actual public calls

The [report](../results/l40s-short-optimized.json) and
[matching chart](../results/l40s-short-optimized.svg) retain all 24 forward,
backward and combined graph/eager rows. The independent evaluation seed is
2992779, with 1000 ms warmup and nine alternating 100 ms rounds. Profiling is
separate. Combined training medians, microseconds:

| Input / visibility | Public graph | Basic flags-off graph | FA2 graph | SDPA graph | Public eager | Basic flags-off eager |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 dense | 9.35 | 10.27 | 13.92 | 18.09 | 203.07 | 202.93 |
| BF16 causal | 9.02 | 9.49 | 14.51 | 18.66 | 202.60 | 202.38 |
| FP16 dense | 8.82 | 9.90 | 13.94 | 18.01 | 205.80 | 204.16 |
| FP16 causal | 9.06 | 9.54 | 14.52 | 18.76 | 205.80 | 205.95 |

Complete graph calls are 1.49–1.61 times faster than standalone FA2 in this matrix.
The pristine-class control below independently confirms the improvement. Eager
results are mixed and close to the flags-disabled baseline; this study does not
establish a general eager speedup. FP16 dense eager is 1.64 µs (0.8%) slower
in the primary matrix. All samples and variability remain visible. FA2 and
SDPA are complete implementation comparisons, with their own arithmetic and adapters.

The same-process baseline uses the actual preceding PR #81 Python namespace
with both shared-class flags disabled. It is explicitly labeled **basic, flags
disabled**, rather than relabeled as the pristine original class. That disabled
class had a 0.7–1.2% graph regression in the earlier compatibility study.

A separate actual-public comparison therefore retains the pristine original
class under the same native bridge. Persistent child processes isolate the
native family cache; the original class is installed before its first compile,
with no per-call patch. Actual rounded Q/K/V/dO hashes and mapped native-library
hashes match across workers. Startup/compilation are excluded; warm windows
alternate. Its independent seed is 1992773:

| Input / visibility | Pristine original graph | Optimized public graph | Ratio |
| --- | ---: | ---: | ---: |
| BF16 dense | 10.63 | 9.82 | 1.083× |
| BF16 causal | 9.91 | 9.50 | 1.043× |
| FP16 dense | 10.27 | 9.28 | 1.107× |
| FP16 causal | 9.93 | 9.53 | 1.042× |

Eight shared-storage graph controls additionally reverse allocation/capture
order for each dtype/visibility case. Eight no-grad and eight long BF16 D128
controls retain the preceding paths. All samples, paired differences, ties and
regressions remain in the report. The original basic-owner study remains
available as a [dated control](l40s-short-owned.md).

## Code, safety and compatibility

All 15 API/policy/cache/metadata methods pass without skips, covering four valid
entrypoints, live gradient flags, parameters/subclasses/parametrized modules,
cold/warm reuse, changed metadata, fallbacks and compiled calls. The full public
route passes 18 finite directed cases, statistics, changed poisoned graphs and
negative views. A separate FP16 causal fixture retains its expected final-cast
infinity diagnostic, checking finite subsets and infinity signs. Per-head
output/dQ/dK/dV limits remain 1% BF16 and 0.5% FP16.

CPU qualification ran 399 tests with 192 expected skips; all 24 publication
checks pass. Sixteen strict public/FA2 forward/backward bindings retain exact
launched symbols and loaded cubin/PTX/SASS. All four selected backward entries
are byte-identical to the private L40S flag-policy probe. Its four sanitizer
passes, 12 cases each, are reused only after that exact-code proof.

Measured selected backward resources and whole-function instruction sites:

| Family | Registers | Shared bytes | Local bytes | HMMA | LDSM | LDS | BAR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 dense | 164 | 99840 | 0 | 112 | 96 | 28 | 3 |
| FP16 dense | 174 | 99840 | 0 | 112 | 72 | 22 | 3 |
| BF16 causal | 128 | 99840 | 0 | 104 | 72 | 56 | 3 |
| FP16 causal | 131 | 99840 | 0 | 104 | 72 | 50 | 3 |

These static sites are not executed-transaction counts or runtime percentages.
Both basic and optimized routes retain two kernels per complete training call.
Systems correlates five eager calls per case and independently records graph
nodes; exact variant identity comes from the loaded-code bindings. Actual native
libraries and build commands are retained from timing-worker mappings. The
17 configured fields plus family tag and 14-tensor launch ABI remain unchanged.

Complete source/modes/commands are frozen outside the GPU host before launch.
Closed raw artifacts, operands, binaries and captures are physically SHA-verified
off-host before publication. The report generator checks stage receipts,
inventories and code equivalence, rather than inferring a pass from installed
packages or similar symbols.

## Reproduction

Use the pinned `sm80-fa4` stack: Python 3.13.18, PyTorch 2.15.0+cu132,
CuTe DSL 4.8.1, TVM FFI 0.1.15.post0 and quack 0.6.5. Retain captures outside
the checkout:

```bash
python -m benchmarks.l40s_short_optimized_controls --prepare-parent
python -m benchmarks.l40s_short_controls --prepare-parent
python -m benchmarks.l40s_public_metadata_controls --prepare-parent
python -m benchmarks.validate_l40s_short_public --output "$CAPTURE_DIR/qualification.json"
python -m unittest tests.test_l40s_short tests.test_l40s_short_optimized tests.test_l40s_native_predicate -v
python -m benchmarks.compare_attention_variants \
  --operations '[{"label":"public","factory":"benchmarks.l40s_short_optimized_controls:make_attention","kwargs":{"mode":"public"}},{"label":"basic_default_false","factory":"benchmarks.l40s_short_optimized_controls:make_attention","kwargs":{"mode":"parent"}},{"label":"fa2","backend":"fa2"},{"label":"sdpa","backend":"sdpa"}]' \
  --seq 128 --dim 16 --heads 16 --dtype bf16 fp16 --visibility dense causal \
  --phases forward backward joint --seed 2992779 --warmup-ms 1000 \
  --window-ms 100 --repeats 9 --output "$CAPTURE_DIR/paired/report.json"
```

The frozen launch also retains the pristine-class persistent-worker and
same-storage capture-order controls. Older L40S charts and the historical
four-device grid keep their original measurement scope.
