# A100 long causal work order

The A100 policy interleaves causal key tiles across groups of heads for the
measured 4096-token cases: H8 selects group4 and H16 selects group16. Dense
attention, other lengths and head counts keep their preceding policy. The
public eligibility boundary is unchanged; short sequences remain separate from
this change and from the explicit inference API.

The update changes CTA indexing, keeping the selected tiles, pipeline depths,
operand casts, FP32 products/accumulation, actual rounded O delta and late dQ/dK
scale. The central explicit `Schedule` defaults remain unchanged. Floating-point
dQ atomic reduction order can change, so gradients retain their declared norm
and directed rounded-equation gates rather than requiring bitwise equality.

## Evidence and scope

The initial same-process G1/G2/G4/G16 study used seed371. On the full 108-SM
A100-SXM4-40GB, G4 was best for long H8 and G16 for long H16; S128 grouped
variants regressed and H8/S1024 was mixed. Those short families were not enabled.
A separate actual public API evaluation used seed4093, followed by the final
merged-source evaluation with seed4243. All stages preserve the pinned Python
3.13.18, Torch 2.15 CUDA 13.2, CUTLASS DSL 4.8.1 and FA2 2.8.4 stack, with the
same 350 W device. No Q2 pipeline change is included.

The final merged-source run passed public correctness/statistic/poisoned-graph
checks, all four sanitizer tools, 128 actual API timing rows and all 16
loaded-kernel associations. The [compact result](../results/a100-causal-order.json)
retains samples, per-head error summaries, source/cache proofs and resources.
The earlier composed-source evaluation is retained separately and is not
relabelled as this final run. The retained source hashes describe that measured
snapshot. The subsequent merge of the GB10-only geometry update changes its
SM120/SM121 helper files and dispatch argument, while the A100 schedule, native
runtime and GPU methods remain unchanged. Final rebased CPU checks passed
358 tests, with 177 skipped; all 24 publication-guard tests passed.

| Long causal family | Eager joint speedup over G1 | Graph joint speedup over G1 | Graph joint over FA2 | Graph joint over SDPA |
| --- | ---: | ---: | ---: | ---: |
| H8, group4 | 1.128× | 1.165× | 1.165× | 1.240× |
| H16, group16 | 1.090× | 1.091× | 1.081× | 1.162× |

Each row is the geometric mean of D64/D128 and BF16/FP16 at batch1/S4096.
Every one of those eight cases wins against G1, FA2 and SDPA in both joint
boundaries. Individual graph gains over G1 are 14–18% for H8 and 7–11% for H16.
These results do not establish a win for other shapes, and the public policy
keeps those families unchanged.

No forward speedup is claimed. Other measured rows have geometric G1/candidate
ratios of 0.992 eager and 0.996 graph, with individual ratios down to 0.953.
All rows and samples remain in the result rather than being filtered to wins.
The captured representative forward ELF/SASS images are identical before/after;
this is a backward work-order change.

The measured comparison executes exact public API bodies with independent
previous/current selected-entry and schedule modules. Both use the same native
runtime, and each accepts changing operands. The H8 medium causal exclusion can
warm both generic-loader caches; cache proofs retain the actual state rather
than assuming every comparison stays cold.

## Reproduction

```sh
python -m benchmarks.compare_attention_variants \
  --operations '[{"label":"previous_public","factory":"benchmarks.a100_group_api_controls:make_attention","kwargs":{"previous":true}},{"label":"candidate_public","factory":"benchmarks.a100_group_api_controls:make_attention"},{"label":"fa2","backend":"fa2"},{"label":"sdpa","backend":"sdpa"}]' \
  --seq 2048 4096 --dim 64 128 --heads 16 --dtype bf16 fp16 \
  --phases forward joint --visibility dense causal --no-grad-forward \
  --seed 4243 --repeats 7 --warmup-ms 1000 --window-ms 50 \
  --output artifacts/a100-causal-order-h16.json
```

Repeat with `--heads 8` for the other measured family. Forward uses NoGrad;
joint includes forward and all gradients. Each round rotates backend order.
Compilation and warmup precede the seven windows, each targeting 50 ms. Eager
wall time includes preparation and launch; CUDA-event graph replay is reported
separately. FA2 and SDPA are complete backend comparisons, not numerical oracles.

## Correctness and code

Each grouped core passed the 74 directed cases, residual head/batch groups,
partial tiles, literal statistics, poisoned changing graphs and all four
sanitizer tools. The final public path checks selected schedules on actual
runtime tensors, saved statistics and each gradient independently. It checks
CPU/device reference agreement outside sanitizers before using identical CPU
diagnostics while instrumenting the implementation. The per-head limits stay
1% BF16 and 0.5% FP16, with actual O for the rounded backward reference.

Loaded-code evidence captures actual CUPTI module-load ELF bytes and matches
exact graph symbols/register resources, using load phase when necessary to
resolve a unique image. Driver module queries are separate: the callback handle
union has stream semantics and is not treated as a CUmodule handle. Static SASS
counts are not runtime percentages. Counter permission and hardware ceilings
are not inferred from these associations. Referenced ELF/SASS files must pass
their off-host SHA checks before a result states they are retained.

The 16 graph kernels have unique loaded-image associations. All 18 referenced
ELF/SASS files (4,289,634 bytes) were copied off-host and checked against their
recorded SHA values. This verifies the referenced code, not a final archive of
every campaign artifact. The representative backward functions keep 320 HMMA
instructions and their existing async-copy/barrier counts in the static
function listing. Head decoding adds index bookkeeping; one H8 case introduces
16 local bytes while improving latency. Static counts and resource-derived
occupancy alone do not explain or quantify the runtime gain.
