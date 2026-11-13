# Ada experiment catalog

See the [shared invocation and status guidance](../README.md#source-catalogs).
Evidence links identify each study family, not a measured result for every helper.

## L40S shapes and public selection

Short/medium shapes, causal groups and host metadata. [L40S evidence](../../../results/l40s.json), [retained rejected controls](../results/l40s-controls.json).

| Module | Invocation / role | Question or retained purpose |
| --- | --- | --- |
| [compare_l40s](compare_l40s.py) | CLI | Compare L40S schedules, precision-matched/released controls and native SDPA. |
| [l40s_causal_groups](l40s_causal_groups.py) | Imported helper / snapshot; `make_attention`, `make_direct` | Reorder causal key/head CTAs while keeping the selected L40S geometry. |
| [l40s_medium_parent](l40s_medium_parent.py) | Imported helper / snapshot | Load the actual archived PR88 native bridge, not a source-equal rebuild. |
| [l40s_medium_public_controls](l40s_medium_public_controls.py) | Imported helper / snapshot; `make_attention` | Actual PR88 binary and Python namespace versus the bounded public candidate. |
| [l40s_public_metadata_controls](l40s_public_metadata_controls.py) | CLI; `make_attention` | Python mirror for live metadata checks and historical comparison controls. |
| [l40s_schedules](l40s_schedules.py) | Imported helper / snapshot; `make_attention`, `make_previous` | Direct controls for the bounded public L40S schedule and preceding dispatch. |
| [l40s_short_controls](l40s_short_controls.py) | CLI; `make_attention` | Exact preceding Python predicate and isolated public-API comparison controls. |
| [l40s_short_optimized_controls](l40s_short_optimized_controls.py) | CLI; `make_attention` | Exact preceding Python predicate and isolated public-API comparison controls. |
| [summarize_l40s_causal_groups](summarize_l40s_causal_groups.py) | CLI | Summarize the qualified L40S causal ordering and render its current chart. |
| [summarize_l40s_medium](summarize_l40s_medium.py) | CLI | Generate the bounded medium-causal report from verified retained artifacts. |
| [summarize_l40s_native_metadata](summarize_l40s_native_metadata.py) | CLI | Build the measured L40S host-guard report and paired-variability chart. |
| [summarize_l40s_short](summarize_l40s_short.py) | CLI | Build the bounded L40S short-training report from verified retained captures. |
| [summarize_l40s_short_optimized](summarize_l40s_short_optimized.py) | CLI | Summarize the independently qualified L40S public short-kernel policy. |
| [summarize_l40s_training](summarize_l40s_training.py) | CLI | Build the compact L40S training report and separate candidate chart. |
| [validate_l40s_causal_groups](validate_l40s_causal_groups.py) | CLI | Apply unchanged L40S arithmetic gates to the proposed causal CTA groups. |
| [validate_l40s_medium_public](validate_l40s_medium_public.py) | CLI | Public medium-causal arithmetic, changed graphs, and live routing checks. |
| [validate_l40s_short_public](validate_l40s_short_public.py) | CLI; `make_attention`, `make_parent_attention` | Qualify the static S128 D16 public entry under the maintained arithmetic. |
| [validate_l40s_static](validate_l40s_static.py) | CLI | Directed arithmetic, tails, and changed/poisoned graphs for the shared static L40S schedule. |

## RTX4090 and shared Ada comparisons

Short paths, automatic selection and qualified S1024 host dispatch. [RTX4090 evidence](../../../results/rtx4090.json), [backend guide](../../../docs/backends/rtx4090.md).

| Module | Invocation / role | Question or retained purpose |
| --- | --- | --- |
| [compare_ada](compare_ada.py) | CLI | Compare RTX 4090 scheduling with its preceding CuTe path and native SDPA. |
| [qualify_rtx4090_s1024_host](qualify_rtx4090_s1024_host.py) | CLI | Qualify the guarded S1024 path through actual public APIs and live policies. |
| [rtx4090_automatic_controls](rtx4090_automatic_controls.py) | CLI; `make_attention` | Actual preceding public API and automatic RTX 4090 comparison controls. |
| [rtx4090_causal_controls](rtx4090_causal_controls.py) | Imported helper / snapshot; `make_attention` | Compare the bounded public policy with its preceding N64 policy. |
| [rtx4090_host_controls](rtx4090_host_controls.py) | Imported helper / snapshot; `make_attention` | Bounded host controls; unchanged S1024 GPU schedules and operands. |
| [rtx4090_predicate_candidates](rtx4090_predicate_candidates.py) | Imported helper / snapshot; `make_attention` | Preserve the preceding Python predicate as a complete API control. |
| [rtx4090_shared_candidates](rtx4090_shared_candidates.py) | Imported helper / snapshot; `make_attention` | Explicit native-host qualification of the full RTX 4090 shared schedules. |
| [rtx4090_short_controls](rtx4090_short_controls.py) | Imported helper / snapshot; `make_attention` | Complete preceding and selected public RTX 4090 entries. |
| [rtx4090_short_optimized_controls](rtx4090_short_optimized_controls.py) | Imported helper / snapshot; `make_attention` | Preserve the initial full-row public entry as the matched baseline. |
| [validate_rtx4090_shared](validate_rtx4090_shared.py) | CLI | Directed arithmetic, tails, and changed/poisoned graphs for the shared static RTX 4090 schedule. |
| [validate_rtx4090_short](validate_rtx4090_short.py) | CLI; `make_attention` | Qualify the static S128 D16 public entry under the maintained arithmetic. |
