# Ampere experiment catalog

See the [shared invocation and status guidance](../README.md#source-catalogs).
Evidence links identify each study family, not a measured result for every helper.

## A100 D256

Compare shared/evaluation schedules, rounding and public API costs. [A100 measured scope](../../../results/a100.json).

| Module | Invocation / role | Question or retained purpose |
| --- | --- | --- |
| [a100_d256_api_controls](a100_d256_api_controls.py) | Imported helper / snapshot; `make_attention` | Exact public API controls with isolated old/new schedule module sources. |
| [a100_d256_evaluation_controls](a100_d256_evaluation_controls.py) | Imported helper / snapshot; `make_attention` | Exact public API controls with isolated old/new schedule module sources. |
| [a100_d256_evaluation_entry](a100_d256_evaluation_entry.py) | Imported helper / snapshot | Optional qualified SM80 schedule; other calls retain native CuTe dispatch. |
| [a100_d256_previous_entry](a100_d256_previous_entry.py) | Imported helper / snapshot | Optional qualified SM80 schedule; other calls retain native CuTe dispatch. |
| [a100_d256_previous_schedules](a100_d256_previous_schedules.py) | Imported helper / snapshot | Measured A100 policies; explicit experimental Schedule defaults stay stable. |
| [a100_d256_shared_probe](a100_d256_shared_probe.py) | CLI; `make_attention` | Experimental SM80 width enabling; public dispatch stays unchanged. |
| [a100_shared_api_controls](a100_shared_api_controls.py) | Imported helper / snapshot; `make_attention` | Exact public API controls with isolated old/new schedule module sources. |
| [a100_shared_previous_entry](a100_shared_previous_entry.py) | Imported helper / snapshot | Optional qualified SM80 schedule; other calls retain native CuTe dispatch. |
| [a100_shared_previous_schedules](a100_shared_previous_schedules.py) | Imported helper / snapshot | Measured A100 policies; explicit experimental Schedule defaults stay stable. |
| [bind_a100_d256](bind_a100_d256.py) | CLI | Bind exact A100 public group graph functions to CUPTI-loaded ELF images. |
| [bind_a100_d256_shared](bind_a100_d256_shared.py) | CLI | Bind exact A100 public group graph functions to CUPTI-loaded ELF images. |
| [diagnose_a100_d256_rounding](diagnose_a100_d256_rounding.py) | CLI | Preserved D256 operands: audit independent reference reduction policies. |
| [validate_a100_d256](validate_a100_d256.py) | CLI | Broader D256 evaluation envelope; final automatic selection is checked separately. |
| [validate_a100_d256_selection](validate_a100_d256_selection.py) | CLI | Final narrowed public selection; kernel math/tools are qualified separately. |
| [validate_a100_d256_shared](validate_a100_d256_shared.py) | CLI | Held-out actual public shared-V long calls and changed graph operands. |

## A100 medium shapes and CTA grouping

Compare preceding/current selectors and medium-shape schedules. [A100 backend guide](../../../docs/backends/a100.md); snapshots are historical controls, not current selectors.

| Module | Invocation / role | Question or retained purpose |
| --- | --- | --- |
| [a100_group_api_controls](a100_group_api_controls.py) | Imported helper / snapshot; `make_attention` | Exact public API controls with isolated old/new schedule module sources. |
| [a100_medium_controls](a100_medium_controls.py) | Imported helper / snapshot; `make_parent`, `make_python`, `make_direct` | Pristine preceding public API, loaded once with its own selector globals. |
| [a100_medium_g1_schedules](a100_medium_g1_schedules.py) | Imported helper / snapshot | Measured A100 policies; explicit experimental Schedule defaults stay stable. |
| [a100_medium_g4_api_controls](a100_medium_g4_api_controls.py) | Imported helper / snapshot; `make_previous` | Exact previous public body with the preceding G1 schedule, initialized once. |
| [a100_medium_group_controls](a100_medium_group_controls.py) | Imported helper / snapshot; `make_attention`, `make_direct` | Isolate one causal head-order field behind exact public API bodies. |
| [a100_medium_parent_api](a100_medium_parent_api.py) | Imported helper / snapshot | Strict CuTe attention and automatic per-call native SDPA selection. |
| [a100_medium_parent_entry](a100_medium_parent_entry.py) | Imported helper / snapshot | Optional qualified SM80 schedule; other calls retain native CuTe dispatch. |
| [a100_medium_python_entry](a100_medium_python_entry.py) | Imported helper / snapshot | Optional qualified SM80 schedule; other calls retain native CuTe dispatch. |
| [a100_previous_fa4_sm80](a100_previous_fa4_sm80.py) | Imported helper / snapshot | Optional qualified SM80 schedule; other calls retain native CuTe dispatch. |
| [a100_previous_host_entry](a100_previous_host_entry.py) | Imported helper / snapshot | Optional qualified SM80 schedule; other calls retain native CuTe dispatch. |
| [a100_previous_schedules](a100_previous_schedules.py) | Imported helper / snapshot | Measured A100 policies; explicit experimental Schedule defaults stay stable. |
| [a100_public_api_controls](a100_public_api_controls.py) | Imported helper / snapshot; `make_attention` | Isolate exact old/new public API bodies and their generic-loader caches. |
| [a100_selected_schedules](a100_selected_schedules.py) | Imported helper / snapshot; `make_attention` | Explicit selected A100 schedule against the unchanged native control. |
| [bind_a100_medium_causal](bind_a100_medium_causal.py) | CLI | Bind exact A100 public group graph functions to CUPTI-loaded ELF images. |
| [bind_a100_medium_g4_public](bind_a100_medium_g4_public.py) | CLI | Bind exact A100 public group graph functions to CUPTI-loaded ELF images. |
| [bind_a100_medium_groups](bind_a100_medium_groups.py) | CLI | Bind exact A100 public group graph functions to CUPTI-loaded ELF images. |
| [bind_a100_public_groups](bind_a100_public_groups.py) | CLI | Bind exact A100 public group graph functions to CUPTI-loaded ELF images. |
| [profile_a100_medium_backward](profile_a100_medium_backward.py) | CLI (top-level execution) | Compare G1/G4 backward through public APIs, direct runners and resident buffers; diagnostic timings exclude different host costs. |
| [validate_a100_medium_groups](validate_a100_medium_groups.py) | CLI | Check medium causal head-order controls with fixed rounded equations. |
| [validate_a100_medium_modes](validate_a100_medium_modes.py) | CLI | Check actual runtime gradient-mode boundaries for medium causal selection. |
| [validate_a100_medium_reference](validate_a100_medium_reference.py) | CLI | Audit CPU/device references on identical captured medium-attention operands. |
| [validate_a100_public_groups](validate_a100_public_groups.py) | CLI | Check actual long public selection, literal statistics and changing graphs. |

## A40 selection and shapes

Evaluate native predicates, head groups and short D256. [A40 measured scope](../../../results/a40.json).

| Module | Invocation / role | Question or retained purpose |
| --- | --- | --- |
| [a40_native_head_groups](a40_native_head_groups.py) | Imported helper / snapshot; `make_attention` | Native causal work-order controls with unchanged arithmetic and layouts. |
| [a40_predicate_candidates](a40_predicate_candidates.py) | Imported helper / snapshot; `make_attention` | Matched A40 public controls: native metadata or the qualified Python policy. |
| [a40_shared_candidates](a40_shared_candidates.py) | Imported helper / snapshot; `make_attention` | Explicit native-host qualification of the full A40 shared schedules. |
| [summarize_a40_d256](summarize_a40_d256.py) | CLI | Create the compact, repository-relative A40 D256 qualification report. |
| [validate_a40_d256](validate_a40_d256.py) | CLI | Actual public D256 selection, literal statistics, fallbacks and live graphs. |
| [validate_a40_head_groups](validate_a40_head_groups.py) | CLI | Check causal CTA remapping across residual head groups and batches. |
| [validate_a40_shared](validate_a40_shared.py) | CLI | Directed arithmetic, tails, and changed/poisoned graphs for the shared static A40 schedule. |
| [validate_a40_short_d256](validate_a40_short_d256.py) | CLI | Actual public short D256 selection, literal statistics, fallbacks and live graphs. |

## RTX3090 and shared warp studies

Explicit opt-in, short-shape ownership and shared SM80/SM86 comparisons. [RTX3090 guide](../../../docs/backends/rtx3090.md), [retained negative controls](../results/controls.json).

| Module | Invocation / role | Question or retained purpose |
| --- | --- | --- |
| [compare_rtx3090_optin](compare_rtx3090_optin.py) | CLI (top-level execution) | Measure the bounded public RTX 3090 opt-in entry. |
| [compare_sm80_eager](compare_sm80_eager.py) | CLI | Matched complete eager wall time for native, static schedules, and SDPA. |
| [compare_sm80_host](compare_sm80_host.py) | CLI | Compare complete eager SM80 calls with Python and native preparation. |
| [compare_sm86](compare_sm86.py) | CLI | Compare RTX 3090 Ti schedules, their released control and default SDPA. |
| [sm80_native_host](sm80_native_host.py) | Imported helper / snapshot; `make_attention` | Explicit flat-host SM80 comparison; automatic dispatch is unchanged. |
| [sm80_schedule_candidates](sm80_schedule_candidates.py) | CLI; `make_attention` | Bounded resident-V, warp ownership, and causal forward experiments. |
| [static_sm80_experiments](static_sm80_experiments.py) | Imported helper / snapshot; `make_attention` | Explicit static schedule and unchanged native control for qualification. |
| [static_sm80_schedules](static_sm80_schedules.py) | CLI; `make_attention` | One-factor dO-stage experiment for the shared static warp-MMA schedule. |
| [validate_rtx3090_optin](validate_rtx3090_optin.py) | CLI; `make_attention` | Qualify the bounded ordinary RTX 3090 public opt-in entry. |
| [validate_rtx3090_short](validate_rtx3090_short.py) | CLI; `make_attention`, `make_parent_attention` | Qualify the static S128 D16 public entry under the maintained arithmetic. |
| [validate_rtx3090_short_d128](validate_rtx3090_short_d128.py) | CLI; `make_attention`, `make_parent_attention` | Qualify the static S128 D128 public entry under the maintained arithmetic. |
| [validate_rtx3090_short_d64](validate_rtx3090_short_d64.py) | CLI; `make_attention`, `make_parent_attention` | Qualify the static S128 D64 public entry under the maintained arithmetic. |
| [validate_sm80_residency](validate_sm80_residency.py) | CLI | Check changed graph inputs and tail ownership for explicit SM80 schedules. |
| [validate_sm80_static](validate_sm80_static.py) | CLI | Qualify the explicit shifted SM80 forward and its joint backward schedule. |
