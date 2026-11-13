# GB10 experiment catalog

See the [shared invocation and status guidance](../README.md#source-catalogs).
Evidence links identify each study family, not a measured result for every helper.

## Cache and metadata

Isolate host classification and native metadata preparation. [GB10 evidence](../../../results/gb10.json); [frozen source controls](../../../controls/README.md).

| Module | Invocation / role | Question or retained purpose |
| --- | --- | --- |
| [gb10_device_cache](gb10_device_cache.py) | CLI; `make_attention` | Controlled immutable-device classification probe and complete-call factories. |
| [gb10_metadata_controls](gb10_metadata_controls.py) | Imported helper / snapshot | Load the complete preceding GB10 host policy without changing its package context. |
| [gb10_metadata_parent](gb10_metadata_parent.py) | Imported helper / snapshot | Explicit full-device GB10 shared schedules with ordinary CuTe fallbacks. |
| [gb10_native_metadata](gb10_native_metadata.py) | CLI | Compare the production native-metadata GB10 API with its complete preceding host policy. |
| [validate_gb10_native_metadata](validate_gb10_native_metadata.py) | CLI (top-level execution) | Compare live native metadata decisions with the complete published parent. |

## Causal ordering and loaded-code association

Compare CTA head grouping and bind measured code to selected paths. [GB10 measured scope](../../../results/gb10.json).

| Module | Invocation / role | Question or retained purpose |
| --- | --- | --- |
| [bind_gb10_loaded](bind_gb10_loaded.py) | CLI | Retain actual GB10 module-load images and associate exact graph resources. |
| [bind_gb10_long_causal](bind_gb10_long_causal.py) | CLI | Retain actual GB10 module-load images and associate exact graph resources. |
| [bind_gb10_selected_group](bind_gb10_selected_group.py) | CLI | Retain actual GB10 module-load images and associate exact graph resources. |
| [gb10_native_head_groups](gb10_native_head_groups.py) | Imported helper / snapshot; `make_attention`, `make_public`, `make_long_public` | GB10 causal CTA ordering controls; all arithmetic and geometry stay fixed. |
| [paired_gb10_schedules](paired_gb10_schedules.py) | CLI | Alternate complete GB10 M64/M32 calls and standalone FA2 in one process. |
| [validate_gb10_head_groups](validate_gb10_head_groups.py) | CLI | Check causal CTA remapping across residual head groups and batches. |

## Geometry, inference and qualification

Evaluate widths, forward geometry and complete public calls. [GB10 backend guide](../../../docs/backends/gb10.md); per-entry validation is distinct from a new universal qualification.

| Module | Invocation / role | Question or retained purpose |
| --- | --- | --- |
| [experimental_gb10_shared](experimental_gb10_shared.py) | Imported helper / snapshot; `make_attention`, `make_public_attention` | Explicit GB10 schedules using the shared shifted warp-MMA implementation. |
| [gb10_forward_geometry](gb10_forward_geometry.py) | Imported helper / snapshot; `make_public`, `make_attention` | Compare dense GB10 forward tiles while preserving its public backward. |
| [gb10_inference](gb10_inference.py) | CLI | Measure the complete explicit GB10 inference API against its previous policy. |
| [gb10_long_d256](gb10_long_d256.py) | CLI | Measure the complete GB10 long D256 forward-only API against its preceding policy. |
| [gb10_long_d64](gb10_long_d64.py) | CLI | Measure the complete GB10 long D64 forward-only API against its preceding policy. |
| [gb10_short_widths](gb10_short_widths.py) | CLI | Measure the complete explicit GB10 inference API against its previous policy. |
| [gb10_width_controls](gb10_width_controls.py) | Imported helper / snapshot | Load a preceding native metadata bridge without sharing its changed predicate. |
| [preflight_gb10_shared](preflight_gb10_shared.py) | CLI | Compile actual GB10 storage variants, inspect resources, and launch no kernels. |
| [profile_gb10](profile_gb10.py) | CLI | Profile the complete public GB10 or standalone FA2 forward/backward call. |
| [qualify_gb10_shared](qualify_gb10_shared.py) | CLI | Qualify explicit GB10 shared schedules against the existing arithmetic gates. |
| [validate_gb10_forward_geometry](validate_gb10_forward_geometry.py) | CLI | Qualify dense GB10 forward geometry and its declared backward rounding. |
| [validate_gb10_inference](validate_gb10_inference.py) | CLI | Qualify the actual explicit GB10 forward-only API and its mode boundaries. |
| [validate_gb10_long_d256](validate_gb10_long_d256.py) | CLI | Qualify actual GB10 D256 inference and preserve live training/fallback semantics. |
| [validate_gb10_long_d64](validate_gb10_long_d64.py) | CLI | Qualify actual GB10 D64 inference and preserve live training/fallback semantics. |
| [validate_gb10_short_widths](validate_gb10_short_widths.py) | CLI | Qualify experimental full-GB10 S128 forward-only widths on saved inputs. |
| [validate_gb10_width_policy](validate_gb10_width_policy.py) | CLI | Exercise new GB10 forward-only dispatch and its live fallback boundaries. |
