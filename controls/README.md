# Frozen comparison controls

These files are historical source inputs to explicit experiments. They are not
public APIs or standalone scripts. Do not update them to match current dispatch:
that would change the preceding-policy comparison.

| Snapshot | Loader / owning comparison |
| --- | --- |
| [gb10_before_cache.py](gb10_before_cache.py) | [gb10_device_cache.previous_module / make_attention](../benchmarks/experiments/gb10/gb10_device_cache.py) |
| [gb10_before_groups.py](gb10_before_groups.py) | [gb10_native_head_groups.make_public](../benchmarks/experiments/gb10/gb10_native_head_groups.py) |
| [gb10_before_long_causal.py](gb10_before_long_causal.py) | [gb10_native_head_groups.make_long_public](../benchmarks/experiments/gb10/gb10_native_head_groups.py) |
| [a100_medium_parent_entry.py](../benchmarks/experiments/ampere/a100_medium_parent_entry.py) | [a100_medium_controls.make_parent](../benchmarks/experiments/ampere/a100_medium_controls.py) |
| [a100_shared_previous_entry.py](../benchmarks/experiments/ampere/a100_shared_previous_entry.py) | [a100_shared_api_controls.make_attention](../benchmarks/experiments/ampere/a100_shared_api_controls.py) |

The GB10 cache-control loader assigns the package name
`kohakufa_cute.fa4_sm80._gb10_cache_control`; relative imports consequently resolve
against the actual backend package. The A100 loaders either use a package-qualified
module name or construct an isolated package with aliased sibling modules.
Source-byte loading is deliberate, so ordinary import-reference searches do not
show every dependency.

The two A100 entry snapshots above currently have identical bytes. They belong
to different comparison harnesses, which record source hashes and isolate selector
globals. Equality alone is not grounds for deletion or replacing them with an
import of the current backend. A future consolidation must preserve both
experiments' source identity and package context.

## Snapshot identity

The following SHA256 values identify the retained bytes at the navigation audit.
They are not assertions of an original upstream revision. The files do not embed
a verified origin revision; recover and verify that separately from campaign
records or Git history before describing an exact historical checkout.
Other device-study snapshots are listed alongside their consuming helpers in the
[experiment catalogs](../benchmarks/experiments/README.md).

| Repository-relative source | SHA256 |
| --- | --- |
| `controls/gb10_before_cache.py` | `ba50700db22de0660838daea57ed4dfe47a1ce1cc1f83993ec932da31e87849a` |
| `controls/gb10_before_groups.py` | `61b8f78d8c653ee8ddd67f3858618669e5dcbf8f94cbba1c5438c8930d5673a9` |
| `controls/gb10_before_long_causal.py` | `b44645ef80338d4e0c6efa5a2ed5adbbbd2385799f63efb8411514cdb3ca6a81` |
| `benchmarks/experiments/ampere/a100_medium_parent_entry.py` | `814a31327180e9a2d0f2e3068042db6275e15a88afcad772a5ecd5fbfc6c99dc` |
| `benchmarks/experiments/ampere/a100_shared_previous_entry.py` | `814a31327180e9a2d0f2e3068042db6275e15a88afcad772a5ecd5fbfc6c99dc` |

Measured family context: [GB10](../results/gb10.json), [A100](../results/a100.json).
