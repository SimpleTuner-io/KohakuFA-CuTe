# Test map

Use Python 3.13 and `unittest`. This directory contains unit/regression tests;
`benchmarks/neighborhood/test_*.py` are experiment runners, not this test suite.
The [validation guide](../docs/validation.md) owns numerical acceptance criteria.
The [source map](../docs/internals/source-map.md) identifies implementation owners.

## Choose the execution environment first

| Work | Entry points | Environment |
| --- | --- | --- |
| Source and local-link inspection | `python3 tools/check_repository.py` | Standard-library inspection; does not import the ML runtime |
| Publication guard | `python3 -m unittest tests.test_public_text -v` | Standard library; temporary files and Git fixtures |
| Catalog/report checks | [test_results_catalog.py](test_results_catalog.py), [test_blackwell_init_association.py](test_blackwell_init_association.py), [test_neighborhood_performance_summary.py](test_neighborhood_performance_summary.py) | Saved-data/schema/rendering checks; no GPU computation. Inspect imports before extending a local selection. |
| Framework host/dispatch checks | [test_fa4_sm80.py](test_fa4_sm80.py), [test_native_build.py](test_native_build.py), device predicate tests below | May mock CUDA operands but still import PyTorch/package code. Run on the configured test host. |
| CPU numerical checks | [test_row_metrics.py](test_row_metrics.py), [test_baseline_precision.py](test_baseline_precision.py), reference fixtures | Numerical work belongs on the configured remote host, even with CUDA hidden. |
| GPU contract and backend checks | [test_kernels.py](test_kernels.py) and device suites below | Qualified runtime and actual target GPU. Skip conditions are not qualification results. |

Install the development dependencies on the configured remote host before full
CPU discovery. NumPy is required by the operand-hashing tests through PyTorch's
NumPy bridge:

```sh
python3 -m pip install -e '.[dev]'
CUDA_VISIBLE_DEVICES='' python3 -m unittest discover -v -f
```

This command still imports PyTorch and executes CPU numerical references; it is
not a lightweight laptop inspection command. For GPU changes, run relevant
architecture/contract tests with the target GPU visible, and sanitizer checks
when memory ownership or asynchronous synchronization changes. Do not infer a
GPU pass from CPU discovery skipping the GPU suite.

## Test families

These are navigation groups, not blanket claims about CPU/GPU requirements.
A file can contain both mocked host tests and GPU tests; inspect its imports and
class/method skip conditions before choosing a command.

### Publication and saved evidence

[public_text](test_public_text.py), [results_catalog](test_results_catalog.py).

### Ordinary arithmetic and references

[baseline_precision](test_baseline_precision.py), [determinism](test_determinism.py), [key_precision](test_key_precision.py), [length_precision](test_length_precision.py), [precision_policy](test_precision_policy.py), [value_precision](test_value_precision.py), [weight_precision](test_weight_precision.py).

### Public API, native host and generic kernels

| Concern | Tests |
| --- | --- |
| API and autograd | [api](test_api.py), [center_keys](test_center_keys.py), [training](test_training.py), [warp_inference](test_warp_inference.py) |
| Native build, cache and logical views | [native_build](test_native_build.py), [native_view_flags](test_native_view_flags.py), [configured_host](test_configured_host.py), [selected_validation](test_selected_validation.py) |
| Generic kernels and launch geometry | [kernels](test_kernels.py), [causal_head_groups](test_causal_head_groups.py), [launch_limits](test_launch_limits.py) |
| Independent references and adversarial inputs | [reference](test_reference.py), [adversarial](test_adversarial.py) |

### Ampere and RTX3090

[a100_d256](test_a100_d256.py), [a100_d256_shared](test_a100_d256_shared.py), [a100_group_policy](test_a100_group_policy.py), [a100_public_controls](test_a100_public_controls.py), [a40_native_predicate](test_a40_native_predicate.py), [a40_shared](test_a40_shared.py), [fa4_sm80](test_fa4_sm80.py), [rtx3090](test_rtx3090.py), [sm80_native_host](test_sm80_native_host.py), [sm86](test_sm86.py).

### Ada and RTX4090/L40S

[sm89_forward](test_sm89_forward.py), [ada](test_ada.py), [l40s](test_l40s.py), [l40s_medium](test_l40s_medium.py), [l40s_native_predicate](test_l40s_native_predicate.py), [l40s_short](test_l40s_short.py), [l40s_short_optimized](test_l40s_short_optimized.py), [l40s_static](test_l40s_static.py), [rtx4090_automatic](test_rtx4090_automatic.py), [rtx4090_native_predicate](test_rtx4090_native_predicate.py), [rtx4090_shared](test_rtx4090_shared.py), [rtx4090_short](test_rtx4090_short.py), [rtx4090_short_optimized](test_rtx4090_short_optimized.py).

### Hopper

[hopper_eligibility](test_hopper_eligibility.py), [hopper_mask_partition](test_hopper_mask_partition.py), [hopper_runner](test_hopper_runner.py), [hopper_scheduler](test_hopper_scheduler.py), [hopper_statistics](test_hopper_statistics.py), [sm90_forward](test_sm90_forward.py).

### Blackwell and SM120

[b200_wide](test_b200_wide.py), [blackwell_init_association](test_blackwell_init_association.py), [blackwell_launch_limits](test_blackwell_launch_limits.py), [sm120_d256](test_sm120_d256.py), [sm120_outliers](test_sm120_outliers.py), [sm120_public_host](test_sm120_public_host.py), [tune_sm120](test_tune_sm120.py).

### GB10

[gb10_benchmark](test_gb10_benchmark.py), [gb10_long_inference](test_gb10_long_inference.py), [gb10_shared](test_gb10_shared.py).

### Neighborhood

[neighborhood](test_neighborhood.py), [neighborhood_accuracy](test_neighborhood_accuracy.py), [neighborhood_backends](test_neighborhood_backends.py), [neighborhood_benchmark](test_neighborhood_benchmark.py), [neighborhood_key_precision](test_neighborhood_key_precision.py), [neighborhood_performance_summary](test_neighborhood_performance_summary.py), [neighborhood_precision](test_neighborhood_precision.py), [neighborhood_schedule](test_neighborhood_schedule.py), [neighborhood_tuning](test_neighborhood_tuning.py), [tune_neighborhood](test_tune_neighborhood.py).

[3D neighborhood tests](test_neighborhood_3d.py): independent forward/inverse volume traversal, LTX layout, corrected gradients, graph replay and plane seams (CPU and GPU sections).

### Benchmark adapters and reporting

| Concern | Tests |
| --- | --- |
| Backend adapters and pinned dependencies | [flashattention_backends](test_flashattention_backends.py), [flashattention_revision](test_flashattention_revision.py) |
| Capture and retained compiler artifacts | [benchmark_capture](test_benchmark_capture.py), [compiled_artifacts](test_compiled_artifacts.py) |
| Application and profile reporting | [clef_forward_benchmark](test_clef_forward_benchmark.py), [profile_attention](test_profile_attention.py), [row_metrics](test_row_metrics.py), [tracker_regressions](test_tracker_regressions.py) |

### Standalone memory diagnostics

[tma4d_diagnostic](test_tma4d_diagnostic.py), [tmem_init](test_tmem_init.py), [tmem_init_controls](test_tmem_init_controls.py).

## Shared fixtures and the large generic suite

[attention_reference.py](attention_reference.py) provides the independent
attention comparison; [hopper_runtime.py](hopper_runtime.py) supports Hopper
runtime checks. The benchmark [rounded reference](../benchmarks/reference.py)
uses actual rounded forward output for backward diagnosis. Preserve the directed
[baseline](test_baseline_precision.py), [key](test_key_precision.py) and
[value](test_value_precision.py) arithmetic fixtures when changing organization.

`test_kernels.py` contains several distinct concerns. Search method prefixes and
names for short-query split forward, warm native dispatch/cache, compiler dynamic
shapes, SM121 specialized owners, ordinary numerical edge cases, masks and graph
replay with changed inputs. Its size is a reason to use this map, not to drop
cases or replace the declared rounded computation with a different oracle.
