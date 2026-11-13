# Blackwell experiment catalog

See the [shared invocation and status guidance](../README.md#source-catalogs).
Evidence links identify each study family, not a measured result for every helper.

## Scheduling and bounded public paths

B200/SM103/SM120 candidates and memory checks. [B200](../../../results/b200.json), [B300](../../../results/b300.json), [RTX5090](../../../results/rtx5090.json); [negative pipeline controls](../results/rtx5090-pipeline.json).

| Module | Invocation / role | Question or retained purpose |
| --- | --- | --- |
| [b200_schedule_control](b200_schedule_control.py) | Imported helper / snapshot | Explicit previous-schedule control for matched B200 D512 comparisons. |
| [check_b200_wide_memory](check_b200_wide_memory.py) | CLI | Qualify both wide ownership kernels, tails, and changed-input graph replay. |
| [compare_sm120](compare_sm120.py) | CLI | Compare RTX 5090 outlier schedules with their preceding path and SDPA. |
| [experimental_b200_wide](experimental_b200_wide.py) | Imported helper / snapshot | Explicit SM100 experiment for the retained 256-wide Blackwell backward. |
| [sm120_dout_experiments](sm120_dout_experiments.py) | Imported helper / snapshot; `make_attention` | Matched D64 Q2 pipeline controls; D128 retains the qualified single dO stage. |
| [sm120_group_controls](sm120_group_controls.py) | Imported helper / snapshot; `make_attention` | Actual public entry and matched explicit configured controls; no kernel duplication. |
| [tune_sm100](tune_sm100.py) | CLI | Tune the sm_100 warp-MMA attention path on datacentre Blackwell. |
| [tune_sm103](tune_sm103.py) | CLI | Tune B300 attention tiles with precision checks and CUDA-graph timings. |
| [tune_sm120](tune_sm120.py) | CLI | Tune Blackwell attention tiles with precision checks and CUDA-graph timings. |

## Generated code, synchronization and readout

Measurement/instrumentation controls and their plotters. [Retained controls](../results/controls.json). These probes do not by themselves establish complete-call speedups.

| Module | Invocation / role | Question or retained purpose |
| --- | --- | --- |
| [compare_blackwell_codegen](compare_blackwell_codegen.py) | CLI | Compare production Blackwell code generation with a pinned baseline source. |
| [compare_blackwell_native](compare_blackwell_native.py) | CLI | Compare complete native compile/dispatch variants and pinned Gluon kernels. |
| [compare_blackwell_probabilities](compare_blackwell_probabilities.py) | CLI | Compare Blackwell probability tile changes against an explicit baseline source. |
| [dump_blackwell_assembly](dump_blackwell_assembly.py) | CLI | Capture validated CuTe/Gluon PTX, cubins and SASS for one production shape. |
| [experimental_blackwell_register_aux](experimental_blackwell_register_aux.py) | Imported helper / snapshot | Expose the wide backward directly for readout/reference measurements. |
| [instrument_blackwell_barriers](instrument_blackwell_barriers.py) | CLI | Generate a separate Blackwell kernel source with matrix-warp wait timers. |
| [instrument_blackwell_delay](instrument_blackwell_delay.py) | CLI | Generate diagnostic KV role delays in a frozen register-auxiliary source. |
| [instrument_blackwell_roles](instrument_blackwell_roles.py) | CLI | Generate a separate Blackwell source with sampled, shared-buffered role traces. |
| [measure_blackwell_barriers](measure_blackwell_barriers.py) | CLI | Measure separately generated matrix-warp instrumentation on Blackwell. |
| [measure_blackwell_mma_reference](measure_blackwell_mma_reference.py) | CLI | Measure a resident-operand reference for D512 backward matrix throughput. |
| [measure_blackwell_readout](measure_blackwell_readout.py) | CLI | Measure frozen backward prototypes against production and SDPA. |
| [measure_blackwell_roles](measure_blackwell_roles.py) | CLI | Measure sampled Blackwell role timelines with an uninstrumented control. |
| [measure_blackwell_sensitivity](measure_blackwell_sensitivity.py) | CLI | Measure backward sensitivity to delays in diagnostic KV role modules. |
| [plot_blackwell_codegen](plot_blackwell_codegen.py) | CLI | Render paired latency reductions and captured CUDA-graph kernel timelines. |
| [plot_blackwell_probabilities](plot_blackwell_probabilities.py) | CLI | Render paired probability-tile speed changes and actual instrumented CTA intervals. |
| [plot_blackwell_readout](plot_blackwell_readout.py) | CLI | Render the observed Nsight kernel sequence from the readout profiling report. |
| [plot_blackwell_roles](plot_blackwell_roles.py) | CLI | Plot sampled warp-role intervals and export Chrome/Perfetto trace events. |
| [plot_blackwell_sensitivity](plot_blackwell_sensitivity.py) | CLI | Render per-role latency sensitivity relative to each zero-delay control. |
