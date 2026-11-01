# RTX 5090 configured training schedule

The pinned optional installation selects a warp-MMA path for contiguous FP16/BF16 `[1,16,4096,64]` and `[1,16,4096,128]` Q/K/V on a full 170-SM RTX 5090, for dense or causal training. All three inputs must require gradients and gradient recording must be enabled. It requires equal Q/K/V shapes, no mask or dropout, and a positive normal FP32 attention scale. Other calls retain their existing dispatch, including inference, compiled calls, shorter sequences, views, and other devices.

Use the [pinned optional installation](a100-sm80-fa4.md#installation-and-selection). Measurements use Python 3.13, Torch 2.15.0+cu132, CuTe DSL 4.8.1, TVM FFI 0.1.15.post0 and Quack 0.6.5, on a full device with a 600 W power limit. The dependency gate retains this qualified stack to compare matching compilation and arithmetic policies. Standalone FA2 2.8.4 was built for SM120 from revision `4616f5116d33e3ba3cbb71d582caffcfcb49649c`; its loaded extension hash is retained in the results.

The schedule uses shared V, M64/N64 backward tiles, two query stages, one output-gradient stage, 256 threads and 4/2/2 MMA warp layouts. Forward uses M128/N64 for D64 and M64/N64 for D128. The configured C++ host supplies fresh descriptors and streams. The [default arithmetic contract](precision.md), including operand conversions, normalization statistics and gradient-scale placement, is unchanged.

Causal calls interleave key tiles across groups of 16 heads. Later key tiles have fewer visible query tiles; remapping their assignment spreads that unequal work across heads. The logical tiles, products, tail predicates and per-CTA synchronization remain the same. Dense calls retain group one. This uses the existing [causal head-group control](causal-head-groups.md), qualified separately on SM120.

## Complete-call comparison

B1/H16/S4096; median complete warmed eager forward plus all three gradients, in milliseconds:

| D | Dtype | Visibility | Configured G1 control | Selected public CuTe | FA2 | Selected SDPA |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 64 | BF16 | dense | 1.341 | 1.343 | 1.405 | 1.447 |
| 64 | BF16 | causal | 0.751 | 0.686 | 0.798 | 0.803 |
| 64 | FP16 | dense | 1.373 | 1.371 | 1.421 | 1.463 |
| 64 | FP16 | causal | 0.759 | 0.701 | 0.809 | 0.811 |
| 128 | BF16 | dense | 2.500 | 2.492 | 2.612 | 2.662 |
| 128 | BF16 | causal | 1.364 | 1.223 | 1.401 | 1.442 |
| 128 | FP16 | dense | 2.548 | 2.536 | 2.640 | 2.685 |
| 128 | FP16 | causal | 1.385 | 1.244 | 1.416 | 1.451 |

Across the four causal calls, the actual public path is 1.08–1.12× the G1 control for combined forward/backward and 1.15–1.18× for backward. Combined causal calls are 1.14–1.16× standalone FA2 and 1.16–1.18× default SDPA. Dense scheduling remains G1. The G1 control uses the same configured C++ entry directly, excluding public selection overhead. FA2 and SDPA retain their own arithmetic policies; their results compare complete implementations.

[Current compact results](../results/rtx5090-causal-g16.json) retain all 32 backward/combined eager/graph rows, seven samples per backend, source hashes, runtime versions and eight selected-code bindings. Each mode uses a one-second sustained warmup, rotates backend order and measures approximately 100 ms windows after compilation and allocation. Backward reuses a retained forward; combined includes forward and all gradients. Profiling and sanitizer runs are separate from timing. The [initial Q2 host comparison](../results/rtx5090-configured-host.json) retains the preceding generic-entry comparison and S1024 calibration. S1024 captured device time improved, but complete eager selection was inconsistent, so its automatic path remains unchanged.

The BF16 causal main kernels use 160 registers and 58,368 shared bytes for D64, and 218 registers and 99,328 shared bytes for D128, with zero local allocation. G1 uses 165 and 216 registers respectively. These are compiled resources, not measured occupancy or stall percentages. Exact captured graph symbols were matched to loaded cubins, compiler-returned ELF for CuTe, and selected PTX/SASS for both CuTe and FA2. CuTe PTX declares ISA 9.4 and `sm_120a`; the numeric CUDA function architecture attributes are recorded separately.

Qualified resident-V, larger-tile and K-fragment controls remain experimental. The transferred Ada M32/N128 preset was slower here. A D64 Q2/shared-V G1 K-residency control reduced steady-loop LDSM sites from 60 to 52 at equal 80 HMMA sites, but increased non-padding instruction sites from 528 to 634 and registers from 165 to 178. Long backward latency increased by 2.7–5.5%. These static counts exclude known always-false predicate sites and do not establish runtime stall fractions.

These are synthetic aligned MHA inputs, not a complete model or serving benchmark. Separate source-derived BSHD and fused-QKV view experiments also passed correctness and sanitizer checks; this selection does not enable their experimental strided constructor.

## Qualification

G2/G4/G16 controls passed 96 strong fixtures plus 48 batch/head/partial-tile cases per group, changed/poisoned graphs, and all four sanitizer tools. The actual public G16 path then passed its matrix/frontdoor, changed-graph and selection/fallback methods, all four tools, 32 timing rows and six exact selected-code bindings.

After integrating onto current main, three public GPU methods, independent output/dQ/dK/dV and literal-statistic checks, CPU/GPU reference agreement, bounded partial-tile runs under memcheck/racecheck/synccheck/initcheck, the 32-row comparison and two new public-code bindings passed again. The existing long public sanitizer passes were retained for the unchanged backward implementation. Reference equations run on exact CPU copies under sanitizers; GPU kernels, graph replay and poisoned storage remain instrumented. The per-head mathematical norm gates remain 1% for BF16 and 0.5% for FP16, with the existing rounded directed checks. CPU discovery ran 342 tests total with 177 GPU/runtime skips and no failures.

```sh
PYTHONPATH=src python3 -m unittest tests.test_sm120_public_host -v
PYTHONPATH=src python3 -m benchmarks.compare_attention_variants \
  --operations '[{"label":"public","backend":"cute"},{"label":"g1","factory":"benchmarks.sm120_group_controls:make_attention","kwargs":{"implementation":"configured-g1"}},{"label":"g16","factory":"benchmarks.sm120_group_controls:make_attention","kwargs":{"implementation":"configured-g16"}},{"label":"fa2","backend":"fa2"},{"label":"sdpa","backend":"sdpa"}]' \
  --seq 4096 --dim 64 128 --heads 16 --dtype bf16 fp16 \
  --visibility dense causal --phases backward joint --repeats 7 \
  --warmup-ms 1000 --window-ms 100 --output rtx5090-groups.json
```

Full raw machine captures, source snapshots, failed harnesses and exploratory variants remain outside the tracked tree.
