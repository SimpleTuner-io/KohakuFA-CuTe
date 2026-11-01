# RTX 5090 configured training schedule

The optional pinned installation selects a configured warp-MMA path for contiguous FP16/BF16 `[1,16,4096,64]` and `[1,16,4096,128]` Q/K/V on a full 170-SM RTX 5090, for dense or causal training. All three inputs must require gradients and gradient recording must be enabled. It requires equal Q/K/V shapes, no mask or dropout, and a positive normal FP32 attention scale. Other calls retain native CuTe dispatch, including inference, compiled calls, short sequences, views, and other devices.

Use the [pinned optional installation](a100-sm80-fa4.md#installation-and-selection). The measured environment was Python 3.13.18, Torch 2.15.0+cu132, CuTe DSL 4.8.1, TVM FFI 0.1.15.post0 and Quack 0.6.5. The dependency gate deliberately retains that qualified stack; this change does not establish compatibility with other compiler versions. Standalone FA2 2.8.4 was built for SM120 from revision `4616f5116d33e3ba3cbb71d582caffcfcb49649c`; its loaded extension hash is retained in the results.

The schedule uses shared V, M64/N64 backward tiles, two query stages, one output-gradient stage, 256 threads and 4/2/2 MMA warp layouts. Forward uses M128/N64 for D64 and M64/N64 for D128. The existing configured C++ host supplies fresh descriptors and streams and launches the same GPU methods. It retains the [default arithmetic contract](precision.md); this is scheduling and host preparation work.

The representative main backward kernels use 165 registers and 58,368 shared bytes for D64, and 204 registers and 99,328 shared bytes for D128, with no local allocation. These are compiled resources, not measurements of stalls or achieved occupancy. Resident-V and larger shared-V tile alternatives were qualified separately; their results did not justify replacing this schedule. In particular, the transferred Ada M32/N128 schedule was slower on this SM120 workload.

## Complete-call comparison

B1/H16/S4096; median complete warmed eager forward plus all three gradients, in milliseconds:

| D | Dtype | Visibility | Earlier generic CuTe entry | Selected public CuTe | FA2 | Selected SDPA |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 64 | BF16 | dense | 1.709 | 1.342 | 1.405 | 1.441 |
| 64 | BF16 | causal | 1.014 | 0.748 | 0.805 | 0.806 |
| 64 | FP16 | dense | 1.716 | 1.373 | 1.421 | 1.464 |
| 64 | FP16 | causal | 1.020 | 0.755 | 0.808 | 0.811 |
| 128 | BF16 | dense | 3.317 | 2.493 | 2.611 | 2.664 |
| 128 | BF16 | causal | 1.974 | 1.351 | 1.395 | 1.443 |
| 128 | FP16 | dense | 3.330 | 2.537 | 2.640 | 2.684 |
| 128 | FP16 | causal | 1.942 | 1.367 | 1.406 | 1.451 |

The selected public path is 1.25–1.46× the earlier generic entry, 1.03–1.08× standalone FA2 and 1.06–1.08× default SDPA for these eight combined calls. The earlier generic control calls its C++ entry directly, excluding its previous public selector overhead. The comparison includes an explicit configured-runner control to expose the new public selector cost. Standalone FA2 and SDPA implement their own arithmetic policies; they are complete implementation comparisons, not arithmetic-identical scheduling controls. Graph resources identify the actual SDPA FlashAttention kernels; the backend name alone is not used to infer selection.

[Compact results](../results/rtx5090-configured-host.json) retain seven samples for every backend and all 96 forward, backward and combined eager/graph rows, with source hashes and backend metadata. Each mode uses a one-second sustained warmup, rotates backend order and measures approximately 100 ms windows after compilation and allocation. Forward retains autograd; backward reuses a retained forward; combined includes forward and all gradients. Profiling and sanitizer runs are separate from timing. Short S1024 calibration remains in the data: captured device time improved, while complete eager public selection was inconsistent against the direct generic entry. S1024 therefore retains its existing automatic path.

These are synthetic aligned MHA inputs, not a complete model or serving benchmark. Separate source-derived BSHD and fused-QKV view experiments also passed correctness and sanitizer checks, but this public change does not enable the experimental strided constructor. Consumer reshapes, projection fusion and inference remain separate work.

## Qualification

The actual public path passed three GPU test methods covering the selected timing matrix, directed rounding and negative maxima, public API frontdoors, changed-data and poisoned graph replay, literal saved maxima/log-normalizers, and independent output/dQ/dK/dV checks. The established per-head mathematical norm gates remain 1% for BF16 and 0.5% for FP16; directed fixtures use the existing rounded reference and actual forward output.

Memcheck, racecheck, synccheck and initcheck passed. The last three evaluate unchanged reference equations on exact copied CPU tensors to avoid instrumenting reference GEMMs; CPU/GPU reference agreement was checked outside sanitizers. Kernel execution, graph replay and poisoned storage checks remain on the GPU. A final selector smoke checks S4096 selection and S1024 fallback after narrowing the bound. CPU discovery ran 276 tests total with 160 GPU/runtime skips and no failures.

The reproducible public tests are:

```sh
PYTHONPATH=src python3 -m unittest tests.test_sm120_public_host -v
```

The standard comparison can also exercise the selected public backend:

```sh
PYTHONPATH=src python3 -m benchmarks.compare_flashattention \
  --backends cute fa2 sdpa --seq 4096 --dim 64 128 --heads 16 \
  --dtype bf16 fp16 --modes dense causal --repeats 7 \
  --window-ms 100 --output rtx5090-public.json
```

That command uses the standard harness's timing boundary; the retained table above uses complete synchronized eager windows with a separate graph control. Full raw machine captures and exploratory variants remain outside the tracked tree.
