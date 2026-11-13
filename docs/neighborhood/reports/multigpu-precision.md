# Neighborhood precision across GPUs

Historical method labels below describe the recorded campaign. The combined dQ+dK method with column-staged statistics is now the public neighborhood default; see [the current contract](../../precision.md#neighborhood-default).

This report covers the explicit combined method on five devices, following its separate H100 qualification. Start with the [current guide](../README.md) for the default-policy distinction and usage. [Machine-readable evidence](../../../results/neighborhood/multigpu-precision.json) retains all samples and component errors.

The explicit dQ+dK experiment uses the same arithmetic and retained rounded inputs on every architecture. Default dispatch is unchanged.

The [H100 dQ+dK study](dk-precision.md) specifies the experiment and its explicit selection controls.

The mathematical reference uses native FP64 on identical rounded Q/K/V/dO. The independent rounded-arithmetic reference uses the actual rounded forward output; it does not emulate tensor-core reduction order.

The 21 historical fixtures gate output, dQ, dK and dV independently. The eight zero/constant-dP controls retain their declared output/dK/dV mathematical gates; dQ formula agreement and observed FP64 error remain reported. Limits are 0.5% for FP16 and 1% for BF16 per head and component.

Measurements apply to the actual device configurations and recorded power limits below.

| GPU | Compute capability | SMs | VRAM (GiB) | Driver | Power limit (W) |
| --- | --- | ---: | ---: | --- | ---: |
| NVIDIA B200 | 10.0 | 148 | 178.4 | 580.126.09 | 1000 |
| NVIDIA B300 SXM6 AC | 10.3 | 148 | 267.7 | 620.70.06 | 1100 |
| NVIDIA GeForce RTX 4090 | 8.9 | 128 | 23.5 | 595.99.02 | 350 |
| NVIDIA GeForce RTX 5090 | 12.0 | 170 | 31.4 | 595.71.05 | 400 |
| NVIDIA L40S | 8.9 | 142 | 44.4 | 595.71.05 | 350 |

| GPU | Saved-input qualification | GPU methods | Sanitizers | Complete qualification |
| --- | --- | --- | --- | --- |
| b200 | pass | 15 (pass; skipped=0) | 4/4 | pass |
| b300 | pass | 15 (pass; skipped=0) | 4/4 | pass |
| rtx4090 | pass | 15 (pass; skipped=0) | 4/4 | pass |
| rtx5090 | pass | 15 (pass; skipped=0) | 4/4 | pass |
| l40s | pass | 15 (pass; skipped=0) | 4/4 | pass |

Worst per-head relative norm errors on the 21 historical fixtures:

| GPU | Dtype | Output | dQ | dK | dV |
| --- | --- | ---: | ---: | ---: | ---: |
| b200 | bfloat16 | 0.20378% | 0.28013% | 0.28530% | 0.25569% |
| b200 | float16 | 0.02624% | 0.04288% | 0.04084% | 0.03098% |
| b300 | bfloat16 | 0.20378% | 0.28013% | 0.28530% | 0.25569% |
| b300 | float16 | 0.02624% | 0.04288% | 0.04084% | 0.03098% |
| rtx4090 | bfloat16 | 0.20378% | 0.28013% | 0.28530% | 0.25569% |
| rtx4090 | float16 | 0.02624% | 0.03556% | 0.04084% | 0.03098% |
| rtx5090 | bfloat16 | 0.20378% | 0.28013% | 0.28530% | 0.25569% |
| rtx5090 | float16 | 0.02624% | 0.04288% | 0.04084% | 0.03098% |
| l40s | bfloat16 | 0.20378% | 0.28013% | 0.28530% | 0.25569% |
| l40s | float16 | 0.02624% | 0.03556% | 0.04084% | 0.03098% |

Secondary resident forward-plus-backward CUDA graph timings follow. They include the complete advertised call, including required backend layout copies. Compilation and immutable geometry preparation are outside this boundary. Missing or failed rows do not establish a speedup.

Baseline, prior dQ, combined dQ+dK, FlexAttention and NATTEN use different precision policies. These are complete-implementation comparisons, rather than isolated scheduling comparisons under identical arithmetic.

| GPU | Workload | Implementation | Status | Median (µs) | Change vs prior dQ |
| --- | --- | --- | --- | ---: | ---: |
| b200 | 64sq-w7-fp16 | baseline | pass | 152.975 | -10.272% |
| b200 | 64sq-w7-fp16 | fp32_p_weighted_dq | pass | 170.488 | +0.000% |
| b200 | 64sq-w7-fp16 | weighted_dq_reconstructed_dk | pass | 139.828 | -17.983% |
| b200 | 64sq-w7-fp16 | flex | pass | 140.585 | -17.540% |
| b200 | 64sq-w7-fp16 | natten | pass | 219.550 | +28.777% |
| b200 | 64sq-w13-fp16 | baseline | pass | 260.323 | -11.513% |
| b200 | 64sq-w13-fp16 | fp32_p_weighted_dq | pass | 294.194 | +0.000% |
| b200 | 64sq-w13-fp16 | weighted_dq_reconstructed_dk | pass | 239.870 | -18.465% |
| b200 | 64sq-w13-fp16 | flex | pass | 199.622 | -32.146% |
| b200 | 64sq-w13-fp16 | natten | pass | 219.229 | -25.481% |
| b200 | tail-33x49-w7-fp16 | baseline | pass | 80.089 | -4.122% |
| b200 | tail-33x49-w7-fp16 | fp32_p_weighted_dq | pass | 83.533 | +0.000% |
| b200 | tail-33x49-w7-fp16 | weighted_dq_reconstructed_dk | pass | 85.434 | +2.276% |
| b200 | tail-33x49-w7-fp16 | flex | pass | 106.061 | +26.970% |
| b200 | tail-33x49-w7-fp16 | natten | pass | 161.687 | +93.561% |
| b200 | 64sq-w7-bf16 | baseline | pass | 148.597 | -8.091% |
| b200 | 64sq-w7-bf16 | fp32_p_weighted_dq | pass | 161.678 | +0.000% |
| b200 | 64sq-w7-bf16 | weighted_dq_reconstructed_dk | pass | 151.285 | -6.428% |
| b200 | 64sq-w7-bf16 | flex | pass | 140.772 | -12.931% |
| b200 | 64sq-w7-bf16 | natten | pass | 207.826 | +28.543% |
| b200 | 64sq-w13-bf16 | baseline | pass | 257.967 | -8.593% |
| b200 | 64sq-w13-bf16 | fp32_p_weighted_dq | pass | 282.216 | +0.000% |
| b200 | 64sq-w13-bf16 | weighted_dq_reconstructed_dk | pass | 263.098 | -6.774% |
| b200 | 64sq-w13-bf16 | flex | pass | 199.248 | -29.399% |
| b200 | 64sq-w13-bf16 | natten | pass | 218.988 | -22.404% |
| b200 | tail-33x49-w7-bf16 | baseline | pass | 84.048 | -4.145% |
| b200 | tail-33x49-w7-bf16 | fp32_p_weighted_dq | pass | 87.683 | +0.000% |
| b200 | tail-33x49-w7-bf16 | weighted_dq_reconstructed_dk | pass | 88.868 | +1.351% |
| b200 | tail-33x49-w7-bf16 | flex | pass | 105.658 | +20.500% |
| b200 | tail-33x49-w7-bf16 | natten | pass | 156.634 | +78.638% |
| b300 | 64sq-w7-fp16 | baseline | pass | 149.540 | -8.900% |
| b300 | 64sq-w7-fp16 | fp32_p_weighted_dq | pass | 164.150 | +0.000% |
| b300 | 64sq-w7-fp16 | weighted_dq_reconstructed_dk | pass | 134.296 | -18.187% |
| b300 | 64sq-w7-fp16 | flex | pass | 141.314 | -13.911% |
| b300 | 64sq-w7-fp16 | natten | pass | 199.709 | +21.663% |
| b300 | 64sq-w13-fp16 | baseline | pass | 258.191 | -9.458% |
| b300 | 64sq-w13-fp16 | fp32_p_weighted_dq | pass | 285.160 | +0.000% |
| b300 | 64sq-w13-fp16 | weighted_dq_reconstructed_dk | pass | 234.631 | -17.720% |
| b300 | 64sq-w13-fp16 | flex | pass | 200.112 | -29.825% |
| b300 | 64sq-w13-fp16 | natten | pass | 199.651 | -29.986% |
| b300 | tail-33x49-w7-fp16 | baseline | pass | 78.994 | -2.841% |
| b300 | tail-33x49-w7-fp16 | fp32_p_weighted_dq | pass | 81.304 | +0.000% |
| b300 | tail-33x49-w7-fp16 | weighted_dq_reconstructed_dk | pass | 83.913 | +3.209% |
| b300 | tail-33x49-w7-fp16 | flex | pass | 100.896 | +24.098% |
| b300 | tail-33x49-w7-fp16 | natten | pass | 148.530 | +82.685% |
| b300 | 64sq-w7-bf16 | baseline | pass | 141.511 | -8.480% |
| b300 | 64sq-w7-bf16 | fp32_p_weighted_dq | pass | 154.623 | +0.000% |
| b300 | 64sq-w7-bf16 | weighted_dq_reconstructed_dk | pass | 140.855 | -8.904% |
| b300 | 64sq-w7-bf16 | flex | pass | 141.462 | -8.512% |
| b300 | 64sq-w7-bf16 | natten | pass | 201.210 | +30.129% |
| b300 | 64sq-w13-bf16 | baseline | pass | 245.464 | -8.955% |
| b300 | 64sq-w13-bf16 | fp32_p_weighted_dq | pass | 269.608 | +0.000% |
| b300 | 64sq-w13-bf16 | weighted_dq_reconstructed_dk | pass | 251.688 | -6.647% |
| b300 | 64sq-w13-bf16 | flex | pass | 198.841 | -26.248% |
| b300 | 64sq-w13-bf16 | natten | pass | 200.294 | -25.709% |
| b300 | tail-33x49-w7-bf16 | baseline | pass | 83.402 | -0.710% |
| b300 | tail-33x49-w7-bf16 | fp32_p_weighted_dq | pass | 83.998 | +0.000% |
| b300 | tail-33x49-w7-bf16 | weighted_dq_reconstructed_dk | pass | 85.529 | +1.822% |
| b300 | tail-33x49-w7-bf16 | flex | pass | 100.926 | +20.152% |
| b300 | tail-33x49-w7-bf16 | natten | pass | 148.964 | +77.342% |
| rtx4090 | 64sq-w7-fp16 | baseline | pass | 149.341 | -7.870% |
| rtx4090 | 64sq-w7-fp16 | fp32_p_weighted_dq | pass | 162.099 | +0.000% |
| rtx4090 | 64sq-w7-fp16 | weighted_dq_reconstructed_dk | pass | 165.591 | +2.154% |
| rtx4090 | 64sq-w7-fp16 | flex | pass | 192.881 | +18.990% |
| rtx4090 | 64sq-w7-fp16 | natten | pass | 343.046 | +111.628% |
| rtx4090 | 64sq-w13-fp16 | baseline | pass | 266.783 | -7.535% |
| rtx4090 | 64sq-w13-fp16 | fp32_p_weighted_dq | pass | 288.525 | +0.000% |
| rtx4090 | 64sq-w13-fp16 | weighted_dq_reconstructed_dk | pass | 295.234 | +2.325% |
| rtx4090 | 64sq-w13-fp16 | flex | pass | 274.126 | -4.990% |
| rtx4090 | 64sq-w13-fp16 | natten | pass | 368.071 | +27.570% |
| rtx4090 | tail-33x49-w7-fp16 | baseline | pass | 88.831 | -2.371% |
| rtx4090 | tail-33x49-w7-fp16 | fp32_p_weighted_dq | pass | 90.988 | +0.000% |
| rtx4090 | tail-33x49-w7-fp16 | weighted_dq_reconstructed_dk | pass | 95.164 | +4.589% |
| rtx4090 | tail-33x49-w7-fp16 | flex | pass | 96.543 | +6.105% |
| rtx4090 | tail-33x49-w7-fp16 | natten | pass | 210.745 | +131.618% |
| rtx4090 | 64sq-w7-bf16 | baseline | pass | 155.982 | -7.934% |
| rtx4090 | 64sq-w7-bf16 | fp32_p_weighted_dq | pass | 169.424 | +0.000% |
| rtx4090 | 64sq-w7-bf16 | weighted_dq_reconstructed_dk | pass | 169.397 | -0.016% |
| rtx4090 | 64sq-w7-bf16 | flex | pass | 189.272 | +11.714% |
| rtx4090 | 64sq-w7-bf16 | natten | pass | 337.866 | +99.420% |
| rtx4090 | 64sq-w13-bf16 | baseline | pass | 281.279 | -8.290% |
| rtx4090 | 64sq-w13-bf16 | fp32_p_weighted_dq | pass | 306.705 | +0.000% |
| rtx4090 | 64sq-w13-bf16 | weighted_dq_reconstructed_dk | pass | 299.976 | -2.194% |
| rtx4090 | 64sq-w13-bf16 | flex | pass | 269.675 | -12.073% |
| rtx4090 | 64sq-w13-bf16 | natten | pass | 367.081 | +19.685% |
| rtx4090 | tail-33x49-w7-bf16 | baseline | pass | 91.217 | -5.805% |
| rtx4090 | tail-33x49-w7-bf16 | fp32_p_weighted_dq | pass | 96.838 | +0.000% |
| rtx4090 | tail-33x49-w7-bf16 | weighted_dq_reconstructed_dk | pass | 96.248 | -0.610% |
| rtx4090 | tail-33x49-w7-bf16 | flex | pass | 93.468 | -3.480% |
| rtx4090 | tail-33x49-w7-bf16 | natten | pass | 209.712 | +116.560% |
| rtx5090 | 64sq-w7-fp16 | baseline | pass | 134.088 | -10.611% |
| rtx5090 | 64sq-w7-fp16 | fp32_p_weighted_dq | pass | 150.005 | +0.000% |
| rtx5090 | 64sq-w7-fp16 | weighted_dq_reconstructed_dk | pass | 146.899 | -2.071% |
| rtx5090 | 64sq-w7-fp16 | flex | pass | 170.761 | +13.837% |
| rtx5090 | 64sq-w7-fp16 | natten | pass | 387.968 | +158.637% |
| rtx5090 | 64sq-w13-fp16 | baseline | pass | 234.917 | -10.891% |
| rtx5090 | 64sq-w13-fp16 | fp32_p_weighted_dq | pass | 263.628 | +0.000% |
| rtx5090 | 64sq-w13-fp16 | weighted_dq_reconstructed_dk | pass | 260.085 | -1.344% |
| rtx5090 | 64sq-w13-fp16 | flex | pass | 240.549 | -8.754% |
| rtx5090 | 64sq-w13-fp16 | natten | pass | 430.594 | +63.334% |
| rtx5090 | tail-33x49-w7-fp16 | baseline | pass | 83.389 | -5.046% |
| rtx5090 | tail-33x49-w7-fp16 | fp32_p_weighted_dq | pass | 87.820 | +0.000% |
| rtx5090 | tail-33x49-w7-fp16 | weighted_dq_reconstructed_dk | pass | 88.140 | +0.364% |
| rtx5090 | tail-33x49-w7-fp16 | flex | pass | 78.731 | -10.350% |
| rtx5090 | tail-33x49-w7-fp16 | natten | pass | 207.294 | +136.043% |
| rtx5090 | 64sq-w7-bf16 | baseline | pass | 128.018 | -7.460% |
| rtx5090 | 64sq-w7-bf16 | fp32_p_weighted_dq | pass | 138.337 | +0.000% |
| rtx5090 | 64sq-w7-bf16 | weighted_dq_reconstructed_dk | pass | 142.718 | +3.167% |
| rtx5090 | 64sq-w7-bf16 | flex | pass | 156.258 | +12.954% |
| rtx5090 | 64sq-w7-bf16 | natten | pass | 376.492 | +172.155% |
| rtx5090 | 64sq-w13-bf16 | baseline | pass | 232.565 | -6.532% |
| rtx5090 | 64sq-w13-bf16 | fp32_p_weighted_dq | pass | 248.817 | +0.000% |
| rtx5090 | 64sq-w13-bf16 | weighted_dq_reconstructed_dk | pass | 256.583 | +3.121% |
| rtx5090 | 64sq-w13-bf16 | flex | pass | 219.104 | -11.942% |
| rtx5090 | 64sq-w13-bf16 | natten | pass | 424.606 | +70.650% |
| rtx5090 | tail-33x49-w7-bf16 | baseline | pass | 86.632 | -1.604% |
| rtx5090 | tail-33x49-w7-bf16 | fp32_p_weighted_dq | pass | 88.045 | +0.000% |
| rtx5090 | tail-33x49-w7-bf16 | weighted_dq_reconstructed_dk | pass | 90.618 | +2.922% |
| rtx5090 | tail-33x49-w7-bf16 | flex | pass | 78.068 | -11.332% |
| rtx5090 | tail-33x49-w7-bf16 | natten | pass | 204.023 | +131.726% |
| l40s | 64sq-w7-fp16 | baseline | pass | 137.008 | -6.040% |
| l40s | 64sq-w7-fp16 | fp32_p_weighted_dq | pass | 145.815 | +0.000% |
| l40s | 64sq-w7-fp16 | weighted_dq_reconstructed_dk | pass | 147.996 | +1.496% |
| l40s | 64sq-w7-fp16 | flex | pass | 161.581 | +10.812% |
| l40s | 64sq-w7-fp16 | natten | pass | 330.012 | +126.322% |
| l40s | 64sq-w13-fp16 | baseline | pass | 245.175 | -6.124% |
| l40s | 64sq-w13-fp16 | fp32_p_weighted_dq | pass | 261.168 | +0.000% |
| l40s | 64sq-w13-fp16 | weighted_dq_reconstructed_dk | pass | 272.646 | +4.395% |
| l40s | 64sq-w13-fp16 | flex | pass | 232.304 | -11.052% |
| l40s | 64sq-w13-fp16 | natten | pass | 355.968 | +36.298% |
| l40s | tail-33x49-w7-fp16 | baseline | pass | 74.094 | -3.710% |
| l40s | tail-33x49-w7-fp16 | fp32_p_weighted_dq | pass | 76.949 | +0.000% |
| l40s | tail-33x49-w7-fp16 | weighted_dq_reconstructed_dk | pass | 80.077 | +4.065% |
| l40s | tail-33x49-w7-fp16 | flex | pass | 90.642 | +17.795% |
| l40s | tail-33x49-w7-fp16 | natten | pass | 174.121 | +126.281% |
| l40s | 64sq-w7-bf16 | baseline | pass | 144.418 | -5.973% |
| l40s | 64sq-w7-bf16 | fp32_p_weighted_dq | pass | 153.592 | +0.000% |
| l40s | 64sq-w7-bf16 | weighted_dq_reconstructed_dk | pass | 150.893 | -1.757% |
| l40s | 64sq-w7-bf16 | flex | pass | 161.950 | +5.442% |
| l40s | 64sq-w7-bf16 | natten | pass | 323.308 | +110.498% |
| l40s | 64sq-w13-bf16 | baseline | pass | 257.413 | -6.961% |
| l40s | 64sq-w13-bf16 | fp32_p_weighted_dq | pass | 276.673 | +0.000% |
| l40s | 64sq-w13-bf16 | weighted_dq_reconstructed_dk | pass | 275.292 | -0.499% |
| l40s | 64sq-w13-bf16 | flex | pass | 234.027 | -15.414% |
| l40s | 64sq-w13-bf16 | natten | pass | 350.134 | +26.551% |
| l40s | tail-33x49-w7-bf16 | baseline | pass | 80.192 | -2.642% |
| l40s | tail-33x49-w7-bf16 | fp32_p_weighted_dq | pass | 82.368 | +0.000% |
| l40s | tail-33x49-w7-bf16 | weighted_dq_reconstructed_dk | pass | 81.688 | -0.826% |
| l40s | tail-33x49-w7-bf16 | flex | pass | 91.147 | +10.658% |
| l40s | tail-33x49-w7-bf16 | natten | pass | 172.499 | +109.425% |

All per-head errors, declared gates, observed control failures, source and input hashes, raw timing samples and compiled-path flags are retained in the companion JSON. Compiler artifacts establish target and selected flags; graph replay and numerical tests supply separate execution evidence. Results qualify these operator workloads and do not establish model-level quality or performance on other shapes.
