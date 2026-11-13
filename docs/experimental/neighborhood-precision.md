# Experimental neighborhood precision

Historical method labels below describe the recorded campaign. The combined dQ+dK method with column-staged statistics is now the public neighborhood default; see [the current contract](../precision.md#neighborhood-default).

The default arithmetic is unchanged. These explicit experiments compare stronger backward equations on identical rounded FP16/BF16 operands. They measure the neighborhood operator; no HDiT model or task quality was evaluated.

The combined method improves measured dQ accuracy, but dK remains a material limitation: the sharp negative-scale fixture has 98.12% BF16 and 9.81% FP16 dK relative norm error against mathematical FP64. Passing the prescribed rounded equations does not establish near-FP64 accuracy for every gradient.

Measured on NVIDIA H100 80GB HBM3 (132 SMs), PyTorch 2.13.0+cu130, CUDA 13.0, CUTLASS DSL 4.8.1, NATTEN 0.21.7+torch2130cu130. The qualified stack is retained for matched comparisons. Exact source/library metadata, all per-head component errors, sample medians/ranges and negative results are in [the compact JSON](neighborhood-precision.json).

At qualification time, [PyTorch stable documentation](https://docs.pytorch.org/docs/stable/) resolves to 2.15. The [official NATTEN installation matrix](https://natten.org/install/) supplies 0.21.7 prebuilt libnatten wheels for PyTorch 2.13 and 2.12, including the retained 2.13/CUDA 13.0 pairing. This campaign keeps that already-qualified matching binary stack for same-process comparisons rather than changing framework and compiler policy during the experiment.

FP64 means native `torch.float64` on the same already-quantized inputs. Small fixtures measure complete tensors. Large timing cases probe recorded boundary/interior query and inverse-key owners over all heads and channels; those errors are sampled, rather than complete large-tensor norms. NATTEN and FlexAttention are independently assessed implementations.

The retained generated CuTe PTX identifies CUDA compiler 13.4.59 and PTX ISA 9.4. This loaded-code evidence is distinct from PyTorch's CUDA 13.0 runtime/build tag. The compiler metadata is retained in JSON.

## Arithmetic

`fp32_p` keeps reconstructed P in FP32 through dS formation, then rounds dS once to the input dtype. dV still uses input-rounded P. `weighted_dq` uses the default dS and subtracts `sum(dS_low) * (P_low @ K) / sum(P_low)` from the unscaled FP32 dQ accumulation. `fp32_p_weighted_dq` combines both. The scale is applied after accumulation; delta comes from the actual rounded forward output in every GPU variant. Forward arithmetic is shared.

The normalized weighted-key correction removes the spurious common-key term caused when rounded dS no longer sums to zero. This is a stronger numerical method, with an extra query-gradient product and reductions; it changes the default rounded equations. It does not recover discarded input bits.

## Large common-key offset

Worst-head relative L2 errors against mathematical FP64, in percent. Each 5×9 fixture uses a 3×3 shifted neighborhood and common Q/K component 1024, with exactly representable FP32 score differences. The JSON retains every head, signed-scale fixture and component.

| dtype | implementation | O | dQ | dK | dV |
| --- | --- | ---: | ---: | ---: | ---: |
| float16 | baseline | 0.02568% | 25.67980% | 0.04016% | 0.03098% |
| float16 | fp32_p | 0.02568% | 20.84322% | 0.03594% | 0.03098% |
| float16 | weighted_dq | 0.02568% | 0.03999% | 0.04016% | 0.03098% |
| float16 | fp32_p_weighted_dq | 0.02568% | 0.03140% | 0.03594% | 0.03098% |
| float16 | flex | 0.30015% | 21.42498% | 0.48652% | 0.47834% |
| float16 | natten | 0.03025% | 27.61043% | 0.56041% | 0.56659% |
| bfloat16 | baseline | 0.19563% | 194.09470% | 0.36442% | 0.23818% |
| bfloat16 | fp32_p | 0.19563% | 176.75346% | 0.25541% | 0.23818% |
| bfloat16 | weighted_dq | 0.19563% | 0.34000% | 0.36442% | 0.23818% |
| bfloat16 | fp32_p_weighted_dq | 0.19563% | 0.25989% | 0.25541% | 0.23818% |
| bfloat16 | flex | 0.31345% | 163.33017% | 0.52967% | 0.51539% |
| bfloat16 | natten | 0.28393% | 213.48852% | 0.60490% | 0.65719% |

The correction targets a real cancellation error: multiplying a nonzero rounded dS row sum by a large common K component. Keeping P in FP32 alone leaves dS rounding and rounded-output delta error. The improvement is specific measured evidence; it does not establish universal pointwise accuracy or a task-level winner.

## Remaining sharp-attention dK error

The combined dQ correction does not repair every gradient. On the complete small sharp-attention fixtures, dK still differs substantially from mathematical FP64, despite passing the unchanged prescribed-equation gates. dK retains delta from the actual rounded O and input-rounded dS. The measurements identify this remaining limitation; they do not independently establish its complete numerical cause.

Each row below selects the combined variant's worst dK head and compares that same head in the other implementations. Reference L2 and maximum absolute discrepancy provide scale for cancellation-sensitive errors; a small reference norm does not turn a large relative discrepancy into an accuracy pass. In particular, BF16 negative-scale head 1 has reference L2 0.175785 and maximum absolute error 0.061478, alongside 98.12% relative error. FlexAttention has a comparable error on that fixture, which does not make either implementation a mathematical oracle.

| dtype | scale | head | FP64 dK L2 | baseline dK | combined dK | Flex dK | NATTEN dK | combined max absolute error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| float16 | 0.125 | 0 | 22.073502 | 0.20570% | 0.20570% | 0.20564% | 0.89428% | 0.015680 |
| float16 | -0.125 | 1 | 0.908967 | 9.85962% | 9.81425% | 9.81474% | nonfinite | 0.025569 |
| bfloat16 | 0.125 | 0 | 10.713372 | 2.17412% | 2.14846% | 2.14847% | 11.47049% | 0.082758 |
| bfloat16 | -0.125 | 1 | 0.175785 | 98.12111% | 98.12044% | 98.12048% | nonfinite | 0.061478 |

## Remaining dQ error

These CPU counterfactuals separate operand boundaries using FP64 final products. Ideal delta is derived in FP64 and then rounded to FP32 before dS. Omitting the dS cast also omits the final low-precision output cast in that diagnostic. The final-cast floor rounds mathematical FP64 dQ directly to the output dtype.

| dtype | default operands | FP32 P | ideal delta + FP32 P | additionally omit dS cast | ideal final-cast floor |
| --- | ---: | ---: | ---: | ---: | ---: |
| float16 | 25.67975% | 20.84321% | 13.52236% | 0.01090% | 0.02567% |
| bfloat16 | 194.09730% | 176.73049% | 94.54408% | 0.00455% | 0.19621% |

The corrected kernel's remaining error must be compared with its dtype's final-cast floor per head. Further improvements can require another rounding policy or residual products and must be evaluated with their complete-call cost. The counterfactuals are diagnostic equations, rather than measured GPU variants.

## Complete forward plus backward cost

Resident CUDA graph device-event medians in microseconds: 7 alternating timing rounds, 8 complete calls per replay. Compilation, allocation and immutable geometry preparation are outside the resident interval; required per-call layout copies and all forward/backward kernels are inside. Preparation costs remain in JSON. Changed-input graph checks precede timing. Profiling is separate from uninstrumented samples.

| case | baseline | FP32 P | weighted dQ | combined | FlexAttention | NATTEN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64sq-w7-fp16 | 118.13 | 117.30 | 135.88 | 132.88 | 123.18 | 254.78 |
| 64sq-w13-fp16 | 205.10 | 203.78 | 237.37 | 232.09 | 172.11 | 244.80 |
| tail-33x49-w7-fp16 | 79.60 | 78.06 | 82.07 | 80.26 | 80.46 | 177.03 |
| 64sq-w7-bf16 | 117.86 | 116.34 | 132.27 | 132.32 | 122.98 | 255.33 |
| 64sq-w13-bf16 | 207.35 | 204.68 | 231.03 | 231.87 | 173.01 | 254.84 |
| tail-33x49-w7-bf16 | 78.95 | 76.89 | 80.64 | 80.77 | 79.95 | 177.10 |

Sampled mathematical FP64 errors for the ordinary timing inputs:

| case | implementation | O | dQ | dK | dV |
| --- | --- | ---: | ---: | ---: | ---: |
| 64sq-w7-fp16 | baseline | 0.02813% | 0.04264% | 0.03970% | 0.03215% |
| 64sq-w7-fp16 | fp32_p | 0.02813% | 0.03413% | 0.03518% | 0.03215% |
| 64sq-w7-fp16 | weighted_dq | 0.02813% | 0.03842% | 0.03970% | 0.03215% |
| 64sq-w7-fp16 | fp32_p_weighted_dq | 0.02813% | 0.02934% | 0.03518% | 0.03215% |
| 64sq-w7-fp16 | flex | 0.02818% | 0.03413% | 0.03518% | 0.03215% |
| 64sq-w7-fp16 | natten | 0.02865% | 0.03436% | 0.03384% | 0.03214% |
| 64sq-w13-fp16 | baseline | 0.02790% | 0.03750% | 0.04292% | 0.03044% |
| 64sq-w13-fp16 | fp32_p | 0.02790% | 0.03279% | 0.03312% | 0.03044% |
| 64sq-w13-fp16 | weighted_dq | 0.02790% | 0.03696% | 0.04292% | 0.03044% |
| 64sq-w13-fp16 | fp32_p_weighted_dq | 0.02790% | 0.02994% | 0.03312% | 0.03044% |
| 64sq-w13-fp16 | flex | 0.02776% | 0.03263% | 0.03303% | 0.03044% |
| 64sq-w13-fp16 | natten | 0.02923% | 0.03217% | 0.03362% | 0.03044% |
| tail-33x49-w7-fp16 | baseline | 0.02670% | 0.03776% | 0.04246% | 0.03086% |
| tail-33x49-w7-fp16 | fp32_p | 0.02670% | 0.03404% | 0.03144% | 0.03086% |
| tail-33x49-w7-fp16 | weighted_dq | 0.02670% | 0.03505% | 0.04246% | 0.03086% |
| tail-33x49-w7-fp16 | fp32_p_weighted_dq | 0.02670% | 0.03232% | 0.03144% | 0.03086% |
| tail-33x49-w7-fp16 | flex | 0.02682% | 0.03386% | 0.03138% | 0.03086% |
| tail-33x49-w7-fp16 | natten | 0.02785% | 0.03960% | 0.03212% | 0.03090% |
| 64sq-w7-bf16 | baseline | 0.22016% | 0.31096% | 0.30788% | 0.25969% |
| 64sq-w7-bf16 | fp32_p | 0.22016% | 0.26032% | 0.24387% | 0.25969% |
| 64sq-w7-bf16 | weighted_dq | 0.22016% | 0.28718% | 0.30788% | 0.25969% |
| 64sq-w7-bf16 | fp32_p_weighted_dq | 0.22016% | 0.22936% | 0.24387% | 0.25969% |
| 64sq-w7-bf16 | flex | 0.22106% | 0.25946% | 0.24367% | 0.25969% |
| 64sq-w7-bf16 | natten | 0.22759% | 0.43345% | 0.24890% | 0.25969% |
| 64sq-w13-bf16 | baseline | 0.22649% | 0.32736% | 0.33184% | 0.26866% |
| 64sq-w13-bf16 | fp32_p | 0.22649% | 0.25178% | 0.25489% | 0.26866% |
| 64sq-w13-bf16 | weighted_dq | 0.22649% | 0.31864% | 0.33184% | 0.26866% |
| 64sq-w13-bf16 | fp32_p_weighted_dq | 0.22649% | 0.23999% | 0.25489% | 0.26866% |
| 64sq-w13-bf16 | flex | 0.22524% | 0.25191% | 0.25391% | 0.26832% |
| 64sq-w13-bf16 | natten | 0.22228% | 0.27908% | 0.25028% | 0.26866% |
| tail-33x49-w7-bf16 | baseline | 0.21157% | 0.32172% | 0.31637% | 0.25075% |
| tail-33x49-w7-bf16 | fp32_p | 0.21157% | 0.25804% | 0.26772% | 0.25075% |
| tail-33x49-w7-bf16 | weighted_dq | 0.21157% | 0.29676% | 0.31637% | 0.25075% |
| tail-33x49-w7-bf16 | fp32_p_weighted_dq | 0.21157% | 0.24721% | 0.26772% | 0.25075% |
| tail-33x49-w7-bf16 | flex | 0.21331% | 0.25676% | 0.26362% | 0.25182% |
| tail-33x49-w7-bf16 | natten | 0.22003% | 0.30069% | 0.25192% | 0.25075% |

## Qualification and limitations

The experiment retains independent prescribed-equation gates at 0.5% FP16 and 1% BF16 per head and component, plus ordinary mathematical FP64 gates. These are norm criteria and do not require universal bitwise agreement across FP32 reduction orders. Formula errors, mathematical errors and final-cast floors remain separate. Paired unchanged-component checks and changed-input graph checks are retained in JSON.

There are 4 retained non-success rows across the directed and timing reports, plus 8 completed comparator diagnostics with nonfinite results. Their classifications are:

- 3: NATTEN public kernel_size API requires both window dimensions >= 2.
- 1: numerical qualification failure; unchanged acceptance criteria.
- 8: Nonfinite comparator output/gradients on a negative-scale compatibility fixture.

The float16 `fp32_p` variant fails its unchanged formula gate on `large-common-offset-negative` (dq, head 1): relative norm error 3.60726%, maximum absolute discrepancy 0.125. This failure remains recorded; it is not relabelled as a passing gate.

A separate immutable retained-operand diagnosis reproduces all four original outputs bytewise. CPU and CUDA IEEE FP32 references agree on the failed dQ gate. Replacing their dP with native FP16-input products with FP32 output reproduces both heads' final dQ bitwise; independently reconstructed probability and actual saved normalization statistics give the same result. This identifies an FP32 reduction-order difference amplified by the declared dS cast, with no acceptance threshold changed.

At query 7/key 15 in head 1, pre-cast dS is −1.4194334745407104 with the IEEE product and −1.4194337129592896 with the FP16-input product. Those values lie one FP32 ULP on either side of the FP16 midpoint −1.41943359375. Rounding selects −1.4189453125 versus −1.419921875: one FP16 step, which the common key and signed scale turn into the observed dQ discrepancy of exactly 0.125. The compact JSON retains the original failed gate, diagnostic source/operand/report hashes, reconstruction metrics and actual successful diagnostic exit receipt.

Qualification receipts:

| stage | result | scope |
| --- | --- | --- |
| cpu | pass | 517 tests, 223 skipped |
| experimental_gpu | pass | 2 methods, 5 variants, 2 dtypes |
| default_neighborhood_gpu | pass | 10 methods |
| memcheck | pass |  |
| racecheck | pass |  |
| initcheck | pass |  |
| synccheck | pass |  |

Execution errors and unsupported calls have no successful timing. Worst-head comparisons can select different heads; all matched per-head records are retained. FP64 is finite precision, initial input quantization is excluded, and measurements on this H100 do not establish behavior on another GPU. The original default and its acceptance criteria remain unchanged.

Reproduce the compact artifacts from retained completed reports:

```sh
python3.13 -m benchmarks.summarize_neighborhood_precision \
  --directed RESULTS/directed/report.json --timing RESULTS/timing/report.json \
  --output docs/experimental/neighborhood-precision.json \
  --markdown docs/experimental/neighborhood-precision.md
```

Use `--forensic`, `--forensic-receipt`, `--compiler-metadata` and `--qualification` with the retained compact metadata to include the supplementary evidence.
