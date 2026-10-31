# Gradient precision

Attention backward reconstructs probabilities rather than storing the full attention matrix. When one key dominates, query/key gradients subtract nearly equal quantities. A small reconstruction error can overwhelm the true derivative while forward outputs and loss still look correct. [KohakuFA's precision analysis](https://github.com/KohakuBlueleaf/KohakuFA/blob/efd5b72622ce04fcafbe6ca3e368649d8b7b9527/docs/precision.md) provides the original investigation and derivations.

## Arithmetic choices

For positive scale, write the reconstruction as:

```text
S = Q @ Kᵀ
c = scale · log₂(e)
m = max(S) per row
log_l = log₂(sum(exp₂((S - m) · c)))
P = exp₂((S - m) · c - log_l)
dP = dO @ Vᵀ
dS = P · (dP - sum(P · dP))
dQ = (dS @ K) · scale
dK = (dSᵀ @ Q) · scale
dV = Pᵀ @ dO
```

| Choice | Why it matters |
| --- | --- |
| Store the unscaled maximum and logarithmic normalization separately | Combining them in one FP32 statistic can lose the small normalization term |
| Subtract the maximum before scaling | Scaling large scores first can lose differences between nearby scores |
| Convert unscaled dS for tensor-core products; apply scale after FP32 accumulation | Applying scale before low-precision conversion can underflow small derivatives earlier |
| Keep the centering base and FP32 residual correction separate | Subtracting nearly equal quantities in one rounded value can erase the remaining gradient |

The centered backward uses the output/dO dot product as a base, then recomputes the residual softmax correction with FP32 probabilities. It retains the full P-weighted-key correction; the rounded output is not the final softmax correction. These choices apply across the supported GPU architectures.

## Forward normalization rounding

Storing the maximum and logarithmic normalization separately avoids one source of cancellation, but does not make the running denominator exact. On RTX 5090 FP16 D256, the 16-key-column forward sweep accumulated enough denominator error to bias a directed key gradient by 2.7%. Capturing FP32 dS and multiplying it by Q in FP64 reproduced that bias, establishing that it preceded the low-precision operand conversion.

The qualified forward owner retains one FP32 error carry per row for the running denominator. It compensates both addition of each new key tile and rescaling when the maximum changes, then combines the carry at the denominator epilogue. Explicit round-to-nearest add, subtract, multiply and FMA instructions preserve those rounding boundaries. Maximum/logsum storage and the external workspace stay unchanged.

The controlled normalizer correction plus scaled derivative residual reduces the directed coordinate error to about 0.08%, with the existing 2% gate retained. This policy is scoped to the checked unmasked, equal-head RTX 5090 FP16 D256 path. The short forward owner already reduces its complete key tile in one step. Other devices and paths retain their qualified normalization policy. The [5090 analysis](rtx5090-outliers.md) records the independent controls, resource costs and complete-call timings.

## Low components of backward weights

Tensor-core products consume input-dtype operands. Rounding reconstructed FP32 probabilities or derivatives once can erase a small gradient after cancellation across query rows. This affects square calls as well as rectangles, and can change a coordinate's sign even when the complete tensor passes a norm-relative gate.

The qualified RTX 3090 Ti, L40S and RTX 5090 paths retain two components of these weights. For `W = dS` in the key-gradient product, or `W = P` in the value-gradient product:

```text
W_hi = round_input(W)
W_lo = round_input((W - FP32(W_hi)) * residual_scale)
G_hi = accumulate_FP32(W_hiᵀ @ X)
G_lo = accumulate_FP32(W_loᵀ @ X)
G = round_input(G_hi + G_lo / residual_scale)
```

Here `X` is Q for dK or dO for dV; dK also applies the attention scale after accumulation. The two FP32 accumulators remain separate until the epilogue. Adding each low product into the larger running sum loses part of the correction through FP32 rounding. A scalar correction multiplied by one query row is also insufficient: that row may be unrelated or invisible to the key.

BF16 residuals use scale one. Qualified FP16 probability residuals use **2²⁴**: a probability can be smaller than FP16's representable range while its product with a large finite dO is representable. Scaling the residual preserves that product without another shared plane or accumulator. Since P is bounded by one, the scaled half-rounding residual stays in range. Qualified generic D16/D64/D128 and full-width D512 FP16 derivative residuals use **2¹⁰**. RTX 5090 D256 uses that scale together with compensated forward normalization. RTX 3090 Ti and L40S D256 retain their separately qualified unscaled derivative residual; scaling alone did not pass the retained RTX 3090 Ti coordinate check.

The policy runs after device scheduling and retains query ownership, centering preparation, mask and runtime-length guards. Generic D16/D64/D128 owners retain both components; the qualified unmasked wide owners use separate geometry to fit the extra products and accumulators within the 99-KiB block limit. Grouped or masked calls outside a wide owner's qualification keep their existing implementation. These changes do not establish equivalent qualification for other GPUs.

The deterministic [key-gradient regressions](../tests/test_key_precision.py) cover D16/64/128/256/512. The [value-gradient regressions](../tests/test_value_precision.py) include short square embeddings, rectangles and a probability-underflow case. They compare output, dQ, dK and dV independently with centered FP64 on identical rounded inputs. The cancellation coordinates require the correct sign and at most 2% relative error with no absolute allowance; the FP16 probability-underflow coordinate requires at most 0.5%.

For example, the 1,023-query/17-key D128 value fixture has an FP64 coordinate near **−0.01824**. A single rounded probability operand gives **+0.37109 in BF16** or **+0.008705 in FP16**, despite passing the whole-tensor norm gate. Separate high/low accumulation gives approximately **−0.01819** and **−0.018234**. A different FP16 fixture has an FP64 dV coordinate near **+0.67997**; unscaled half probabilities and residuals lose it entirely, while the scaled residual returns **+0.67969**. Default SDPA also loses that coordinate on the tested operands.

Extra tensor-core products have a real cost. The device analyses and retained four-backend comparisons separate the preceding released computation, the preceding schedule with corrected weights, the selected schedule with corrected weights, and SDPA. A faster computation that fails a precision check remains a diagnostic result.

## Remaining numerical limits

The high/low split preserves an FP32 weight more faithfully; it cannot recover information already lost while forming that weight. A retained BF16 large-gradient diagnostic produces two distinct FP64 probabilities that round to the same FP32 value. Both the high and low products then lose their difference, and the gradient norm gate fails. This is a limitation of the current FP32 probability reconstruction, rather than a hardware performance ceiling or an unavoidable output-dtype limit.

Final output rounding is a separate constraint. In additional FP16 diagnostics with dO scaled by `1e-5`, direct FP64-to-FP16 rounding alone exceeds the unchanged 0.5% gradient norm gate for some components. Some selected paths also retain avoidable loss beyond that floor. These are recorded failures. The [5090 analysis](rtx5090-outliers.md#numerical-limits) separates component errors, output-rounding floors and lost representable values; ordinary grid qualification does not imply those diagnostics pass.

## Check gradients, including their magnitude

The saturated BF16 regression has an FP64 query derivative of `1.92874984796e-21`; CuTe returns `1.93229391092e-21`. A separate 100-digit decimal finite difference checks the reference. See the [GPU regressions](../tests/test_kernels.py) and [reference tests](../tests/test_reference.py).

Finite gradients alone are insufficient: zero is finite too. Compare output and Q/K/V gradients on **identical rounded inputs** against a stable FP64 reference, recording relative error, absolute error, reference norm, norm ratio, and cosine. The [precision benchmark](../benchmarks/precision.py) reports those metrics.

This is still low-precision attention. Extremely small derivatives can fall below FP16/BF16's representable range; correctly rounded zero then has relative error 1 against a nonzero FP64 value. At extreme logits, FP32 score rounding can still affect accuracy. The kernels do not promise exact FP64 gradients.

## Optional key centering

`attention(..., center_keys=True)` subtracts the mean key per batch and K/V head in FP32, then casts once back to the input dtype. In exact arithmetic, this shifts every visible logit in a query row equally and leaves softmax unchanged, including with causal attention, boolean masks, and GQA.

In finite precision it can reduce score and dQ errors from a shared key component. Autograd differentiates through the mean subtraction, including its correction to dK. It adds a reduction and temporary key storage and defaults to `False`. This implements [KohakuFA's key-centering addition](https://github.com/KohakuBlueleaf/KohakuFA/commit/043d4afe873f5c254dad6d701f7ca70aeb0692be) alongside the always-enabled centered backward.
