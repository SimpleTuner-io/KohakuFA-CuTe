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

## Check gradients, including their magnitude

The saturated BF16 regression has an FP64 query derivative of `1.92874984796e-21`; CuTe returns `1.93229391092e-21`. A separate 100-digit decimal finite difference checks the reference. See the [GPU regressions](../tests/test_kernels.py) and [reference tests](../tests/test_reference.py).

Finite gradients alone are insufficient: zero is finite too. Compare output and Q/K/V gradients on **identical rounded inputs** against a stable FP64 reference, recording relative error, absolute error, reference norm, norm ratio, and cosine. The [precision benchmark](../benchmarks/precision.py) reports those metrics.

This is still low-precision attention. Extremely small derivatives can fall below FP16/BF16's representable range; correctly rounded zero then has relative error 1 against a nonzero FP64 value. At extreme logits, FP32 score rounding can still affect accuracy. The kernels do not promise exact FP64 gradients.

## Optional key centering

`attention(..., center_keys=True)` subtracts the mean key per batch and K/V head in FP32, then casts once back to the input dtype. In exact arithmetic, this shifts every visible logit in a query row equally and leaves softmax unchanged, including with causal attention, boolean masks, and GQA.

In finite precision it can reduce score and dQ errors from a shared key component. Autograd differentiates through the mean subtraction, including its correction to dK. It adds a reduction and temporary key storage and defaults to `False`. This implements [KohakuFA's key-centering addition](https://github.com/KohakuBlueleaf/KohakuFA/commit/043d4afe873f5c254dad6d701f7ca70aeb0692be) alongside the always-enabled centered backward.

## Compensated key/value products on L40S

The qualified L40S backward paths retain separate high and low components of probabilities and softmax derivatives before tensor-core products. Each component accumulates in its own FP32 sum, combined in the epilogue. This fixes cancellation cases where a single rounded operand gives a wrong-sign dK or dV despite a small whole-tensor norm error. FP16 residual storage is scaled to avoid losing representable gradients to intermediate underflow.

Automatic activation is restricted to the full L40S device family qualified here and FP16/BF16 head widths 16/64/128/256/512. D256 uses the new owner only without an explicit mask and with equal query/key-value head counts. D512 uses it without an explicit mask, including the separately checked grouped-head path. Other wide configurations retain their preceding fallback. Other GPU families retain their existing arithmetic until separately enabled and qualified.

The [L40S analysis](l40s-outliers.md) records exact FP64 regressions, unchanged component gates, latency costs, scheduling gains and source provenance. Cancellation coordinates have a zero-atol 2% relative gate; the FP16 probability-underflow coordinate retains a 0.5% gate.
