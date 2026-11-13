# Gradient precision

Ordinary attention targets low-precision attention with FP32 accumulation. It preserves
separate unscaled score maxima and logarithmic normalization, shifts scores before
scaling, and applies the attention scale after gradient accumulation. These choices
address lost normalization information at large logits and premature gradient
underflow from scaling before conversion.

## Arithmetic contract

For positive scale, backward reconstructs probabilities using the forward statistics:

```text
S = Q @ Kᵀ                          # FP32 accumulation
P = exp₂((S - m) · scale · log₂(e) - log_l)
P_low = round_input(P)              # one FP16/BF16 conversion
Delta = sum(FP32(O) · FP32(dO))
dP = dO @ Vᵀ                       # FP32 accumulation
dS_low = round_input(FP32(P_low) · (dP - Delta))
dQ = round_input(accumulate_FP32(dS_low @ K) · scale)
dK = round_input(accumulate_FP32(dS_lowᵀ @ Q) · scale)
dV = round_input(accumulate_FP32(P_lowᵀ @ dO))
```

Negative scales use the corresponding signed-score maximum. Masks, causal
visibility, GQA, partial tiles, and all-hidden rows retain their API semantics.
The output/dO reduction uses the actual rounded forward output. The accumulation
order and forward tiling can still produce ordinary floating-point differences.

P can remain in FP32 storage after conversion to the input dtype and back; that
storage does not restore the discarded probability bits.

The probability and derivative operand conversions are part of the computation.
A probability that rounds to zero contributes zero to dV even when multiplying its
unrounded value by a large dO would produce a representable result. Cancellation
between rounded operands can also change a small coordinate's sign. These effects
do not establish an indexing, reconstruction, or synchronization defect.

## Optional stronger accuracy experiments

For ordinary attention, residual P/dS products, compensated reductions, and
weighted-key dQ corrections remain separate accuracy experiments. They are
excluded from ordinary-attention default dispatch on every GPU. The [experiment record](../benchmarks/experiments/README.md) retains
the previous-policy comparison and precision tradeoffs.

## Neighborhood default

`neighborhood_attention()` uses the qualified normalized dQ correction and
reconstructed dK correction by default on every supported GPU. No benchmark
context is required. This is a stronger numerical contract than ordinary
attention above; the measured complete-call cost is accepted in exchange for
better accuracy on the defined workloads. There is no device-specific fallback
to the former neighborhood arithmetic.

Reconstruct P in FP32 from the separate unscaled extrema and logarithmic
normalization. Let `delta_O = sum(FP32(O) * FP32(dO))`, using actual rounded O,
and `dP = dO @ V.T` with FP32 accumulation. The query pass forms
`dS_q = round_input(P32 * (dP - delta_O))` and subtracts the rounded dS row sum
times the normalized, input-rounded-P weighted key mean from unscaled dQ.
It also accumulates `t = sum_FP32(P32 * (dP - delta_O)) / sum_FP32(P32)`.
The KV pass uses `dS_k = round_input(P32 * ((dP - delta_O) - t))` with two
separate subtractions. Actual-output delta and t occupy distinct FP32 buffers.
Gradient products accumulate in FP32; scaling follows dQ/dK accumulation and
precedes final input-dtype conversion. Forward and input-rounded P for dV are
unchanged. KV statistics use the qualified `column_staged` schedule.

The independent equations are in
`benchmarks/neighborhood_key_precision_reference.py`. The
[H100 study](experimental/neighborhood-key-precision.md) and
[five-device replay](experimental/neighborhood-multigpu-precision.md) retain
the original measurements made while this method was explicitly selected.
Their historical labels and source hashes are preserved. The old `baseline`
policy remains an explicit benchmark control, not the public default.

## Diagnose rounding separately from implementation errors

[FP64 equations](../benchmarks/reference.py) diagnose mathematical accuracy on
identical rounded inputs. `rounded_reference` additionally represents the specified
P/dS conversions and accepts the actual O for the backward delta. Its score/dP
products use IEEE FP32; its final gradient products use FP64 accumulation to
distinguish operand rounding from the separate FP32 accumulation floor. It does
not reproduce a GPU's instruction order.

The retained [key](../tests/test_key_precision.py) and
[value](../tests/test_value_precision.py) cancellation fixtures check the rounded
computation. Their CPU tests retain the original unrounded equations and closed
forms. The [baseline tests](../tests/test_baseline_precision.py) explicitly preserve
examples where rounded and unrounded expectations differ. Random-input checks
still compare independent output/dQ/dK/dV norm errors against FP64.

Check each component, first/last rows, masks, negative maxima, tails, signed and
zero scales, changed graph inputs, and poisoned storage. A finite result alone is
insufficient. Conversely, a zero required by an operand conversion does not need
extra residual products to qualify the baseline. Do not expand tolerances to hide
an implementation defect or impose an unrounded-coordinate gate on this contract.

## Optional key centering

`attention(..., center_keys=True)` subtracts the mean key per batch and KV head in
FP32, then casts once to the input dtype. In exact arithmetic it shifts every
visible logit in a query row equally. It can reduce errors from a shared key
component, including with causal attention, boolean masks, and GQA.

Autograd differentiates through the mean subtraction and its dK correction. The
option adds a reduction and temporary key storage, and defaults to `False`.
