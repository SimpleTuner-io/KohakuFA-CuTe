"""Independent small-neighborhood equations for explicit precision experiments.

These are diagnostic equations, not a tensor-core reduction-order emulator.
The final products use FP64, isolating operand rounding from FP32 accumulation.
No production kernel imports this module.
"""

import torch

from benchmarks.neighborhood.reference import neighborhood_mask
from benchmarks.reference import fp64_reference, metrics


@torch.no_grad()
def precision_reference(inputs, dout, spatial, window, scale, output, *, fp32_p=False, correction="none"):
    """Evaluate the declared P/dS and weighted-key correction boundaries.

    ``weighted`` subtracts sum(dS_low) times P_low@K; ``normalized`` also
    divides that weighted key by sum(P_low). The latter exactly removes a
    constant key component in these final FP64 diagnostic products. Both
    change the default rounded computation, including when P is kept FP32.
    dV always uses P_low; dK always uses dS_low and post-accumulation scale.
    """
    if correction not in ("none", "weighted", "normalized"):
        raise ValueError("Unknown weighted-key correction.")
    q, k, v = [x.detach() for x in inputs]
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("This diagnostic requires equal Q/K/V head counts and shapes.")
    precision = torch.backends.cuda.matmul.fp32_precision
    try:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        with torch.autocast(device_type=q.device.type, enabled=False):
            scores = (q.float() @ k.float().transpose(-1, -2)).double()
            dp = dout.float() @ v.float().transpose(-1, -2)
    finally:
        torch.backends.cuda.matmul.fp32_precision = precision
    visible = neighborhood_mask(spatial, window, device=q.device)
    signed = (scores if scale >= 0 else -scores).masked_fill(~visible, -torch.inf)
    shifted = ((signed - signed.amax(-1, keepdim=True)) * abs(scale)).masked_fill(~visible, -torch.inf)
    weights = torch.exp(shifted)
    probability = weights / weights.sum(-1, keepdim=True)
    rounded_p = probability.float().to(q.dtype).float()
    derivative_p = probability.float() if fp32_p else rounded_p
    delta = (output.float() * dout.float()).sum(-1, keepdim=True)
    ds = (derivative_p * (dp - delta)).to(q.dtype).double()
    dq = ds @ k.double()
    if correction != "none":
        weighted_key = rounded_p.double() @ k.double()
        if correction == "normalized":
            weighted_key /= rounded_p.double().sum(-1, keepdim=True)
        dq -= ds.sum(-1, keepdim=True) * weighted_key
    return (
        probability @ v.double(),
        dq * scale,
        (ds.transpose(-1, -2) @ q.double()) * scale,
        rounded_p.double().transpose(-1, -2) @ dout.double(),
    )


def component_metrics(actual, expected):
    """Keep every head and output/gradient independently visible."""
    result = []
    for component, got, want in zip(("out", "dq", "dk", "dv"), actual, expected):
        got, want = got.detach().cpu(), want.detach().cpu()
        for batch in range(got.shape[0]):
            for head in range(got.shape[1]):
                result.append(
                    dict(component=component, batch=batch, head=head, **metrics(got[batch, head], want[batch, head]))
                )
    return result


def reference_gate(actual, expected, dtype):
    """Existing norm limits, without universal pointwise/bitwise requirements."""
    limit = 0.005 if dtype == torch.float16 else 0.01
    errors = component_metrics(actual, expected)
    for error in errors:
        error["passed"] = error["finite"] and (
            error["relative_error"] < limit if error["reference_norm"] else error["max_absolute_error"] == 0
        )
    return dict(passed=all(error["passed"] for error in errors), limit=limit, errors=errors)


@torch.no_grad()
def mathematical_reference(inputs, dout, spatial, window, scale):
    return fp64_reference(inputs, dout, neighborhood_mask(spatial, window, device=inputs[0].device), scale=scale)


@torch.no_grad()
def counterfactual_dq(inputs, dout, spatial, window, scale, output):
    """Separate error sources; these counterfactuals are not GPU candidates.

    Ideal-delta forms the scalar from ideal FP64 P and dP, then rounds it to
    FP32 before subtraction; other score/dP work stays IEEE FP32. Removing the
    dS cast does not remove that FP32 product floor.
    The cast floor excludes input quantization, as do all these diagnostics.
    """
    q, k, v = inputs
    visible = neighborhood_mask(spatial, window, device=q.device)
    precision = torch.backends.cuda.matmul.fp32_precision
    try:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        scores = (q.float() @ k.float().transpose(-1, -2)).double()
        dp = dout.float() @ v.float().transpose(-1, -2)
    finally:
        torch.backends.cuda.matmul.fp32_precision = precision
    signed = (scores if scale >= 0 else -scores).masked_fill(~visible, -torch.inf)
    shifted = ((signed - signed.amax(-1, keepdim=True)) * abs(scale)).masked_fill(~visible, -torch.inf)
    p = torch.softmax(shifted, -1).float()
    exact_scores = (q.double() @ k.double().transpose(-1, -2)) * scale
    exact_p = exact_scores.masked_fill(~visible, -torch.inf).softmax(-1)
    exact_dp = dout.double() @ v.double().transpose(-1, -2)
    ideal_delta = (exact_p * exact_dp).sum(-1, keepdim=True).float()
    actual_delta = (output.float() * dout.float()).sum(-1, keepdim=True)
    result = {}
    for name, full_p, ideal, full_ds in (
        ("default", False, False, False),
        ("fp32_p", True, False, False),
        ("ideal_delta", False, True, False),
        ("fp32_p_ideal_delta", True, True, False),
        ("fp32_p_ideal_delta_unrounded_ds", True, True, True),
    ):
        operand_p = p if full_p else p.to(q.dtype).float()
        ds = operand_p * (dp - (ideal_delta if ideal else actual_delta))
        if not full_ds:
            ds = ds.to(q.dtype)
        result[name] = (ds.double() @ k.double()) * scale
    exact = mathematical_reference(inputs, dout, spatial, window, scale)[1]
    result["output_cast_floor"] = exact.to(q.dtype).double()
    return result
