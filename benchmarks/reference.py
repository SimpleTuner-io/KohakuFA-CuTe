"""FP64 attention backward with a centered softmax correction."""

import math

import torch


@torch.no_grad()
def fp64_reference(inputs, dout, mask=None, causal=False, scale=None):
    q, k, v = [tensor.double() for tensor in inputs]
    dout = dout.double()
    groups = q.shape[1] // k.shape[1]
    keys, values = [tensor.repeat_interleave(groups, dim=1) for tensor in (k, v)]
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    logits = (q @ keys.transpose(-1, -2)) * scale
    if mask is not None:
        logits.masked_fill_(~mask, -torch.inf)
    if causal:
        visible = torch.arange(k.shape[2], device=q.device)[None, :] <= torch.arange(q.shape[2], device=q.device)[:, None]
        logits.masked_fill_(~visible, -torch.inf)
    valid = torch.isfinite(logits).any(-1, keepdim=True)
    probability = torch.softmax(logits.masked_fill(~valid, 0), -1).masked_fill(~valid, 0)
    dp = dout @ values.transpose(-1, -2)
    centered = dp - dp.gather(-1, logits.argmax(-1, keepdim=True))
    ds = probability * (centered - (probability * centered).sum(-1, keepdim=True))
    dq = (ds @ keys) * scale
    dk = (ds.transpose(-1, -2) @ q) * scale
    dv = probability.transpose(-1, -2) @ dout
    dk, dv = [tensor.reshape(k.shape[0], k.shape[1], groups, *k.shape[2:]).sum(2) for tensor in (dk, dv)]
    return probability @ values, dq, dk, dv


def stable_norm(tensor):
    largest = float(tensor.abs().max())
    return largest * float((tensor / largest).norm()) if largest else 0.0


def metrics(got, reference):
    got, reference = got.detach().double().flatten(), reference.detach().double().flatten()
    denominator = stable_norm(reference)
    norm = stable_norm(got)
    finite = bool(torch.isfinite(got).all())
    error = stable_norm(got - reference) if finite else None
    relative_error = error / denominator if finite and denominator > 0 else None
    norm_ratio = norm / denominator if finite and denominator > 0 else None
    return {
        "finite": finite,
        "reference_norm": denominator,
        "relative_error": relative_error if relative_error is not None and math.isfinite(relative_error) else None,
        "log10_relative_error": (
            math.log10(error) - math.log10(denominator) if finite and error > 0 and denominator > 0 else None
        ),
        "norm_ratio": norm_ratio if norm_ratio is not None and math.isfinite(norm_ratio) else None,
        "cosine": float((got / norm).dot(reference / denominator)) if finite and norm > 0 and denominator > 0 else None,
        "max_absolute_error": float((got - reference).abs().max()) if finite else None,
    }


def row_metrics(got, reference):
    """Error per last-dimension vector, flattened in batch/head/sequence order.

    Keep zero-reference rows undefined rather than assigning them an artificial
    relative error. Normalize before computing norms so tiny gradients survive.
    """
    if got.shape != reference.shape or got.ndim < 1 or got.shape[-1] == 0:
        raise ValueError("row metrics require matching tensors with a nonempty final dimension")
    shape = list(got.shape[:-1])
    got, reference = [tensor.detach().double().reshape(-1, tensor.shape[-1]) for tensor in (got, reference)]

    def norm(tensor):
        largest = tensor.abs().amax(-1)
        divisor = torch.where(largest > 0, largest, 1)
        return largest * (tensor / divisor[:, None]).norm(dim=-1)

    finite = torch.isfinite(got).all(-1) & torch.isfinite(reference).all(-1)
    denominator = norm(reference)
    error = norm(got - reference)
    relative = error / denominator
    log_relative = error.log10() - denominator.log10()

    def values(tensor, valid):
        return [value if math.isfinite(value) else None for value in tensor.masked_fill(~valid, torch.nan).tolist()]

    return {
        "shape": shape,
        "finite": finite.tolist(),
        "reference_norm": values(denominator, torch.isfinite(reference).all(-1)),
        "relative_error": values(relative, finite & (denominator > 0)),
        "log10_relative_error": values(log_relative, finite & (denominator > 0) & (error > 0)),
        "max_absolute_error": values((got - reference).abs().amax(-1), finite),
    }


@torch.no_grad()
def rounded_reference(inputs, dout, mask=None, causal=False, scale=None, *, output=None):
    """Diagnose the single-rounded-P/dS contract, separately from FP64 accuracy.

    Accumulate the final products in FP64 to expose operand rounding without
    adding another reduction-order dependency. Runtime FP32 accumulation still
    has its own rounding floor. Scores and dP use IEEE FP32 GEMMs; their
    reduction order can differ from tensor-core products. Supplying O isolates backward
    from a forward output-rounding difference in delta = sum(O * dO).
    """
    q, k, v = [tensor.double() for tensor in inputs]
    dtype = inputs[0].dtype
    groups = q.shape[1] // k.shape[1]
    keys, values = [tensor.repeat_interleave(groups, dim=1) for tensor in (k, v)]
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    precision = torch.backends.cuda.matmul.fp32_precision
    try:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        with torch.autocast(device_type=q.device.type, enabled=False):
            scores = (q.float() @ keys.float().transpose(-1, -2)).double()
            dp = dout.float() @ values.float().transpose(-1, -2)
    finally:
        torch.backends.cuda.matmul.fp32_precision = precision
    # Shift the unscaled scores before multiplication, preserving small gaps.
    signed = scores if scale >= 0 else -scores
    visible = torch.ones_like(scores, dtype=torch.bool)
    if mask is not None:
        visible &= mask
    if causal:
        visible &= torch.arange(k.shape[2], device=q.device)[None] <= torch.arange(q.shape[2], device=q.device)[:, None]
    signed = signed.masked_fill(~visible, -torch.inf)
    valid = visible.any(-1, keepdim=True)
    maximum = signed.amax(-1, keepdim=True).masked_fill(~valid, 0)
    weights = torch.exp((signed - maximum) * abs(scale)).masked_fill(~visible, 0)
    probability = weights / weights.sum(-1, keepdim=True).masked_fill(~valid, 1)
    out = probability @ values
    base = (out.to(dtype) if output is None else output).float()
    delta = (base * dout.float()).sum(-1, keepdim=True)
    rounded_p = probability.float().to(dtype).float()
    ds = (rounded_p * (dp - delta)).to(dtype).double()
    dq = (ds @ keys) * scale
    dk = (ds.transpose(-1, -2) @ q) * scale
    dv = rounded_p.double().transpose(-1, -2) @ dout.double()
    dk, dv = [tensor.reshape(k.shape[0], k.shape[1], groups, *k.shape[2:]).sum(2) for tensor in (dk, dv)]
    return out, dq, dk, dv
