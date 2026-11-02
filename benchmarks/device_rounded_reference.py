"""Evaluate the existing device-rounded reference with CPU FP64 diagnostics.

Independent IEEE device FP32 GEMMs establish scores and dP, as in
``reference.rounded_reference`` on CUDA. Capture its actual-O FP32 delta too.
CPU evaluation then preserves those intermediates instead of substituting a
different BLAS reduction order immediately before the P/dS rounding boundaries.
This changes reference execution location, not the kernel arithmetic or gates.
"""

import torch

from benchmarks.reference import fp64_reference, metrics, rounded_reference


@torch.no_grad()
def captured_reference(inputs, dout, output, *, causal=False, scale=None):
    q, k, v = inputs
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    if q.device.type != "cuda" or q.shape != k.shape or k.shape != v.shape:
        raise ValueError("This diagnostic requires square CUDA MHA operands.")
    old = torch.backends.cuda.matmul.fp32_precision
    try:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        with torch.autocast(device_type="cuda", enabled=False):
            scores = (q.float() @ k.float().transpose(-1, -2)).cpu()
            dp = (dout.float() @ v.float().transpose(-1, -2)).cpu()
    finally:
        torch.backends.cuda.matmul.fp32_precision = old
    delta = (output.float() * dout.float()).sum(-1, keepdim=True).cpu()
    q, k, v, dout = (t.detach().cpu().double() for t in (*inputs, dout))
    signed = scores.double() if scale >= 0 else -scores.double()
    visible = torch.ones_like(signed, dtype=torch.bool)
    if causal:
        visible &= torch.ones(signed.shape[-2:], dtype=torch.bool).tril()
    signed = signed.masked_fill(~visible, -torch.inf)
    maximum = signed.amax(-1, keepdim=True)
    weights = ((signed - maximum) * abs(scale)).exp().masked_fill(~visible, 0)
    probability = weights / weights.sum(-1, keepdim=True)
    low_p = probability.float().to(inputs[0].dtype).float()
    low_ds = (low_p * (dp - delta)).to(inputs[0].dtype).double()
    result = (
        probability @ v,
        (low_ds @ k) * scale,
        (low_ds.transpose(-1, -2) @ q) * scale,
        low_p.double().transpose(-1, -2) @ dout,
    )
    audit = {
        "score_fp64": metrics(scores, q @ k.transpose(-1, -2)),
        "dp_fp64": metrics(dp, dout @ v.transpose(-1, -2)),
    }
    return result, audit


@torch.no_grad()
def assess_device_intermediates(actual, inputs, dout, *, causal, scale, directed, agreement=False):
    records = []
    limit = 0.01 if inputs[0].dtype == torch.bfloat16 else 0.005
    for head in range(inputs[0].shape[1]):
        part = tuple(t[:, head : head + 1] for t in inputs)
        grad = dout[:, head : head + 1]
        observed = tuple(t[:, head : head + 1] for t in actual)
        copied = tuple(t.detach().cpu() for t in part)
        mathematical = fp64_reference(copied, grad.cpu(), causal=causal, scale=scale)
        rounded, audit = captured_reference(part, grad, observed[0], causal=causal, scale=scale)
        if agreement:
            original_gpu = rounded_reference(part, grad, causal=causal, scale=scale, output=observed[0])
            original_cpu = rounded_reference(copied, grad.cpu(), causal=causal, scale=scale, output=observed[0].cpu())
        for index, (label, got, exact, reference) in enumerate(
            zip(("out", "dq", "dk", "dv"), observed, mathematical, rounded)
        ):
            math_error, contract_error = metrics(got.cpu(), exact), metrics(got.cpu(), reference)
            required = contract_error if directed else math_error
            assert required["finite"], (head, label, required)
            if required["reference_norm"]:
                assert required["relative_error"] < limit, (head, label, required)
            else:
                assert required["max_absolute_error"] == 0, (head, label, required)
            row = {
                "head": head,
                "component": label,
                "fp64": math_error,
                "rounded": contract_error,
                "intermediate_audit": audit,
            }
            if agreement:
                difference = metrics(reference, original_gpu[index].cpu())
                # This checks two reference evaluations, never kernel accuracy.
                if difference["reference_norm"]:
                    assert difference["relative_error"] < 1e-10, (head, label, difference)
                else:
                    assert difference["max_absolute_error"] == 0
                row["reference_cpu_gpu_agreement"] = difference
                row["native_cpu_rounded_diagnostic"] = metrics(got.cpu(), original_cpu[index])
                row["original_gpu_rounded_diagnostic"] = metrics(got.cpu(), original_gpu[index].cpu())
            records.append(row)
    return records
