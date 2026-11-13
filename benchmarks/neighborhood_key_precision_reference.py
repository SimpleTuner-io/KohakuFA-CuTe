"""Independent equations for residual-centered neighborhood key gradients.

The new correction is a separate numerical policy. Production's default and
the prior weighted-query experiment retain their original equations.
"""

import torch

from benchmarks.neighborhood_precision_reference import precision_reference
from benchmarks.neighborhood_reference import neighborhood_mask, neighborhoods

METHODS = ("baseline", "fp32_p_weighted_dq", "reconstructed_dk", "weighted_dq_reconstructed_dk")
NEW_METHODS = ("reconstructed_dk", "weighted_dq_reconstructed_dk")


@torch.no_grad()
def key_terms(inputs, dout, spatial, window, scale, output):
    """FP32 residuals, normalized correction, and rounded key operands.

    FP32 sums specify the independent arithmetic boundary, rather than an
    exact replay of warp shuffles, tensor-core products, or traversal order.
    Score shifting precedes scaling. The two subtractions before dS are kept
    separate so a small correction need not be added to a large scalar delta.
    """
    q, k, v = inputs
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
    weights = torch.exp((signed - signed.amax(-1, keepdim=True)) * abs(scale))
    p = (weights / weights.sum(-1, keepdim=True)).float()
    delta = (output.float() * dout.float()).sum(-1, keepdim=True)
    residual = dp - delta
    mass = p.sum(-1, keepdim=True)
    correction = (p * residual).sum(-1, keepdim=True) / mass
    centered = residual - correction
    ds = (p * centered).to(q.dtype)
    return dict(
        probability=p, dp=dp, delta=delta, residual=residual, mass=mass, correction=correction, centered=centered, ds=ds
    )


@torch.no_grad()
def key_precision_reference(inputs, dout, spatial, window, scale, output, *, method):
    """Return O/dQ/dK/dV under one independently prescribed experiment.

    New methods only change dK relative to their corresponding query method.
    They keep P input-rounded for dV and dS input-rounded for gradient products;
    final reference products use FP64 to isolate operand/reduction rounding.
    """
    if method not in METHODS:
        raise ValueError("Unknown key-gradient precision experiment.")
    full_p = method != "baseline"
    weighted = method in ("fp32_p_weighted_dq", "weighted_dq_reconstructed_dk")
    result = precision_reference(
        inputs, dout, spatial, window, scale, output, fp32_p=full_p, correction="normalized" if weighted else "none"
    )
    if method not in NEW_METHODS:
        return result
    terms = key_terms(inputs, dout, spatial, window, scale, output)
    dk = (terms["ds"].double().transpose(-1, -2) @ inputs[0].double()) * scale
    return result[0], result[1], dk, result[3]


@torch.no_grad()
def sparse_key_precision_reference(inputs, dout, spatial, window, scale, output, *, method, positions, chunk_size=64):
    """Bounded CPU policy reference at independently selected spatial owners.

    Enumerate legal windows independently, then find every contributing query
    by forward-list membership for each selected key. This never constructs an
    N-by-N matrix. Local FP32 reductions remain an independent reduction order,
    while final products and accumulation use FP64 as in the dense reference.
    """
    if method not in METHODS or chunk_size < 1:
        raise ValueError("Expected a declared method and positive reference chunk size.")
    q, k, v = [value.detach().cpu() for value in inputs]
    gradient, out = dout.detach().cpu(), output.detach().cpu()
    edges = neighborhoods(spatial, window)
    selected = sorted(set(positions))
    if not selected or any(index < 0 or index >= len(edges) for index in selected):
        raise ValueError("Reference positions must identify valid query and key owners.")
    query_slot = {query: slot for slot, query in enumerate(selected)}
    key_slot = torch.full((len(edges),), -1, dtype=torch.long)
    key_slot[selected] = torch.arange(len(selected))
    owners = set(selected)
    needed = [query for query, keys in enumerate(edges) if query in owners or owners.intersection(keys)]
    shape = (*q.shape[:2], len(selected), q.shape[-1])
    result = [torch.zeros(shape, dtype=torch.float64) for _ in range(4)]
    full_p = method != "baseline"
    weighted = method in ("fp32_p_weighted_dq", "weighted_dq_reconstructed_dk")
    precision = torch.backends.cuda.matmul.fp32_precision
    try:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        for batch in range(q.shape[0]):
            for head in range(q.shape[1]):
                for first in range(0, len(needed), chunk_size):
                    queries = needed[first : first + chunk_size]
                    indices = torch.tensor([edges[query] for query in queries])
                    qd = q[batch, head, queries].double()
                    kd, vd = k[batch, head, indices].double(), v[batch, head, indices].double()
                    grad = gradient[batch, head, queries].double()
                    scores = torch.bmm(qd.float()[:, None], kd.float().transpose(1, 2)).squeeze(1).double()
                    dp = torch.bmm(grad.float()[:, None], vd.float().transpose(1, 2)).squeeze(1)
                    signed = scores if scale >= 0 else -scores
                    weights = torch.exp((signed - signed.amax(-1, keepdim=True)) * abs(scale))
                    probability = weights / weights.sum(-1, keepdim=True)
                    p32 = probability.float()
                    low_p = p32.to(q.dtype).float()
                    delta = (out[batch, head, queries].float() * gradient[batch, head, queries].float()).sum(
                        -1, keepdim=True
                    )
                    residual = dp - delta
                    query_ds = ((p32 if full_p else low_p) * residual).to(q.dtype).double()
                    key_ds = query_ds
                    if method in NEW_METHODS:
                        correction = (p32 * residual).sum(-1, keepdim=True) / p32.sum(-1, keepdim=True)
                        key_ds = (p32 * (residual - correction)).to(q.dtype).double()
                    dq = torch.bmm(query_ds[:, None], kd).squeeze(1)
                    if weighted:
                        mean = torch.bmm(low_p.double()[:, None], kd).squeeze(1)
                        mean /= low_p.double().sum(-1, keepdim=True)
                        dq -= query_ds.sum(-1, keepdim=True) * mean
                    dq *= scale
                    expected_out = torch.bmm(probability[:, None], vd).squeeze(1)
                    for local, query in enumerate(queries):
                        if query in query_slot:
                            slot = query_slot[query]
                            result[0][batch, head, slot] = expected_out[local]
                            result[1][batch, head, slot] = dq[local]
                    slots = key_slot[indices]
                    query_indices, neighbor_indices = (slots >= 0).nonzero(as_tuple=True)
                    destination = slots[query_indices, neighbor_indices]
                    result[2][batch, head].index_add_(
                        0, destination, key_ds[query_indices, neighbor_indices, None] * qd[query_indices]
                    )
                    result[3][batch, head].index_add_(
                        0, destination, low_p.double()[query_indices, neighbor_indices, None] * grad[query_indices]
                    )
    finally:
        torch.backends.cuda.matmul.fp32_precision = precision
    result[2] *= scale
    return dict(positions=selected, policy=tuple(result))
