"""Small, independent neighborhood enumeration for qualification only.

Production traversal must not import this module or construct its dense mask.
"""

from itertools import product
from math import prod

import torch


def neighborhoods(spatial_shape, window_size):
    """Choose the legal window whose anchor is nearest the query on each axis.

    Enumerating every legal placement avoids reusing production's clamp or
    inverse formulas. The anchor is window//2, including even-size windows.
    """
    if len(spatial_shape) not in (2, 3) or len(window_size) != len(spatial_shape):
        raise ValueError("Expected a 2D or 3D shape and matching window.")
    if any(not 1 <= window <= extent for extent, window in zip(spatial_shape, window_size)):
        raise ValueError("Window must fit the spatial shape.")
    strides = [prod(spatial_shape[axis + 1 :]) for axis in range(len(spatial_shape))]
    result = []
    for query in product(*(range(extent) for extent in spatial_shape)):
        starts = [
            min(range(extent - window + 1), key=lambda candidate: abs(candidate + window // 2 - position))
            for position, extent, window in zip(query, spatial_shape, window_size)
        ]
        result.append(
            tuple(
                sum(position * stride for position, stride in zip(key, strides))
                for key in product(*(range(start, start + window) for start, window in zip(starts, window_size)))
            )
        )
    return tuple(result)


def neighborhood_mask(spatial_shape, window_size, *, device="cpu"):
    edges = neighborhoods(spatial_shape, window_size)
    mask = torch.zeros(len(edges), len(edges), dtype=torch.bool, device=device)
    for query, keys in enumerate(edges):
        mask[query, list(keys)] = True
    return mask


@torch.no_grad()
def two_term_forward_reference(inputs, spatial_shape, window_size, scale):
    """Independent rounding diagnostic for 3D's two-term forward probability.

    One global row maximum and FP64 final products isolate the high/low operand
    conversions. This is not an exact replay of online tile rescaling or FP32
    tensor-core accumulation; retain mathematical FP64 alongside this result.
    """
    q, k, v = inputs
    precision = torch.backends.cuda.matmul.fp32_precision
    try:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        with torch.autocast(device_type=q.device.type, enabled=False):
            scores = q.float() @ k.float().transpose(-1, -2)
    finally:
        torch.backends.cuda.matmul.fp32_precision = precision
    visible = neighborhood_mask(spatial_shape, window_size, device=q.device)
    signed = (scores if scale >= 0 else -scores).masked_fill(~visible, -torch.inf)
    shifted = ((signed - signed.amax(-1, keepdim=True)) * abs(scale)).masked_fill(~visible, -torch.inf)
    weights = shifted.double().exp().float()
    high = weights.to(q.dtype).float()
    low = (weights - high).to(q.dtype).float()
    numerator = high.double() @ v.double() + low.double() @ v.double()
    return numerator / weights.sum(-1, keepdim=True).double()
