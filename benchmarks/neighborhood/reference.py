"""Small, independent neighborhood enumeration for qualification only.

Production traversal must not import this module or construct its dense mask.
"""

import torch


def neighborhoods(spatial_shape, window_size):
    """Choose the legal window whose anchor is nearest the query on each axis.

    Enumerating every legal placement avoids reusing production's clamp or
    inverse formulas. The anchor is window//2, including even-size windows.
    """
    height, width = spatial_shape
    wh, ww = window_size
    if not 1 <= wh <= height or not 1 <= ww <= width:
        raise ValueError("Window must fit the spatial shape.")
    result = []
    for y in range(height):
        top = min(range(height - wh + 1), key=lambda candidate: abs(candidate + wh // 2 - y))
        for x in range(width):
            left = min(range(width - ww + 1), key=lambda candidate: abs(candidate + ww // 2 - x))
            result.append(tuple(row * width + col for row in range(top, top + wh) for col in range(left, left + ww)))
    return tuple(result)


def neighborhood_mask(spatial_shape, window_size, *, device="cpu"):
    edges = neighborhoods(spatial_shape, window_size)
    mask = torch.zeros(len(edges), len(edges), dtype=torch.bool, device=device)
    for query, keys in enumerate(edges):
        mask[query, list(keys)] = True
    return mask
