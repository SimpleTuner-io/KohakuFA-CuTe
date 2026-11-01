"""Bounded RTX 5090 training selection for the qualified warp schedule."""

import math
from functools import lru_cache

import torch


@lru_cache(maxsize=16)
def target_device(device):
    if torch.cuda.get_device_capability(device) != (12, 0):
        return False
    properties = torch.cuda.get_device_properties(device)
    return (
        properties.multi_processor_count == 170
        and "5090" in properties.name
        and properties.shared_memory_per_block_optin == 101376
    )


def compatible(q, k, v, mask, dropout, causal, scale, gqa):
    if (
        torch.compiler.is_compiling()
        or torch.are_deterministic_algorithms_enabled()
        or getattr(q, "ndim", None) != 4
        or q.device.type != "cuda"
        or not torch.is_grad_enabled()
        or not all(getattr(t, "requires_grad", False) for t in (q, k, v))
        or not target_device(q.device)
        or mask is not None
        or dropout != 0
        or q.shape != k.shape
        or q.shape != v.shape
        or q.shape[0] != 1
        or q.shape[1] != 16
        or q.shape[2] != 4096
        or q.shape[3] not in (64, 128)
        or q.device != k.device
        or q.device != v.device
        or q.device.index != torch.cuda.current_device()
        or q.dtype not in (torch.bfloat16, torch.float16)
        or q.dtype != k.dtype
        or q.dtype != v.dtype
        or not all(t.is_contiguous() and t.data_ptr() % 16 == 0 for t in (q, k, v))
    ):
        return False
    return scale is None or (
        isinstance(scale, (float, int)) and math.isfinite(scale) and 1.1754943508222875e-38 <= scale <= 2e38
    )


@lru_cache(maxsize=2)
def schedule(dimension):
    from ._attention import Schedule

    if dimension not in (64, 128):
        raise ValueError("The selected SM120 schedule requires D64 or D128.")
    return Schedule(
        forward_m=128 if dimension == 64 else 64,
        forward_n=64,
        forward_stages=1,
        query_in_registers=False,
        backward_m=64,
        backward_n=64,
        query_stages=2,
        dout_stages=1,
        score_m_warps=4,
        kv_m_warps=2,
        dq_m_warps=2,
        v_in_regs=False,
    )
