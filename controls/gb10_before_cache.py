"""Explicit full-device GB10 shared schedules with ordinary CuTe fallbacks."""

import math
from functools import lru_cache

import torch

from ._attention import Schedule
from ._host import native_runner


@lru_cache(maxsize=12)
def schedule(sequence, causal, heads=4, dtype=torch.bfloat16):
    """Preserve M64 short/causal work and use the measured wider dense tile."""
    if sequence not in (512, 1024, 4096) or (causal and sequence == 1024):
        raise ValueError("The explicit GB10 schedule requires a qualified sequence family.")
    wide_dense = not causal and ((sequence == 1024 and heads == 4) or (sequence == 4096 and dtype == torch.float16))
    return Schedule(
        forward_m=64,
        backward_m=32 if wide_dense else 64,
        backward_n=128 if wide_dense else 64,
        query_stages=1 if wide_dense else 2,
        dout_stages=1,
        threads=256,
        score_m_warps=2 if wide_dense else 4,
        kv_m_warps=2,
        dq_m_warps=2,
        v_in_regs=not wide_dense,
    )


def compatible(q, k, v, mask=None, dropout=0.0, causal=False, scale=None, gqa=False):
    """Bound the opt-in path to the complete GPU and measured D128 families."""
    if (
        torch.compiler.is_compiling()
        or mask is not None
        or dropout != 0
        or gqa
        or torch.are_deterministic_algorithms_enabled()
        or getattr(q, "ndim", None) != 4
        or q.shape != k.shape
        or q.shape != v.shape
        or q.device.type != "cuda"
        or q.device != k.device
        or q.device != v.device
        or q.device.index != torch.cuda.current_device()
        or q.dtype not in (torch.float16, torch.bfloat16)
        or q.dtype != k.dtype
        or q.dtype != v.dtype
        or q.shape[0] != 1
        or q.shape[1] not in (4, 16)
        or q.shape[2] not in (512, 1024, 4096)
        or (causal and q.shape[2] == 1024)
        or q.shape[-1] != 128
        or not all(t.is_contiguous() for t in (q, k, v))
        or any(t.data_ptr() % 16 for t in (q, k, v))
        or torch.cuda.get_device_capability(q.device) != (12, 1)
    ):
        return False
    properties = torch.cuda.get_device_properties(q.device)
    if (
        properties.name != "NVIDIA GB10"
        or properties.multi_processor_count != 48
        or properties.shared_memory_per_block_optin != 101376
    ):
        return False
    if scale is not None and (
        not isinstance(scale, (float, int)) or not math.isfinite(scale) or not 1.1754943508222875e-38 <= scale <= 2e38
    ):
        return False
    from . import available

    return available()


def scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    """Run the explicit measured family and preserve the public fallback API."""
    if compatible(q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa):
        return native_runner(schedule(q.shape[2], is_causal, q.shape[1], q.dtype))(q, k, v, scale=scale, causal=is_causal)
    from kohakufa_cute import api

    return api.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal, scale=scale, enable_gqa=enable_gqa)
