"""Explicit A40 shared joint-backward schedules and native fallbacks."""

import math
from functools import lru_cache

import torch

from ._attention import Schedule
from ._host import native_runner


@lru_cache(maxsize=32)
def _is_device(index):
    """Device topology is immutable for a CUDA index during this process."""
    properties = torch.cuda.get_device_properties(index)
    return (
        (properties.major, properties.minor) == (8, 6)
        and properties.name == "NVIDIA A40"
        and properties.multi_processor_count == 84
        and properties.shared_memory_per_block_optin == 101376
    )


@lru_cache(maxsize=32)
def schedule(dimension, causal, sequence=4096):
    """Keep the wider dense key tile separate from causal work distribution."""
    if dimension not in (64, 128):
        raise ValueError("The explicit A40 schedule requires D64 or D128.")
    if dimension == 64:
        return Schedule(
            forward_m=64,
            backward_n=128,
            query_stages=2,
            dout_stages=1,
            score_m_warps=2,
            kv_m_warps=4,
            dq_m_warps=4,
            v_in_regs=True,
            causal_head_group=(16 if sequence == 1024 else 4) if causal else 1,
        )
    wide_dense = not causal
    narrow_query = wide_dense and dimension == 128
    return Schedule(
        forward_m=64,
        forward_n=64,
        forward_stages=1,
        query_in_registers=False,
        backward_m=32 if narrow_query else 64,
        backward_n=128 if wide_dense else 64,
        query_stages=1,
        dout_stages=1,
        threads=256,
        score_m_warps=2 if wide_dense else 4,
        kv_m_warps=2,
        dq_m_warps=2,
        v_in_regs=not narrow_query,
        causal_head_group=4 if causal else 1,
    )


def compatible(q, k, v, mask=None, dropout=0.0, causal=False, scale=None, gqa=False):
    """Bound the opt-in path to complete devices and measured shape families."""
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
        or q.shape[1] != 16
        or q.shape[2] not in (1024, 4096)
        or not (q.shape[-1] == 128 or (q.shape[-1] == 64 and causal))
        or not all(t.is_contiguous() for t in (q, k, v))
        or any(t.data_ptr() % 16 for t in (q, k, v))
        or not _is_device(q.device.index)
    ):
        return False
    if scale is not None and (
        not isinstance(scale, (float, int)) or not math.isfinite(scale) or not 1.1754943508222875e-38 <= scale <= 2e38
    ):
        return False
    from . import available

    return available()


def scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    """Use the explicit measured family; retain ordinary CuTe for other calls."""
    if compatible(q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa):
        return native_runner(schedule(q.shape[-1], is_causal, q.shape[2]))(q, k, v, scale=scale, causal=is_causal)
    from kohakufa_cute import api

    return api.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal, scale=scale, enable_gqa=enable_gqa)
