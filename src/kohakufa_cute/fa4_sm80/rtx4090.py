"""Explicit RTX 4090 shared joint-backward schedules and native fallbacks."""

import math
from functools import lru_cache

import torch

from ._attention import Schedule
from ._host import native_runner


@lru_cache(maxsize=1)
def _metadata_predicate():
    """Cache the function handle, while checking live tensors every call."""
    from kohakufa_cute.native import extension

    return extension().rtx4090_metadata_compatible


@lru_cache(maxsize=12)
def schedule(dimension, causal, sequence=1024):
    """Keep the wider dense key tile separate from causal work distribution."""
    if dimension not in (64, 128):
        raise ValueError("The explicit RTX 4090 schedule requires D64 or D128.")
    wide_dense = not causal
    narrow_query = wide_dense and dimension == 128
    wide_d64 = wide_dense and dimension == 64
    long_causal_d64 = causal and dimension == 64 and sequence == 4096
    wide_backward = wide_dense or long_causal_d64
    return Schedule(
        forward_m=128 if dimension == 64 and not causal else 64,
        forward_n=64,
        forward_stages=1,
        query_in_registers=False,
        backward_m=32 if narrow_query else 64,
        backward_n=128 if wide_backward else 64,
        query_stages=1,
        dout_stages=1,
        threads=256,
        score_m_warps=2 if wide_backward else 4,
        kv_m_warps=4 if wide_d64 or long_causal_d64 else 2,
        dq_m_warps=4 if wide_d64 or long_causal_d64 else 2,
        v_in_regs=not narrow_query,
        causal_head_group=16 if causal else 1,
    )


def compatible(q, k, v, mask=None, dropout=0.0, causal=False, scale=None, gqa=False):
    """Keep policy guards live and combine mutable tensor metadata checks."""
    if (
        torch.compiler.is_compiling()
        or mask is not None
        or dropout != 0
        or gqa
        or torch.are_deterministic_algorithms_enabled()
    ):
        return False
    if scale is not None and (
        not isinstance(scale, (float, int)) or not math.isfinite(scale) or not 1.1754943508222875e-38 <= scale <= 2e38
    ):
        return False
    from . import available

    return available() and _metadata_predicate()(q, k, v, causal)


def scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    """Use the explicit measured family; retain ordinary CuTe for other calls."""
    if compatible(q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa):
        return native_runner(schedule(q.shape[-1], is_causal, q.shape[-2]))(q, k, v, scale=scale, causal=is_causal)
    from kohakufa_cute import api

    return api.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal, scale=scale, enable_gqa=enable_gqa)
