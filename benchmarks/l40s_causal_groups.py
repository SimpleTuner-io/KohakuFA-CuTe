"""Reorder causal key/head CTAs while keeping the selected L40S geometry."""

from dataclasses import replace
from functools import lru_cache


def make_attention(*, head_group=1):
    if head_group not in (1, 2, 4, 8, 16):
        raise ValueError("Select one of the supported causal head groups.")
    from kohakufa_cute import api
    from kohakufa_cute.fa4_sm80 import _l40s
    from kohakufa_cute.fa4_sm80._host import native_runner

    @lru_cache(maxsize=16)
    def schedule(dimension, sequence, causal):
        original = _l40s.schedule(dimension, sequence, causal)
        return replace(original, causal_head_group=head_group if causal else 1)

    def operation(q, k, v, *, is_causal=False, scale=None, enable_gqa=False):
        if _l40s.compatible(q, k, v, None, 0.0, is_causal, scale, enable_gqa):
            return native_runner(schedule(q.shape[-1], q.shape[2], is_causal))(q, k, v, causal=is_causal, scale=scale)
        return api.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)

    return operation


def make_direct(*, native=True):
    """Exercise the proposed groups on tails and head counts outside public bounds."""
    from benchmarks.l40s_schedules import make_attention as direct_attention

    operations = {
        64: direct_attention(native=native, head_group=8),
        128: direct_attention(native=native, head_group=16),
    }

    def operation(q, k, v, **kwargs):
        return operations[q.shape[-1]](q, k, v, **kwargs)

    return operation
