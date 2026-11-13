"""Measured A100 policies; explicit experimental Schedule defaults stay stable."""

from dataclasses import replace
from functools import lru_cache

from ._attention import Schedule


@lru_cache(maxsize=32)
def a100_schedule(dimension, sequence, causal):
    schedule = Schedule()
    if dimension == 64 and sequence >= 2048:
        return replace(
            schedule,
            backward_m=128,
            query_stages=1,
            dout_stages=1,
            score_m_warps=4,
            kv_m_warps=4,
            dq_m_warps=4,
        )
    if dimension == 128 and causal:
        return replace(schedule, forward_m=64)
    return schedule
