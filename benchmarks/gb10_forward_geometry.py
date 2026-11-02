"""Compare dense GB10 forward tiles while preserving its public backward."""

from dataclasses import replace
from functools import lru_cache
from types import FunctionType


def make_public():
    from kohakufa_cute.fa4_sm80 import gb10

    return gb10.scaled_dot_product_attention


def make_attention(*, forward_m=64, forward_n=64):
    if forward_m not in (64, 128) or forward_n not in (32, 64):
        raise ValueError("Select an explicit M64/M128 and N32/N64 geometry.")
    from kohakufa_cute.fa4_sm80 import gb10

    @lru_cache(maxsize=32)
    def schedule(sequence, causal, heads=4, dtype=None):
        original = gb10.schedule(sequence, causal, heads, dtype)
        return original if causal else replace(original, forward_m=forward_m, forward_n=forward_n)

    original = gb10.scaled_dot_product_attention
    scoped = dict(original.__globals__, schedule=schedule)
    operation = FunctionType(original.__code__, scoped, original.__name__, original.__defaults__, original.__closure__)
    operation.__kwdefaults__ = original.__kwdefaults__
    return operation
