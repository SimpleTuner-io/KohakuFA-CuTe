"""Reuse a compiled host runner without changing the selected GPU schedule."""

from dataclasses import asdict
from functools import lru_cache


@lru_cache(maxsize=32)
def native_runner(schedule):
    from kohakufa_cute.native import extension

    return extension().AttentionRunner("warp", asdict(schedule))
