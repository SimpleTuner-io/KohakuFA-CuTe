"""Actual public entry and matched explicit configured controls; no kernel duplication."""

from dataclasses import replace
from functools import lru_cache


def make_attention(*, implementation="public"):
    from kohakufa_cute import api
    from kohakufa_cute.fa4_sm80 import _sm120
    from kohakufa_cute.fa4_sm80._host import native_runner

    if implementation == "public":
        return api.scaled_dot_product_attention
    if implementation not in ("configured-g1", "configured-g16"):
        raise ValueError(implementation)
    import torch

    from kohakufa_cute import fa4_sm80

    if not _sm120.target_device(torch.device("cuda", torch.cuda.current_device())) or not fa4_sm80.available():
        raise ValueError("The explicit controls require the qualified full RTX 5090 runtime.")

    @lru_cache(maxsize=4)
    def runner(dimension, causal):
        group = 16 if causal and implementation == "configured-g16" else 1
        return native_runner(replace(_sm120.schedule(dimension, False), causal_head_group=group))

    def run(q, k, v, *, is_causal=False, scale=None):
        return runner(q.shape[-1], is_causal)(q, k, v, causal=is_causal, scale=scale)

    return run
