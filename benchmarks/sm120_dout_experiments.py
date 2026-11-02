"""Matched D64 Q2 pipeline controls; D128 retains the qualified single dO stage."""

import math
from dataclasses import replace
from functools import lru_cache


@lru_cache(maxsize=8)
def schedule_for(dimension, causal, dout_stages):
    from kohakufa_cute.fa4_sm80 import _sm120

    if dout_stages not in (1, 2):
        raise ValueError("Select one or two output-gradient stages.")
    return replace(_sm120.schedule(dimension, causal), dout_stages=dout_stages if dimension == 64 else 1)


def make_attention(*, dout_stages=1):
    import torch

    from kohakufa_cute import api, fa4_sm80
    from kohakufa_cute.fa4_sm80 import _sm120
    from kohakufa_cute.fa4_sm80._host import native_runner

    if dout_stages not in (1, 2):
        raise ValueError("Select one or two output-gradient stages.")
    if not _sm120.target_device(torch.device("cuda", torch.cuda.current_device())) or not fa4_sm80.available():
        raise ValueError("This explicit experiment requires the qualified full RTX 5090 runtime.")

    @lru_cache(maxsize=4)
    def runner(dimension, causal):
        return native_runner(schedule_for(dimension, causal, dout_stages))

    def run(q, k, v, *, is_causal=False, scale=None, enable_gqa=False):
        amount = q.shape[-1] ** -0.5 if scale is None else scale
        if not isinstance(amount, (float, int)) or not math.isfinite(amount) or amount < torch.finfo(torch.float32).tiny:
            return api._extension().sdpa(q, k, v, None, 0.0, is_causal, scale, enable_gqa)
        return runner(q.shape[-1], is_causal)(q, k, v, causal=is_causal, scale=scale)

    return run
