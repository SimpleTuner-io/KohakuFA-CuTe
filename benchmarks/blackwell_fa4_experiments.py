"""Explicit factory for the retained experimental SM103 schedule."""

import torch


def make_static_attention():
    """Use the statically retained, source-qualified SM103 schedule explicitly."""
    from kohakufa_cute.fa4_blackwell.attention import attention as run

    def attention(q, k, v, *, is_causal=False, scale=None, enable_gqa=False):
        if (
            q.shape != k.shape
            or k.shape != v.shape
            or q.shape[-1] not in (64, 128)
            or torch.cuda.get_device_capability(q.device) != (10, 3)
        ):
            raise ValueError("The retained SM103 experiment requires square MHA with D64 or D128.")
        if scale is not None and scale <= 0:
            raise ValueError("The retained SM103 experiment requires a positive scale.")
        return run(q, k, v, causal=is_causal, scale=scale)

    return attention
