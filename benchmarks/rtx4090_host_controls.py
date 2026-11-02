"""Bounded host controls; unchanged S1024 GPU schedules and operands."""

import math

import torch


def make_attention(*, mode="fused", dimension=64, causal=False):
    from kohakufa_cute import api
    from kohakufa_cute.fa4_sm80 import available, rtx4090
    from kohakufa_cute.fa4_sm80._host import native_runner
    from kohakufa_cute.native import extension

    if mode == "parent":
        from benchmarks.rtx4090_automatic_controls import parent_api

        return parent_api().scaled_dot_product_attention
    if mode == "public":
        return api.scaled_dot_product_attention
    if mode == "explicit":
        return rtx4090.scaled_dot_product_attention
    if mode not in ("direct", "live", "fused"):
        raise ValueError("Select public, explicit, direct, live or fused.")
    if dimension not in (64, 128) or not available():
        raise ValueError("The qualified runtime and S1024 D64/D128 are required.")
    module = extension()
    runner = native_runner(rtx4090.schedule(dimension, causal, 1024))
    metadata = module.rtx4090_metadata_compatible
    fused = module.rtx4090_guarded_call

    def operation(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
        # Direct is an explicit overhead control for already-qualified inputs.
        # Live/fused retain public policy checks, without caching tensor state.
        policy = (
            not torch.compiler.is_compiling()
            and not torch.are_deterministic_algorithms_enabled()
            and attn_mask is None
            and dropout_p == 0
            and is_causal == causal
            and (scale is None or (isinstance(scale, (int, float)) and math.isfinite(scale) and 2**-126 <= scale <= 2e38))
        )
        if policy:
            if mode == "direct":
                return runner(q, k, v, scale=scale, causal=is_causal)
            if mode == "live":
                if metadata(q, k, v, is_causal):
                    return runner(q, k, v, scale=scale, causal=is_causal)
            else:
                output = fused(runner, q, k, v, dimension, scale, is_causal)
                if output is not None:
                    return output
        return api.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal, scale=scale, enable_gqa=enable_gqa)

    return operation
