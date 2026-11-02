"""Preserve the initial full-row public entry as the matched baseline."""

from functools import lru_cache


@lru_cache(maxsize=1)
def _previous_runner():
    from kohakufa_cute.native import extension

    return extension().AttentionRunner("short", {})


def make_attention(*, policy="selected"):
    from kohakufa_cute import api
    from kohakufa_cute.fa4_sm80 import rtx4090

    if policy == "selected":
        return rtx4090.scaled_dot_product_attention
    if policy != "previous":
        raise ValueError("Select the initial or optimized short public entry.")

    def previous(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
        if getattr(q, "ndim", None) == 4 and q.shape[-1] == 16:
            if rtx4090._short_compatible(q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa):
                return _previous_runner()(q, k, v, scale=scale, causal=is_causal)
        if rtx4090.compatible(q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa):
            return rtx4090.native_runner(rtx4090.schedule(q.shape[-1], is_causal, q.shape[-2]))(
                q, k, v, scale=scale, causal=is_causal
            )
        return api.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal, scale=scale, enable_gqa=enable_gqa)

    return previous
