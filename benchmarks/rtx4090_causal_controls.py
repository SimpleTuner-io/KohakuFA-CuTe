"""Compare the bounded public policy with its preceding N64 policy."""

from functools import lru_cache


@lru_cache(maxsize=4)
def previous_schedule(dimension, causal):
    from kohakufa_cute.fa4_sm80 import rtx4090

    return rtx4090.schedule(dimension, causal, 1024)


def make_attention(source=None, dependencies=None, *, policy="selected"):
    if policy not in ("selected", "previous"):
        raise ValueError("Select the public policy or its preceding policy.")
    from benchmarks.rtx4090_shared_candidates import make_attention as qualified

    qualified(source, dependencies)
    from kohakufa_cute import api
    from kohakufa_cute.fa4_sm80 import rtx4090

    if policy == "selected":
        return rtx4090.scaled_dot_product_attention

    def previous(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
        if rtx4090.compatible(q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa):
            return rtx4090.native_runner(previous_schedule(q.shape[-1], is_causal))(q, k, v, scale=scale, causal=is_causal)
        return api.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal, scale=scale, enable_gqa=enable_gqa)

    return previous
