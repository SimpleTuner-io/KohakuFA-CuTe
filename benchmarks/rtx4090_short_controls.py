"""Complete preceding and selected public RTX 4090 entries."""


def make_attention(*, policy="selected"):
    from kohakufa_cute import api
    from kohakufa_cute.fa4_sm80 import rtx4090

    if policy == "selected":
        return rtx4090.scaled_dot_product_attention
    if policy != "previous":
        raise ValueError("Select the previous or selected public entry.")

    def previous(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
        if rtx4090.compatible(q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa):
            return rtx4090.native_runner(rtx4090.schedule(q.shape[-1], is_causal, q.shape[-2]))(
                q, k, v, scale=scale, causal=is_causal
            )
        return api.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal, scale=scale, enable_gqa=enable_gqa)

    return previous
