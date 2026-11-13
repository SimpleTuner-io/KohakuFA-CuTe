"""Explicit GB10 schedules using the shared shifted warp-MMA implementation."""

from functools import lru_cache


def configuration(
    dimension, sequence, causal, *, forward_m=64, dout_stages=1, query_stages=2, dense_m32=False, stage1_shared_v=False
):
    """Retain the earlier GB10 geometry while changing one stage independently."""
    options = dict(
        forward_m=forward_m,
        backward_m=64,
        backward_n=128 if dimension == 64 and sequence >= 4096 and not causal else 64,
        query_stages=query_stages,
        dout_stages=dout_stages,
        threads=256,
        score_m_warps=2 if dimension == 64 else 4,
        kv_m_warps=4 if dimension == 64 else 2,
        dq_m_warps=4 if dimension == 64 else 2,
        v_in_regs=True,
    )
    if stage1_shared_v and dimension == 128 and not causal:
        options.update(query_stages=1, dout_stages=1, v_in_regs=False)
    if dense_m32 and dimension == 128 and not causal:
        options.update(
            backward_m=32,
            backward_n=128,
            query_stages=1,
            dout_stages=1,
            score_m_warps=2,
            kv_m_warps=2,
            dq_m_warps=2,
            v_in_regs=False,
        )
    return options


def make_attention(*, host="native", forward_m=64, dout_stages=1, query_stages=2, dense_m32=False, stage1_shared_v=False):
    """Return the complete explicit call; production dispatch stays unchanged."""
    import torch

    from kohakufa_cute.fa4_sm80 import available
    from kohakufa_cute.fa4_sm80._attention import Schedule, attention
    from kohakufa_cute.native import extension

    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor, properties.multi_processor_count) != (12, 1, 48) or not available():
        raise RuntimeError("This experiment requires the full GB10 and qualified optional runtime.")
    if host not in ("native", "python") or forward_m not in (64, 128):
        raise ValueError("Select native/python preparation and forward M64/M128.")
    if not 1 <= dout_stages <= query_stages <= 2:
        raise ValueError("Select one or two Q/dO stages, with dO stages no greater than Q stages.")

    @lru_cache(maxsize=8)
    def runner(dimension, sequence, causal):
        options = configuration(
            dimension,
            sequence,
            causal,
            forward_m=forward_m,
            dout_stages=dout_stages,
            query_stages=query_stages,
            dense_m32=dense_m32,
            stage1_shared_v=stage1_shared_v,
        )
        return extension().AttentionRunner("warp", options) if host == "native" else Schedule(**options)

    def selected(q, k, v, *, is_causal=False, scale=None, enable_gqa=False):
        if enable_gqa or q.shape != k.shape or q.shape != v.shape or q.shape[-1] not in (64, 128):
            raise ValueError("The explicit GB10 schedule requires square D64/D128 MHA.")
        selected_runner = runner(q.shape[-1], q.shape[2], is_causal)
        if host == "native":
            return selected_runner(q, k, v, scale=scale, causal=is_causal)
        if scale is not None and scale <= 0:
            raise ValueError("Signed and zero scales retain native production fallback.")
        return attention(q, k, v, causal=is_causal, scale=scale, schedule=selected_runner)

    return selected


def make_public_attention():
    """Return the bounded opt-in entry for complete API timing and qualification."""
    from kohakufa_cute.fa4_sm80.gb10 import scaled_dot_product_attention

    return scaled_dot_product_attention
