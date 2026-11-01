"""GB10 causal CTA ordering controls; all arithmetic and geometry stay fixed."""

from dataclasses import replace
from functools import lru_cache


def make_attention(*, head_group=1, direct=False, variant="q2-register"):
    from kohakufa_cute.fa4_sm80 import gb10
    from kohakufa_cute.fa4_sm80._attention import Schedule
    from kohakufa_cute.fa4_sm80._host import native_runner

    @lru_cache(maxsize=12)
    def runner(sequence, heads, dtype):
        base = (
            Schedule(
                forward_m=64,
                backward_n=64,
                query_stages=2,
                dout_stages=1,
                score_m_warps=4,
                kv_m_warps=2,
                dq_m_warps=2,
                v_in_regs=True,
            )
            if direct
            else gb10.schedule(sequence, True, heads, dtype)
        )
        if variant not in ("q2-register", "q1-shared", "q2-shared"):
            raise ValueError("Select a declared pipeline/storage control.")
        if variant != "q2-register":
            base = replace(base, query_stages=1 if variant == "q1-shared" else 2, v_in_regs=False)
        return native_runner(replace(base, causal_head_group=head_group))

    def operation(q, k, v, *, is_causal=False, scale=None, enable_gqa=False):
        if not is_causal:
            raise ValueError("This control changes causal ordering only.")
        if not direct and not gb10.compatible(q, k, v, causal=True, scale=scale, gqa=enable_gqa):
            raise ValueError("Select a qualified public GB10 family for timing.")
        if q.shape != k.shape or q.shape != v.shape or q.shape[-1] != 128 or enable_gqa:
            raise ValueError("This study retains D128 square MHA.")
        return runner(q.shape[2], q.shape[1], q.dtype)(q, k, v, causal=True, scale=scale)

    return operation


def make_public(*, previous=False):
    if previous:
        from benchmarks.gb10_device_cache import previous_module

        return previous_module("controls/gb10_before_groups.py").scaled_dot_product_attention
    from kohakufa_cute.fa4_sm80.gb10 import scaled_dot_product_attention

    return scaled_dot_product_attention
