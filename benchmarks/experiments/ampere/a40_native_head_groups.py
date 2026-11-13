"""Native causal work-order controls with unchanged arithmetic and layouts."""

from functools import lru_cache


def make_attention(*, head_group=1):
    from kohakufa_cute.fa4_sm80._attention import Schedule
    from kohakufa_cute.fa4_sm80._host import native_runner

    @lru_cache(maxsize=2)
    def schedule(dimension):
        if dimension not in (64, 128):
            raise ValueError("These work-order controls require D64 or D128.")
        return Schedule(
            forward_m=64,
            backward_n=128 if dimension == 64 else 64,
            query_stages=2 if dimension == 64 else 1,
            dout_stages=1,
            score_m_warps=2 if dimension == 64 else 4,
            kv_m_warps=4 if dimension == 64 else 2,
            dq_m_warps=4 if dimension == 64 else 2,
            v_in_regs=True,
            causal_head_group=head_group,
        )

    def operation(q, k, v, *, is_causal=False, scale=None):
        return native_runner(schedule(q.shape[-1]))(q, k, v, causal=is_causal, scale=scale)

    return operation
