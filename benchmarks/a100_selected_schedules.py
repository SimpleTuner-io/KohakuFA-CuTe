"""Explicit selected A100 schedule against the unchanged native control."""


def make_attention():
    from benchmarks.static_sm80_experiments import _select_native_control
    from kohakufa_cute import fa4_sm80

    if not fa4_sm80.available():
        raise ValueError("Requires the qualified optional stack.")
    _select_native_control()

    def selected(q, k, v, *, is_causal=False, scale=None, enable_gqa=False):
        if (
            q.shape != k.shape
            or q.shape != v.shape
            or q.shape[-1] not in (64, 128)
            or not all(t.is_contiguous() and t.data_ptr() % 16 == 0 for t in (q, k, v))
            or (scale is not None and scale <= 0)
        ):
            raise ValueError("Requires aligned square positive-scale D64/D128 MHA.")
        return fa4_sm80.attention(q, k, v, causal=is_causal, scale=scale)

    return selected
