"""Explicit flat-host SM80 comparison; automatic dispatch is unchanged."""


def attention(q, k, v, *, is_causal=False, scale=None, enable_gqa=False):
    from kohakufa_cute.native import extension

    return extension().sm80_attention(q, k, v, q.shape[-1] ** -0.5 if scale is None else scale, is_causal)


def make_attention():
    from benchmarks.static_sm80_experiments import _select_native_control
    from kohakufa_cute.fa4_sm80 import available

    if not available():
        raise ValueError("The flat-host comparison requires the qualified optional SM80 runtime.")
    _select_native_control()
    return attention
