"""Explicit configured native host calls; public dispatch remains separate."""

import atexit
from unittest.mock import patch


def make_attention(configuration=None, family="warp"):
    from kohakufa_cute.native import extension

    runner = extension().AttentionRunner(family, dict(configuration or {}))
    if family == "warp":
        from kohakufa_cute import fa4_sm80 as backend
    elif family == "hopper":
        from kohakufa_cute import fa4_hopper as backend
    else:
        raise ValueError("Select the warp or hopper kernel family.")
    control = patch.object(backend, "compatible", return_value=False)
    control.start()
    atexit.register(control.stop)

    def selected(q, k, v, *, is_causal=False, scale=None, enable_gqa=False):
        return runner(q, k, v, scale=scale, causal=is_causal)

    return selected
