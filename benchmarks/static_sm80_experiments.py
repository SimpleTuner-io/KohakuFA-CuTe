"""Explicit static schedule and unchanged native control for qualification."""

import atexit
from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def native_control():
    """Keep the existing CuTe backend as the baseline in a matched process."""
    from kohakufa_cute import fa4_sm80

    with patch.object(fa4_sm80, "compatible", return_value=False):
        yield


def _select_native_control():
    """Keep native CuTe as the shared driver's baseline for this process."""
    from kohakufa_cute import fa4_sm80

    control = patch.object(fa4_sm80, "compatible", return_value=False)
    control.start()
    atexit.register(control.stop)


def make_attention():
    """Return the selected static schedule with no SDPA fallback."""
    import torch

    from kohakufa_cute.fa4_sm80 import available
    from kohakufa_cute.fa4_sm80._attention import attention

    if torch.cuda.get_device_capability() != (8, 0) or not available():
        raise ValueError("The explicit static schedule requires the qualified SM80 runtime.")
    _select_native_control()

    def selected(q, k, v, *, is_causal=False, scale=None, enable_gqa=False):
        if (
            q.shape != k.shape
            or q.shape != v.shape
            or q.shape[-1] not in (64, 128)
            or q.dtype not in (torch.bfloat16, torch.float16)
            or not all(t.is_contiguous() for t in (q, k, v))
            or (scale is not None and scale <= 0)
        ):
            raise ValueError("The static comparison requires square contiguous positive-scale FP16/BF16 MHA.")
        return attention(q, k, v, causal=is_causal, scale=scale)

    return selected
