"""Matched A40 public controls: native metadata or the qualified Python policy."""

import math
from functools import lru_cache
from types import FunctionType

import torch


@lru_cache(maxsize=32)
def _python_device(index):
    properties = torch.cuda.get_device_properties(index)
    return (
        (properties.major, properties.minor) == (8, 6)
        and properties.name == "NVIDIA A40"
        and properties.multi_processor_count == 84
        and properties.shared_memory_per_block_optin == 101376
    )


def python_compatible(q, k, v, mask=None, dropout=0.0, causal=False, scale=None, gqa=False):
    """Keep the preceding public predicate as an explicit benchmark control."""
    if (
        torch.compiler.is_compiling()
        or mask is not None
        or dropout != 0
        or gqa
        or torch.are_deterministic_algorithms_enabled()
        or getattr(q, "ndim", None) != 4
        or q.shape != k.shape
        or q.shape != v.shape
        or q.device.type != "cuda"
        or q.device != k.device
        or q.device != v.device
        or q.device.index != torch.cuda.current_device()
        or q.dtype not in (torch.float16, torch.bfloat16)
        or q.dtype != k.dtype
        or q.dtype != v.dtype
        or q.shape[0] != 1
        or q.shape[1] != 16
        or (q.shape[2] not in (1024, 4096) and not (q.shape[2] == 128 and q.shape[-1] == 256))
        or not (q.shape[-1] in (128, 256) or (q.shape[-1] == 64 and causal))
        or not all(t.is_contiguous() for t in (q, k, v))
        or any(t.data_ptr() % 16 for t in (q, k, v))
        or not _python_device(q.device.index)
    ):
        return False
    if scale is not None and (
        not isinstance(scale, (float, int)) or not math.isfinite(scale) or not 1.1754943508222875e-38 <= scale <= 2e38
    ):
        return False
    from kohakufa_cute.fa4_sm80 import available

    return available()


def make_attention(mode="native"):
    from kohakufa_cute.fa4_sm80 import a40

    if mode == "native":
        return a40.scaled_dot_product_attention
    if mode != "python":
        raise ValueError("Select the native metadata predicate or preceding Python control.")
    original = a40.scaled_dot_product_attention
    result = FunctionType(
        original.__code__,
        dict(original.__globals__, compatible=python_compatible),
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    result.__kwdefaults__ = original.__kwdefaults__
    return result
