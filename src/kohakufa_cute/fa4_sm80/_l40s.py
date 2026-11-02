"""Full-device L40S schedule selection under the shared rounded arithmetic."""

import math
from functools import lru_cache

import torch


@lru_cache(maxsize=32)
def is_device(device):
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (8, 9):
        return False
    properties = torch.cuda.get_device_properties(device)
    return (
        properties.name == "NVIDIA L40S"
        and properties.multi_processor_count == 142
        and properties.shared_memory_per_block_optin == 101376
    )


def _cold_compatible(q, k, v, mask, dropout, causal, scale, gqa):
    from . import available

    if (
        not torch.is_grad_enabled()
        or not any(getattr(t, "requires_grad", False) for t in (q, k, v))
        or torch.compiler.is_compiling()
        or torch.are_deterministic_algorithms_enabled()
        or mask is not None
        or dropout != 0
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
        or q.shape[2] not in (1024, 4096)
        or (causal and q.shape[-1] == 64 and q.shape[2] != 4096)
        or q.shape[-1] not in (64, 128)
        or not all(t.is_contiguous() and t.data_ptr() % 16 == 0 for t in (q, k, v))
        or not is_device(q.device)
    ):
        return False
    if scale is not None and (
        not isinstance(scale, (float, int)) or not math.isfinite(scale) or not 1.1754943508222875e-38 <= scale <= 2e38
    ):
        return False
    return available()


@lru_cache(maxsize=1)
def _metadata_predicate():
    """Cache only the function handle; every tensor is checked on each call."""
    from kohakufa_cute.native import extension

    return extension().l40s_metadata_compatible


def compatible(q, k, v, mask, dropout, causal, scale, gqa):
    if (
        torch.compiler.is_compiling()
        or torch.are_deterministic_algorithms_enabled()
        or mask is not None
        or dropout != 0
    ):
        return False
    if scale is not None and (
        not isinstance(scale, (float, int)) or not math.isfinite(scale) or not 1.1754943508222875e-38 <= scale <= 2e38
    ):
        return False
    from . import available

    if not available():
        return False
    # Unsupported cold calls keep the preceding fallback without loading a
    # native extension merely to reject their metadata. The hot path checks all
    # tensor metadata in C++; only the immutable function handle is cached.
    if _metadata_predicate.cache_info().currsize == 0 and not _cold_compatible(
        q, k, v, mask, dropout, causal, scale, gqa
    ):
        return False
    # Equal-shape MHA remains eligible with either value of enable_gqa.
    return _metadata_predicate()(q, k, v, causal)


@lru_cache(maxsize=32)
def schedule(dimension, sequence, causal):
    from ._attention import Schedule

    narrow_query = dimension == 128 and not causal
    return Schedule(
        forward_m=128 if dimension == 64 or (sequence == 4096 and not causal) else 64,
        forward_n=32 if dimension == 128 and sequence == 4096 and not causal else 64,
        backward_m=32 if narrow_query else 64,
        backward_n=128 if dimension == 64 or narrow_query else 64,
        query_stages=2 if dimension == 64 else 1,
        dout_stages=1,
        score_m_warps=2 if dimension == 64 or narrow_query else 4,
        kv_m_warps=4 if dimension == 64 else 2,
        dq_m_warps=4 if dimension == 64 else 2,
        v_in_regs=not narrow_query,
        causal_head_group=(
            8
            if causal and dimension == 64 and sequence == 4096
            else 16 if causal and dimension == 128 and sequence in (1024, 4096) else 1
        ),
    )


def attention(q, k, v, *, causal=False, scale=None):
    from ._host import native_runner

    return native_runner(schedule(q.shape[-1], q.shape[2], causal))(q, k, v, causal=causal, scale=scale)
