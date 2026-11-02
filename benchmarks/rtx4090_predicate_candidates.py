"""Preserve the preceding Python predicate as a complete API control."""

import math

import torch


def python_compatible(q, k, v, mask=None, dropout=0.0, causal=False, scale=None, gqa=False):
    """Bound the opt-in path to complete devices and measured shape families."""
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
        or q.shape[2] not in (1024, 4096)
        or q.shape[-1] not in (64, 128)
        or not all(t.is_contiguous() for t in (q, k, v))
        or any(t.data_ptr() % 16 for t in (q, k, v))
        or torch.cuda.get_device_capability(q.device) != (8, 9)
    ):
        return False
    properties = torch.cuda.get_device_properties(q.device)
    if (
        properties.name != "NVIDIA GeForce RTX 4090"
        or properties.multi_processor_count != 128
        or properties.shared_memory_per_block_optin != 101376
    ):
        return False
    if scale is not None and (
        not isinstance(scale, (float, int)) or not math.isfinite(scale) or not 1.1754943508222875e-38 <= scale <= 2e38
    ):
        return False
    from kohakufa_cute.fa4_sm80 import available

    return available()


def make_attention(source=None, dependencies=None, *, policy="native"):
    if policy not in ("native", "python"):
        raise ValueError("Select the native or preceding Python metadata predicate.")
    from benchmarks.rtx4090_shared_candidates import make_attention as qualified

    qualified(source, dependencies)
    from kohakufa_cute import api
    from kohakufa_cute.fa4_sm80 import rtx4090

    if policy == "native":
        return rtx4090.scaled_dot_product_attention

    def previous(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
        if python_compatible(q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa):
            return rtx4090.native_runner(rtx4090.schedule(q.shape[-1], is_causal, q.shape[-2]))(
                q, k, v, scale=scale, causal=is_causal
            )
        return api.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal, scale=scale, enable_gqa=enable_gqa)

    return previous
