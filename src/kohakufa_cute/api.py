"""Strict CuTe attention and automatic per-call native SDPA selection."""

from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

import torch
from packaging.version import Version

from . import _ops, fa4_sm80

_CAPABILITIES = {(8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (10, 3), (12, 0), (12, 1)}
_CENTERED_CAPABILITIES = {(8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (10, 3), (12, 0), (12, 1)}
_native_sdpa = torch.nn.functional.scaled_dot_product_attention


@lru_cache(maxsize=1)
def _extension():
    from .native import extension

    return extension()


@torch.compiler.assume_constant_result
def _runtime_available():
    for package, minimum in (
        ("nvidia-cutlass-dsl", "4.8"),
        ("apache-tvm-ffi", "0.1.14"),
        ("ninja", "1.11"),
        ("triton", "3.6"),
    ):
        try:
            installed = version(package)
        except PackageNotFoundError:
            return False
        if Version(installed) < Version(minimum):
            return False
    return True


def _compatible(q, k, v, mask, dropout, causal, gqa):
    return (
        dropout == 0.0
        and all(t.ndim == 4 for t in (q, k, v))
        and q.device.type == "cuda"
        and k.device == q.device == v.device
        and q.dtype in (torch.float16, torch.bfloat16)
        and k.dtype == q.dtype == v.dtype
        and all(q.shape)
        and all(k.shape)
        and k.shape == v.shape
        and q.shape[0] == k.shape[0]
        and q.shape[3] == k.shape[3]
        and q.shape[3] <= 512
        and q.shape[1] % k.shape[1] == 0
        and (gqa or q.shape[1] == k.shape[1])
        and (mask is None or (mask.dtype == torch.bool and mask.device == q.device and not causal))
        and torch.cuda.get_device_capability(q.device) in _CENTERED_CAPABILITIES
    )


def _validate(q, k, v, mask, dropout, causal, gqa, *, combined_mask=False):
    if dropout != 0.0:
        raise ValueError("CuTe attention does not support dropout.")
    if any(t.ndim != 4 for t in (q, k, v)):
        raise ValueError("CuTe attention requires 4D [batch, heads, sequence, dim] Q/K/V.")
    if q.device.type != "cuda" or k.device != q.device or v.device != q.device:
        raise ValueError("CuTe attention requires Q/K/V on the same CUDA device.")
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("CuTe attention requires matching FP16/BF16 Q/K/V; FP32 attention is unsupported.")
    if not all(q.shape) or not all(k.shape) or k.shape != v.shape or q.shape[0] != k.shape[0] or q.shape[3] != k.shape[3]:
        raise ValueError("CuTe attention requires nonempty matching batch/head dimensions and equal K/V shapes.")
    if q.shape[1] % k.shape[1] or not 1 <= q.shape[3] <= 512:
        raise ValueError("CuTe attention requires head dimensions 1..512 and query heads divisible by K/V heads.")
    if not gqa and q.shape[1] != k.shape[1]:
        raise ValueError("CuTe SDPA requires enable_gqa=True for differing Q and K/V head counts.")
    if mask is not None:
        if mask.dtype != torch.bool or mask.device != q.device:
            raise ValueError("CuTe attention requires a boolean mask on the Q/K/V device.")
        if causal and not combined_mask:
            raise ValueError("CuTe attention requires causal visibility folded into the boolean mask.")
    if torch.cuda.get_device_capability(q.device) not in _CAPABILITIES:
        raise ValueError("CuTe attention requires sm_80, sm_86, sm_89, sm_90, sm_100, sm_103, sm_120 or sm_121.")


def _center_keys(k):
    """Subtract each batch/head's mean key in FP32 and round once to the input dtype."""
    keys = k.float()
    return (keys - keys.mean(dim=-2, keepdim=True)).to(k.dtype)


def _blackwell_attention(q, k, v, mask, dropout, causal, scale, gqa):
    if torch.compiler.is_compiling() or q.device.type != "cuda":
        return None
    from . import fa4_blackwell

    if fa4_blackwell.compatible(q, k, v, mask, dropout, causal, scale, gqa):
        return fa4_blackwell.run_attention(q, k, v, causal=causal, scale=scale)
    return None


def attention(q, k, v, *, mask=None, causal=False, scale=None, center_keys=False):
    """Strict CUDA attention; Q/K/V have shape [B, H, S, D], with implicit GQA.

    ``center_keys`` subtracts the mean key per batch and K/V head in FP32 before
    rounding back to the input dtype. Softmax is invariant to this shared logit
    shift in exact arithmetic, including with masks. Autograd differentiates
    through the mean, projecting the key gradient back to the original keys.
    """
    if center_keys or torch.compiler.is_compiling() or _extension.cache_info().currsize == 0:
        _validate(q, k, v, mask, 0.0, causal, True, combined_mask=True)
    if center_keys:
        k = _center_keys(k)
    if torch.compiler.is_compiling():
        return _ops.attention(q, k, v, mask=mask, causal=causal, scale=scale)
    if fa4_sm80.compatible(q, k, v, mask, 0, causal, scale, True):
        return fa4_sm80.attention(q, k, v, causal=causal, scale=scale)
    selected = _blackwell_attention(q, k, v, mask, 0.0, causal, scale, True)
    if selected is not None:
        return selected
    return _extension().attention(q, k, v, mask, scale, causal)


def scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
):
    """Strict CuTe implementation with the PyTorch SDPA argument convention."""
    if torch.compiler.is_compiling() or _extension.cache_info().currsize == 0:
        _validate(query, key, value, attn_mask, dropout_p, is_causal, enable_gqa)
        if torch.compiler.is_compiling():
            return _ops.attention(query, key, value, mask=attn_mask, causal=is_causal, scale=scale)
    if fa4_sm80.compatible(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa):
        return fa4_sm80.attention(query, key, value, causal=is_causal, scale=scale)
    selected = _blackwell_attention(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)
    if selected is not None:
        return selected
    return _extension().sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)


def automatic_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
):
    """Select centered CuTe kernels on supported GPUs, native SDPA for unsupported calls."""
    if query.device.type == "cuda":
        if not torch.compiler.is_compiling() and fa4_sm80.compatible(
            query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
        ):
            return fa4_sm80.attention(query, key, value, causal=is_causal, scale=scale)
        selected = _blackwell_attention(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)
        if selected is not None:
            return selected
        if not torch.compiler.is_compiling() and _extension.cache_info().currsize:
            return _extension().automatic_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)
        if _compatible(query, key, value, attn_mask, dropout_p, is_causal, enable_gqa) and _runtime_available():
            return scaled_dot_product_attention(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)
    return _native_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale=scale, enable_gqa=enable_gqa)
