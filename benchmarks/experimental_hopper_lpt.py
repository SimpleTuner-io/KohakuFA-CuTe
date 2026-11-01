"""Unqualified causal head-interleaving experiment; never selected by default.

FA3 selects SingleTileBwdLPTScheduler for ordinary causal backward, while the
pinned FA4 SM90 entry selects its equivalent only in deterministic mode. This
experiment changes that scheduler selection, retaining deterministic=False,
the existing atomic accumulation and the same GPU mainloop and conversions.

Use make_control_attention and make_attention as a paired comparison. Both
include the same maximum padding and Python preparation as static_hopper.
"""

import math
from functools import lru_cache


def head_swizzle(sequence, head_dim, value_dim, element_size=2, l2_bytes=50 << 20):
    """CPU model of the pinned scheduler's per-head L2 working-set estimate."""
    size_one_head = sequence * ((head_dim + value_dim) * element_size + head_dim * 4)
    capacity = max(1, l2_bytes // size_one_head)
    return 1 << (capacity.bit_length() - 1)


def decode_work(index, key_tiles, heads, batches, swizzle):
    """CPU model of SingleTileLPTBwdScheduler, with its SPT flag false.

    This models the logical mapping, not CUDA's admission order or concurrency.
    The residual-head group uses its actual width, as the deployed scheduler does.
    """
    if not 0 <= index < key_tiles * heads * batches:
        raise ValueError("The physical CTA index is outside this grid.")
    group, residual = divmod(index, swizzle * key_tiles)
    full_groups, remainder = divmod(heads * batches, swizzle)
    width = swizzle if group < full_groups else remainder
    key_tile, head_in_group = divmod(residual, width)
    batch, head = divmod(group * swizzle + head_in_group, heads)
    return key_tile, head, batch


@lru_cache(maxsize=16)
def _api(persistent, interleave, ordinary_stores=False, separate_masking=False):
    from kohakufa_cute.fa4_hopper import _attention
    from kohakufa_cute.fa4_hopper._backward import FlashAttentionBackwardSm90
    from kohakufa_cute.fa4_hopper._private_api import make_private_api

    class CausalHeadInterleavedBackward(FlashAttentionBackwardSm90):
        causal_head_interleave = True

    backward = CausalHeadInterleavedBackward if interleave else FlashAttentionBackwardSm90
    if ordinary_stores:
        from benchmarks.hopper_ordinary_gradient_stores import OrdinaryGradientStores

        class OrdinaryBackward(OrdinaryGradientStores, backward):
            pass

        backward = OrdinaryBackward
    if separate_masking:

        class SeparateMaskingBackward(backward):
            separate_causal_masking = True

        backward = SeparateMaskingBackward
    control = _attention._api(persistent)
    return make_private_api(
        FlashAttentionForwardSm90=control.FlashAttentionForwardSm90,
        FlashAttentionBackwardSm90=backward,
        FlashAttentionBackwardPreprocess=control.FlashAttentionBackwardPreprocess,
        _tile_size_bwd_sm90=control._tile_size_bwd_sm90,
    )


def _make_attention(interleave, ordinary_stores=False, separate_masking=False):
    import torch

    from kohakufa_cute import fa4_hopper

    if not fa4_hopper.available():
        raise RuntimeError("This experiment requires the pinned optional Hopper runtime.")

    class Attention(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, causal, scale):
            persistent = not causal and q.shape[-1] == 128 and q.shape[-2] == 4096
            module = _api(persistent, interleave, ordinary_stores, separate_masking)
            maximum = torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
            output = torch.empty_like(q)
            _, logsum, *_ = module._flash_attn_fwd(
                *(tensor.transpose(1, 2) for tensor in (q, k, v)),
                softmax_scale=scale,
                causal=causal,
                pack_gqa=False,
                out=output.transpose(1, 2),
                return_lse=True,
                aux_tensors=[maximum],
            )
            ctx.save_for_backward(q, k, v, output, maximum, logsum)
            ctx.module, ctx.causal, ctx.scale = module, causal, scale
            return output

        @staticmethod
        def backward(ctx, dout):
            q, k, v, out, maximum, logsum = ctx.saved_tensors
            padding = (-maximum.shape[-1]) % 640
            if padding:
                maximum = torch.nn.functional.pad(maximum, (0, padding))
            gradients = ctx.module._flash_attn_bwd(
                *(tensor.transpose(1, 2) for tensor in (q, k, v, out, dout)),
                logsum,
                ctx.scale,
                ctx.causal,
                0.0,
                aux_tensors=[maximum],
            )
            return *(gradient.transpose(1, 2) for gradient in gradients[:3]), None, None

    def attention(q, k, v, *, is_causal=False, scale=None, enable_gqa=False):
        selected = q.shape[-1] ** -0.5 if scale is None else scale
        if not isinstance(selected, (int, float)) or not math.isfinite(selected):
            raise ValueError("The Hopper experiment requires a finite numeric scale.")
        if selected <= 0:
            from kohakufa_cute import native

            return native.extension().sdpa(q, k, v, None, 0.0, is_causal, scale, enable_gqa)
        if not torch.finfo(torch.float32).tiny <= selected <= 2e38:
            raise ValueError("The positive Hopper experiment requires a normal FP32 scale.")
        if (
            torch.cuda.get_device_capability(q.device) != (9, 0)
            or q.shape != k.shape
            or k.shape != v.shape
            or q.shape[-1] not in (64, 128)
            or q.dtype not in (torch.float16, torch.bfloat16)
            or not all(tensor.is_contiguous() for tensor in (q, k, v))
            or enable_gqa
        ):
            raise ValueError("The scheduler experiment requires contiguous square Hopper MHA D64/D128.")
        return Attention.apply(q, k, v, is_causal, selected)

    return attention


def make_attention():
    return _make_attention(True)


def make_control_attention():
    return _make_attention(False)


def make_ordinary_attention():
    """Independent producer-instruction control, requiring separate math gates."""
    return _make_attention(True, ordinary_stores=True)


def make_separate_masking_attention():
    """D64 causal diagonal and steady loops, retaining the interleaved schedule."""
    return _make_attention(True, separate_masking=True)
