"""Joint SM103 backward with separate unscaled normalization statistics."""

from functools import lru_cache

import torch

from ._private_api import make_private_api


@lru_cache(maxsize=1)
def _api():
    from ._backward import FlashAttentionBackwardSm100
    from ._forward import FlashAttentionForwardSm100
    from ._preprocess import FlashAttentionBackwardPreprocess

    return make_private_api(
        FlashAttentionForwardSm100=FlashAttentionForwardSm100,
        FlashAttentionBackwardSm100=FlashAttentionBackwardSm100,
        FlashAttentionBackwardPreprocess=FlashAttentionBackwardPreprocess,
    )


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, scale):
        maximum = torch.empty((*q.shape[:-2], ((q.shape[-2] + 511) // 512) * 512), dtype=torch.float32, device=q.device)
        out, logsum, *_ = _api()._flash_attn_fwd(
            *(t.transpose(1, 2) for t in (q, k, v)),
            softmax_scale=scale,
            causal=causal,
            return_lse=True,
            aux_tensors=[maximum],
        )
        out = out.transpose(1, 2)
        ctx.save_for_backward(q, k, v, out, maximum, logsum)
        ctx.causal, ctx.scale = causal, scale
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, maximum, logsum = ctx.saved_tensors
        gradients = _api()._flash_attn_bwd(
            *(t.transpose(1, 2) for t in (q, k, v, out, dout)),
            logsum,
            ctx.scale,
            ctx.causal,
            0.0,
            aux_tensors=[maximum],
        )
        return *(g.transpose(1, 2) for g in gradients[:3]), None, None


def attention(q, k, v, *, causal=False, scale=None):
    return _Attention.apply(q, k, v, causal, q.shape[-1] ** -0.5 if scale is None else scale)
