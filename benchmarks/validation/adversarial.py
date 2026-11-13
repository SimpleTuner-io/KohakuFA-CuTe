"""Finite-input regressions derived from NVIDIA cuDNN SDPA reports.

Sources: cudnn-frontend issues 160/1072 and pull request 628. References use
the identical rounded inputs, rather than cuDNN's rounded saved output/LSE.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Case:
    name: str
    queries: int = 1
    keys: int = 128
    dim: int = 64
    shift: float = -128.0
    heads: int = 2
    kvheads: int = 2
    scale: float | None = None
    causal: bool = False
    mask: str = "none"
    layout: str = "sbhd"
    literal: bool = False


def cases(extended=True):
    result = [
        Case("issue160", queries=2, keys=2, dim=8, heads=1, kvheads=1, literal=True),
        Case("issue1072"),
        Case("zero_shift_control", shift=0),
        Case("key97_control", keys=97),
    ]
    result.extend(Case(f"kv_tail_{length}", queries=65, keys=length) for length in (63, 64, 65, 127, 129, 191, 192, 193))
    if not extended:
        return result
    result.extend(
        Case(f"negative_lse_square_{length}", queries=length, keys=length) for length in (64, 192, 1000, 1216, 1280)
    )
    result.extend(
        Case(f"single_query_{length}", keys=length) for length in (1, 63, 64, 65, 127, 129, 192, 193, 255, 256, 257)
    )
    result.extend(Case(f"shift_{abs(shift)}", queries=65, keys=192, shift=shift) for shift in (-64, -96, -512, -1024, 128))
    result.extend(Case(f"head_dim_{dim}", queries=33, keys=129, dim=dim) for dim in (80, 128, 256, 512))
    result.extend(
        [
            Case("negative_scale", queries=65, keys=193, scale=-0.125),
            Case("zero_scale", queries=65, keys=193, scale=0),
            Case("causal_tail", queries=65, keys=193, causal=True),
            Case("single_query_causal", causal=True),
            Case("gqa_tail", queries=65, keys=193, heads=4),
            Case("bhsd_tail", queries=65, keys=193, layout="bhsd"),
            Case("padding_mask", queries=65, keys=193, mask="padding"),
            Case("fully_masked_row", queries=65, keys=193, mask="empty_row"),
            Case("fully_masked_tile", queries=65, keys=193, mask="late_keys"),
        ]
    )
    return result


def make_case(case, *, device, dtype, seed=123):
    generator = torch.Generator(device=device).manual_seed(seed)

    def tensor(length, heads):
        shape = (length, 1, heads, case.dim) if case.layout == "sbhd" else (1, heads, length, case.dim)
        result = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        return result.permute(1, 2, 0, 3) if case.layout == "sbhd" else result

    q, k, v = tensor(case.queries, case.heads), tensor(case.keys, case.kvheads), tensor(case.keys, case.kvheads)
    dout = tensor(case.queries, case.heads) * 0.05
    if case.literal:
        # Rounded numeric fixture from NVIDIA/cudnn-frontend#160.
        values = (
            [
                [-1.7109, 0.7227, -1.5234, -2.1250, 2.4844, 0.4277, 2.4844, 0.3008],
                [-0.9023, 1.1094, -1.6094, -1.8359, -4.2500, 1.4531, 1.3281, 1.1172],
            ],
            [
                [-0.6367, 0.3945, -0.8867, -0.6250, 70.0000, 1.2109, 1.2969, -0.1934],
                [-0.7031, 1.2031, -0.7383, -0.7188, 70.5000, 1.9297, 0.7891, 1.2500],
            ],
            [
                [0.7422, 0.1875, -0.5664, 0.2383, -0.5195, 0.3789, 0.1260, -1.3047],
                [-0.7031, -0.5859, 0.3613, -0.2852, 0.1729, 0.4023, 1.0312, -1.0000],
            ],
            [
                [3.8126e-9, -1.3970e-8, 5.7335e-9, -1.0303e-8, -4.9477e-9, 1.3388e-8, -1.1292e-8, -1.4115e-9],
                [-1.3504e-8, -1.0594e-8, 1.2282e-8, 1.4144e-8, 5.0932e-9, 1.3853e-8, 1.4727e-8, 2.0606e-8],
            ],
        )
        for target, value in zip((q, k, v, dout), values):
            target.copy_(torch.tensor(value, device=device, dtype=dtype)[None, None])
    else:
        q[..., 0] = case.dim**0.5
        k[..., 0] = -case.shift if case.scale is not None and case.scale < 0 else case.shift
    mask = None
    if case.mask != "none":
        mask = torch.ones(case.queries, case.keys, dtype=torch.bool, device=device)
        if case.mask == "padding":
            mask[:, -1] = False
        elif case.mask == "empty_row":
            mask[0] = False
        elif case.mask == "late_keys":
            mask[:, :128] = False
        else:
            raise ValueError("Unknown mask fixture")
    return (q, k, v), dout, mask


@torch.no_grad()
def logits_statistics(inputs, mask=None, *, causal=False, scale=None):
    q, k, _ = inputs
    keys = k.double().repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    scores = q.double() @ keys.transpose(-1, -2)
    scores *= q.shape[-1] ** -0.5 if scale is None else scale
    if mask is not None:
        scores.masked_fill_(~mask, -torch.inf)
    if causal:
        visible = torch.arange(k.shape[2], device=q.device)[None] <= torch.arange(q.shape[2], device=q.device)[:, None]
        scores.masked_fill_(~visible, -torch.inf)
    maxima, lse = scores.amax(-1), scores.logsumexp(-1)
    valid = torch.isfinite(lse)
    return {
        "row_max_min": float(maxima[valid].min()),
        "row_max_max": float(maxima[valid].max()),
        "lse_min": float(lse[valid].min()),
        "lse_max": float(lse[valid].max()),
        "fully_masked_rows": int((~valid).sum()),
    }
