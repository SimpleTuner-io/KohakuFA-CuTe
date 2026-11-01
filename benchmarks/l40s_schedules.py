"""Direct controls for the bounded public L40S schedule and preceding dispatch."""

import math


def make_attention(*, native=True, forward_m=None):
    import torch

    from kohakufa_cute import api, fa4_sm80
    from kohakufa_cute.fa4_sm80._attention import attention
    from kohakufa_cute.fa4_sm80._host import native_runner
    from kohakufa_cute.fa4_sm80._l40s import schedule

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    if (
        properties.name != "NVIDIA L40S"
        or properties.multi_processor_count != 142
        or properties.shared_memory_per_block_optin != 101376
        or not fa4_sm80.available()
    ):
        raise ValueError("These controls require the qualified full-device L40S runtime.")

    def operation(q, k, v, *, is_causal=False, scale=None, enable_gqa=False):
        if scale is not None and (not math.isfinite(scale) or scale < 2**-126):
            return api.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError("These controls require square MHA.")
        selected = schedule(q.shape[-1], q.shape[2], is_causal)
        if forward_m is not None:
            from dataclasses import replace

            selected = replace(selected, forward_m=forward_m)
        if native:
            return native_runner(selected)(q, k, v, causal=is_causal, scale=scale)
        return attention(q, k, v, causal=is_causal, scale=scale, schedule=selected)

    return operation


def make_previous():
    from kohakufa_cute.native import extension

    module = extension()

    def operation(q, k, v, *, is_causal=False, scale=None, enable_gqa=False):
        return module.attention(q, k, v, None, scale, is_causal)

    return operation
