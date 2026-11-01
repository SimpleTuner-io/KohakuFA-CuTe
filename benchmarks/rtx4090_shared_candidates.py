"""Explicit native-host qualification of the full RTX 4090 shared schedules."""

import importlib
import sys
from dataclasses import replace
from pathlib import Path


def make_attention(source=None, dependencies=None, *, direct=False, forward_control="selected", causal_head_group=None):
    if forward_control not in ("selected", "stage2", "register-q"):
        raise ValueError("Select a bounded independent forward pipeline control.")
    if dependencies:
        sys.path.insert(0, str(Path(dependencies).resolve()))
    if source:
        package = importlib.import_module("flash_attn")
        path = str((Path(source) / "flash_attn").resolve())
        if path not in package.__path__:
            package.__path__.append(path)
    import torch

    from kohakufa_cute import api, fa4_sm80
    from kohakufa_cute.fa4_sm80 import rtx4090

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    if (
        (properties.major, properties.minor) != (8, 9)
        or properties.name != "NVIDIA GeForce RTX 4090"
        or properties.multi_processor_count != 128
        or properties.shared_memory_per_block_optin != 101376
        or not fa4_sm80.available()
    ):
        raise ValueError("The explicit native qualification requires the pinned full RTX 4090 runtime.")
    if not direct and forward_control == "selected" and causal_head_group is None:
        return rtx4090.scaled_dot_product_attention

    def selected(q, k, v, *, is_causal=False, scale=None, enable_gqa=False):
        actual_scale = q.shape[-1] ** -0.5 if scale is None else scale
        if enable_gqa or not 1.1754943508222875e-38 <= actual_scale <= 2e38:
            return api.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
        configuration = rtx4090.schedule(q.shape[-1], is_causal)
        if is_causal and causal_head_group is not None:
            configuration = replace(configuration, causal_head_group=causal_head_group)
        if q.shape[-1] == 128:
            if forward_control == "stage2":
                configuration = replace(configuration, forward_stages=2)
            elif forward_control == "register-q":
                configuration = replace(configuration, query_in_registers=True)
        return rtx4090.native_runner(configuration)(q, k, v, scale=actual_scale, causal=is_causal)

    return selected
