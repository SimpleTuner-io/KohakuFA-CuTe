"""Explicit native-host qualification of the full A40 shared schedules."""

import importlib
import sys
from pathlib import Path


def make_attention(source=None, dependencies=None, *, direct=False, forward_m=64, host="native", head_group=None):
    if host not in ("native", "python") or forward_m not in (64, 128):
        raise ValueError("Select an explicit native/Python host and M64/M128 forward control.")
    if dependencies:
        sys.path.insert(0, str(Path(dependencies).resolve()))
    if source:
        package = importlib.import_module("flash_attn")
        path = str((Path(source) / "flash_attn").resolve())
        if path not in package.__path__:
            package.__path__.append(path)
    from dataclasses import replace
    from functools import lru_cache
    from types import FunctionType

    import torch

    from kohakufa_cute import api, fa4_sm80
    from kohakufa_cute.fa4_sm80 import a40

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    if (
        (properties.major, properties.minor) != (8, 6)
        or properties.name != "NVIDIA A40"
        or properties.multi_processor_count != 84
        or properties.shared_memory_per_block_optin != 101376
        or not fa4_sm80.available()
    ):
        raise ValueError("The explicit native qualification requires the pinned full A40 runtime.")

    @lru_cache(maxsize=32)
    def selected_schedule(dimension, causal, sequence):
        result = replace(a40.schedule(dimension, causal, sequence), forward_m=forward_m)
        return result if head_group is None else replace(result, causal_head_group=head_group)

    if not direct and head_group is not None:
        # Keep the exact public guard/entry bytecode while changing only the
        # cached schedule control; no global patch survives this factory.
        original = a40.scaled_dot_product_attention
        scope = dict(original.__globals__, schedule=selected_schedule)
        cloned = FunctionType(original.__code__, scope, original.__name__, original.__defaults__, original.__closure__)
        cloned.__kwdefaults__ = original.__kwdefaults__
        return cloned
    if not direct:
        return a40.scaled_dot_product_attention

    def selected(q, k, v, *, is_causal=False, scale=None, enable_gqa=False):
        actual_scale = q.shape[-1] ** -0.5 if scale is None else scale
        if enable_gqa or not 1.1754943508222875e-38 <= actual_scale <= 2e38:
            return api.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
        schedule = selected_schedule(q.shape[-1], is_causal, q.shape[2])
        if host == "python":
            from kohakufa_cute.fa4_sm80._attention import attention

            return attention(q, k, v, scale=actual_scale, causal=is_causal, schedule=schedule)
        return a40.native_runner(schedule)(q, k, v, scale=actual_scale, causal=is_causal)

    return selected
