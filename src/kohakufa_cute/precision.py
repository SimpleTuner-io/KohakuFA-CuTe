"""Precision policy shared by the qualified warp-MMA device schedules."""


def schedule(program, dimension, *, device=None):
    if dimension in (16, 64, 128):
        from .weight_precision import schedule as generic_schedule

        return generic_schedule(program, dimension)
    if dimension not in (256, 512):
        raise ValueError("Compensated weights require a separately qualified head dimension.")

    from .kernels import Attention

    if dimension == 256:
        from .medium_backward import MediumKeyValue as KeyValue
    else:
        from .wide_backward import WideKeyValue as KeyValue

    options = {}
    if dimension == 512 and device is not None:
        import torch

        if (
            torch.cuda.get_device_capability(device) == (12, 0)
            and torch.cuda.get_device_name(device) == "NVIDIA GeForce RTX 5090"
        ):
            options["packed_bf16_causal"] = True

    memo = {}

    def visit(owner):
        identity = id(owner)
        if identity in memo:
            return memo[identity]
        memo[identity] = owner
        if type(owner) is Attention and owner.mode == "kv" and not owner.masked:
            assert owner.centered
            replacement = KeyValue(owner.causal, vector=owner.vector, copy_cache_global=owner.copy_cache_global, **options)
            memo[identity] = replacement
            return replacement
        for attribute in (
            "keyvalue",
            "fallback",
            "tuned",
            "short",
            "compact",
            "checked",
            "complete",
            "precise",
            "medium",
            "long",
        ):
            child = getattr(owner, attribute, None)
            if child is not None:
                setattr(owner, attribute, visit(child))
        return owner

    return visit(program)
