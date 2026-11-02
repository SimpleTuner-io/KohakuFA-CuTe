"""Isolate one causal head-order field behind exact public API bodies."""

import atexit
import hashlib
import importlib
import json
import os
import sys
from dataclasses import asdict, replace
from functools import lru_cache
from pathlib import Path
from types import ModuleType

_modules = []


def _proof():
    destination = os.environ.get("A100_MEDIUM_GROUP_API_PROOF")
    if destination:
        Path(destination).write_text(
            json.dumps(
                [{**record, "generic_cache_size": module._extension.cache_info().currsize} for module, record in _modules],
                indent=2,
            )
            + "\n"
        )


atexit.register(_proof)


def schedule(group):
    from kohakufa_cute.fa4_sm80._schedules import a100_schedule

    if group not in (1, 4, 8):
        raise ValueError("This bounded study accepts groups1/4/8 only.")
    return replace(a100_schedule(128, 2048, True, heads=8), causal_head_group=group)


def make_attention(*, group=1):
    from kohakufa_cute import api, fa4_sm80
    from kohakufa_cute.fa4_sm80 import _schedules

    selected = schedule(group)
    package_name = "kohakufa_cute._medium_group_" + str(group)
    package = ModuleType(package_name)
    package.__package__ = package_name
    package.__path__ = []
    package.__file__ = __file__
    sys.modules[package_name] = package
    for name in ("_host", "_attention", "_private_api", "_l40s", "_sm120"):
        sys.modules[package_name + "." + name] = importlib.import_module("kohakufa_cute.fa4_sm80." + name)
    schedules = ModuleType(package_name + "._schedules")
    schedules.__package__ = package_name
    schedules.__file__ = __file__

    @lru_cache(maxsize=32)
    def policy(dimension, sequence, causal, heads=16):
        if dimension == 128 and sequence == 2048 and causal and heads == 8:
            return selected
        return _schedules.a100_schedule(dimension, sequence, causal, heads=heads)

    schedules.a100_schedule = policy
    sys.modules[schedules.__name__] = schedules
    entry_source, api_source = Path(fa4_sm80.__file__).read_bytes(), Path(api.__file__).read_bytes()
    exec(compile(entry_source, __file__, "exec"), package.__dict__)
    module = ModuleType("kohakufa_cute._medium_group_api_" + str(group))
    module.__package__ = "kohakufa_cute"
    module.__file__ = __file__
    sys.modules[module.__name__] = module
    exec(compile(api_source, __file__, "exec"), module.__dict__)
    module.fa4_sm80 = package
    assert module.scaled_dot_product_attention.__code__.co_code == api.scaled_dot_product_attention.__code__.co_code
    _modules.append(
        (
            module,
            {
                "group": group,
                "schedule": asdict(selected),
                "api_sha256": hashlib.sha256(api_source).hexdigest(),
                "entry_sha256": hashlib.sha256(entry_source).hexdigest(),
                "factory_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "initial_generic_cache_size": module._extension.cache_info().currsize,
            },
        )
    )
    return module.scaled_dot_product_attention


def make_direct(*, group=1):
    """Exercise the same schedule on explicit partial tiles outside dispatch."""
    from kohakufa_cute.fa4_sm80._host import native_runner

    runner = native_runner(schedule(group))

    def attention(q, k, v, *, is_causal=False, scale=None):
        assert q.shape[0:2] == (1, 8) and q.shape[-1] == 128 and is_causal
        return runner(q, k, v, causal=True, scale=scale)

    return attention
