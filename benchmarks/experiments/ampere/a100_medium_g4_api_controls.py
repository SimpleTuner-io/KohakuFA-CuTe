"""Exact previous public body with the preceding G1 schedule, initialized once."""

import atexit
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from types import ModuleType

_owners = []


def _proof():
    target = os.environ.get("A100_MEDIUM_G4_API_PROOF")
    if target:
        Path(target).write_text(
            json.dumps(
                [{**row, "generic_cache_size": module._extension.cache_info().currsize} for module, row in _owners], indent=2
            )
            + "\n"
        )


atexit.register(_proof)


def make_previous():
    from kohakufa_cute import api, fa4_sm80

    name = "kohakufa_cute._medium_g4_previous"
    package = ModuleType(name)
    package.__package__ = name
    package.__path__ = []
    package.__file__ = __file__
    sys.modules[name] = package
    for child in ("_host", "_attention", "_private_api", "_l40s", "_sm120"):
        sys.modules[name + "." + child] = importlib.import_module("kohakufa_cute.fa4_sm80." + child)
    schedules = ModuleType(name + "._schedules")
    schedules.__package__ = name
    schedules.__file__ = str(Path(__file__).with_name("a100_medium_g1_schedules.py"))
    source = Path(schedules.__file__).read_bytes()
    exec(compile(source, schedules.__file__, "exec"), schedules.__dict__)
    sys.modules[schedules.__name__] = schedules
    entry = Path(fa4_sm80.__file__).read_bytes()
    exec(compile(entry, __file__, "exec"), package.__dict__)
    module = ModuleType("kohakufa_cute._medium_g4_previous_api")
    module.__package__ = "kohakufa_cute"
    module.__file__ = __file__
    sys.modules[module.__name__] = module
    code = Path(api.__file__).read_bytes()
    exec(compile(code, __file__, "exec"), module.__dict__)
    module.fa4_sm80 = package
    assert module.scaled_dot_product_attention.__code__.co_code == api.scaled_dot_product_attention.__code__.co_code
    _owners.append(
        (
            module,
            {
                "api_sha256": hashlib.sha256(code).hexdigest(),
                "entry_sha256": hashlib.sha256(entry).hexdigest(),
                "schedule_sha256": hashlib.sha256(source).hexdigest(),
                "initial_generic_cache_size": module._extension.cache_info().currsize,
            },
        )
    )
    return module.scaled_dot_product_attention
