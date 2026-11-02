"""Exact public API controls with isolated old/new schedule module sources."""

import atexit
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from types import ModuleType

_modules = []


def _write_proof():
    destination = os.environ.get("A100_SHARED_API_PROOF")
    if destination:
        Path(destination).write_text(
            json.dumps(
                [
                    {**record, "actual_generic_cache_size": module._extension.cache_info().currsize}
                    for module, record in _modules
                ],
                indent=2,
            )
            + "\n"
        )


atexit.register(_write_proof)


def make_attention(*, previous=False):
    from kohakufa_cute import api, fa4_sm80
    from kohakufa_cute.fa4_sm80 import _schedules

    folder = Path(__file__).parent
    schedule_source = (
        (folder / "a100_shared_previous_schedules.py").read_bytes() if previous else Path(_schedules.__file__).read_bytes()
    )
    entry_source = (
        (folder / "a100_shared_previous_entry.py").read_bytes() if previous else Path(fa4_sm80.__file__).read_bytes()
    )
    suffix = "previous" if previous else "candidate"
    package_name = "kohakufa_cute.fa4_sm80._a100_shared_" + suffix
    package = ModuleType(package_name)
    package.__package__ = package_name
    package.__path__ = []
    package.__file__ = "benchmarks/a100_shared_api_controls.py"
    sys.modules[package_name] = package
    for name in ("_host", "_attention", "_private_api", "_l40s", "_sm120"):
        sys.modules[package_name + "." + name] = importlib.import_module("kohakufa_cute.fa4_sm80." + name)
    schedules = ModuleType(package_name + "._schedules")
    schedules.__package__ = package_name
    schedules.__file__ = package.__file__
    sys.modules[schedules.__name__] = schedules
    exec(compile(schedule_source, schedules.__file__, "exec"), schedules.__dict__)
    exec(compile(entry_source, package.__file__, "exec"), package.__dict__)

    source = Path(api.__file__).read_bytes()
    module = ModuleType("kohakufa_cute._a100_shared_api_" + suffix)
    module.__package__ = "kohakufa_cute"
    module.__file__ = package.__file__
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    module.fa4_sm80 = package
    assert module._extension.cache_info().currsize == 0
    _modules.append(
        (
            module,
            {
                "previous": previous,
                "api_source_sha256": hashlib.sha256(source).hexdigest(),
                "entry_source_sha256": hashlib.sha256(entry_source).hexdigest(),
                "schedule_source_sha256": hashlib.sha256(schedule_source).hexdigest(),
                "api_bytecode_sha256": hashlib.sha256(module.scaled_dot_product_attention.__code__.co_code).hexdigest(),
                "initial_generic_cache_size": 0,
            },
        )
    )
    return module.scaled_dot_product_attention
