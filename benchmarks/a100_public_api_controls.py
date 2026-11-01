"""Isolate exact old/new public API bodies and their generic-loader caches."""

import atexit
import hashlib
import json
import os
import sys
from pathlib import Path
from types import ModuleType

_modules = []


def _write_proof():
    destination = os.environ.get("A100_PUBLIC_CONTROL_PROOF")
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

    source_root = Path(__file__).parent
    selected_source = (
        (source_root / "a100_previous_fa4_sm80.py").read_bytes() if previous else Path(fa4_sm80.__file__).read_bytes()
    )
    suffix = "previous" if previous else "candidate"
    selected_name = "kohakufa_cute.fa4_sm80._a100_public_" + suffix
    selected = ModuleType(selected_name)
    selected.__package__ = "kohakufa_cute.fa4_sm80"
    selected.__file__ = "benchmarks/a100_public_api_controls.py"
    sys.modules[selected_name] = selected
    exec(compile(selected_source, selected.__file__, "exec"), selected.__dict__)

    api_source = Path(api.__file__).read_bytes()
    name = "kohakufa_cute._a100_public_api_" + suffix
    module = ModuleType(name)
    module.__package__ = "kohakufa_cute"
    module.__file__ = "benchmarks/a100_public_api_controls.py"
    sys.modules[name] = module
    exec(compile(api_source, module.__file__, "exec"), module.__dict__)
    module.fa4_sm80 = selected
    assert module._extension.cache_info().currsize == 0
    _modules.append(
        (
            module,
            {
                "previous": previous,
                "api_source_sha256": hashlib.sha256(api_source).hexdigest(),
                "selected_module_source_sha256": hashlib.sha256(selected_source).hexdigest(),
                "api_bytecode_sha256": hashlib.sha256(module.scaled_dot_product_attention.__code__.co_code).hexdigest(),
                "initial_generic_cache_size": 0,
            },
        )
    )
    return module.scaled_dot_product_attention
