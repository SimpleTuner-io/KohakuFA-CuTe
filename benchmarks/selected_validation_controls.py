"""Compare exact API bodies with independent cold/warm generic-loader caches."""

import atexit
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

_modules = []


def _write_proof():
    destination = os.environ.get("KOH_VALIDATION_CACHE_PROOF")
    if destination:
        records = [{**record, "actual_cache_size": module._extension.cache_info().currsize} for module, record in _modules]
        Path(destination).write_text(json.dumps(records, indent=2) + "\n")


atexit.register(_write_proof)


def make_attention(*, previous=False, warm=False, control_file=None, control_revision="969f1f9"):
    from kohakufa_cute import api

    if previous:
        source = (
            Path(control_file).read_bytes()
            if control_file
            else subprocess.check_output(["git", "show", control_revision + ":src/kohakufa_cute/api.py"])
        )
    else:
        source = Path(api.__file__).read_bytes()
    name = "kohakufa_cute._validation_control_" + ("old" if previous else "new") + ("_warm" if warm else "_cold")
    module = ModuleType(name)
    module.__package__ = "kohakufa_cute"
    module.__file__ = "benchmarks/selected_validation_controls.py"
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    if warm:
        module._extension()
    assert module._extension.cache_info().currsize == int(warm)
    _modules.append(
        (
            module,
            {
                "previous": previous,
                "warm": warm,
                "expected_cache_size": int(warm),
                "source_sha256": hashlib.sha256(source).hexdigest(),
                "api_bytecode_sha256": hashlib.sha256(module.scaled_dot_product_attention.__code__.co_code).hexdigest(),
            },
        )
    )
    return module.scaled_dot_product_attention
