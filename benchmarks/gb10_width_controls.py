"""Load a preceding native metadata bridge without sharing its changed predicate."""

import hashlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def parent(source):
    source = Path(source)
    native_name = "kohakufa_cute._gb10_width_parent_native"
    spec = importlib.util.spec_from_file_location(native_name, source / "parent_native.py")
    native = importlib.util.module_from_spec(spec)
    sys.modules[native_name] = native
    spec.loader.exec_module(native)
    name = "kohakufa_cute.fa4_sm80._gb10_width_parent"
    original = (source / "parent_gb10.py").read_text()
    old = "from kohakufa_cute.native import extension"
    assert original.count(old) == 1
    text = original.replace(old, "from " + native_name + " import extension")
    spec = importlib.util.spec_from_file_location(name, source / "parent_gb10.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(text, str(source / "parent_gb10.py"), "exec"), module.__dict__)
    module.width_baseline_provenance = {
        "scope": "Pristine preceding native.py/native.cpp and GB10 Python body; only the cached metadata-function import is redirected to the separately loaded preceding bridge. GPU implementation and generic fallback unchanged.",
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("parent_*") if p.is_file()
        },
        "loader_python_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    return module
