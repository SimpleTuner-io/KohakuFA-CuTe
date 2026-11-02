"""Load the complete preceding GB10 host policy without changing its package context."""

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def parent():
    name = "kohakufa_cute.fa4_sm80._gb10_metadata_parent"
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("gb10_metadata_parent.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
