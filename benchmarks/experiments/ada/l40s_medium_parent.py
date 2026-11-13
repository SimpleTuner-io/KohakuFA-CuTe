"""Load the actual archived PR88 native bridge, not a source-equal rebuild."""

import atexit
import hashlib
import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def old_bridge():
    import torch

    import kohakufa_cute.native  # Load required Torch/TVM libraries before dlopen.

    root = Path(__file__).resolve().parents[3]
    directory = root / "private-tools"
    if not directory.exists():
        directory = root / ".publication/medium-tools"
    expected = json.loads((directory / "legacy-native.json").read_text())
    binary = directory / expected["filename"]
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == expected["sha256"]
    name = binary.stem
    if name in sys.modules:
        module = sys.modules[name]
        assert (
            hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == expected["sha256"]
        ), "Source-equal rebuild is not the actual old binary"
    else:
        specification = importlib.util.spec_from_file_location(name, binary)
        module = importlib.util.module_from_spec(specification)
        sys.modules[name] = module
        specification.loader.exec_module(module)
    assert module.l40s_metadata_compatible.__module__ == name
    atexit.register(module.clear_cache)
    return module


@lru_cache(maxsize=1)
def activate_old_bridge_for_pilot():
    """Install once before compilation in this explicit benchmark process."""
    from kohakufa_cute import api, native

    module = old_bridge()
    assert native.extension.cache_info().currsize == 0, "Old bridge must be installed before first native load"
    assert api._extension.cache_info().currsize == 0

    @lru_cache(maxsize=1)
    def archived_extension():
        return module

    native.extension = archived_extension
    return module
