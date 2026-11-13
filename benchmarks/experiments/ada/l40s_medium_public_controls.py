"""Actual PR88 binary and Python namespace versus the bounded public candidate."""

from functools import lru_cache

_BINDING_REGISTERED = False


@lru_cache(maxsize=1)
def parent_api():
    import importlib
    import sys
    from dataclasses import asdict
    from pathlib import Path
    from types import ModuleType

    from benchmarks.experiments.ada.l40s_medium_parent import old_bridge
    from kohakufa_cute import _ops
    from kohakufa_cute.fa4_sm80 import _attention, _native, _private_api

    bridge = old_bridge()
    name = "_l40s_medium_actual_pr88"
    root = Path(__file__).resolve().parents[3] / ".publication/medium-parent-source/src/kohakufa_cute"
    package = ModuleType(name)
    package.__path__ = [str(root)]
    package.__package__ = name
    sys.modules[name] = package
    native = ModuleType(name + ".native")
    native.extension = lambda: bridge
    sys.modules[name + ".native"] = native
    sys.modules[name + "._ops"] = _ops
    host = ModuleType(name + ".fa4_sm80._host")

    @lru_cache(maxsize=32)
    def native_runner(schedule):
        return bridge.AttentionRunner("warp", asdict(schedule))

    host.native_runner = native_runner
    sys.modules[host.__name__] = host
    for suffix, module in (("_attention", _attention), ("_native", _native), ("_private_api", _private_api)):
        sys.modules[name + ".fa4_sm80." + suffix] = module
    api = importlib.import_module(name + ".api")
    policy = importlib.import_module(name + ".fa4_sm80._l40s")
    # Bind the original native metadata function once in the isolated namespace.
    policy._metadata_predicate = lru_cache(maxsize=1)(lambda: bridge.l40s_metadata_compatible)
    return api


def make_attention(*, mode="public"):
    from kohakufa_cute import api

    record_host_bindings()
    if mode == "public":
        return api.scaled_dot_product_attention
    if mode == "parent":
        return parent_api().scaled_dot_product_attention
    raise ValueError("Choose public or parent.")


def record_host_bindings():
    """Retain actual mapped bridge bytes from the process that executes the calls."""
    global _BINDING_REGISTERED
    if _BINDING_REGISTERED:
        return
    import atexit
    import hashlib
    import json
    import os
    import shutil
    from pathlib import Path

    from benchmarks.experiments.ada.l40s_medium_parent import old_bridge
    from kohakufa_cute.native import extension

    modules = {"candidate": extension(), "parent": old_bridge()}
    directory = os.environ.get("L40S_MEDIUM_BINDING_DIR")
    if not directory:
        return

    def save():
        root = Path(directory) / str(os.getpid())
        root.mkdir(parents=True, exist_ok=True)
        maps = Path("/proc/self/maps").read_text()
        (root / "maps.txt").write_text(maps)
        records = {}
        for label, module in modules.items():
            binary = Path(module.__file__).resolve()
            assert str(binary) in maps
            shutil.copy2(binary, root / (label + ".so"))
            records[label] = {
                "filename": binary.name,
                "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "metadata_module": module.l40s_metadata_compatible.__module__,
                "mapped": True,
            }
        (root / "binding.json").write_text(json.dumps(records, indent=2) + "\n")

    atexit.register(save)
    _BINDING_REGISTERED = True
