"""Actual preceding public API and automatic RTX 4090 comparison controls."""

from functools import lru_cache


@lru_cache(maxsize=1)
def parent_api():
    import importlib
    import sys
    from pathlib import Path
    from types import ModuleType

    from kohakufa_cute import _ops, native
    from kohakufa_cute.fa4_sm80 import _attention, _host, _native, _private_api

    name = "_rtx4090_preceding_public"
    root = Path(__file__).resolve().parents[3] / ".publication/parent-source/src/kohakufa_cute"
    package = ModuleType(name)
    package.__path__ = [str(root)]
    package.__package__ = name
    sys.modules[name] = package
    sys.modules[name + "._ops"] = _ops
    sys.modules[name + ".native"] = native
    for suffix, module in (
        ("_attention", _attention),
        ("_host", _host),
        ("_native", _native),
        ("_private_api", _private_api),
    ):
        sys.modules[name + ".fa4_sm80." + suffix] = module
    return importlib.import_module(name + ".api")


def make_attention(*, mode="public"):
    from kohakufa_cute import api
    from kohakufa_cute.fa4_sm80 import rtx4090

    if mode == "public":
        return api.scaled_dot_product_attention
    if mode == "automatic":
        return api.automatic_scaled_dot_product_attention
    if mode == "parent":
        return parent_api().scaled_dot_product_attention
    if mode == "explicit":
        return rtx4090.scaled_dot_product_attention
    raise ValueError("Select public, automatic, parent or explicit.")


if __name__ == "__main__":
    import argparse
    import json

    from benchmarks.experiments.ada.l40s_public_metadata_controls import prepare_parent_sources

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-parent", required=True, action="store_true")
    parser.add_argument("--revision", required=True)
    options = parser.parse_args()
    print(json.dumps(prepare_parent_sources(options.revision)))
