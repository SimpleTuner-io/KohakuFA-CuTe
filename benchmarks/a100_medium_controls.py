"""Pristine preceding public API, loaded once with its own selector globals."""

import atexit
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path


def make_parent():
    root = Path(__file__).parent

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    selector = load("kohakufa_cute.fa4_sm80._medium_parent", root / "a100_medium_parent_entry.py")
    module = load("kohakufa_cute._medium_parent_api", root / "a100_medium_parent_api.py")
    module.fa4_sm80 = selector

    def proof():
        target = os.environ.get("A100_MEDIUM_API_PROOF")
        if target:
            from kohakufa_cute import api, fa4_sm80

            report = {
                "parent_api_sha256": hashlib.sha256((root / "a100_medium_parent_api.py").read_bytes()).hexdigest(),
                "parent_selector_sha256": hashlib.sha256((root / "a100_medium_parent_entry.py").read_bytes()).hexdigest(),
                "public_api_sha256": hashlib.sha256(Path(api.__file__).read_bytes()).hexdigest(),
                "public_selector_sha256": hashlib.sha256(Path(fa4_sm80.__file__).read_bytes()).hexdigest(),
                "parent_generic_cache_size": module._extension.cache_info().currsize,
                "public_generic_cache_size": api._extension.cache_info().currsize,
                "same_api_bytecode": module.scaled_dot_product_attention.__code__.co_code
                == api.scaled_dot_product_attention.__code__.co_code,
            }
            Path(target).write_text(json.dumps(report, indent=2) + "\n")

    atexit.register(proof)
    return module.scaled_dot_product_attention


def make_python():
    """Retain the preceding fully qualified Python-metadata selection."""
    root = Path(__file__).parent

    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, root / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    selector = load("kohakufa_cute.fa4_sm80._medium_python", "a100_medium_python_entry.py")
    module = load("kohakufa_cute._medium_python_api", "a100_medium_parent_api.py")
    module.fa4_sm80 = selector

    def proof():
        target = os.environ.get("A100_MEDIUM_PYTHON_PROOF")
        if target:
            from kohakufa_cute import api

            Path(target).write_text(
                json.dumps(
                    {
                        "api_sha256": hashlib.sha256((root / "a100_medium_parent_api.py").read_bytes()).hexdigest(),
                        "selector_sha256": hashlib.sha256((root / "a100_medium_python_entry.py").read_bytes()).hexdigest(),
                        "same_api_bytecode": module.scaled_dot_product_attention.__code__.co_code
                        == api.scaled_dot_product_attention.__code__.co_code,
                        "generic_cache_size": module._extension.cache_info().currsize,
                    },
                    indent=2,
                )
                + "\n"
            )

    atexit.register(proof)
    return module.scaled_dot_product_attention


def make_direct():
    """Explicit runner diagnostic, bypassing the advertised API preparation."""
    from kohakufa_cute.fa4_sm80._host import native_runner
    from kohakufa_cute.fa4_sm80._schedules import a100_schedule

    runner = native_runner(a100_schedule(128, 2048, True, heads=8))

    def attention(q, k, v, *, is_causal=False, scale=None):
        assert tuple(q.shape) == (1, 8, 2048, 128) and is_causal
        return runner(q, k, v, causal=True, scale=scale)

    return attention
