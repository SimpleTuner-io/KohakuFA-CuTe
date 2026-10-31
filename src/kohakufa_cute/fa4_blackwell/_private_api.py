"""Private source-qualified FA4 interface globals and compilation caches."""

import hashlib
import importlib
from pathlib import Path
from types import FunctionType, ModuleType, SimpleNamespace

_EXPECTED_SOURCES = {
    "interface": "7890aa821fe751fc2e088a2b7a2a2c3c4fdaa3832ab2a4e13c6b143f566ba496",
    "flash_bwd_preprocess": "f31922a9ecb817923038db3addcd83a9486a8bfba2b0d74be6ee23f42256e704",
    "flash_bwd_sm100": "b0aff6cfe27d4656bf220e6861cb05e6cd657a6666b67774d0398058eb0329b7",
    "flash_fwd_sm100": "10a767639aa23e444693d053e22d03fb18cff129be5235e232fa56e663837af3",
    "softmax": "aaddf3030dba7b9809c4fefae3caab22d0f22b9ecf5050601d6646aefb611a49",
}


def interface_matches():
    for name, expected in _EXPECTED_SOURCES.items():
        module = importlib.import_module("flash_attn.cute." + name)
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
            return False
    return True


def make_private_api(**overrides):
    from flash_attn.cute import interface

    if not interface_matches():
        raise ValueError("The private experiment requires the recorded FA4 interface source.")
    scope = dict(vars(interface))
    utils = ModuleType(interface.utils.__name__)
    vars(utils).update(vars(interface.utils))
    utils._get_sm100_backend_override = lambda: False
    scope["utils"] = utils
    for name, original in vars(interface).items():
        if not isinstance(original, FunctionType) or original.__module__ != interface.__name__:
            continue
        clone = FunctionType(original.__code__, scope, original.__name__, original.__defaults__, original.__closure__)
        clone.__kwdefaults__ = original.__kwdefaults__
        clone.__annotations__ = dict(original.__annotations__)
        clone.__dict__.update(vars(original))
        if hasattr(original, "compile_cache"):
            clone.compile_cache = {}
        scope[name] = clone
    scope.update(overrides)
    return SimpleNamespace(**scope)
