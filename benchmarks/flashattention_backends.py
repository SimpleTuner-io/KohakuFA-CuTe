"""Explicit standalone FlashAttention adapters for BHSD attention comparisons."""

import hashlib
import importlib
import importlib.metadata
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PureWindowsPath

import torch


@lru_cache(maxsize=64)
def _file_sha256(filename, size, modified_ns):
    digest = hashlib.sha256()
    with Path(filename).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(filename):
    path = Path(filename)
    metadata = path.stat()
    return _file_sha256(str(path), metadata.st_size, metadata.st_mtime_ns)


def configuration_metadata(value):
    """Retain configuration values while identifying external paths by hash."""
    if isinstance(value, dict):
        return {key: configuration_metadata(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [configuration_metadata(item) for item in value]
    if isinstance(value, str) and (Path(value).is_absolute() or PureWindowsPath(value).is_absolute()):
        return {"external_path_sha256": hashlib.sha256(value.encode()).hexdigest()}
    return value


class UnsupportedCase(ValueError):
    """The requested computation has no equivalent supported comparison here."""


@dataclass
class Backend:
    name: str
    function: object
    module: object
    distribution: str

    def __call__(self, q, k, v, *, is_causal=False, scale=None, enable_gqa=False):
        if self.name in ("cute", "sdpa"):
            return self.function(q, k, v, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
        if is_causal and q.shape[2] != k.shape[2]:
            raise UnsupportedCase("Rectangular causal visibility differs between the APIs.")
        if q.shape[1] != k.shape[1] and not enable_gqa:
            raise ValueError("Differing Q/KV heads require enable_gqa=True.")
        if self.name == "fa2" and max(q.shape[-1], v.shape[-1]) > 256:
            raise UnsupportedCase("Standalone FA2 supports head dimensions through 256.")
        kwargs = dict(causal=is_causal, softmax_scale=scale)
        if self.name == "fa2":
            kwargs["dropout_p"] = 0.0
        result = self.function(*(t.transpose(1, 2) for t in (q, k, v)), **kwargs)
        # FA4 currently returns (out, lse); FA2/FA3 normally return out.
        output = result[0] if isinstance(result, tuple) else result
        if not isinstance(output, torch.Tensor):
            raise TypeError("The selected FlashAttention API did not return a tensor output.")
        return output.transpose(1, 2)

    def metadata(self):
        try:
            version = importlib.metadata.version(self.distribution)
        except importlib.metadata.PackageNotFoundError:
            version = None
        filename = getattr(self.module, "__file__", None)
        entrypoint = sys.modules.get(getattr(self.function, "__module__", ""))
        entrypoint_file = getattr(entrypoint, "__file__", None)
        binaries = {
            name: file_sha256(module.__file__)
            for name, module in list(sys.modules.items())
            if name.startswith("flash_attn")
            and getattr(module, "__file__", None)
            and Path(module.__file__).suffix in (".so", ".pyd", ".dylib")
        }
        return {
            "name": self.name,
            "module": self.module.__name__,
            "distribution": self.distribution,
            "version": version,
            "module_source_sha256": file_sha256(filename) if filename else None,
            "entrypoint_module": getattr(self.function, "__module__", None),
            "entrypoint_source_sha256": file_sha256(entrypoint_file) if entrypoint_file else None,
            "loaded_flash_extension_sha256": binaries,
            "layout_adapter": "BHSD views to BSHD and back" if self.name.startswith("fa") else "BHSD",
            "fallback": False,
        }


def load_backend(name):
    """Load exactly the selected implementation; import errors never select SDPA."""
    specifications = {
        "cute": ("kohakufa_cute", "scaled_dot_product_attention", "kohakufa-cute"),
        "sdpa": ("torch.nn.functional", "scaled_dot_product_attention", "torch"),
        "fa2": ("flash_attn", "flash_attn_func", "flash-attn"),
        "fa3": ("flash_attn_3.flash_attn_interface", "flash_attn_func", "flash-attn-3"),
        "fa4": ("flash_attn.cute", "flash_attn_func", "flash-attn-4"),
    }
    module_name, function_name, distribution = specifications[name]
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    if not callable(function):
        raise TypeError("The selected attention entry point is not callable.")
    return Backend(name, function, module, distribution)
