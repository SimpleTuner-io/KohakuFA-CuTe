"""Exact preceding Python predicate and isolated public-API comparison controls."""

import math
from functools import lru_cache

import torch


def python_compatible(q, k, v, mask, dropout, causal, scale, gqa):
    from kohakufa_cute.fa4_sm80 import _l40s, available

    if (
        not torch.is_grad_enabled()
        or not any(getattr(t, "requires_grad", False) for t in (q, k, v))
        or torch.compiler.is_compiling()
        or torch.are_deterministic_algorithms_enabled()
        or mask is not None
        or dropout != 0
        or getattr(q, "ndim", None) != 4
        or q.shape != k.shape
        or q.shape != v.shape
        or q.device.type != "cuda"
        or q.device != k.device
        or q.device != v.device
        or q.device.index != torch.cuda.current_device()
        or q.dtype not in (torch.float16, torch.bfloat16)
        or q.dtype != k.dtype
        or q.dtype != v.dtype
        or q.shape[0] != 1
        or q.shape[1] != 16
        or q.shape[2] not in (1024, 4096)
        or (causal and q.shape[-1] == 64 and q.shape[2] != 4096)
        or q.shape[-1] not in (64, 128)
        or not all(t.is_contiguous() and t.data_ptr() % 16 == 0 for t in (q, k, v))
        or not _l40s.is_device(q.device)
    ):
        return False
    if scale is not None and (
        not isinstance(scale, (float, int)) or not math.isfinite(scale) or not 1.1754943508222875e-38 <= scale <= 2e38
    ):
        return False
    return available()


@lru_cache(maxsize=1)
def parent_api():
    """Load the exact preceding public Python namespace, sharing unchanged GPU code."""
    import importlib
    import sys
    from pathlib import Path
    from types import ModuleType

    from kohakufa_cute import _ops, native
    from kohakufa_cute.fa4_sm80 import _attention, _host, _native, _private_api

    name = "_l40s_preceding_public"
    root = Path(__file__).resolve().parents[1] / ".publication/parent-source/src/kohakufa_cute"
    package = ModuleType(name)
    package.__path__ = [str(root)]
    package.__package__ = name
    sys.modules[name] = package
    # Python policy/API files are pristine parent bytes. Unchanged CUDA methods,
    # Schedule identity and the native runner are shared for a matched comparison.
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

    if mode == "public":
        return api.scaled_dot_product_attention
    if mode == "parent":
        return parent_api().scaled_dot_product_attention
    raise ValueError("Select the actual public or preceding public namespace.")


def prepare_parent_sources(revision="3f6ba0465a2f0887a0992831f0caa904beb939b3"):
    """Recover the measured preceding namespace from the repository history."""
    import hashlib
    import io
    import re
    import subprocess
    import tarfile
    from pathlib import Path

    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Use a complete parent revision.")
    repository = Path(__file__).resolve().parents[1]
    target = repository / ".publication/parent-source"
    data = subprocess.check_output(["git", "archive", revision, "src/kohakufa_cute"], cwd=repository)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        expected = {member.name: hashlib.sha256(archive.extractfile(member).read()).hexdigest() for member in members}
        for name, checksum in expected.items():
            path = target / name
            if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
                raise ValueError("Retained parent source differs; preserve it before selecting another revision.")
        target.mkdir(parents=True, exist_ok=True)
        archive.extractall(target, filter="data")
    assert all(
        hashlib.sha256((target / name).read_bytes()).hexdigest() == checksum for name, checksum in expected.items()
    )
    return {"revision": revision, "source_files": len(expected), "target": ".publication/parent-source"}


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-parent", action="store_true", required=True)
    parser.add_argument("--revision", default="3f6ba0465a2f0887a0992831f0caa904beb939b3")
    options = parser.parse_args()
    print(json.dumps(prepare_parent_sources(options.revision)))
