"""Poison allocator result slots before tcgen05 allocation, for initcheck only.

The invalid sentinel must be replaced by the real allocation result for any
attention calculation to succeed. This adds stores/barriers and is not a default
kernel or a performance comparison.
"""

import ast
import hashlib
import importlib
import importlib.util
import os
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from types import FunctionType, SimpleNamespace

import torch

_temporary = tempfile.TemporaryDirectory(prefix="blackwell-pointer-control-")
_EXPECTED = {
    "forward": "c64f0823b0d77ec59f47f37ae2f950f00340d4b6c31fd271b693287dd4dc6cd0",
    "backward": "5c3820c3bd43d0643ca0b7510ad65a38f414a012e6cd1a4bec2c7349051845b4",
}


def _replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Inspect the retained source before adapting the pointer diagnostic.")
    return source.replace(old, new, 1)


def _load(name, source):
    ast.parse(source)
    directory = Path(os.environ.get("CUTE_DSL_DUMP_DIR", _temporary.name)) / "experimental-sources"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name.rsplit(".", 1)[-1] + ".py")
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _function():
    from kohakufa_cute.fa4_blackwell import _backward, _forward, _preprocess
    from kohakufa_cute.fa4_blackwell._private_api import make_private_api

    classes = {}
    marker = "        tmem = cutlass.utils.TmemAllocator("
    injection = """        if cute.arch.thread_idx()[0] == 0:
            cute.make_tensor(storage.tmem_holding_buf.ptr, cute.make_layout((1,)))[0] = Int32(0x7fffffff)
        cute.arch.sync_threads()
"""
    for name, original in (("forward", _forward), ("backward", _backward)):
        source = Path(original.__file__).read_text()
        if hashlib.sha256(source.encode()).hexdigest() != _EXPECTED[name]:
            raise ValueError("The pointer control requires its recorded numerical source.")
        source = _replace_once(source, marker, injection + marker)
        classes[name] = _load("kohakufa_cute.fa4_blackwell._poison_allocator_" + name, source)
    source = Path(classes["forward"].__file__).read_text()
    marker = "            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)"
    if source.count(marker) != 3:
        raise ValueError("Inspect every TMEM pointer reader before recording the allocation.")
    source = source.replace(
        marker,
        marker
        + "\n            if cute.arch.thread_idx()[0] == self.mma_warp_id * 32:\n                aux_data.tensors[1][cute.arch.block_idx()[0]] = cute.make_tensor(storage.tmem_holding_buf.ptr, cute.make_layout((1,)))[0]",
        1,
    )
    classes["forward"] = _load("kohakufa_cute.fa4_blackwell._recorded_allocator_forward", source)
    api = make_private_api(
        FlashAttentionForwardSm100=classes["forward"].FlashAttentionForwardSm100,
        FlashAttentionBackwardSm100=classes["backward"].FlashAttentionBackwardSm100,
        FlashAttentionBackwardPreprocess=_preprocess.FlashAttentionBackwardPreprocess,
    )
    original_fwd = api._flash_attn_fwd
    owned = []

    def record_fwd(q, k, v, *args, **kwargs):
        result_slots = torch.full((1024,), -1, dtype=torch.int32, device=q.device)
        owned.append(result_slots)
        kwargs["aux_tensors"] = list(kwargs["aux_tensors"]) + [result_slots]
        result = original_fwd(q, k, v, *args, **kwargs)
        if not torch.cuda.is_current_stream_capturing():
            values = result_slots.cpu().tolist()
            values = [v for v in values if v >= 0]
            if not values or 0x7FFFFFFF in values:
                raise AssertionError("Allocation did not replace the invalid shared sentinel.")
            print("post-allocation pointer values", sorted(set(values)), "written slots", len(values), flush=True)
        return result

    api = SimpleNamespace(_flash_attn_fwd=record_fwd, _flash_attn_bwd=api._flash_attn_bwd)
    module = importlib.import_module("kohakufa_cute.fa4_blackwell.attention")
    scope = dict(vars(module), _api=lambda: api)
    methods = {}
    for name in ("forward", "backward"):
        original = getattr(module._Attention, name)
        function = FunctionType(original.__code__, scope, original.__name__, original.__defaults__, original.__closure__)
        function.__kwdefaults__ = original.__kwdefaults__
        methods[name] = staticmethod(function)
    return type("_PoisonedAllocatorAttention", (torch.autograd.Function,), methods)


def _factory():
    function = _function()

    def attention(q, k, v, *, is_causal=False, scale=None, enable_gqa=False):
        return function.apply(q, k, v, is_causal, q.shape[-1] ** -0.5 if scale is None else scale)

    return attention


def make_pointer_poisoned_attention():
    return _factory()
