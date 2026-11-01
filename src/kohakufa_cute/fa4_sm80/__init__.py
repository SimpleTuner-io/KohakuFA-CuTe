"""Optional qualified SM80 schedule; other calls retain native CuTe dispatch."""

import math
import sys
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

import torch


@lru_cache(maxsize=1)
def available():
    """Require the measured runtime and exact optional source dependency."""
    if sys.version_info[:2] != (3, 13):
        return False
    try:
        version("flash-attn-4")
        for package, qualified in (
            ("nvidia-cutlass-dsl", "4.8.1"),
            ("apache-tvm-ffi", "0.1.15.post0"),
            ("quack-kernels", "0.6.5"),
            ("torch-c-dlpack-ext", "0.1.5"),
        ):
            if version(package) != qualified:
                return False
    except PackageNotFoundError:
        return False
    if torch.__version__.split("+")[0] != "2.15.0" or torch.version.cuda != "13.2":
        return False
    from ._private_api import interface_matches

    return interface_matches()


def compatible(q, k, v, mask, dropout, causal, scale, gqa):
    """Select only the full-device shape and arithmetic families qualified here."""
    if (
        torch.compiler.is_compiling()
        or mask is not None
        or dropout != 0
        or torch.are_deterministic_algorithms_enabled()
        or getattr(q, "ndim", None) != 4
        or q.shape != k.shape
        or k.shape != v.shape
        or q.device.type != "cuda"
        or q.device != k.device
        or q.device != v.device
        or q.device.index != torch.cuda.current_device()
        or q.dtype not in (torch.float16, torch.bfloat16)
        or q.dtype != k.dtype
        or q.dtype != v.dtype
        or q.shape[0] != 1
        or not 8 <= q.shape[1] <= 16
        or not 2048 <= q.shape[2] <= 4096
        or q.shape[-1] not in (64, 128)
        or not all(t.is_contiguous() for t in (q, k, v))
        or any(t.data_ptr() % 16 for t in (q, k, v))
        or torch.cuda.get_device_capability(q.device) != (8, 0)
        or torch.cuda.get_device_properties(q.device).multi_processor_count != 108
    ):
        return False
    # This medium-size causal family had no forward wall-time gain after host
    # preparation, although its captured joint device time improved.
    if causal and q.shape[-1] == 128 and q.shape[1] < 16 and q.shape[2] < 4096:
        return False
    if scale is not None and (
        not isinstance(scale, (float, int)) or not math.isfinite(scale) or not 1.1754943508222875e-38 <= scale <= 2e38
    ):
        return False
    return available()


def attention(q, k, v, *, causal=False, scale=None):
    from ._attention import attention as run

    return run(q, k, v, causal=causal, scale=scale)
