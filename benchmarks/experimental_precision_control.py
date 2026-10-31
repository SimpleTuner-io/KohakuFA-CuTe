"""Reconstruct the previous default arithmetic for same-process comparisons.

This is a benchmark control, never production dispatch. Its equivalence to the
preceding source must be checked independently before interpreting timings.
"""

from contextlib import contextmanager
from unittest.mock import patch

import cutlass
import cutlass.cute as cute
import torch

from kohakufa_cute import kernels, native


@contextmanager
def previous_precision():
    compile_program = cute.compile
    selected = []

    def previous(program, *arguments, **kwargs):
        memo = set()

        def visit(owner):
            if id(owner) in memo:
                return
            memo.add(id(owner))
            if isinstance(owner, kernels.Attention):
                owner.baseline_precision = False
                if type(owner).__name__ == "D64Values":
                    owner.compensated_keys = owner.compensated_values = True
            for name in (
                "query",
                "keyvalue",
                "fallback",
                "tuned",
                "short",
                "compact",
                "checked",
                "complete",
                "precise",
                "medium",
                "long",
                "score",
                "wide",
                "q",
                "kv",
                "attention",
            ):
                child = getattr(owner, name, None)
                if child is not None:
                    visit(child)

        visit(program)
        if arguments and hasattr(arguments[0], "shape") and len(arguments[0].shape) == 4:
            q = arguments[0]
            device = torch.cuda.current_device()
            capability, name = torch.cuda.get_device_capability(device), torch.cuda.get_device_name(device)
            dimension = q.shape[3]
            # Matches the preceding compiler's final precision boundary.
            eligible = (
                (capability, name)
                in (
                    ((8, 6), "NVIDIA GeForce RTX 3090 Ti"),
                    ((8, 9), "NVIDIA L40S"),
                    ((12, 0), "NVIDIA GeForce RTX 5090"),
                )
                and q.element_type in (cutlass.Float16, cutlass.BFloat16)
                and dimension in (16, 64, 128, 256, 512)
                and (dimension != 256 or q.shape[1] == arguments[1].shape[1])
            )
            mode = (
                "backward"
                if hasattr(program, "query") or hasattr(program, "fallback") and hasattr(program.fallback, "query")
                else getattr(program, "mode", "forward")
            )
            if eligible and (
                mode == "backward"
                or mode == "forward"
                and capability == (12, 0)
                and q.element_type == cutlass.Float16
                and dimension == 256
            ):
                program = kernels.precision_schedule(program, dimension, device=device)
                memo.clear()
                visit(program)
        selected.append(type(program).__name__)
        return compile_program(program, *arguments, **kwargs)

    native.extension().clear_cache()
    kernels._compile_kernel.cache_clear()
    try:
        with patch.object(cute, "compile", previous):
            yield selected
    finally:
        native.extension().clear_cache()
        kernels._compile_kernel.cache_clear()
