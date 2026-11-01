"""Explicit previous-schedule control for matched B200 D512 comparisons."""

from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def previous_backward():
    """Retain native 128-column output owners without changing their arithmetic."""
    import cutlass.cute as cute
    import torch

    from kohakufa_cute import kernels, native
    from kohakufa_cute.blackwell_register_aux import AdaptiveBackward

    if torch.cuda.get_device_capability() != (10, 0) or "B200" not in torch.cuda.get_device_name():
        raise ValueError("The previous-schedule control requires B200.")
    compiler = cute.compile
    retained = []

    def compile_program(program, *args, **kwargs):
        if isinstance(program, AdaptiveBackward):
            program = program.fallback
        function = compiler(program, *args, **kwargs)
        retained.append(function)
        return function

    native.extension().clear_cache()
    kernels._compile_kernel.cache_clear()
    try:
        with patch.object(cute, "compile", compile_program):
            yield retained
    finally:
        native.extension().clear_cache()
        kernels._compile_kernel.cache_clear()
