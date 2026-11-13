"""Explicit SM100 experiment for the retained 256-wide Blackwell backward.

The default SM100 D256/D512 pipeline computes two/four 128-column output
slices. The retained adaptive owner computes one/two 256-column slices,
sharing score work across more output features. This context tests that owner
on B200 without changing production dispatch or enabling SM103 reductions.

GPU compilation, numerical checks, sanitizers and complete-call timings are
required before treating this as a supported or faster B200 implementation.
The retained owner has unresolved racecheck reports on SM103; those reports
must be investigated on the target rather than assumed harmless.
"""

from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def wide_backward(dimension=512):
    """Test one explicitly selected width on eligible baseline BF16 calls.

    The yielded dictionary retains compiled functions after cache resets, so
    previously captured graphs remain valid while comparing another variant.
    Its replacement records describe actual compile-time selection.
    """
    import cutlass
    import cutlass.cute as cute
    import torch

    from kohakufa_cute import kernels, native
    from kohakufa_cute.blackwell_pipeline import Backward
    from kohakufa_cute.blackwell_register_aux import AdaptiveBackward

    dimension = int(dimension)
    if dimension not in (256, 512):
        raise ValueError("The wide-backward experiment supports explicit D256 or D512.")
    device = torch.cuda.current_device()
    name = torch.cuda.get_device_name(device)
    if torch.cuda.get_device_capability(device) != (10, 0) or "B200" not in name or "MIG" in name:
        raise ValueError("The wide-backward experiment requires a full B200.")
    compiler = cute.compile
    retained = {"compiled": [], "replacements": []}

    def compile_program(program, *arguments, **kwargs):
        if isinstance(program, kernels.JointBackward) and len(arguments) >= 2:
            q, k = arguments[:2]
            eligible = (
                q.element_type == cutlass.BFloat16
                and q.shape[3] == dimension
                and q.shape[1] == k.shape[1]
                and q.shape[0] * q.shape[1] >= 8
                and isinstance(program.query, Backward)
                and isinstance(program.keyvalue, Backward)
                and not program.query.masked
                and not program.keyvalue.masked
            )
            if eligible:
                if not all(owner.baseline_precision for owner in (program.query, program.keyvalue)):
                    raise ValueError("The B200 experiment requires the baseline operand policy.")
                program = AdaptiveBackward(program.query.causal, program, program.centering.loader.vector)
                # Match the SM100 compiler policy. The wider owner does not
                # grant SM103-only reduction operations to this architecture.
                for owner in (program.wide.q, program.wide.kv):
                    owner.hardware_reductions = False
                    owner.role_statistics = True
                retained["replacements"].append(
                    {
                        "dtype": "bf16",
                        "dimension": dimension,
                        "query_width": program.wide.q.width,
                        "keyvalue_width": program.wide.kv.width,
                        "hardware_reductions": False,
                        "causal": program.wide.q.causal,
                    }
                )
        function = compiler(program, *arguments, **kwargs)
        retained["compiled"].append(function)
        return function

    native.extension().clear_cache()
    kernels._compile_kernel.cache_clear()
    try:
        with patch.object(cute, "compile", compile_program):
            yield retained
    finally:
        native.extension().clear_cache()
        kernels._compile_kernel.cache_clear()
