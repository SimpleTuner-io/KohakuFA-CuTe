"""Independent backward equations conditioned on the actual rounded forward O."""

import torch

from benchmarks.reference import fp64_reference, rounded_reference


@torch.no_grad()
def attention_reference(inputs, dout, mask=None, causal=False, scale=None, *, output=None):
    if inputs[1].shape[2] == 1:
        # A visible singleton has P=1 and dS=0 exactly. Two independent FP32
        # dP/delta reductions can otherwise invent a tiny reference gradient
        # and incorrectly reject the implementation's exact zero.
        return fp64_reference(inputs, dout, mask, causal, scale)
    # Delta is defined from the implemented forward O, not an ideal FP64 O.
    # Recompute it eagerly when the test constructs its reference before capture;
    # the returned forward value still comes from independent equations.
    if output is None:
        from kohakufa_cute import attention

        output = attention(*inputs, mask=mask, causal=causal, scale=scale)
    return rounded_reference(inputs, dout, mask, causal, scale, output=output)
