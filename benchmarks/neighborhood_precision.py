"""Explicit stronger arithmetic experiments; never automatic dispatch.

All selections retain the production forward, actual rounded-O delta, FP32
products and post-accumulation gradient scaling. ``fp32_p`` keeps reconstructed
P in FP32 only while forming dS; dS and the dV probability operand still round
once to the input dtype. ``weighted_dq`` additionally projects the rounded dS
product onto centered keys using the normalized rounded-P weighted key mean.
The combination applies both changes. These change the declared low-precision
equations and are not baseline-equivalent scheduling optimizations.

``reconstructed_dk`` keeps P in FP32 and reconstructs a correction to the
rounded-O softmax center during the existing query traversal:
``t = sum(P32 * (dP32 - delta_O)) / sum(P32)``. The KV owner forms
``dS = round_input(P32 * ((dP32 - delta_O) - t))``. A separate FP32 row buffer
preserves delta_O, and the two subtractions avoid adding a small correction
back to a large center. No extra neighborhood pass or GEMM is required.
``weighted_dq_reconstructed_dk`` also retains the normalized weighted dQ
projection. Neither variant changes dV, forward arithmetic, or default dispatch.
"""

from contextlib import contextmanager

# Keep the qualified dQ campaign's enumeration stable for its runner and tests.
VARIANTS = ("baseline", "fp32_p", "weighted_dq", "fp32_p_weighted_dq", "legacy_weighted")
KEY_VARIANTS = ("baseline", "fp32_p_weighted_dq", "reconstructed_dk", "weighted_dq_reconstructed_dk")
ALL_VARIANTS = VARIANTS + ("reconstructed_dk", "weighted_dq_reconstructed_dk")
KEY_STATISTICS = ("direct", "column", "staged", "column_staged")


@contextmanager
def variant(name):
    """Select one serialized full forward/backward experiment and restore it.

    The selection is a compile-cache key, so returning to baseline does not
    reuse a stronger-arithmetic executable. Use the same context for warmup,
    graph capture, replay and changed-input eager validation.
    """
    from kohakufa_cute.neighborhood_kernels import set_experiment_precision

    if name not in ALL_VARIANTS:
        raise ValueError("Unknown neighborhood precision experiment.")
    previous = set_experiment_precision(name)
    try:
        yield name
    finally:
        set_experiment_precision(previous)


@contextmanager
def key_statistics(mode):
    """Select a serialized KV statistics loading experiment with identical math.

    ``column`` hoists row statistics across key coordinates. ``staged`` copies
    the four FP32 statistics planes into CTA shared memory before consumption.
    ``column_staged`` combines the two independent scheduling choices. Defaults
    remain ``direct``. The concrete backward selection is a compile-cache key;
    use one selection across warmup, full calls, capture and changed-data replay.
    """
    from kohakufa_cute.neighborhood_kernels import set_experiment_key_statistics

    if mode not in KEY_STATISTICS:
        raise ValueError("Unknown neighborhood KV statistics schedule.")
    previous = set_experiment_key_statistics(mode)
    try:
        yield mode
    finally:
        set_experiment_key_statistics(previous)
