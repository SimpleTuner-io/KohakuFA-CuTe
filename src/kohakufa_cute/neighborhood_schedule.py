"""Framework-free scheduling selection for qualified neighborhood workloads.

The device-specific policy changes tile ownership and traversal only. Every
schedule uses the same operand conversions and FP32 arithmetic contract.
"""

DEFAULT_PRECISION = "weighted_dq_reconstructed_dk"
DEFAULT_KEY_STATISTICS = "column_staged"

BASELINE_COLUMNS = (32, 32, 32)
BASELINE_TRAVERSAL = (False, False, False)


def default_schedule(*, device_name, capability, sm_count, shape, spatial, window, dtype, contiguous):
    """Return concrete forward/query/KV columns and exact-traversal flags.

    This bounded H100 policy is qualified on representative workloads; it does
    not promise a speedup for every geometry inside the selection family.
    """
    baseline = BASELINE_COLUMNS, BASELINE_TRAVERSAL
    if len(shape) != 4 or len(spatial) != 2 or len(window) != 2:
        return baseline
    batch, heads, sequence, dimension = shape
    height, width = spatial
    window_h, window_w = window
    if min(batch, heads, sequence, height, width) < 1 or sequence != height * width:
        return baseline
    owner_ctas = batch * heads * ((sequence + 63) // 64)
    normalized_name = "".join(device_name.upper().split())
    selected = (
        "H100" in normalized_name
        and "80GB" in normalized_name
        and "HBM3" in normalized_name
        and "MIG" not in normalized_name
        and "NVL" not in normalized_name
        and tuple(capability) == (9, 0)
        and sm_count == 132
        and contiguous
        and dtype in ("float16", "bfloat16")
        and dimension == 64
        and sequence >= 1024
        and owner_ctas >= 128
        # Retain the original schedule in a CTA interval with a measured regression.
        and not 2 * sm_count < owner_ctas <= 3 * sm_count
        and 32 <= width <= 64
        and window_h == window_w
        and 5 <= window_h <= 13
        and window_h % 2 == 1
        and window_h <= height
        and window_w <= width
    )
    if not selected:
        return baseline
    exact = width == 64
    return (32, 32, 64), (exact, False, exact)


def default_volume_schedule(*, device_name, capability, sm_count, shape, spatial, window, dtype, vector):
    """Measured H100 PCIe large-window owner; arithmetic never varies by device."""
    normalized_name = "".join(device_name.upper().split())
    if (
        len(spatial) == 3
        and len(shape) == 4
        and "H100PCIE" in normalized_name
        and "MIG" not in normalized_name
        and tuple(capability) == (9, 0)
        and sm_count == 114
        and shape[-1] == 64
        and dtype in ("float16", "bfloat16")
        and vector == 8
        and tuple(window) == (11, 11, 11)
        and spatial[0] >= 11
        and all(extent >= 64 and extent % 8 == 0 for extent in spatial[1:])
    ):
        return ((1, 8, 8), True)
    return False
