"""Preserve the low component of rectangular low-precision value-gradient weights."""

from copy import deepcopy

import cuda.bindings.driver as cuda
import cutlass.cute as cute


class RectangularValues:
    """Keep both sequence lengths in a cached program's accuracy policy."""

    def __init__(self, fallback):
        self.fallback = fallback
        self.precise = deepcopy(fallback)
        self.precise.compensated_values = True
        # Share weights across two output-column warps. Smaller output fragments
        # keep the independent high/low value sums out of local memory.
        self.precise.column_warps = self.precise.score_column_warps = 2
        self.precise.rows = self.precise.grid_rows = 32
        self.precise.narrow_n = False
        self.precise.score_b_matrices = 4
        self.precise.kv_column_statistics = True

    @cute.jit
    def __call__(
        self, q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream: cuda.CUstream
    ):
        if q.shape[2] > k.shape[2]:
            self.precise(q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream)
        else:
            self.fallback(q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream)
