"""Derived FA4 SoftmaxSm100 methods with this project's arithmetic.

Upstream: Dao-AILab/flash-attention, revision
4616f5116d33e3ba3cbb71d582caffcfcb49649c.
BSD-3-Clause; see LICENSES/FlashAttention-BSD-3-Clause.txt and
LICENSES/FlashAttention-AUTHORS.txt. Unchanged methods use the pinned dependency.
"""

# Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.
# All rights reserved.


from flash_attn.cute.softmax import Float32
from flash_attn.cute.softmax import SoftmaxSm100 as _Original
from flash_attn.cute.softmax import cute, cutlass


class SoftmaxSm100(_Original):

    @staticmethod
    def create(
        scale_log2: Float32,
        rescale_threshold: cutlass.Constexpr[float] = 0.0,
        softmax_scale: Float32 | None = None,
        max_offset: cutlass.Constexpr[int] = 0,
    ):
        num_rows = 1
        arch = 100
        row_max = cute.make_rmem_tensor(num_rows, Float32)
        row_sum = cute.make_rmem_tensor(num_rows, Float32)
        return SoftmaxSm100(
            scale_log2,
            num_rows,
            row_max,
            row_sum,
            arch,
            softmax_scale,
            rescale_threshold=rescale_threshold,
            max_offset=max_offset,
        )

    @cute.jit
    def scale_subtract_rowmax(self, acc_S_row: cute.Tensor, row_max: Float32):
        assert cute.size(acc_S_row.shape) % 2 == 0
        for i in cutlass.range(0, cute.size(acc_S_row.shape), 2, unroll_full=True):
            pair = cute.arch.sub_packed_f32x2((acc_S_row[i], acc_S_row[i + 1]), (row_max, row_max))
            pair = cute.arch.mul_packed_f32x2(pair, (self.scale_log2, self.scale_log2))
            acc_S_row[i], acc_S_row[i + 1] = pair
