"""Two-term probability products for explicitly qualified 3D forward accuracy."""

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import warp, warpgroup

from .kernels import hopper_operand, operand_a


@cute.jit
def residual_product(program, weights, source, offset, mma, thread, tid, result):
    residual = cute.make_fragment_like(weights, cutlass.Float32)
    residual.store(weights.load() - weights.load().to(source.element_type).to(cutlass.Float32))
    source_tile = cute.local_tile(source, (program.columns, program.product_width), (0, offset // program.product_width))
    transposed = cute.composition(
        source_tile, cute.make_layout((program.product_width, program.columns), stride=(program.columns, 1))
    )
    if cutlass.const_expr(program.hopper):
        cute.arch.fence_proxy("async.shared", space="cta")
        pmma = cute.make_tiled_mma(
            warpgroup.MmaF16BF16Op(
                source.element_type,
                cutlass.Float32,
                (64, program.product_width, 16),
                warpgroup.OperandSource.RMEM,
                cute.nvgpu.OperandMajorMode.K,
                cute.nvgpu.OperandMajorMode.MN,
            )
        )
        pt = pmma.get_slice(tid)
        result = cute.make_tensor(
            result.iterator, cute.make_layout(pt.partition_shape_C((program.rows, program.product_width)))
        )
        high = hopper_operand(weights, pmma.tv_layout_A.shape[1], source.element_type)
        low = hopper_operand(residual, pmma.tv_layout_A.shape[1], source.element_type)
        rb = pmma.make_fragment_B(pt.partition_B(transposed))
        warpgroup.fence()
        pmma.set(warpgroup.Field.ACCUMULATE, True)
        for chunk in cutlass.range(cute.size(high.shape[2]), unroll_full=True):
            cute.gemm(pmma, result, high[None, None, chunk], rb[None, None, chunk], result)
            cute.gemm(pmma, result, low[None, None, chunk], rb[None, None, chunk], result)
        warpgroup.commit_group()
        warpgroup.wait_group(0)
    else:
        high, low = operand_a(weights, source.element_type), operand_a(residual, source.element_type)
        rb = thread.make_fragment_B(thread.partition_B(transposed))
        copy = cute.make_tiled_copy_B(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), source.element_type), mma
        )
        copier = copy.get_slice(tid)
        src, dst = copier.partition_S(transposed), copier.retile(rb)
        for chunk in cutlass.range_constexpr(program.columns // 16):
            cute.copy(copy, src[None, None, chunk], dst[None, None, chunk])
            cute.gemm(mma, result, high[None, None, chunk], rb[None, None, chunk], result)
            cute.gemm(mma, result, low[None, None, chunk], rb[None, None, chunk], result)
