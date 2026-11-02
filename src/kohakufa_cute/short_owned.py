"""Full-row S128 D16 backward with exclusive gradient ownership."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import warp
from cutlass.cute.runtime import from_dlpack, make_fake_stream
from cutlass.memory import SmemAllocator

from . import kernels
from .kernels import Attention


class ShortOwned(Attention):
    """One CTA owns every query/key row and all three output gradients."""

    def __init__(self, causal):
        super().__init__("q", causal, False, columns=128, width=32, warps=8, query_column_warps=1)
        self.rows = 128
        self.product_width = 32
        self.score_b_matrices = 4

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        unused_mask: cute.Tensor,
        out: cute.Tensor,
        maximum: cute.Tensor,
        logsum: cute.Tensor,
        dout: cute.Tensor,
        unused_delta: cute.Tensor,
        dq: cute.Tensor,
        dk: cute.Tensor,
        dv: cute.Tensor,
        unused_blocks: cute.Tensor,
        unused_base: cute.Tensor,
        scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(q.element_type, cutlass.Float32, (16, 8, 16)),
            (4, 2, 1),
            permutation_mnk=(64, 32, 16),
        )
        self.kernel_owned(q, k, v, out, maximum, logsum, dout, dq, dk, dv, scale, mma).launch(
            grid=(q.shape[0] * q.shape[1], 1, 1), block=(256, 1, 1), stream=stream
        )

    @cute.jit
    def store_direct(self, result, destination, coordinates, batch, head, scale):
        for i in cutlass.range_constexpr(cute.size(coordinates)):
            row, column = coordinates[i]
            if column < 16:
                destination[batch, head, row, column] = (result[i] * scale).to(destination.element_type)

    @cute.jit
    def transposed_product(self, weights, source, mma, thread, tid, result):
        # The logical transpose requires transposed matrix loads as in Joint.
        left = cute.composition(weights, cute.make_layout((128, 128), stride=(128, 1)))
        ra = thread.make_fragment_A(thread.partition_A(left))
        ca = cute.make_tiled_copy_A(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), source.element_type), mma
        )
        ta = ca.get_slice(tid)
        sa, da = ta.partition_S(left), ta.retile(ra)
        right = cute.composition(source, cute.make_layout((32, 128), stride=(128, 1)))
        rb = thread.make_fragment_B(thread.partition_B(right))
        cb = cute.make_tiled_copy_B(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), source.element_type), mma
        )
        tb = cb.get_slice(tid)
        sb, db = tb.partition_S(right), tb.retile(rb)
        for chunk in cutlass.range_constexpr(8):
            cute.copy(ca, sa[None, None, chunk], da[None, None, chunk])
            cute.copy(cb, sb[None, None, chunk], db[None, None, chunk])
            cute.gemm(mma, result, ra[None, None, chunk], rb[None, None, chunk], result)

    @cute.kernel
    def kernel_owned(self, q, k, v, out, maximum, logsum, dout, dq, dk, dv, scale, mma):
        tid, _, _ = cute.arch.thread_idx()
        bh, _, _ = cute.arch.block_idx()
        batch, head = bh // q.shape[1], bh % q.shape[1]
        allocator = SmemAllocator()
        atom = cute.make_composed_layout(cute.make_swizzle(2, 3, 3), 0, cute.make_layout((8, 32), stride=(32, 1)))
        inputs = cute.tile_to_shape(atom, (128, 32), (0, 1))
        weights = cute.tile_to_shape(atom, (128, 128), (0, 1))
        sq = self.shared(allocator, q.element_type, inputs)
        sk = self.shared(allocator, q.element_type, inputs)
        sv = self.shared(allocator, q.element_type, inputs)
        sdo = self.shared(allocator, q.element_type, inputs)
        sp = self.shared(allocator, q.element_type, weights)
        sds = self.shared(allocator, q.element_type, weights)
        delta = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((128,)), byte_alignment=128)
        saved_maximum = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((128,)), byte_alignment=128)
        saved_logsum = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((128,)), byte_alignment=128)
        self.load(q, sq, batch, head, 0, tid)
        self.load(k, sk, batch, head, 0, tid)
        self.load(v, sv, batch, head, 0, tid)
        self.load(dout, sdo, batch, head, 0, tid)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        if tid < 128:
            value = cutlass.Float32(0)
            for d in cutlass.range_constexpr(16):
                value += out[batch, head, tid, d].to(cutlass.Float32) * sdo[tid, d].to(cutlass.Float32)
            delta[tid] = value
            saved_maximum[tid] = maximum[batch, head, tid]
            saved_logsum[tid] = logsum[batch, head, tid]
        cute.arch.sync_threads()
        thread = mma.get_slice(tid)
        scores = self.dot(sq, sk, mma, thread, tid)
        derivatives = self.dot(sdo, sv, mma, thread, tid)
        coordinates = thread.partition_C(cute.make_identity_tensor((128, 128)))
        magnitude = scale if scale >= 0 else -scale
        sign = cutlass.Float32(1) if scale >= 0 else cutlass.Float32(-1)
        for i in cutlass.range_constexpr(cute.size(coordinates)):
            row, column = coordinates[i]
            probability = cutlass.Float32(0)
            visible = True
            if cutlass.const_expr(self.causal):
                visible = column <= row
            if visible:
                probability = cute.math.exp2(
                    (scores[i] * sign - saved_maximum[row]) * magnitude * 1.4426950408889634 - saved_logsum[row],
                    fastmath=True,
                )
            rounded = probability.to(q.element_type)
            sp[row, column] = rounded
            sds[row, column] = (rounded.to(cutlass.Float32) * (derivatives[i] - delta[row])).to(q.element_type)
        cute.arch.sync_threads()
        output_coordinates = thread.partition_C(cute.make_identity_tensor((128, 32)))
        result = cute.make_rmem_tensor(thread.partition_shape_C((128, 32)), cutlass.Float32)
        result.fill(0.0)
        self.product(sds, sk, 0, mma, thread, tid, result, shared_weights=True)
        self.store_direct(result, dq, output_coordinates, batch, head, scale)
        result.fill(0.0)
        self.transposed_product(sds, sq, mma, thread, tid, result)
        self.store_direct(result, dk, output_coordinates, batch, head, scale)
        result.fill(0.0)
        self.transposed_product(sp, sdo, mma, thread, tid, result)
        self.store_direct(result, dv, output_coordinates, batch, head, cutlass.Float32(1))


def compile_kernel(mode, tensors, causal):
    """Static native entry; preserve the ordinary training forward calculation."""
    if mode == "forward":
        return kernels.compile_kernel(
            tensors[0].device.index, "forward", causal, False, tuple(kernels.descriptor(t) for t in tensors)
        )
    if mode != "backward":
        raise ValueError("Select forward or backward for short attention.")
    program = ShortOwned(causal)
    return cute.compile(
        program,
        *[from_dlpack(t.detach(), assumed_align=16) for t in tensors],
        cutlass.Float32(1),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
