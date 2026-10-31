"""RTX 3090 Ti key ownership with query-major scores and resident V operands."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import warp
from cutlass.memory import SmemAllocator

from kohakufa_cute.kernels import Attention, packed_half_conversion, quad_sum, row_view
from kohakufa_cute.launch import block_indices, launch_grid


class QueryMajorValues(Attention):
    """Match the native Q64/K32 score footprint without changing centered math."""

    def __init__(self, causal, vector, *, resident_v=True, release_barrier=True):
        super().__init__(
            "kv", causal, False, columns=64, width=256, warps=8, centered=True, vector=vector, kv_column_warps=8
        )
        self.rows = self.grid_rows = 32
        self.resident_v, self.release_barrier = resident_v, release_barrier
        self.score = Attention("forward", causal, False, columns=32, width=256, warps=8, vector=vector)
        self.score.rows = self.score.grid_rows = 64
        self.score.column_warps = 2

    @cute.jit
    def __call__(
        self, q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream: cuda.CUstream
    ):
        assert q.shape[3] == 256 and q.shape[1] == k.shape[1]
        score_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(q.element_type, cutlass.Float32, (16, 8, 16)),
            (4, 2, 1),
            permutation_mnk=(64, 32, 16),
        )
        gradient_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(q.element_type, cutlass.Float32, (16, 8, 16)),
            (1, 8, 1),
            permutation_mnk=(32, 128, 16),
        )
        self.kernel_values(q, k, v, maximum, logsum, dout, delta, base, dk, dv, scale, score_mma, gradient_mma).launch(
            grid=launch_grid(cute.ceil_div(k.shape[2], 32), q.shape[0] * q.shape[1]), block=(256, 1, 1), stream=stream
        )

    @cute.jit
    def register_b(self, source, mma, thread, tid):
        values = thread.make_fragment_B(thread.partition_B(source))
        copy = cute.make_tiled_copy_B(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), source.element_type), mma
        )
        copier = copy.get_slice(tid)
        src, dst = copier.partition_S(source), copier.retile(values)
        for chunk in cutlass.range_constexpr(cute.size(src.shape[2])):
            cute.copy(copy, src[None, None, chunk], dst[None, None, chunk])
        return values

    @cute.jit
    def dot_with_b(self, source, values, mma, thread, tid):
        result = cute.make_rmem_tensor(thread.partition_shape_C((64, 32)), cutlass.Float32)
        result.fill(0.0)
        operand = thread.make_fragment_A(thread.partition_A(cute.local_tile(source, (64, 32), (0, 0))))
        copy = cute.make_tiled_copy_A(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), source.element_type), mma
        )
        copier = copy.get_slice(tid)
        src, dst = copier.partition_S(source), copier.retile(operand)
        cute.copy(copy, src[None, None, 0], dst[None, None, 0])
        for chunk in cutlass.range_constexpr(16):
            if cutlass.const_expr(chunk + 1 < 16):
                cute.copy(copy, src[None, None, chunk + 1], dst[None, None, (chunk + 1) % 2])
            cute.gemm(mma, result, operand[None, None, chunk % 2], values[None, None, chunk], result)
        return result

    @cute.jit
    def transposed_product(self, weights, source, mma, thread, tid, result):
        transposed_weights = cute.composition(weights, cute.make_layout((32, 64), stride=(64, 1)))
        a = thread.make_fragment_A(thread.partition_A(transposed_weights))
        copy_a = cute.make_tiled_copy_A(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), source.element_type), mma
        )
        copier_a = copy_a.get_slice(tid)
        sa, da = copier_a.partition_S(transposed_weights), copier_a.retile(a)
        transposed_source = cute.composition(source, cute.make_layout((256, 64), stride=(64, 1)))
        b = thread.make_fragment_B(thread.partition_B(transposed_source))
        copy_b = cute.make_tiled_copy_B(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), source.element_type), mma
        )
        copier_b = copy_b.get_slice(tid)
        sb, db = copier_b.partition_S(transposed_source), copier_b.retile(b)
        for chunk in cutlass.range_constexpr(4):
            cute.copy(copy_a, sa[None, None, chunk], da[None, None, chunk])
            cute.copy(copy_b, sb[None, None, chunk], db[None, None, chunk])
            cute.gemm(mma, result, a[None, None, chunk], b[None, None, chunk], result)

    @cute.kernel
    def kernel_values(self, q, k, v, maximum, logsum, dout, delta, base, dk, dv, scale, score_mma, gradient_mma):
        tid, _, _ = cute.arch.thread_idx()
        tile, bh, _ = block_indices(cute.ceil_div(k.shape[2], 32), q.shape[0] * q.shape[1])
        batch, head, start = bh // q.shape[1], bh % q.shape[1], tile * 32
        score_thread, gradient_thread = score_mma.get_slice(tid), gradient_mma.get_slice(tid)
        allocator = SmemAllocator()
        atom = cute.make_composed_layout(cute.make_swizzle(2, 3, 3), 0, cute.make_layout((8, 32), stride=(32, 1)))
        key_layout = cute.tile_to_shape(atom, (32, 256), (0, 1))
        query_layout = cute.tile_to_shape(atom, (64, 256), (0, 1))
        weight_layout = cute.tile_to_shape(atom, (64, 32), (0, 1))
        keys = self.shared(allocator, q.element_type, key_layout)
        values = self.shared(allocator, q.element_type, key_layout)
        queries = self.shared(allocator, q.element_type, query_layout)
        gradients = self.shared(allocator, q.element_type, query_layout)
        derivatives = cute.make_tensor(values.iterator, weight_layout)
        probabilities = cute.make_tensor(values.iterator + cute.cosize(weight_layout), weight_layout)
        self.load(k, keys, batch, head, start, tid)
        self.load(v, values, batch, head, start, tid)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        resident_values = None
        if cutlass.const_expr(self.resident_v):
            resident_values = self.register_b(values, score_mma, score_thread, tid)
        # Every reader finishes the initial V preload before this buffer can
        # become P/dS, including when faster warps enter the first query tile.
        cute.arch.sync_threads()
        accum_k = cute.make_rmem_tensor(gradient_thread.partition_shape_C((32, 256)), cutlass.Float32)
        accum_v = cute.make_rmem_tensor(gradient_thread.partition_shape_C((32, 256)), cutlass.Float32)
        accum_k.fill(0.0)
        accum_v.fill(0.0)
        coords = score_thread.partition_C(cute.make_identity_tensor((64, 32)))
        rc = row_view(coords)
        magnitude = scale if scale >= 0.0 else -scale
        sign = cutlass.Float32(1.0) if scale >= 0.0 else cutlass.Float32(-1.0)
        first = start // 64 if cutlass.const_expr(self.causal) else 0
        for block in range(first, cute.ceil_div(q.shape[2], 64)):
            query = block * 64
            self.load(q, queries, batch, head, query, tid)
            self.load(dout, gradients, batch, head, query, tid)
            if cutlass.const_expr(not self.resident_v):
                self.load(v, values, batch, head, start, tid)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            row_max = cute.make_rmem_tensor((cute.size(rc.shape[0]),), cutlass.Float32)
            row_logsum = cute.make_rmem_tensor(row_max.shape, cutlass.Float32)
            row_center = cute.make_rmem_tensor(row_max.shape, cutlass.Float32)
            row_residual = cute.make_rmem_tensor(row_max.shape, cutlass.Float32)
            for r in cutlass.range_constexpr(cute.size(rc.shape[0])):
                qr = query + rc[r, 0][0]
                row_max[r], row_logsum[r], row_center[r], row_residual[r] = 0.0, 0.0, 0.0, 0.0
                if qr < q.shape[2]:
                    row_max[r] = maximum[batch, head, qr]
                    row_logsum[r] = logsum[batch, head, qr]
                    row_center[r] = base[batch, head, qr]
                    row_residual[r] = delta[batch, head, qr]
            scores = self.score.dot(queries, keys, score_mma, score_thread, tid)
            if cutlass.const_expr(self.resident_v):
                dp = self.dot_with_b(gradients, resident_values, score_mma, score_thread, tid)
            else:
                dp = self.score.dot(gradients, values, score_mma, score_thread, tid)
            sv, pv = row_view(scores), row_view(dp)
            for r in cutlass.range_constexpr(cute.size(rc.shape[0])):
                for c in cutlass.range_constexpr(cute.size(rc.shape[1])):
                    row, col = rc[r, c]
                    visible = query + row < q.shape[2] and start + col < k.shape[2]
                    if cutlass.const_expr(self.causal):
                        visible = visible and start + col <= query + row
                    probability = cutlass.Float32(0.0)
                    if visible:
                        probability = cute.math.exp2(
                            (sv[r, c] * sign - row_max[r]) * magnitude * 1.4426950408889634 - row_logsum[r],
                            fastmath=True,
                        )
                    sv[r, c] = probability * ((pv[r, c] - row_center[r]) - row_residual[r])
                    pv[r, c] = probability
            rounded_scores = cute.make_fragment_like(scores, q.element_type)
            rounded_probabilities = cute.make_fragment_like(dp, q.element_type)
            packed_half_conversion(scores, rounded_scores)
            packed_half_conversion(dp, rounded_probabilities)
            if cutlass.const_expr(not self.resident_v or self.release_barrier):
                cute.arch.sync_threads()
            self.store_packed_weights(rounded_scores, derivatives, coords)
            self.store_packed_weights(rounded_probabilities, probabilities, coords)
            cute.arch.sync_threads()
            self.transposed_product(derivatives, queries, gradient_mma, gradient_thread, tid, accum_k)
            self.transposed_product(probabilities, gradients, gradient_mma, gradient_thread, tid, accum_v)
            # All gradient readers release the aliased V/weight planes and
            # current Q/dO buffers before the next iteration's copies.
            cute.arch.sync_threads()
        self.store_result(accum_k, dk, keys, batch, head, start, 0, gradient_thread, tid, scale)
        self.store_result(accum_v, dv, keys, batch, head, start, 0, gradient_thread, tid, cutlass.Float32(1.0))


class QueryMajorQueries(Attention):
    """Keep centered query math while giving the large K operand four N warps."""

    def __init__(self, causal, vector):
        super().__init__(
            "q", causal, False, columns=32, width=256, warps=8, centered=True, vector=vector, query_column_warps=4
        )
        self.rows = self.grid_rows = 64
        self.score = Attention("forward", causal, False, columns=32, width=256, warps=8, vector=vector)
        self.score.rows = self.score.grid_rows = 64
        self.score.column_warps = 2

    @cute.jit
    def __call__(
        self, q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream: cuda.CUstream
    ):
        score_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(q.element_type, cutlass.Float32, (16, 8, 16)),
            (4, 2, 1),
            permutation_mnk=(64, 32, 16),
        )
        gradient_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(q.element_type, cutlass.Float32, (16, 8, 16)),
            (2, 4, 1),
            permutation_mnk=(64, 64, 16),
        )
        self.kernel_queries(q, k, v, maximum, logsum, dout, delta, base, dq, scale, score_mma, gradient_mma).launch(
            grid=launch_grid(cute.ceil_div(q.shape[2], 64), q.shape[0] * q.shape[1]), block=(256, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel_queries(self, q, k, v, maximum, logsum, dout, delta, base, dq, scale, score_mma, gradient_mma):
        tid, _, _ = cute.arch.thread_idx()
        tile, bh, _ = block_indices(cute.ceil_div(q.shape[2], 64), q.shape[0] * q.shape[1])
        batch, head, start = bh // q.shape[1], bh % q.shape[1], tile * 64
        score_thread, gradient_thread = score_mma.get_slice(tid), gradient_mma.get_slice(tid)
        allocator = SmemAllocator()
        atom = cute.make_composed_layout(cute.make_swizzle(2, 3, 3), 0, cute.make_layout((8, 32), stride=(32, 1)))
        query_layout = cute.tile_to_shape(atom, (64, 256), (0, 1))
        key_layout = cute.tile_to_shape(atom, (32, 256), (0, 1))
        weight_layout = cute.tile_to_shape(atom, (64, 32), (0, 1))
        queries = self.shared(allocator, q.element_type, query_layout)
        gradients = self.shared(allocator, q.element_type, query_layout)
        keys = self.shared(allocator, q.element_type, key_layout)
        values = self.shared(allocator, q.element_type, key_layout)
        derivatives = cute.make_tensor(values.iterator, weight_layout)
        probabilities = cute.make_tensor(values.iterator + cute.cosize(weight_layout), weight_layout)
        # Score N-warps own separate row sums. The gradient MMA has a different
        # row layout, so its epilogue gathers by coordinate after this reduction.
        shared_rows = allocator.allocate_tensor(
            cutlass.Float32, cute.make_layout((2, 2, 64), stride=(128, 64, 1)), byte_alignment=128
        )
        self.load(q, queries, batch, head, start, tid)
        self.load(dout, gradients, batch, head, start, tid)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        accum = cute.make_rmem_tensor(gradient_thread.partition_shape_C((64, 256)), cutlass.Float32)
        mean = cute.make_fragment_like(accum)
        accum.fill(0.0)
        mean.fill(0.0)
        score_coords = score_thread.partition_C(cute.make_identity_tensor((64, 32)))
        score_rows = row_view(score_coords)
        totals = cute.make_rmem_tensor((cute.size(score_rows.shape[0]),), cutlass.Float32)
        rounded = cute.make_fragment_like(totals)
        row_max, row_logsum, center = (
            cute.make_fragment_like(totals),
            cute.make_fragment_like(totals),
            cute.make_fragment_like(totals),
        )
        totals.fill(0.0)
        rounded.fill(0.0)
        for r in cutlass.range_constexpr(cute.size(score_rows.shape[0])):
            qr = start + score_rows[r, 0][0]
            row_max[r], row_logsum[r], center[r] = 0.0, 0.0, 0.0
            if qr < q.shape[2]:
                row_max[r], row_logsum[r], center[r] = (
                    maximum[batch, head, qr],
                    logsum[batch, head, qr],
                    base[batch, head, qr],
                )
        magnitude = scale if scale >= 0.0 else -scale
        sign = cutlass.Float32(1.0) if scale >= 0.0 else cutlass.Float32(-1.0)
        limit = cutlass.min(k.shape[2], start + 64) if cutlass.const_expr(self.causal) else k.shape[2]
        for block in range(cute.ceil_div(limit, 32)):
            key = block * 32
            self.load(k, keys, batch, head, key, tid)
            self.load(v, values, batch, head, key, tid)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            scores = self.score.dot(queries, keys, score_mma, score_thread, tid)
            dp = self.score.dot(gradients, values, score_mma, score_thread, tid)
            sv, pv = row_view(scores), row_view(dp)
            for r in cutlass.range_constexpr(cute.size(score_rows.shape[0])):
                for c in cutlass.range_constexpr(cute.size(score_rows.shape[1])):
                    row, col = score_rows[r, c]
                    visible = start + row < q.shape[2] and key + col < k.shape[2]
                    if cutlass.const_expr(self.causal):
                        visible = visible and key + col <= start + row
                    probability = cutlass.Float32(0.0)
                    if visible:
                        probability = cute.math.exp2(
                            (sv[r, c] * sign - row_max[r]) * magnitude * 1.4426950408889634 - row_logsum[r], fastmath=True
                        )
                    sv[r, c] = probability * (pv[r, c] - center[r])
                    pv[r, c] = probability
                totals[r] += quad_sum(sv[r, None].load().reduce(cute.ReductionOp.ADD, 0.0, 0))
            rounded_scores = cute.make_fragment_like(scores, q.element_type)
            rounded_probabilities = cute.make_fragment_like(dp, q.element_type)
            packed_half_conversion(scores, rounded_scores)
            packed_half_conversion(dp, rounded_probabilities)
            rounded_view = row_view(rounded_scores)
            for r in cutlass.range_constexpr(cute.size(score_rows.shape[0])):
                rounded[r] += quad_sum(rounded_view[r, None].load().to(cutlass.Float32).reduce(cute.ReductionOp.ADD, 0.0, 0))
            # Release every V reader before any score warp packs P/dS there.
            cute.arch.sync_threads()
            self.store_packed_weights(rounded_scores, derivatives, score_coords)
            self.store_packed_weights(rounded_probabilities, probabilities, score_coords)
            cute.arch.sync_threads()
            self.dual_product(derivatives, probabilities, keys, 0, gradient_mma, gradient_thread, tid, accum, mean)
            cute.arch.sync_threads()
        if tid % 4 == 0:
            for r in cutlass.range_constexpr(cute.size(score_rows.shape[0])):
                row = score_rows[r, 0][0]
                shared_rows[0, tid // 128, row] = totals[r]
                shared_rows[1, tid // 128, row] = rounded[r]
        cute.arch.sync_threads()
        if tid < 128 and tid % 4 == 0:
            for r in cutlass.range_constexpr(cute.size(score_rows.shape[0])):
                row = score_rows[r, 0][0]
                if start + row < q.shape[2]:
                    delta[batch, head, start + row] = shared_rows[0, 0, row] + shared_rows[0, 1, row]
        gradient_rows = row_view(gradient_thread.partition_C(cute.make_identity_tensor((64, 256))))
        av, mv = row_view(accum), row_view(mean)
        for r in cutlass.range_constexpr(cute.size(gradient_rows.shape[0])):
            row = gradient_rows[r, 0][0]
            mass = shared_rows[1, 0, row] + shared_rows[1, 1, row]
            av[r, None] = av[r, None].load() - mass * mv[r, None].load()
        self.store_result(accum, dq, queries, batch, head, start, 0, gradient_thread, tid, scale)
