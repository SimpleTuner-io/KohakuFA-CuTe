"""Experimental centered prepass and key-owned joint backward for SM89.

This is an explicit analysis prototype, outside automatic kernel selection.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import warp
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.memory import SmemAllocator

from kohakufa_cute.kernels import Attention, packed_half_conversion, quad_sum, row_view
from kohakufa_cute.launch import launch_grid


@dsl_user_op
def atomic_add_f32(pointer, value, *, loc=None, ip=None):
    """One FP32 contribution; output conversion happens after all owners finish."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Int64(pointer.toint()).ir_value(loc=loc, ip=ip), cutlass.Float32(value).ir_value(loc=loc, ip=ip)],
            "{ red.relaxed.gpu.global.add.f32 [$1], $2; mov.u32 $0, 0; }",
            "=r,l,f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


class Delta(Attention):
    """Only compute the complete-row centered residual, without gradient GEMMs."""

    def __init__(self, causal, dimension, pivot_correction=False, cache_global=False, columns=32):
        super().__init__(
            "q",
            causal,
            False,
            columns=columns,
            width=dimension,
            warps=4,
            centered=True,
            warp_centering=True,
            copy_cache_global=cache_global,
        )
        self.pivot_correction = pivot_correction

    @cute.jit
    def __call__(self, q, k, v, out, maximum, logsum, dout, delta, base, pivots, stream: cuda.CUstream):
        mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(q.element_type, cutlass.Float32, (16, 8, 16)),
            (4, 1, 1),
            permutation_mnk=(64, 16, 16),
        )
        self.kernel_delta(q, k, v, out, maximum, logsum, dout, delta, base, pivots, mma).launch(
            grid=launch_grid(cute.ceil_div(q.shape[2], 64), q.shape[0] * q.shape[1]), block=(128, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel_delta(self, q, k, v, out, maximum, logsum, dout, delta, base, pivots, mma):
        tid, _, _ = cute.arch.thread_idx()
        tile, bh, _ = cute.arch.block_idx()
        batch, head = bh // q.shape[1], bh % q.shape[1]
        start = tile * 64
        atom = cute.make_composed_layout(cute.make_swizzle(2, 3, 3), 0, cute.make_layout((8, 32), stride=(32, 1)))
        alloc = SmemAllocator()
        left = self.shared(alloc, q.element_type, cute.tile_to_shape(atom, (64, self.width), (0, 1)))
        other_left = self.shared(alloc, q.element_type, left.layout)
        right = self.shared(alloc, q.element_type, cute.tile_to_shape(atom, (self.columns, self.width), (0, 1)))
        other_right = self.shared(alloc, q.element_type, right.layout)
        center_base = alloc.allocate_tensor(cutlass.Float32, cute.make_layout((64,)), byte_alignment=128)
        self.load(q, left, batch, head, start, tid)
        self.load(dout, other_left, batch, head, start, tid)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        center_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(q.element_type, cutlass.Float32, (16, 8, 16)), (1, 1, 1), permutation_mnk=(16, 16, 16)
        )
        center_thread = center_mma.get_slice(tid % 32)
        center_coords = row_view(center_thread.partition_C(cute.make_identity_tensor((16, 16))))
        for part in cutlass.range_constexpr(2):
            self.load(out, other_right, batch, head, start + part * 32, tid)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            if tid < 64:
                group = tid // 32
                do_part = cute.local_tile(other_left, (16, self.width), (part * 2 + group, 0))
                out_part = cute.local_tile(other_right, (16, self.width), (group, 0))
                value = self.center_loader.dot(do_part, out_part, center_mma, center_thread, tid % 32)
                value_view = row_view(value)
                for r in cutlass.range_constexpr(cute.size(center_coords.shape[0])):
                    diagonal = cutlass.Float32(0.0)
                    for c in cutlass.range_constexpr(cute.size(center_coords.shape[1])):
                        if center_coords[r, c][0] == center_coords[r, c][1]:
                            diagonal = value_view[r, c]
                    diagonal = quad_sum(diagonal)
                    row = part * 32 + group * 16 + center_coords[r, 0][0]
                    if tid % 4 == 0:
                        center_base[row] = diagonal
                        if start + row < q.shape[2]:
                            base[batch, head, start + row] = diagonal
            cute.arch.sync_threads()
        thread = mma.get_slice(tid)
        coords = row_view(thread.partition_C(cute.make_identity_tensor((64, self.columns))))
        total = cute.make_rmem_tensor((cute.size(coords.shape[0]),), cutlass.Float32)
        total.fill(0.0)
        if cutlass.const_expr(self.pivot_correction):
            best = cute.make_rmem_tensor(total.shape, cutlass.Float32)
            best_index = cute.make_rmem_tensor(total.shape, cutlass.Int32)
            best.fill(-float("inf"))
            best_index.fill(2147483647)
        limit = cutlass.min(k.shape[2], start + 64) if cutlass.const_expr(self.causal) else k.shape[2]
        for block in range(cute.ceil_div(limit, self.columns)):
            self.load(k, right, batch, head, block * self.columns, tid)
            self.load(v, other_right, batch, head, block * self.columns, tid)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            scores = self.dot(left, right, mma, thread, tid)
            dp = self.dot(other_left, other_right, mma, thread, tid)
            sv, pv = row_view(scores), row_view(dp)
            for r in cutlass.range_constexpr(cute.size(coords.shape[0])):
                qr = start + coords[r, 0][0]
                row_max, row_logsum, center = cutlass.Float32(0.0), cutlass.Float32(0.0), cutlass.Float32(0.0)
                if qr < q.shape[2]:
                    row_max, row_logsum, center = (
                        maximum[batch, head, qr],
                        logsum[batch, head, qr],
                        center_base[coords[r, 0][0]],
                    )
                for c in cutlass.range_constexpr(cute.size(coords.shape[1])):
                    kr = block * self.columns + coords[r, c][1]
                    p = cutlass.Float32(0.0)
                    if qr < q.shape[2] and kr < k.shape[2]:
                        if not self.causal or kr <= qr:
                            if cutlass.const_expr(self.pivot_correction):
                                if sv[r, c] > best[r] or (sv[r, c] == best[r] and kr < best_index[r]):
                                    best[r], best_index[r] = sv[r, c], kr
                            p = cute.math.exp2(
                                (sv[r, c] - row_max) * (self.width**-0.5) * 1.4426950408889634 - row_logsum,
                                fastmath=True,
                            )
                    sv[r, c] = p * (pv[r, c] - center)
                total[r] += quad_sum(sv[r, None].load().reduce(cute.ReductionOp.ADD, 0.0, 0))
            cute.arch.sync_threads()
        for r in cutlass.range_constexpr(cute.size(coords.shape[0])):
            qr = start + coords[r, 0][0]
            if cutlass.const_expr(self.pivot_correction):
                for step in cutlass.range_constexpr(2):
                    shift = 2 >> step
                    other_value = cute.arch.shuffle_sync_bfly(best[r], offset=shift, mask=-1, mask_and_clamp=31)
                    other_index = cute.arch.shuffle_sync_bfly(best_index[r], offset=shift, mask=-1, mask_and_clamp=31)
                    if other_value > best[r] or (other_value == best[r] and other_index < best_index[r]):
                        best[r], best_index[r] = other_value, other_index
            if tid % 4 == 0 and qr < q.shape[2]:
                delta[batch, head, qr] = total[r]
                if cutlass.const_expr(self.pivot_correction):
                    delta[batch, head, q.shape[2] + qr] = cutlass.Float32(0.0)
                    pivots[batch, head, qr] = best_index[r]


class Joint(Attention):
    """Keep K/V and dK/dV local while sharing one score pair with dQ."""

    def __init__(
        self,
        causal,
        dimension,
        compensated,
        pivot_correction=False,
        cache_global=False,
        direct_dq=False,
        query_column_warps=2,
        gradient_column_warps=2,
        packed_weights=False,
        overlap_query=False,
        query_row_scores=False,
        resident_kv=0,
        reduce_score_mass=False,
        early_probability=False,
        opaque_accumulator=False,
    ):
        super().__init__(
            "kv",
            causal,
            False,
            columns=64,
            width=dimension,
            warps=8,
            centered=True,
            kv_column_warps=2,
            kv_column_statistics=True,
            copy_cache_global=cache_global,
        )
        self.compensated = compensated
        self.direct_dq = direct_dq
        self.query_column_warps = query_column_warps
        self.gradient_column_warps = gradient_column_warps
        self.packed_weights = packed_weights
        self.overlap_query = overlap_query
        self.query_row_scores = query_row_scores
        self.resident_kv = resident_kv
        self.reduce_score_mass = reduce_score_mass
        self.early_probability = early_probability
        self.opaque_accumulator = opaque_accumulator
        self.pivot_correction = pivot_correction

    @cute.jit
    def __call__(self, q, k, v, maximum, logsum, dout, delta, dq_accum, dk, dv, base, stream: cuda.CUstream):
        mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(q.element_type, cutlass.Float32, (16, 8, 16)),
            (4, 2, 1),
            permutation_mnk=(64, 32, 16),
        )
        self.kernel_joint(q, k, v, maximum, logsum, dout, delta, dq_accum, dk, dv, base, mma).launch(
            grid=launch_grid(cute.ceil_div(k.shape[2], 64), q.shape[0] * q.shape[1]), block=(256, 1, 1), stream=stream
        )

    @cute.jit
    def transposed_query_product(self, weights, source, mma, thread, tid, accum):
        transposed_weights = cute.composition(weights, cute.make_layout((64, 64), stride=(64, 1)))
        ra = thread.make_fragment_A(thread.partition_A(transposed_weights))
        copy_a = cute.make_tiled_copy_A(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), source.element_type), mma
        )
        copier_a = copy_a.get_slice(tid)
        sa, da = copier_a.partition_S(transposed_weights), copier_a.retile(ra)
        transposed_source = cute.composition(source, cute.make_layout((self.width, 64), stride=(64, 1)))
        rb = thread.make_fragment_B(thread.partition_B(transposed_source))
        copy_b = cute.make_tiled_copy_B(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), source.element_type), mma
        )
        copier_b = copy_b.get_slice(tid)
        sb, db = copier_b.partition_S(transposed_source), copier_b.retile(rb)
        for chunk in cutlass.range_constexpr(4):
            cute.copy(copy_a, sa[None, None, chunk], da[None, None, chunk])
            cute.copy(copy_b, sb[None, None, chunk], db[None, None, chunk])
            cute.gemm(mma, accum, ra[None, None, chunk], rb[None, None, chunk], accum)

    @cute.jit
    def register_operand_b(self, source, mma, thread, tid):
        registers = thread.make_fragment_B(thread.partition_B(source))
        copy = cute.make_tiled_copy_B(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), source.element_type), mma
        )
        copier = copy.get_slice(tid)
        src, dst = copier.partition_S(source), copier.retile(registers)
        for chunk in cutlass.range_constexpr(cute.size(src.shape[2])):
            cute.copy(copy, src[None, None, chunk], dst[None, None, chunk])
        return registers

    @cute.jit
    def dot_resident_b(self, source, registers, mma, thread, tid):
        result = cute.make_rmem_tensor(thread.partition_shape_C((64, 64)), cutlass.Float32)
        result.fill(0.0)
        operand = thread.make_fragment_A(thread.partition_A(cute.local_tile(source, (64, 32), (0, 0))))
        copy = cute.make_tiled_copy_A(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), source.element_type), mma
        )
        copier = copy.get_slice(tid)
        src, dst = copier.partition_S(source), copier.retile(operand)
        cute.copy(copy, src[None, None, 0], dst[None, None, 0])
        for chunk in cutlass.range_constexpr(self.width // 16):
            if cutlass.const_expr(chunk + 1 < self.width // 16):
                cute.copy(copy, src[None, None, chunk + 1], dst[None, None, (chunk + 1) % 2])
            cute.gemm(mma, result, operand[None, None, chunk % 2], registers[None, None, chunk], result)
        return result

    @cute.kernel
    def kernel_joint(self, q, k, v, maximum, logsum, dout, delta, dq_accum, dk, dv, base, mma):
        tid, _, _ = cute.arch.thread_idx()
        tile, bh, _ = cute.arch.block_idx()
        batch, head = bh // q.shape[1], bh % q.shape[1]
        start = tile * 64
        alloc = SmemAllocator()
        atom = cute.make_composed_layout(cute.make_swizzle(2, 3, 3), 0, cute.make_layout((8, 32), stride=(32, 1)))
        operand_layout = cute.tile_to_shape(atom, (64, self.width), (0, 1))
        left = self.shared(alloc, q.element_type, operand_layout)
        other_left = self.shared(alloc, q.element_type, operand_layout)
        if cutlass.const_expr(self.overlap_query):
            query_stages = self.shared(alloc, q.element_type, cute.tile_to_shape(atom, (64, self.width, 2), (0, 1, 2)))
            right = query_stages[None, None, 0]
        else:
            right = self.shared(alloc, q.element_type, operand_layout)
        other_right = self.shared(alloc, q.element_type, operand_layout)
        weights_layout = cute.tile_to_shape(atom, (64, 64), (0, 1))
        ds = self.shared(alloc, q.element_type, weights_layout)
        ps = self.shared(alloc, q.element_type, weights_layout)
        if cutlass.const_expr(self.reduce_score_mass):
            shared_mass = alloc.allocate_tensor(
                cutlass.Float32, cute.make_layout((64, 2), stride=(2, 1)), byte_alignment=128
            )
        low_ds = None
        if cutlass.const_expr(self.compensated):
            low_ds = self.shared(alloc, q.element_type, weights_layout)
        self.load(k, left, batch, head, start, tid)
        self.load(v, other_left, batch, head, start, tid)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        thread = mma.get_slice(tid)
        if cutlass.const_expr(self.resident_kv > 0):
            resident_v = self.register_operand_b(other_left, mma, thread, tid)
        if cutlass.const_expr(self.resident_kv > 1):
            resident_k = self.register_operand_b(left, mma, thread, tid)
        gradient_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(q.element_type, cutlass.Float32, (16, 8, 16)),
            (8 // self.gradient_column_warps, self.gradient_column_warps, 1),
            permutation_mnk=(16 * (8 // self.gradient_column_warps), 16 * self.gradient_column_warps, 16),
        )
        gradient_thread = gradient_mma.get_slice(tid)
        accum_k = cute.make_rmem_tensor(gradient_thread.partition_shape_C((64, self.width)), cutlass.Float32)
        accum_v = cute.make_fragment_like(accum_k)
        accum_k.fill(0.0)
        accum_v.fill(0.0)
        score_coords = thread.partition_C(cute.make_identity_tensor((64, 64)))
        query_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(q.element_type, cutlass.Float32, (16, 8, 16)),
            (8 // self.query_column_warps, self.query_column_warps, 1),
            permutation_mnk=(16 * (8 // self.query_column_warps), 16 * self.query_column_warps, 16),
        )
        query_thread = query_mma.get_slice(tid)
        q_coords = query_thread.partition_C(cute.make_identity_tensor((64, self.width)))
        first = start // 64 if cutlass.const_expr(self.causal) else 0
        stop = cute.ceil_div(q.shape[2], 64)
        if cutlass.const_expr(self.overlap_query):
            if first < stop:
                self.load(q, right, batch, head, first * 64, tid)
                self.load(dout, other_right, batch, head, first * 64, tid)
                cute.arch.cp_async_commit_group()
        for block in range(first, stop):
            query = block * 64
            if cutlass.const_expr(self.overlap_query):
                right = query_stages[None, None, (block - first) % 2]
            else:
                self.load(q, right, batch, head, query, tid)
                self.load(dout, other_right, batch, head, query, tid)
                cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            if cutlass.const_expr(self.query_row_scores):
                if cutlass.const_expr(self.resident_kv > 1):
                    scores = self.dot_resident_b(right, resident_k, mma, thread, tid)
                else:
                    scores = self.dot(right, left, mma, thread, tid)
                if cutlass.const_expr(self.early_probability):
                    early = cute.make_fragment_like(scores)
                    coordinates = row_view(score_coords)
                    values, early_view = row_view(scores), row_view(early)
                    for r in cutlass.range_constexpr(cute.size(coordinates.shape[0])):
                        qr = query + coordinates[r, 0][0]
                        row_max, row_logsum = cutlass.Float32(0.0), cutlass.Float32(0.0)
                        if qr < q.shape[2]:
                            row_max, row_logsum = maximum[batch, head, qr], logsum[batch, head, qr]
                        for c in cutlass.range_constexpr(cute.size(coordinates.shape[1])):
                            kr = start + coordinates[r, c][1]
                            visible = qr < q.shape[2] and kr < k.shape[2]
                            if cutlass.const_expr(self.causal):
                                visible = visible and kr <= qr
                            probability = cutlass.Float32(0.0)
                            if visible:
                                probability = cute.math.exp2(
                                    (values[r, c] - row_max) * (self.width**-0.5) * 1.4426950408889634 - row_logsum,
                                    fastmath=True,
                                )
                            early_view[r, c] = probability
                    early_rounded = cute.make_fragment_like(early, q.element_type)
                    packed_half_conversion(early, early_rounded)
                    self.store_packed_weights(early_rounded, ps, score_coords)
                if cutlass.const_expr(self.resident_kv > 0):
                    probabilities = self.dot_resident_b(other_right, resident_v, mma, thread, tid)
                else:
                    probabilities = self.dot(other_right, other_left, mma, thread, tid)
                coordinates = row_view(score_coords)
                values, derivatives = row_view(scores), row_view(probabilities)
                if cutlass.const_expr(self.early_probability):
                    early_view = row_view(early)
                for r in cutlass.range_constexpr(cute.size(coordinates.shape[0])):
                    qr = query + coordinates[r, 0][0]
                    row_max, row_logsum, residual, center = (cutlass.Float32(0.0),) * 4
                    if qr < q.shape[2]:
                        if cutlass.const_expr(not self.early_probability):
                            row_max, row_logsum = maximum[batch, head, qr], logsum[batch, head, qr]
                        residual, center = delta[batch, head, qr], base[batch, head, qr]
                    for c in cutlass.range_constexpr(cute.size(coordinates.shape[1])):
                        kr = start + coordinates[r, c][1]
                        visible = qr < q.shape[2] and kr < k.shape[2]
                        if cutlass.const_expr(self.causal):
                            visible = visible and kr <= qr
                        probability, derivative = cutlass.Float32(0.0), cutlass.Float32(0.0)
                        if visible:
                            if cutlass.const_expr(self.early_probability):
                                probability = early_view[r, c]
                            else:
                                probability = cute.math.exp2(
                                    (values[r, c] - row_max) * (self.width**-0.5) * 1.4426950408889634 - row_logsum,
                                    fastmath=True,
                                )
                            derivative = probability * ((derivatives[r, c] - center) - residual)
                        values[r, c], derivatives[r, c] = derivative, probability
            else:
                scores, probabilities = self.kv_weights(
                    left,
                    right,
                    other_left,
                    other_right,
                    q,
                    (maximum, logsum, delta, base),
                    (batch, head, start, query, q.shape[2], k.shape[2]),
                    query >= start + 64,
                    self.width**-0.5,
                    cutlass.Float32(1.0),
                    mma,
                    thread,
                    tid,
                )
            if cutlass.const_expr(self.packed_weights):
                rounded_scores = cute.make_fragment_like(scores, q.element_type)
                rounded_probabilities = cute.make_fragment_like(probabilities, q.element_type)
                packed_half_conversion(scores, rounded_scores)
                packed_half_conversion(probabilities, rounded_probabilities)
                self.store_packed_weights(rounded_scores, ds, score_coords)
                if cutlass.const_expr(not self.early_probability):
                    self.store_packed_weights(rounded_probabilities, ps, score_coords)
                if cutlass.const_expr(self.compensated):
                    residual = cute.make_fragment_like(scores)
                    for i in cutlass.range_constexpr(cute.size(score_coords)):
                        residual[i] = scores[i] - rounded_scores[i].to(cutlass.Float32)
                    rounded_residual = cute.make_fragment_like(scores, q.element_type)
                    packed_half_conversion(residual, rounded_residual)
                    self.store_packed_weights(rounded_residual, low_ds, score_coords)
            else:
                for i in cutlass.range_constexpr(cute.size(score_coords)):
                    row, col = score_coords[i]
                    hi = scores[i].to(q.element_type)
                    ds[row, col] = hi
                    ps[row, col] = probabilities[i].to(q.element_type)
                    if cutlass.const_expr(self.compensated):
                        low_ds[row, col] = (scores[i] - hi.to(cutlass.Float32)).to(q.element_type)
            if cutlass.const_expr(not self.reduce_score_mass):
                cute.arch.sync_threads()
            if cutlass.const_expr(self.reduce_score_mass):
                rounded = rounded_scores
                rounded_view, mass_coords = row_view(rounded), row_view(score_coords)
                for r in cutlass.range_constexpr(cute.size(mass_coords.shape[0])):
                    mass = cutlass.Float32(0.0)
                    for c in cutlass.range_constexpr(cute.size(mass_coords.shape[1])):
                        mass += rounded_view[r, c].to(cutlass.Float32)
                    mass = quad_sum(mass)
                    if tid % 4 == 0:
                        shared_mass[mass_coords[r, 0][0], tid // 128] = mass
                cute.arch.sync_threads()
            if cutlass.const_expr(self.overlap_query):
                if block + 1 < stop:
                    # The alternate Q slot was released by the previous dK/barrier.
                    next_query = query_stages[None, None, (block - first + 1) % 2]
                    self.load(q, next_query, batch, head, query + 64, tid)
                    cute.arch.cp_async_commit_group()
            if cutlass.const_expr(self.query_row_scores):
                self.transposed_query_product(ps, other_right, gradient_mma, gradient_thread, tid, accum_v)
            else:
                self.product(ps, other_right, 0, gradient_mma, gradient_thread, tid, accum_v)
            if cutlass.const_expr(self.overlap_query):
                # dV is the last reader of dO; every warp must release it.
                cute.arch.sync_threads()
                if block + 1 < stop:
                    self.load(dout, other_right, batch, head, query + 64, tid)
                    cute.arch.cp_async_commit_group()
            else:
                if cutlass.const_expr(self.query_row_scores):
                    self.transposed_query_product(ds, right, gradient_mma, gradient_thread, tid, accum_k)
                else:
                    self.product(ds, right, 0, gradient_mma, gradient_thread, tid, accum_k)
            partial_q = cute.make_rmem_tensor(query_thread.partition_shape_C((64, self.width)), cutlass.Float32)
            partial_q.fill(0.0)
            if cutlass.const_expr(self.query_row_scores):
                self.product(ds, left, 0, query_mma, query_thread, tid, partial_q)
                if cutlass.const_expr(self.compensated):
                    self.product(low_ds, left, 0, query_mma, query_thread, tid, partial_q)
            elif cutlass.const_expr(self.direct_dq):
                self.transposed_query_product(ds, left, query_mma, query_thread, tid, partial_q)
                if cutlass.const_expr(self.compensated):
                    self.transposed_query_product(low_ds, left, query_mma, query_thread, tid, partial_q)
            else:
                # P is dead after dV; readers finish before transposed dS replaces it.
                cute.arch.sync_threads()
                for i in cutlass.range_constexpr(cute.size(score_coords)):
                    row, col = score_coords[i]
                    ps[col, row] = ds[row, col]
                cute.arch.sync_threads()
                self.product(ps, left, 0, query_mma, query_thread, tid, partial_q)
                if cutlass.const_expr(self.compensated):
                    for i in cutlass.range_constexpr(cute.size(score_coords)):
                        row, col = score_coords[i]
                        ds[col, row] = low_ds[row, col]
                    cute.arch.sync_threads()
                    self.product(ds, left, 0, query_mma, query_thread, tid, partial_q)
            if cutlass.const_expr(self.reduce_score_mass):
                if tid < 64 and query + tid < q.shape[2]:
                    mass = shared_mass[tid, 0] + shared_mass[tid, 1]
                    pointer = delta.iterator + cute.crd2idx((batch, head, q.shape[2] + query + tid), delta.layout)
                    atomic_add_f32(pointer, mass)
            else:
                if cutlass.const_expr(self.pivot_correction):
                    local_query = tid // 4
                    mass = cutlass.Float32(0.0)
                    for column in cutlass.range_constexpr(16):
                        if cutlass.const_expr(self.query_row_scores):
                            mass += ds[local_query, column * 4 + tid % 4].to(cutlass.Float32)
                        elif cutlass.const_expr(self.direct_dq):
                            mass += ds[column * 4 + tid % 4, local_query].to(cutlass.Float32)
                        else:
                            mass += ps[local_query, column * 4 + tid % 4].to(cutlass.Float32)
                    mass = quad_sum(mass)
                    if tid % 4 == 0 and query + local_query < q.shape[2]:
                        pointer = delta.iterator + cute.crd2idx(
                            (batch, head, q.shape[2] + query + local_query), delta.layout
                        )
                        atomic_add_f32(pointer, mass)
            for i in cutlass.range_constexpr(cute.size(q_coords)):
                row, col = q_coords[i]
                if query + row < q.shape[2] and col < q.shape[3]:
                    if cutlass.const_expr(self.opaque_accumulator):
                        opaque_row, opaque_col = tid // 32 + 8 * (i % 8), tid % 32 + 32 * (i // 8)
                        pointer = dq_accum.iterator + cute.crd2idx(
                            (batch, head, query + opaque_row, opaque_col), dq_accum.layout
                        )
                    else:
                        pointer = dq_accum.iterator + cute.crd2idx((batch, head, query + row, col), dq_accum.layout)
                    atomic_add_f32(pointer, partial_q[i])
            if cutlass.const_expr(self.overlap_query):
                # Current Q and dS stay alive through dQ/RED and this final dK.
                if cutlass.const_expr(self.query_row_scores):
                    self.transposed_query_product(ds, right, gradient_mma, gradient_thread, tid, accum_k)
                else:
                    self.product(ds, right, 0, gradient_mma, gradient_thread, tid, accum_k)
            cute.arch.sync_threads()
        self.store_result(
            accum_k, dk, left, batch, head, start, 0, gradient_thread, tid, cutlass.Float32(self.width**-0.5)
        )
        self.store_result(accum_v, dv, left, batch, head, start, 0, gradient_thread, tid, cutlass.Float32(1.0))

    @cute.jit
    def convert_opaque(self, scratch, dq, k, delta, pivots, stream: cuda.CUstream):
        mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(dq.element_type, cutlass.Float32, (16, 8, 16)),
            (8 // self.query_column_warps, self.query_column_warps, 1),
            permutation_mnk=(16 * (8 // self.query_column_warps), 16 * self.query_column_warps, 16),
        )
        self.kernel_convert_opaque(scratch, dq, k, delta, pivots, mma).launch(
            grid=launch_grid(cute.ceil_div(dq.shape[2], 64), dq.shape[0] * dq.shape[1]),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel_convert_opaque(self, scratch, dq, k, delta, pivots, mma):
        tid, _, _ = cute.arch.thread_idx()
        tile, bh, _ = cute.arch.block_idx()
        batch, head = bh // dq.shape[1], bh % dq.shape[1]
        query = tile * 64
        thread = mma.get_slice(tid)
        coordinates = thread.partition_C(cute.make_identity_tensor((64, self.width)))
        accum = cute.make_rmem_tensor(thread.partition_shape_C((64, self.width)), cutlass.Float32)
        accum.fill(0.0)
        for i in cutlass.range_constexpr(cute.size(coordinates)):
            row, col = coordinates[i]
            if query + row < dq.shape[2]:
                opaque_row, opaque_col = tid // 32 + 8 * (i % 8), tid % 32 + 32 * (i // 8)
                value = scratch[batch, head, query + opaque_row, opaque_col]
                if cutlass.const_expr(self.pivot_correction):
                    pivot = pivots[batch, head, query + row]
                    value -= delta[batch, head, dq.shape[2] + query + row] * k[batch, head, pivot, col].to(
                        cutlass.Float32
                    )
                accum[i] = value
        alloc = SmemAllocator()
        atom = cute.make_composed_layout(cute.make_swizzle(2, 3, 3), 0, cute.make_layout((8, 32), stride=(32, 1)))
        shared = self.shared(alloc, dq.element_type, cute.tile_to_shape(atom, (64, self.width), (0, 1)))
        self.store_result(accum, dq, shared, batch, head, query, 0, thread, tid, cutlass.Float32(self.width**-0.5))


class JointBackward:
    """Default-scale, equal-head D64/D128 analysis prototype with explicit scratch."""

    def __init__(
        self,
        causal,
        dimension,
        compensated=True,
        pivot_correction=False,
        cache_global=False,
        direct_dq=False,
        query_column_warps=2,
        gradient_column_warps=2,
        packed_weights=False,
        overlap_query=False,
        query_row_scores=False,
        resident_kv=0,
        reduce_score_mass=False,
        early_probability=False,
        opaque_accumulator=False,
        delta_columns=32,
    ):
        if dimension not in (64, 128):
            raise ValueError("The joint analysis prototype supports D64/D128 only.")
        self.dimension = dimension
        self.opaque_accumulator = opaque_accumulator
        if pivot_correction and compensated:
            raise ValueError("Choose pivot or two-component correction for this experiment.")
        self.pivot_correction = pivot_correction
        if delta_columns not in (32, 64, 128):
            raise ValueError("The delta experiment supports 32/64/128 key columns.")
        self.delta = Delta(causal, dimension, pivot_correction, cache_global, delta_columns)
        if query_column_warps not in (2, 4):
            raise ValueError("The joint dQ warp layout requires two or four column warps.")
        if gradient_column_warps not in (2, 4):
            raise ValueError("The joint dK/dV warp layout requires two or four column warps.")
        if overlap_query and (compensated or not direct_dq or not pivot_correction):
            raise ValueError("Query overlap requires the direct dQ, single-component pivot path.")
        if query_row_scores and not direct_dq:
            raise ValueError("Query-row scores require direct gradient products.")
        if resident_kv not in (0, 1, 2) or (resident_kv and not query_row_scores):
            raise ValueError("Resident K/V requires query-row score orientation.")
        if reduce_score_mass and not (query_row_scores and pivot_correction and packed_weights):
            raise ValueError("Score-register mass reduction requires query-row packed pivot correction.")
        if early_probability and not (query_row_scores and packed_weights):
            raise ValueError("Early probability requires query-row packed stores.")
        self.joint = Joint(
            causal,
            dimension,
            compensated,
            pivot_correction,
            cache_global,
            direct_dq,
            query_column_warps,
            gradient_column_warps,
            packed_weights,
            overlap_query,
            query_row_scores,
            resident_kv,
            reduce_score_mass,
            early_probability,
            opaque_accumulator,
        )

    @cute.jit
    def __call__(
        self,
        q,
        k,
        v,
        pivots,
        out,
        maximum,
        logsum,
        dout,
        delta,
        dq,
        dk,
        dv,
        scratch,
        base,
        stream: cuda.CUstream,
    ):
        assert q.element_type in (cutlass.BFloat16, cutlass.Float16)
        assert q.shape == k.shape and q.shape == v.shape and q.shape == dout.shape
        assert q.shape == out.shape and q.shape == dq.shape and q.shape == dk.shape and q.shape == dv.shape
        assert q.shape[3] == self.dimension and q.shape[2] > 0
        assert scratch.shape[0] == q.shape[0] and scratch.shape[1] == q.shape[1] and scratch.shape[3] == q.shape[3]
        if cutlass.const_expr(self.opaque_accumulator):
            assert scratch.shape[2] == cute.ceil_div(q.shape[2], 64) * 64
        else:
            assert scratch.shape[2] == q.shape[2]
        for tensor in (q, k, v, out, dout, dq, dk, dv):
            assert tensor.stride[3] == 1 and tensor.stride[2] == q.shape[3]
            assert tensor.stride[1] == q.shape[2] * q.shape[3]
            assert tensor.stride[0] == q.shape[1] * q.shape[2] * q.shape[3]
        for tensor in (q, k, v, out, dout, dq, dk, dv):
            assert tensor.element_type == q.element_type
        for tensor in (maximum, logsum, base, pivots):
            assert tensor.shape == (q.shape[0], q.shape[1], q.shape[2])
        assert maximum.element_type == logsum.element_type == base.element_type == cutlass.Float32
        assert pivots.element_type == cutlass.Int32
        assert delta.shape == (q.shape[0], q.shape[1], q.shape[2] * (2 if self.pivot_correction else 1))
        assert delta.element_type == cutlass.Float32
        assert scratch.element_type == cutlass.Float32
        assert scratch.stride[3] == 1 and scratch.stride[0] == q.shape[1] * scratch.shape[2] * q.shape[3]
        assert (scratch.stride[1] == scratch.shape[2] * q.shape[3] and scratch.stride[2] == q.shape[3]) or (
            scratch.stride[1] == q.shape[3] and scratch.stride[2] == q.shape[1] * q.shape[3]
        )
        scale = cutlass.Float32(self.dimension**-0.5)
        self.clear(scratch).launch(
            grid=(cute.ceil_div(cute.size(scratch), 256), 1, 1), block=(256, 1, 1), stream=stream
        )
        self.delta(q, k, v, out, maximum, logsum, dout, delta, base, pivots, stream)
        self.joint(q, k, v, maximum, logsum, dout, delta, scratch, dk, dv, base, stream)
        if cutlass.const_expr(self.opaque_accumulator):
            self.joint.convert_opaque(scratch, dq, k, delta, pivots, stream)
        else:
            self.convert(scratch, dq, k, delta, pivots, scale).launch(
                grid=(cute.ceil_div(cute.size(dq), 256), 1, 1), block=(256, 1, 1), stream=stream
            )

    @cute.kernel
    def clear(self, scratch):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        i = block * 256 + tid
        linear = cute.make_tensor(scratch.iterator, cute.make_layout(cute.size(scratch)))
        if i < cute.size(scratch):
            linear[i] = cutlass.Float32(0.0)

    @cute.kernel
    def convert(self, scratch, dq, k, delta, pivots, scale: cutlass.Float32):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        i = block * 256 + tid
        target = cute.make_tensor(dq.iterator, cute.make_layout(cute.size(dq)))
        if i < cute.size(dq):
            row = (i // self.dimension) % dq.shape[2]
            head = (i // (self.dimension * dq.shape[2])) % dq.shape[1]
            batch = i // (self.dimension * dq.shape[2] * dq.shape[1])
            feature = i % self.dimension
            value = scratch[batch, head, row, feature]
            if cutlass.const_expr(self.pivot_correction):
                pivot = pivots[batch, head, row]
                value -= delta[batch, head, dq.shape[2] + row] * k[batch, head, pivot, feature].to(cutlass.Float32)
            target[i] = (value * scale).to(dq.element_type)
