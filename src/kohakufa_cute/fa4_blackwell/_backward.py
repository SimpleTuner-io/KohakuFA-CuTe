"""Derived FA4 FlashAttentionBackwardSm100 methods with this project's arithmetic.

Upstream: Dao-AILab/flash-attention, revision
4616f5116d33e3ba3cbb71d582caffcfcb49649c.
BSD-3-Clause; see LICENSES/FlashAttention-BSD-3-Clause.txt and
LICENSES/FlashAttention-AUTHORS.txt. Unchanged methods use the pinned dependency.
"""

# Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.
# All rights reserved.


from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
from flash_attn.cute.flash_bwd_sm100 import (
    AuxData,
    BlockInfo,
    BlockSparseTensors,
    Callable,
    CtaSchedule,
    FastDivmodDivisorV2,
)
from flash_attn.cute.flash_bwd_sm100 import FlashAttentionBackwardSm100 as _Original
from flash_attn.cute.flash_bwd_sm100 import (
    Float32,
    Int32,
    Int64,
    LayoutEnum,
    NamedBarrierBwdSm100,
    Optional,
    ParamsBase,
    PipelineAsync,
    SeqlenInfoQK,
    SingleTileLPTBwdScheduler,
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    TensorMapManager,
    TensorMapUpdateMode,
    TileSchedulerArguments,
    assume_tensor_aligned,
    const_expr,
    copy_utils,
    cpasync,
    cuda,
    cute,
    cutlass,
    get_block_sparse_iteration_info_bwd,
    get_m_block_from_iter_bwd,
    layout_utils,
    math,
    pack_gqa_layout,
    partial,
    pipeline,
    produce_block_sparse_q_loads_bwd_sm100_2cta_hdim192,
    produce_block_sparse_q_loads_bwd_sm100_default,
    sequence_row_offset,
    sm100_utils_basic,
    tcgen05,
    utils,
)


class _RoleBarrier(cutlass.pipeline.NamedBarrier):
    """Count every participating role without asserting whole-CTA convergence."""

    @dsl_user_op
    def arrive(self, *, loc=None, ip=None):
        cute.arch.barrier_arrive(
            barrier_id=self.barrier_id, number_of_threads=self.num_threads, aligned=False, loc=loc, ip=ip
        )

    @dsl_user_op
    def arrive_and_wait(self, *, loc=None, ip=None):
        llvm.inline_asm(
            None,
            [],
            f"barrier.sync {self.barrier_id}, {self.num_threads};",
            "~{memory}",
            has_side_effects=True,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def wait(self, *, loc=None, ip=None):
        self.arrive_and_wait(loc=loc, ip=ip)

    @dsl_user_op
    def wait_unaligned(self, *, loc=None, ip=None):
        self.arrive_and_wait(loc=loc, ip=ip)


class FlashAttentionBackwardSm100(_Original):

    def _setup_smem_layout(self):
        # S.T = K @ Q.T
        sK_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_S,
            self.mma_tiler_kq,
            self.k_dtype,
            1,
        )
        self.sK_layout = cute.slice_(sK_layout, (None, None, None, 0))
        self.sQ_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_S,
            self.mma_tiler_kq,
            self.q_dtype,
            self.Q_stage,
        )
        # dP.T = V @ dO.T
        sV_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dP,
            self.mma_tiler_vdo,
            self.v_dtype,
            1,
        )
        self.sV_layout = cute.slice_(sV_layout, (None, None, None, 0))
        self.sdOt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dP,
            self.mma_tiler_vdo,
            self.do_dtype,
            self.dO_stage,
        )
        # dV += P.T @ dO
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dV,
            self.mma_tiler_pdo,
            self.do_dtype,
            1,
        )
        self.tP_layout = cute.slice_(tP_layout, (None, None, None, 0))
        self.sdO_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dV,
            self.mma_tiler_pdo,
            self.do_dtype,
            self.dO_stage,
        )
        # dK += dS.T @ Q
        sdSt_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.ds_dtype,
            1,
        )
        self.sdSt_layout = cute.slice_(sdSt_layout, (None, None, None, 0))
        tdS_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.ds_dtype,
            1,
        )
        self.tdS_layout = cute.slice_(tdS_layout, (None, None, None, 0))
        self.sQt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.q_dtype,
            self.Q_stage,
        )
        # dQ = dS @ K
        sdS_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dQ,
            self.mma_tiler_dsk,
            self.ds_dtype,
            1,
        )
        self.sdS_layout = cute.slice_(sdS_layout, (None, None, None, 0))
        if self.acc_m64:
            self._check_dS_exchange_layouts()
        sKt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dQ,
            self.mma_tiler_dsk,
            self.k_dtype,
            1,
        )
        self.sKt_layout = cute.slice_(sKt_layout, (None, None, None, 0))
        self.sdS_xchg_layout = cute.make_layout(shape=(self.tile_n, self.tile_m // 2))

        self.sdQaccum_layout = cute.make_layout((self.tile_m * self.dQ_reduce_ncol, self.sdQaccum_stage))
        self.sLSE_layout = cute.make_layout(
            shape=(self.tile_m * 2, self.Q_stage), stride=(1, cute.round_up(self.tile_m, 64) * 2)
        )
        self.sdPsum_layout = cute.make_layout(
            shape=(self.tile_m, self.dO_stage),
            stride=(1, cute.round_up(self.tile_m, 64)),
        )
        self.sdK_epi_tile = (
            self.tile_n,
            math.gcd(128 // (self.dk_dtype.width // 8), self.tile_hdim // 2),  # 64 or 32
        )  # subtiles mma_tiler_dsq[:2] = mma_tiler_pdo[:2]
        self.sdV_epi_tile = (
            self.tile_n,
            math.gcd(128 // (self.dk_dtype.width // 8), self.tile_hdimv // 2),  # 64 or 32
        )  # subtiles mma_tiler_dsq[:2] = mma_tiler_pdo[:2]
        # headdim_64 gets 1 stage
        self.num_epi_stages = max(1, (self.tile_hdim // 2) // self.sdK_epi_tile[1])
        self.num_epi_stages_v = max(1, (self.tile_hdimv // 2) // self.sdV_epi_tile[1])
        self.sdK_flat_epi_tile = self.tile_n * (self.tile_hdim // 2) // self.num_epi_stages
        self.sdV_flat_epi_tile = self.tile_n * (self.tile_hdimv // 2) // self.num_epi_stages_v
        if const_expr(not self.dKV_postprocess):
            # One (tile_n, epi) stage per compute warpgroup; with the M=64 layout the whole CTA
            # stages the (tile_n, hdim) tile, i.e. hdim / epi stages (the same bytes for hdim 128).
            self.sdK_layout = sm100_utils_basic.make_smem_layout_epi(
                self.dk_dtype,
                LayoutEnum.ROW_MAJOR,
                self.sdK_epi_tile,
                self.tile_hdim // self.sdK_epi_tile[1] if self.acc_m64 else 2,
            )
            self.sdV_layout = sm100_utils_basic.make_smem_layout_epi(
                self.dv_dtype,
                LayoutEnum.ROW_MAJOR,
                self.sdV_epi_tile,
                self.tile_hdimv // self.sdV_epi_tile[1] if self.acc_m64 else 2,
            )
        elif self.acc_m64:
            # fp32 row-major staging of half the (tile_n, hdim) tile (NOTE [M=64 accumulator
            # layout] GQA path): all of the K / V storage.
            self.sdK_layout = cute.make_layout((self.tile_n // 2) * self.tile_hdim)
            self.sdV_layout = cute.make_layout((self.tile_n // 2) * self.tile_hdimv)
        else:
            self.sdK_layout = cute.make_layout((self.tile_n * self.dK_reduce_ncol, 2))
            # self.dK_reduce_ncol same for dV
            self.sdV_layout = cute.make_layout((self.tile_n * self.dK_reduce_ncol, 2))

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        mdQ_semaphore: Optional[cute.Tensor] = None,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV_semaphore: Optional[cute.Tensor] = None,
        aux_data: AuxData = AuxData(),
        # Block-sparse tensors (Q direction - for iterating m_blocks per n_block):
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        mCuTotalMBlocks: Optional[cute.Tensor] = None,
        tensormap_workspace: Optional[cute.Tensor] = None,  # (num_ctas, 3, 16) Int64: K, V, Kt descriptors per CTA
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.do_dtype = mdO.element_type
        self.lse_dtype = mLSE.element_type
        self.dpsum_dtype = mdPsum.element_type
        self.dqaccum_dtype = mdQaccum.element_type
        self.dk_dtype = mdK.element_type
        self.dv_dtype = mdV.element_type
        self.ds_dtype = self.q_dtype

        self.is_varlen_k = mCuSeqlensK is not None or mSeqUsedK is not None
        self.is_varlen_q = mCuSeqlensQ is not None or mSeqUsedQ is not None
        # dK / dV stored directly (MHA, or pack_gqa keeping them in TMEM) with a varlen K take
        # the predicated non-TMA store path.
        self.use_tma_store = not ((self.qhead_per_kvhead == 1 or self.pack_gqa) and mCuSeqlensK is not None)
        # self.use_tma_store = not self.qhead_per_kvhead == 1
        self.dKV_postprocess = self.qhead_per_kvhead > 1 and not self.pack_gqa

        if const_expr(self.dKV_postprocess):
            assert self.dk_dtype.width == 32, "Must accumulate dK in float precision for GQA"
            assert self.dv_dtype.width == 32, "Must accumulate dV in float precision for GQA"

        mdQaccum, mdK, mdV, mLSE, mdPsum = [assume_tensor_aligned(t) for t in (mdQaccum, mdK, mdV, mLSE, mdPsum)]

        # (b, s, n, h) --> (s, h, n, b) or (t, n, h) -> (t, h, n)
        QO_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        mQ, mdO = [layout_utils.select(t, mode=QO_layout_transpose) for t in (mQ, mdO)]

        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mK, mV = [layout_utils.select(t, mode=KV_layout_transpose) for t in (mK, mV)]

        # (b, n, s) --> (s, n, b) or (n, t) --> (t, n)
        LSE_dPsum_dQaccum_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        mLSE, mdPsum, mdQaccum = [layout_utils.select(t, mode=LSE_dPsum_dQaccum_transpose) for t in (mLSE, mdPsum, mdQaccum)]

        if const_expr(not self.dKV_postprocess):
            layout_dKV_transpose = KV_layout_transpose
        else:
            layout_dKV_transpose = [2, 1, 0] if const_expr(mCuSeqlensK is None) else [1, 0]
        mdK, mdV = [layout_utils.select(t, mode=layout_dKV_transpose) for t in (mdK, mdV)]
        if const_expr(self.pack_gqa):
            # (s, h, n, b) -> ((qh, s), h, n_kv, b), NOTE [bwd pack_gqa]
            nheads_kv = mK.shape[2]
            mQ = pack_gqa_layout(mQ, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            mdO = pack_gqa_layout(mdO, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            # LSE / dPsum / dQaccum are allocated per KV head in packed row order already
        # (s, h, n, b) --> (h, s, n, b) or (t, h, n) -> (h, t, b)
        dO_transpose = [1, 0, 2, 3] if const_expr(mCuSeqlensQ is None) else [1, 0, 2]
        mdO = layout_utils.select(mdO, mode=dO_transpose)

        # Transposes for 2-CTA K/Q paths (Q follows Q seqlens, K follows K seqlens)
        transpose_sh_q = dO_transpose
        transpose_sh_k = [1, 0, 2, 3] if const_expr(mCuSeqlensK is None) else [1, 0, 2]

        # (b, n, block, stage) -> (block, stage, n, b)
        semaphore_transpose = [2, 3, 1, 0]
        if const_expr(self.deterministic):
            assert mdQ_semaphore is not None
            mdQ_semaphore = layout_utils.select(mdQ_semaphore, mode=semaphore_transpose)

        if const_expr(self.dKV_postprocess and self.deterministic):
            assert mdK_semaphore is not None
            assert mdV_semaphore is not None
            mdK_semaphore, mdV_semaphore = [
                layout_utils.select(t, mode=semaphore_transpose) for t in (mdK_semaphore, mdV_semaphore)
            ]
        else:
            mdK_semaphore = None
            mdV_semaphore = None

        self._setup_attributes()
        (
            self.tiled_mma_S,
            self.tiled_mma_dP,
            self.tiled_mma_dK,
            self.tiled_mma_dV,
            self.tiled_mma_dQ,
        ) = self._get_tiled_mma()
        self._setup_smem_layout()

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (self.tiled_mma_S.thr_id.shape,),
        )
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_q_do_mcast = self.num_mcast_ctas_b > 1

        if const_expr(not self.dKV_postprocess):
            self.mdK_layout_enum = LayoutEnum.from_tensor(mdK)
            self.mdV_layout_enum = LayoutEnum.from_tensor(mdV)
            dK_major_mode = self.mdK_layout_enum.mma_major_mode()
            dV_major_mode = self.mdV_layout_enum.mma_major_mode()
            if const_expr(dK_major_mode != cute.nvgpu.OperandMajorMode.K):
                raise RuntimeError("The layout of mdK is wrong")
            if const_expr(dV_major_mode != cute.nvgpu.OperandMajorMode.K):
                raise RuntimeError("The layout of mdV is wrong")

        if const_expr(self.use_tma_store and not self.dKV_postprocess):
            tma_copy_op_dKV = cpasync.CopyBulkTensorTileS2GOp()
            tma_atom_dK, mdK_tma_tensor = cpasync.make_tiled_tma_atom(
                tma_copy_op_dKV,
                mdK,
                cute.select(self.sdK_layout, mode=[0, 1]),
                self.sdK_epi_tile,
                1,  # no mcast
            )
            tma_atom_dV, mdV_tma_tensor = cpasync.make_tiled_tma_atom(
                tma_copy_op_dKV,
                mdV,
                cute.select(self.sdV_layout, mode=[0, 1]),
                self.sdV_epi_tile,
                1,  # no mcast
            )
        else:
            mdV_tma_tensor = mdV
            mdK_tma_tensor = mdK
            tma_atom_dV = None
            tma_atom_dK = None

        if const_expr(not self.dKV_postprocess):
            thr_layout_r2s_dKV = cute.make_ordered_layout((128, 1), order=(1, 0))  # 128 threads
            val_layout_r2s_dKV = cute.make_ordered_layout(
                (1, 128 // self.dk_dtype.width), order=(1, 0)
            )  # 4 or 8 vals for 16 byte store
            copy_atom_r2s_dKV = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.dk_dtype,
                num_bits_per_copy=128,
            )
            tiled_copy_r2s_dKV = cute.make_tiled_copy_tv(copy_atom_r2s_dKV, thr_layout_r2s_dKV, val_layout_r2s_dKV)
        else:
            tiled_copy_r2s_dKV = copy_utils.tiled_copy_1d(Float32, 128, num_copy_elems=128 // Float32.width)

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        # Bound the K and V TMA loads by the logical seqlen_k (cu_seqlens_k and/or seqused_k) so
        # rows in [seqlen_k, tile_n) arrive zero-filled instead of as whatever the buffer holds.
        # The masked P / dS are exactly zero there, but 0 * NaN = NaN in dP = V @ dO.T and
        # dQ = dS @ K. The load warp rewrites its CTA's copies of the K, V (and Kt) descriptors
        # per work tile with the sequence's row extent (tensormap.replace); TMA then zero-fills
        # rows at or past the end and never touches memory beyond it. A descriptor with a
        # per-launch extent cannot do this: TMA prefetches past the footprint it is given (#3068).
        self.ragged_kv_tma = mCuSeqlensK is not None or mSeqUsedK is not None
        if const_expr(self.ragged_kv_tma):
            assert tensormap_workspace is not None, "ragged K/V TMA loads need a tensormap workspace"
        mK_tma, mV_tma = mK, mV
        mK_gmem, mV_gmem = (mK, mV) if const_expr(self.ragged_kv_tma) else (None, None)
        mKt_gmem = None
        # S.T = K @ Q.T
        tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mK_tma,
            cute.select(self.sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_kq,
            self.tiled_mma_S,
            self.cluster_layout_vmnk.shape,
        )
        Q_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_B(self.cluster_shape_mnk, self.tiled_mma_S.thr_id)
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_B(
            Q_tma_op,
            mQ,
            cute.select(self.sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_kq,
            self.tiled_mma_S,
            self.cluster_layout_vmnk.shape,
        )
        # dP.T = V @ dO.T
        tma_atom_V, tma_tensor_V = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mV_tma,
            cute.select(self.sV_layout, mode=[0, 1, 2]),
            self.mma_tiler_vdo,
            self.tiled_mma_dP,
            self.cluster_layout_vmnk.shape,
        )
        # dV = P.T @ dO
        dO_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_B(self.cluster_shape_mnk, self.tiled_mma_dV.thr_id)
        tma_atom_dO, tma_tensor_dO = cute.nvgpu.make_tiled_tma_atom_B(
            dO_tma_op,
            mdO,
            cute.select(self.sdO_layout, mode=[0, 1, 2]),
            self.mma_tiler_pdo,
            self.tiled_mma_dV,
            self.cluster_layout_vmnk.shape,
        )
        # ------------------------------------------------------------
        # 2-CTA
        # ------------------------------------------------------------
        tma_atom_dOt = tma_tensor_dOt = None
        if const_expr(self.use_2cta_instrs):
            tma_atom_dOt, tma_tensor_dOt = cute.nvgpu.make_tiled_tma_atom_B(
                dO_tma_op,
                layout_utils.select(mdO, mode=transpose_sh_q),
                cute.select(self.sdOt_layout, mode=[0, 1, 2]),
                self.mma_tiler_vdo,
                self.tiled_mma_dP,
                self.cluster_layout_vmnk.shape,
            )
        tma_atom_Qt = tma_tensor_Qt = None
        if const_expr(self.use_2cta_instrs):
            tma_atom_Qt, tma_tensor_Qt = cute.nvgpu.make_tiled_tma_atom_B(
                Q_tma_op,
                layout_utils.select(mQ, mode=transpose_sh_q),
                cute.select(self.sQt_layout, mode=[0, 1, 2]),
                self.mma_tiler_dsq,
                self.tiled_mma_dK,
                self.cluster_layout_vmnk.shape,
            )
        tma_atom_Kt = tma_tensor_Kt = None
        if const_expr(self.use_2cta_instrs):
            Kt_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_B(self.cluster_shape_mnk, self.tiled_mma_dQ.thr_id)
            mKt_tma = layout_utils.select(mK, mode=transpose_sh_k)
            if const_expr(self.ragged_kv_tma):
                mKt_gmem = mKt_tma  # (hdim, seqlen_k, ...): the rows are mode 1
            tma_atom_Kt, tma_tensor_Kt = cute.nvgpu.make_tiled_tma_atom_B(
                Kt_tma_op,
                mKt_tma,
                cute.select(self.sKt_layout, mode=[0, 1, 2]),
                self.mma_tiler_dsk,
                self.tiled_mma_dQ,
                self.cluster_layout_vmnk.shape,
            )

        self.tma_copy_bytes = {
            name: self.cta_group_size * cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1, 2]))
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mV, self.sV_layout),
                ("dO", mdO, self.sdO_layout),
            ]
        }
        self.tma_copy_bytes["LSE"] = self.tile_m * Float32.width // 8 * 2
        self.tma_copy_bytes["dPsum"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dQ"] = self.tile_m * self.dQ_reduce_ncol * Float32.width // 8
        self.tma_copy_bytes["dKacc"] = self.tile_n * self.dK_reduce_ncol * Float32.width // 8
        self.tma_copy_bytes["dS"] = cute.size_in_bytes(self.ds_dtype, self.sdS_layout)
        self.tma_copy_bytes["sdS_xchg"] = self.tma_copy_bytes["dS"] // 2  # Half of dS for exchange

        # TileScheduler = SingleTileScheduler
        if const_expr(self.is_varlen_k):
            TileScheduler = SingleTileVarlenScheduler
        elif const_expr(self.deterministic):
            TileScheduler = SingleTileLPTBwdScheduler
        else:
            TileScheduler = SingleTileScheduler
        if const_expr(self.spt_override is None):
            self.spt = (self.is_causal or self.is_local) and self.deterministic
        else:
            assert self.spt_override is not None
            self.spt = self.spt_override and self.deterministic
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mK.shape[0]), self.cta_tiler[0]),  # num_blocks
            cute.size(mQ.shape[2]),  # num_heads = num_query_heads
            (
                cute.size(mK.shape[3]) if const_expr(mCuSeqlensK is None) else cute.size(mCuSeqlensK.shape[0] - 1)
            ),  # num_batches
            1,  # num_splits
            cute.size(mQ.shape[0]),  # pass seqlen_q or total_q for seqlen_k
            mQ.shape[1],  # headdim
            mV.shape[1],  # headdim_v
            total_q=(
                cute.size(mK.shape[0])  # pass total_k for total_q
                if const_expr(mCuSeqlensK is not None)
                else cute.size(mK.shape[0]) * cute.size(mK.shape[3])
            ),
            tile_shape_mn=self.cta_tiler[:2],  # (tile_n, tile_m)
            cluster_shape_mn=self.cluster_shape_mnk[:2],
            mCuSeqlensQ=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedK,
            # the backward tiles KV rows: the Q-row packing factor is for block_info / mask only
            qhead_per_kvhead_packgqa=1,
            element_size=self.k_dtype.width // 8,
            is_persistent=self.is_persistent,  # persistent mode not tested
            cu_total_m_blocks_ptr=mCuTotalMBlocks,
            lpt=self.spt,
            head_swizzle=self.deterministic,
        )

        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        # Compute allocation sizes for shared buffers that are reused
        # sQ is reused for sdK, sdO is reused for sdV
        sQ_alloc_bytes = max(
            cute.size_in_bytes(self.q_dtype, self.sQ_layout),
            cute.size_in_bytes(self.dk_dtype, self.sdK_layout),
        )
        sdO_alloc_bytes = max(
            cute.size_in_bytes(self.dv_dtype, self.sdV_layout),
            cute.size_in_bytes(self.do_dtype, self.sdO_layout),
        )

        sdK_bytes = cute.size_in_bytes(self.dk_dtype, self.sdK_layout)
        sdV_bytes = cute.size_in_bytes(self.dv_dtype, self.sdV_layout)
        assert sdV_bytes <= sdO_alloc_bytes, "sdV doesn't fit in sdO storage allocation"
        assert sdK_bytes <= sQ_alloc_bytes, "sdK doesn't fit in sQ storage allocation"
        # 2-CTA: sdV reuses sV, sdK reuses sK
        sV_bytes = cute.size_in_bytes(self.v_dtype, self.sV_layout)
        sK_bytes = cute.size_in_bytes(self.k_dtype, self.sK_layout)
        if const_expr(self.use_2cta_instrs):
            assert sdV_bytes <= sV_bytes, "sdV doesn't fit in sV storage allocation (2-CTA)"
            assert sdK_bytes <= sK_bytes, "sdK doesn't fit in sK storage allocation (2-CTA)"

        if const_expr(self.use_2cta_instrs):
            # The serial schedule aliases Qt / dOt into Q / dO and sdS_xchg into sdQaccum; the
            # M=64 layout has no exchange staging buffer at all (see xchg_aliases_dQaccum).
            own_xchg = const_expr(self.cta_schedule == CtaSchedule.TWO_CTA_PIPELINED)
            sQt_size = cute.cosize(self.sQt_layout) if own_xchg else 0
            sdOt_size = cute.cosize(self.sdOt_layout) if own_xchg else 0
            sdS_xchg_size = cute.cosize(self.sdS_xchg_layout) if own_xchg and self.has_sdS_xchg else 0
            # M=64 layout: P^T and this CTA's dS^T are the SMEM A operands of the dV / dK MMAs.
            sP_size = cute.cosize(self.tP_layout) if const_expr(self.acc_m64) else 0
            sdS_dK_size = cute.cosize(self.sdSt_layout) if const_expr(self.acc_m64) else 0
            if const_expr(self.xchg_aliases_dQaccum):
                assert cute.size_in_bytes(self.ds_dtype, self.sdS_xchg_layout) <= cute.size_in_bytes(
                    self.dqaccum_dtype, self.sdQaccum_layout
                ), "sdS_xchg does not fit in the sdQaccum storage it aliases"

            @cute.struct
            class SharedStorage:
                Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
                dO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
                LSE_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
                dPsum_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
                S_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dP_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dS_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dKV_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.sdKVaccum_stage]
                dQ_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
                dQ_cluster_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.dQaccum_reduce_stage // 2]
                dQ_cluster_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.dQaccum_reduce_stage // 2]
                tmem_holding_buf: Int32
                tmem_dealloc_mbar: cutlass.Int64

                # 2-CTA
                Qt_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
                Kt_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dS_cluster_empty_mbar_ptr: cutlass.Int64
                dS_cluster_full_mbar_ptr: cutlass.Int64
                dS_cluster_leader_mbar_ptr: cutlass.Int64
                dQaccum_empty_mbar_ptr: cutlass.Int64

                sQ: cute.struct.Align[
                    cute.struct.MemRange[self.q_dtype, cute.cosize(self.sQ_layout)],
                    self.buffer_align_bytes,
                ]
                sK: cute.struct.Align[
                    cute.struct.MemRange[self.k_dtype, cute.cosize(self.sK_layout)],
                    self.buffer_align_bytes,
                ]
                sV: cute.struct.Align[
                    cute.struct.MemRange[self.v_dtype, cute.cosize(self.sV_layout)],
                    self.buffer_align_bytes,
                ]
                sdO: cute.struct.Align[
                    cute.struct.MemRange[self.do_dtype, cute.cosize(self.sdO_layout)],
                    self.buffer_align_bytes,
                ]
                sQt: cute.struct.Align[
                    cute.struct.MemRange[self.q_dtype, sQt_size],
                    self.buffer_align_bytes,
                ]
                sdOt: cute.struct.Align[
                    cute.struct.MemRange[self.do_dtype, sdOt_size],
                    self.buffer_align_bytes,
                ]
                sdS_xchg: cute.struct.Align[
                    cute.struct.MemRange[self.ds_dtype, sdS_xchg_size],
                    self.buffer_align_bytes,
                ]
                sP: cute.struct.Align[
                    cute.struct.MemRange[self.do_dtype, sP_size],
                    self.buffer_align_bytes,
                ]
                sdS_dK: cute.struct.Align[
                    cute.struct.MemRange[self.ds_dtype, sdS_dK_size],
                    self.buffer_align_bytes,
                ]
                sKt: cute.struct.Align[
                    cute.struct.MemRange[self.k_dtype, cute.cosize(self.sKt_layout)],
                    self.buffer_align_bytes,
                ]
                sdS: cute.struct.Align[
                    cute.struct.MemRange[self.ds_dtype, cute.cosize(self.sdSt_layout)],
                    self.buffer_align_bytes,
                ]
                sLSE: cute.struct.Align[
                    cute.struct.MemRange[self.lse_dtype, cute.cosize(self.sLSE_layout)],
                    128,
                ]
                sdPsum: cute.struct.Align[
                    cute.struct.MemRange[self.dpsum_dtype, cute.cosize(self.sdPsum_layout)],
                    128,
                ]
                sdQaccum: cute.struct.Align[
                    cute.struct.MemRange[self.dqaccum_dtype, cute.cosize(self.sdQaccum_layout)],
                    self.buffer_align_bytes if sdS_xchg_size == 0 else 128,
                ]

        else:

            @cute.struct
            class SharedStorage:
                Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
                dO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
                LSE_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
                dPsum_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
                S_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dP_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dS_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                P_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.P_stage]
                dKV_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.sdKVaccum_stage]
                dQ_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
                dQ_cluster_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.dQaccum_reduce_stage // 2]
                dQ_cluster_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.dQaccum_reduce_stage // 2]
                tmem_holding_buf: Int32
                tmem_dealloc_mbar: Int64

                sQ: cute.struct.Align[
                    cute.struct.MemRange[cute.Uint8, sQ_alloc_bytes],
                    self.buffer_align_bytes,
                ]
                sK: cute.struct.Align[
                    cute.struct.MemRange[self.k_dtype, cute.cosize(self.sK_layout)],
                    self.buffer_align_bytes,
                ]
                sV: cute.struct.Align[
                    cute.struct.MemRange[self.v_dtype, cute.cosize(self.sV_layout)],
                    self.buffer_align_bytes,
                ]
                sdO: cute.struct.Align[
                    cute.struct.MemRange[cute.Uint8, sdO_alloc_bytes],
                    self.buffer_align_bytes,
                ]
                sdS: cute.struct.Align[
                    cute.struct.MemRange[self.ds_dtype, cute.cosize(self.sdSt_layout)],
                    128,
                ]
                sLSE: cute.struct.Align[
                    cute.struct.MemRange[self.lse_dtype, cute.cosize(self.sLSE_layout)],
                    128,
                ]
                sdPsum: cute.struct.Align[
                    cute.struct.MemRange[self.dpsum_dtype, cute.cosize(self.sdPsum_layout)],
                    128,
                ]
                sdQaccum: cute.struct.Align[
                    cute.struct.MemRange[self.dqaccum_dtype, cute.cosize(self.sdQaccum_layout)],
                    self.buffer_align_bytes,
                ]

        self.shared_storage = SharedStorage

        LOG2_E = math.log2(math.e)
        if const_expr(self.score_mod is None):
            # Without score_mod: bake scale into log2
            softmax_scale_log2 = softmax_scale * LOG2_E
        else:
            # With score_mod: score_mod applied to S * softmax_scale, then use LOG2_E only
            softmax_scale_log2 = LOG2_E

        if const_expr(window_size_left is not None):
            window_size_left = Int32(window_size_left)
        if const_expr(window_size_right is not None):
            window_size_right = Int32(window_size_right)

        fastdiv_mods = None
        if const_expr(aux_data.tensors is not None):
            seqlen_q = cute.size(mQ.shape[0]) // (self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1)
            seqlen_k = cute.size(mK.shape[0])
            seqlen_q_divmod = FastDivmodDivisorV2(seqlen_q)
            seqlen_k_divmod = FastDivmodDivisorV2(seqlen_k)
            fastdiv_mods = (seqlen_q_divmod, seqlen_k_divmod)
        self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)
        if const_expr(self.use_block_sparsity and self.use_2cta_instrs):
            # Both CTAs of a cluster must map to the same sparse KV column or they deadlock.
            assert (
                self.kv_subtile_factor % self.cta_group_size == 0
            ), "2-CTA block-sparse backward requires kv_subtile_factor % cta_group_size == 0"

        # 2-CTA: 231424 and 1-CTA: 232448
        # print("SMEM: ", self.shared_storage.size_in_bytes())
        if const_expr(self.use_block_sparsity):
            assert all(
                x is None for x in (mCuSeqlensQ, mCuSeqlensK, mSeqUsedQ, mSeqUsedK)
            ), "Variable sequence length is not supported yet for blocksparse in bwd"

        self.kernel(
            tma_tensor_Q,
            tma_tensor_Qt,
            tma_tensor_K,
            tma_tensor_Kt,
            tma_tensor_V,
            mK_gmem,
            mV_gmem,
            mKt_gmem,
            tensormap_workspace if const_expr(self.ragged_kv_tma) else None,
            mLSE,
            mdPsum,
            tma_tensor_dO,
            tma_tensor_dOt,
            mdV,
            mdK,
            mdQaccum,
            mdV_tma_tensor,
            mdK_tma_tensor,
            mdQ_semaphore,
            mdK_semaphore,
            mdV_semaphore,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            tma_atom_Q,
            tma_atom_Qt,
            tma_atom_K,
            tma_atom_Kt,
            tma_atom_V,
            tma_atom_dO,
            tma_atom_dOt,
            tma_atom_dV,
            tma_atom_dK,
            self.sQ_layout,
            self.sQt_layout,
            self.sK_layout,
            self.sKt_layout,
            self.sV_layout,
            self.sLSE_layout,
            self.sdPsum_layout,
            self.sdO_layout,
            self.sdOt_layout,
            self.sdSt_layout,
            self.sdS_layout,
            self.sdS_xchg_layout,
            self.sdQaccum_layout,
            self.sdK_layout,
            self.sdV_layout,
            self.tP_layout,
            self.tdS_layout,
            self.tiled_mma_S,
            self.tiled_mma_dP,
            self.tiled_mma_dV,
            self.tiled_mma_dK,
            self.tiled_mma_dQ,
            tiled_copy_r2s_dKV,
            softmax_scale,
            softmax_scale_log2,
            window_size_left,
            window_size_right,
            tile_sched_params,
            aux_data,
            fastdiv_mods,
            blocksparse_tensors,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk if cute.size(self.cluster_shape_mnk) > 1 else None,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mQt: Optional[cute.Tensor],
        mK: cute.Tensor,
        mKt: Optional[cute.Tensor],
        mV: cute.Tensor,
        mK_gmem: Optional[cute.Tensor],  # gmem tensors behind tma_atom_K/V/Kt (ragged K/V only)
        mV_gmem: Optional[cute.Tensor],
        mKt_gmem: Optional[cute.Tensor],
        tensormap_workspace: Optional[cute.Tensor],  # (num_ctas, 3, 16) Int64 (ragged K/V only)
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdO: cute.Tensor,
        mdOt: Optional[cute.Tensor],
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdV_tma_tensor: Optional[cute.Tensor],
        mdK_tma_tensor: Optional[cute.Tensor],
        mdQ_semaphore: Optional[cute.Tensor],
        mdK_semaphore: Optional[cute.Tensor],
        mdV_semaphore: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_Qt: Optional[cute.CopyAtom],
        tma_atom_K: cute.CopyAtom,
        tma_atom_Kt: Optional[cute.CopyAtom],
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dOt: Optional[cute.CopyAtom],
        tma_atom_dV: Optional[cute.CopyAtom],
        tma_atom_dK: Optional[cute.CopyAtom],
        sQ_layout: cute.ComposedLayout,
        sQt_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sKt_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sLSE_layout: cute.Layout,
        sdPsum_layout: cute.Layout,
        sdO_layout: cute.ComposedLayout,
        sdOt_layout: cute.ComposedLayout,
        sdSt_layout: cute.ComposedLayout,
        sdS_layout: cute.ComposedLayout,
        sdS_xchg_layout: cute.Layout,
        sdQaccum_layout: cute.Layout,
        sdK_layout: cute.ComposedLayout | cute.Layout,
        sdV_layout: cute.ComposedLayout | cute.Layout,
        tP_layout: cute.ComposedLayout,
        tdS_layout: cute.ComposedLayout,
        tiled_mma_S: cute.TiledMma,
        tiled_mma_dP: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        tiled_copy_r2s_dKV: cute.TiledCopy,
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        tile_sched_params: ParamsBase,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % self.cta_group_size
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())

        # Prefetch tma descriptor
        if warp_idx == self.load_warp_id:
            with cute.arch.elect_one():
                cpasync.prefetch_descriptor(tma_atom_Q)
                if const_expr(tma_atom_Qt is not None):
                    cpasync.prefetch_descriptor(tma_atom_Qt)
                cpasync.prefetch_descriptor(tma_atom_K)
                if const_expr(tma_atom_Kt is not None):
                    cpasync.prefetch_descriptor(tma_atom_Kt)
                cpasync.prefetch_descriptor(tma_atom_V)
                if const_expr(tma_atom_dOt is not None):
                    cpasync.prefetch_descriptor(tma_atom_dOt)
                cpasync.prefetch_descriptor(tma_atom_dO)
                if const_expr(tma_atom_dV is not None):
                    cpasync.prefetch_descriptor(tma_atom_dV)
                if const_expr(tma_atom_dK is not None):
                    cpasync.prefetch_descriptor(tma_atom_dK)

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_S.thr_id.shape,),
        )

        # Alloc
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        dQ_cluster_full_mbar_ptr = storage.dQ_cluster_full_mbar_ptr.data_ptr()
        dQ_cluster_empty_mbar_ptr = storage.dQ_cluster_empty_mbar_ptr.data_ptr()

        if const_expr(self.use_2cta_instrs):
            dS_cluster_full_mbar_ptr = storage.dS_cluster_full_mbar_ptr
            dS_cluster_empty_mbar_ptr = storage.dS_cluster_empty_mbar_ptr
            dS_cluster_leader_mbar_ptr = storage.dS_cluster_leader_mbar_ptr
        else:
            dS_cluster_full_mbar_ptr = None
            dS_cluster_empty_mbar_ptr = None
            dS_cluster_leader_mbar_ptr = None
        # The dQaccum_empty handshake exists only where sdS_xchg aliases sdQaccum; the roles
        # specialize on this handle (and on sdS_xchg) rather than on the configuration.
        if const_expr(self.xchg_aliases_dQaccum):
            dQaccum_empty_mbar_ptr = storage.dQaccum_empty_mbar_ptr
        else:
            dQaccum_empty_mbar_ptr = None

        # Barrier initialization
        if const_expr(self.use_2cta_instrs):
            if const_expr(dQaccum_empty_mbar_ptr is not None):
                if warp_idx == 2:
                    cute.arch.mbarrier_init(
                        dQaccum_empty_mbar_ptr,
                        len(self.reduce_warp_ids),
                    )
            if warp_idx == 4:
                cute.arch.mbarrier_init(dS_cluster_full_mbar_ptr, 1)
                cute.arch.mbarrier_init(dS_cluster_empty_mbar_ptr, 1)
                cute.arch.mbarrier_init(dS_cluster_leader_mbar_ptr, 2)

        if const_expr(self.cluster_reduce_dQ):
            if warp_idx == 4:
                for i in range(self.dQaccum_reduce_stage // 2):
                    cute.arch.mbarrier_init(dQ_cluster_full_mbar_ptr + i, 1)
                    cute.arch.mbarrier_init(dQ_cluster_empty_mbar_ptr + i, 1)

        tmem_alloc_barrier = _RoleBarrier(
            barrier_id=int(NamedBarrierBwdSm100.TmemPtr),
            num_threads=cute.arch.WARP_SIZE * len((self.mma_warp_id, *self.compute_warp_ids, *self.reduce_warp_ids)),
        )
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        # UMMA producers and AsyncThread consumers
        pipeline_producer_group_MMA_AsyncThread = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        pipeline_consumer_group_MMA_AsyncThread = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len(self.compute_warp_ids) * self.cta_group_size
        )
        # With split_P_dS, this pipeline guards only the S TMEM buffer (P has pipeline_P).
        pipeline_S_P = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.S_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        pipeline_dP = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.dP_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        pipeline_dKV = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=2,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.dKV_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        pipeline_consumer_group_MMA_AsyncThread_dQ = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            len(self.reduce_warp_ids) * self.cta_group_size,
        )  # Compute
        pipeline_dQ = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread_dQ,
            barrier_storage=storage.dQ_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # AsyncThread producers and UMMA consumers
        # Only 1 thread per warp will signal
        pipeline_PdS_producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            len(self.compute_warp_ids) * self.cta_group_size,
        )  # Compute
        pipeline_PdS_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id])
        )  # MMA
        pipeline_dS = cutlass.pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=pipeline_PdS_producer_group,
            consumer_group=pipeline_PdS_consumer_group,
            barrier_storage=storage.dS_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        if const_expr(self.split_P_dS):
            pipeline_P = cutlass.pipeline.PipelineAsyncUmma.create(
                num_stages=1,
                producer_group=pipeline_PdS_producer_group,
                consumer_group=pipeline_PdS_consumer_group,
                barrier_storage=storage.P_mbar_ptr.data_ptr(),
                cta_layout_vmnk=cluster_layout_vmnk,
            )
        else:
            pipeline_P = None

        # TMA producer and UMMA consumers
        pipeline_producer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, len([self.load_warp_id]))
        # The arrive count is the number of mcast size
        pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id]) * self.num_mcast_ctas_b
        )
        pipeline_consumer_group_compute = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            len(self.compute_warp_ids) * 1,
        )
        pipeline_LSE = cutlass.pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.LSE_mbar_ptr.data_ptr(),
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group_compute,
            tx_count=self.tma_copy_bytes["LSE"],
            # cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipeline_dPsum = cutlass.pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.dPsum_mbar_ptr.data_ptr(),
            num_stages=self.dO_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group_compute,
            tx_count=self.tma_copy_bytes["dPsum"],
            # cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipeline_Q = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.Q_mbar_ptr.data_ptr(),
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["Q"],
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        if const_expr(self.use_2cta_instrs):
            if const_expr(self.cta_schedule == CtaSchedule.TWO_CTA_SERIAL):
                pipeline_Qt = pipeline_Q
            else:
                pipeline_Qt = pipeline.PipelineTmaUmma.create(
                    barrier_storage=storage.Qt_mbar_ptr.data_ptr(),
                    num_stages=self.Q_stage,
                    producer_group=pipeline_producer_group,
                    consumer_group=pipeline_consumer_group,
                    tx_count=self.tma_copy_bytes["Q"],
                    cta_layout_vmnk=cluster_layout_vmnk,
                    defer_sync=True,
                )
            pipeline_Kt = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.Kt_mbar_ptr.data_ptr(),
                num_stages=self.single_stage,
                producer_group=pipeline_producer_group,
                consumer_group=pipeline_consumer_group,
                tx_count=self.tma_copy_bytes["K"],
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
        else:
            pipeline_Qt = pipeline_Kt = pipeline_Q

        pipeline_dO = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.dO_mbar_ptr.data_ptr(),
            num_stages=self.dO_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["dO"],
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=False,
        )

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner, dtype=self.q_dtype)
        if const_expr(self.cta_schedule == CtaSchedule.TWO_CTA_PIPELINED):
            sQt = storage.sQt.get_tensor(sQt_layout.outer, swizzle=sQt_layout.inner, dtype=self.q_dtype)
        else:
            sQt = cute.make_tensor(cute.recast_ptr(sQ.iterator, sQt_layout.inner, dtype=self.q_dtype), sQt_layout.outer)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        if const_expr(self.use_2cta_instrs):
            sKt = storage.sKt.get_tensor(sKt_layout.outer, swizzle=sKt_layout.inner)
        else:
            sKt = cute.make_tensor(cute.recast_ptr(sK.iterator, sKt_layout.inner), sKt_layout.outer)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sdSt = storage.sdS.get_tensor(sdSt_layout.outer, swizzle=sdSt_layout.inner)
        sdS = cute.make_tensor(cute.recast_ptr(sdSt.iterator, sdS_layout.inner), sdS_layout.outer)
        if const_expr(self.has_sdS_xchg):
            if const_expr(self.cta_schedule == CtaSchedule.TWO_CTA_PIPELINED):
                sdS_xchg = storage.sdS_xchg.get_tensor(sdS_xchg_layout)
            else:
                sdS_xchg = storage.sdQaccum.get_tensor(sdS_xchg_layout, dtype=self.ds_dtype)
        else:
            sdS_xchg = None
        if const_expr(self.acc_m64):
            # P^T and this CTA's dS^T: the SMEM A operands of the dV / dK MMAs.
            sP = storage.sP.get_tensor(tP_layout.outer, swizzle=tP_layout.inner, dtype=self.do_dtype)
            sdS_dK = storage.sdS_dK.get_tensor(sdSt_layout.outer, swizzle=sdSt_layout.inner)
        else:
            sP = None
            sdS_dK = None

        sdO = storage.sdO.get_tensor(sdO_layout.outer, swizzle=sdO_layout.inner, dtype=self.do_dtype)
        if const_expr(self.cta_schedule == CtaSchedule.TWO_CTA_PIPELINED):
            sdOt = storage.sdOt.get_tensor(sdOt_layout.outer, swizzle=sdOt_layout.inner, dtype=self.do_dtype)
        else:
            sdOt = cute.make_tensor(
                cute.recast_ptr(sdO.iterator, sdOt_layout.inner, dtype=self.do_dtype),
                sdOt_layout.outer,
            )

        sLSE = cute.make_tensor(
            storage.sLSE.get_tensor(sLSE_layout).iterator,
            cute.make_layout((self.tile_m, self.Q_stage), stride=(1, cute.round_up(self.tile_m, 64) * 2)),
        )
        sdPsum = storage.sdPsum.get_tensor(sdPsum_layout)
        if const_expr(self.use_2cta_instrs):
            if const_expr(not self.dKV_postprocess):
                sdV = storage.sV.get_tensor(sdV_layout.outer, swizzle=sdV_layout.inner, dtype=self.dv_dtype)
                sdK = storage.sK.get_tensor(sdK_layout.outer, swizzle=sdK_layout.inner, dtype=self.dk_dtype)
            else:
                sdV = storage.sV.get_tensor(sdV_layout, dtype=self.dv_dtype)
                sdK = storage.sK.get_tensor(sdK_layout, dtype=self.dk_dtype)
        elif const_expr(not self.dKV_postprocess):
            sdV = storage.sdO.get_tensor(sdV_layout.outer, swizzle=sdV_layout.inner, dtype=self.dv_dtype)
            sdK = storage.sQ.get_tensor(sdK_layout.outer, swizzle=sdK_layout.inner, dtype=self.dk_dtype)
        else:
            sdV = storage.sdO.get_tensor(sdV_layout, dtype=self.dv_dtype)
            sdK = storage.sQ.get_tensor(sdK_layout, dtype=self.dk_dtype)

        # Buffer sizing is guaranteed by max(...) in SharedStorage declarations
        # for both sQ (reused as sdK) and sdO (reused as sdV)
        sdQaccum = storage.sdQaccum.get_tensor(sdQaccum_layout)

        # TMEM
        # This is a fake tensor, by right need to retrieve tmem_ptr. But we know that we always
        # request 512 columns of tmem, so we know that it starts at 0.
        tmem_ptr = cute.make_ptr(Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16)
        # S
        thr_mma_S = tiled_mma_S.get_slice(mma_tile_coord_v)
        Sacc_shape = thr_mma_S.partition_shape_C(self.mma_tiler_kq[:2])  # (M, N)
        tStS = thr_mma_S.make_fragment_C(Sacc_shape)
        # (MMA, MMA_M, MMA_N)
        tStS = cute.make_tensor(tmem_ptr + self.tmem_S_offset, tStS.layout)
        # dP
        thr_mma_dP = tiled_mma_dP.get_slice(mma_tile_coord_v)
        dPacc_shape = thr_mma_dP.partition_shape_C(self.mma_tiler_vdo[:2])
        tdPtdP = thr_mma_dP.make_fragment_C(dPacc_shape)
        tdPtdP = cute.make_tensor(tmem_ptr + self.tmem_dP_offset, tdPtdP.layout)
        # dV
        thr_mma_dV = tiled_mma_dV.get_slice(mma_tile_coord_v)
        dvacc_shape = thr_mma_dV.partition_shape_C(self.mma_tiler_pdo[:2])
        tdVtdV = thr_mma_dV.make_fragment_C(dvacc_shape)
        tdVtdV = cute.make_tensor(tmem_ptr + self.tmem_dV_offset, tdVtdV.layout)
        tP = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_P_offset, dtype=self.do_dtype), tP_layout.outer)
        # dK
        thr_mma_dK = tiled_mma_dK.get_slice(mma_tile_coord_v)
        dkacc_shape = thr_mma_dK.partition_shape_C(self.mma_tiler_dsq[:2])
        tdKtdK = thr_mma_dK.make_fragment_C(dkacc_shape)
        tdKtdK = cute.make_tensor(tmem_ptr + self.tmem_dK_offset, tdKtdK.layout)
        tdS = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_dS_offset, dtype=self.ds_dtype), tdS_layout.outer)
        # dQ
        thr_mma_dQ = tiled_mma_dQ.get_slice(mma_tile_coord_v)
        dQacc_shape = thr_mma_dQ.partition_shape_C(self.mma_tiler_dsk[:2])
        tdQtdQ = thr_mma_dQ.make_fragment_C(dQacc_shape)
        tdQtdQ = cute.make_tensor(tmem_ptr + self.tmem_dQ_offset, tdQtdQ.layout)

        block_info = BlockInfo(
            self.tile_m,
            # self.tile_n,
            self.tile_n * self.cluster_shape_mnk[0],  # careful, this case is not very well-tested
            self.is_causal,
            self.is_local,
            False,  # is_split_kv
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=self.pack_factor,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1],
            # With ragged K/V TMA tensors mode 0 is a sentinel extent, but cu_seqlens_k or
            # seqused_k is then always present and the static length is unused.
            seqlen_k_static=mK.shape[0] if const_expr(not self.ragged_kv_tma) else Int32(0),
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            tile_m=self.tile_m,
            tile_n=self.tile_n * self.cluster_shape_mnk[0],
        )
        TileSchedulerCls = partial(self.tile_scheduler_cls.create, tile_sched_params)

        AttentionMaskCls = self._generate_attention_mask_cls(window_size_left, window_size_right)
        #  EMPTY
        # (15)
        if warp_idx == self.empty_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_empty)

        #  RELAY
        # (14)
        if warp_idx == self.relay_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_mma if self.use_2cta_instrs else self.num_regs_empty)
            if const_expr(self.use_2cta_instrs):
                self.relay(
                    dS_cluster_full_mbar_ptr,
                    dS_cluster_empty_mbar_ptr,
                    dS_cluster_leader_mbar_ptr,
                    cluster_layout_vmnk,
                    block_info,
                    SeqlenInfoCls,
                    TileSchedulerCls,
                    blocksparse_tensors,
                )

        #  LOAD
        # (13)
        if warp_idx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            self.load(
                thr_mma_S,
                thr_mma_dP,
                thr_mma_dV,
                thr_mma_dK,
                thr_mma_dQ,
                mQ,
                mK,
                mKt,
                mV,
                mK_gmem,
                mV_gmem,
                mKt_gmem,
                tensormap_workspace,
                mdO,
                mQt,
                mdOt,
                mLSE,
                mdPsum,
                sQ,
                sK,
                sKt,
                sV,
                sdO,
                sQt,
                sdOt,
                sLSE,
                sdPsum,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_Kt,
                tma_atom_V,
                tma_atom_dO,
                tma_atom_Qt,
                tma_atom_dOt,
                pipeline_Q,
                pipeline_Qt,
                pipeline_Kt,
                pipeline_dO,
                pipeline_LSE,
                pipeline_dPsum,
                cluster_layout_vmnk,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                blocksparse_tensors,
                should_load_Q=True,
                should_load_dO=True,
                aux_data=aux_data,
            )

        #  MMA
        # (12)
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_mma)

            # Alloc tmem buffer
            tmem.allocate(self.tmem_alloc_cols)
            # TmemAllocator reconstruction drops a custom barrier; use the local owner.
            tmem_alloc_barrier.arrive_and_wait()
            tmem_ptr = tmem.retrieve_ptr(Float32)

            self.mma(
                tiled_mma_S,
                tiled_mma_dP,
                tiled_mma_dV,
                tiled_mma_dK,
                tiled_mma_dQ,
                sQ,
                sQt,
                sK,
                sKt,
                sV,
                sdO,
                sdOt,
                tP,
                sdSt,
                sdS,
                tdS,
                sP,
                sdS_dK,
                tStS,
                tdPtdP,
                tdVtdV,
                tdKtdK,
                tdQtdQ,
                dS_cluster_full_mbar_ptr,
                dS_cluster_empty_mbar_ptr,
                dS_cluster_leader_mbar_ptr,
                pipeline_Q,
                pipeline_Qt,
                pipeline_Kt,
                pipeline_dO,
                pipeline_S_P,
                pipeline_dS,
                pipeline_dKV,
                pipeline_dP,
                pipeline_dQ,
                pipeline_P,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                is_leader_cta,
                blocksparse_tensors,
            )
            # Dealloc the tensor memory buffer
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        # Compute
        # (4, 5, 6, 7, 8, 9, 10, 11) --> 8 warps
        if warp_idx >= self.compute_warp_ids[0] and warp_idx <= self.compute_warp_ids[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_compute)  # 8 warps
            # TmemAllocator reconstruction drops a custom barrier; use the local owner.
            tmem_alloc_barrier.arrive_and_wait()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            self.compute_loop(
                thr_mma_S,
                thr_mma_dP,
                thr_mma_dV,
                thr_mma_dK,
                tStS,
                tdPtdP,
                tdVtdV,
                tdKtdK,
                sLSE,
                sdPsum,
                mdV,
                mdK,
                sdS,
                sdS_xchg,
                sP,
                sdS_dK,
                pipeline_LSE,
                pipeline_dPsum,
                pipeline_S_P,
                pipeline_dS,
                pipeline_dKV,
                pipeline_dP,
                pipeline_P,
                dS_cluster_empty_mbar_ptr,
                dS_cluster_full_mbar_ptr,
                dQaccum_empty_mbar_ptr,
                softmax_scale,
                softmax_scale_log2,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
                sdV,
                sdK,
                mdV_tma_tensor,
                mdK_tma_tensor,
                tma_atom_dV,
                tma_atom_dK,
                tiled_copy_r2s_dKV,
                mdK_semaphore,
                mdV_semaphore,
                aux_data,
                fastdiv_mods,
                blocksparse_tensors,
            )
            tmem_alloc_barrier.arrive()

        # Reduce
        # (0, 1, 2, 3) - dQ
        if warp_idx >= self.reduce_warp_ids[0] and warp_idx <= self.reduce_warp_ids[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_reduce)
            # TmemAllocator reconstruction drops a custom barrier; use the local owner.
            tmem_alloc_barrier.arrive_and_wait()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            self.dQacc_reduce(
                mdQaccum,
                sdQaccum,
                thr_mma_dQ,
                tdQtdQ,
                pipeline_dQ,
                dQaccum_empty_mbar_ptr,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                mdQ_semaphore,
                blocksparse_tensors,
            )
            tmem_alloc_barrier.arrive()

        return

    @cute.jit
    def load(
        self,
        thr_mma_S: cute.ThrMma,
        thr_mma_dP: cute.ThrMma,
        thr_mma_dV: cute.ThrMma,
        thr_mma_dK: cute.ThrMma,
        thr_mma_dQ: cute.ThrMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mKt: Optional[cute.Tensor],
        mV: cute.Tensor,
        mK_gmem: Optional[cute.Tensor],
        mV_gmem: Optional[cute.Tensor],
        mKt_gmem: Optional[cute.Tensor],
        tensormap_workspace: Optional[cute.Tensor],
        mdO: cute.Tensor,
        mQt: Optional[cute.Tensor],
        mdOt: Optional[cute.Tensor],
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sKt: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sQt: cute.Tensor,
        sdOt: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_Kt: Optional[cute.CopyAtom],
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_Qt: Optional[cute.CopyAtom],
        tma_atom_dOt: Optional[cute.CopyAtom],  # 2-CTA only
        pipeline_Q: PipelineAsync,
        pipeline_Qt: PipelineAsync,
        pipeline_Kt: PipelineAsync,
        pipeline_dO: PipelineAsync,
        pipeline_LSE: PipelineAsync,
        pipeline_dPsum: PipelineAsync,
        cluster_layout_vmnk: cute.Layout,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        should_load_Q: bool = True,
        should_load_dO: bool = True,
        aux_data: AuxData = AuxData(),
    ):
        producer_state_Q_LSE = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.Q_stage)
        producer_state_Qt = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.Q_stage)
        producer_state_Kt = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.single_stage
        )
        producer_state_dO_dPsum = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.dO_stage
        )
        producer_state_Q_Qt = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.Q_stage)
        producer_state_O_Ot = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.dO_stage)
        producer_state_LSE = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.Q_stage)
        producer_state_dPsum = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.dO_stage
        )

        # Compute multicast mask for Q & dO buffer full
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        q_do_mcast_mask = None
        if const_expr(self.is_q_do_mcast):
            q_do_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        tensormap_manager = None
        tensormap_gmem = ()
        desc_ptrs = (None, None, None)
        atoms = ()
        if const_expr(self.ragged_kv_tma):
            tensormap_manager = TensorMapManager(TensorMapUpdateMode.GMEM, 128)
            bidx, bidy, bidz = cute.arch.block_idx()
            gdx, gdy, _ = cute.arch.grid_dim()
            cta_linear_idx = bidx + bidy * gdx + bidz * gdx * gdy
            if cta_linear_idx >= cute.size(tensormap_workspace, mode=[0]):
                # An undersized workspace would silently corrupt memory: fail loudly instead.
                cute.arch.inline_ptx("trap;", read_only_args=[])
            n_desc = 3 if const_expr(self.use_2cta_instrs) else 2
            atoms = (tma_atom_K, tma_atom_V, tma_atom_Kt)[:n_desc]
            tensormap_gmem = tuple(
                tensormap_manager.get_tensormap_ptr(tensormap_workspace[(cta_linear_idx, i, None)].iterator)
                for i in range(n_desc)
            )
            desc_ptrs = tuple(tensormap_manager.get_tensormap_ptr(p, cute.AddressSpace.generic) for p in tensormap_gmem) + (
                None,
            ) * (3 - n_desc)
            for atom, gmem_ptr in zip(atoms, tensormap_gmem):
                tensormap_manager.init_tensormap_from_atom(atom, gmem_ptr, self.load_warp_id)
            tensormap_manager.fence_tensormap_initialization()
        # The K/V extent depends on the batch only: rewrite the descriptors when it changes.
        kv_desc_batch_idx = Int32(-1)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            if const_expr(self.ragged_kv_tma) and batch_idx != kv_desc_batch_idx:
                # Shrink the rows extent of K, V (mode 0) and Kt (mode 1) to this sequence's
                # end: packed layouts count absolute rows, dense layouts per-batch rows.
                # Extents must stay >= 1.
                kv_desc_batch_idx = batch_idx
                kv_rows_end = seqlen.offset_k + seqlen.seqlen_k if const_expr(seqlen.has_cu_seqlens_k) else seqlen.seqlen_k
                kv_rows_end = cutlass.max(kv_rows_end, 1)
                bounded = []
                for t, rows_mode in ((mK_gmem, 0), (mV_gmem, 0), (mKt_gmem, 1)):
                    if const_expr(t is not None):
                        shape = list(t.shape)
                        shape[rows_mode] = kv_rows_end
                        bounded.append(cute.make_tensor(t.iterator, cute.make_layout(tuple(shape), stride=t.stride)))
                tensormap_manager.update_tensormap(
                    tuple(bounded), atoms, tensormap_gmem, self.load_warp_id, (None,) * len(atoms)
                )
                for p in tensormap_gmem:
                    tensormap_manager.fence_tensormap_update(p)
                    cute.arch.inline_ptx("prefetch.tensormap [{$r0}];", read_only_args=[p.toint()])
            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block // self.cluster_shape_mnk[0])
            head_idx_kv = self.kv_head_of(head_idx)
            n_block_cta_group = n_block // self.cta_group_size
            n_block_sparse = n_block // self.kv_subtile_factor

            # GMEM tensors (varlen-aware)
            # pack_gqa: the Q row mode is ((qh, seqlen)), so a varlen offset applies to its
            # second mode; LSE / dPsum are per KV head in packed rows, pack_factor per token.
            q_off = seqlen.offset_q if const_expr(not self.pack_gqa) else (None, seqlen.offset_q)
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
            mK_cur, mV_cur = [seqlen.offset_batch_K(t, batch_idx, dim=3)[None, None, head_idx_kv] for t in (mK, mV)]
            kv_block = n_block_cta_group
            if const_expr(not seqlen.has_cu_seqlens_q):
                mdO_cur = mdO[None, None, head_idx, batch_idx]
                mLSE_cur = mLSE[None, head_idx, batch_idx]
                mdPsum_cur = mdPsum[None, head_idx, batch_idx]
            else:
                mdO_cur = cute.domain_offset((0, q_off), mdO[None, None, head_idx])
                row_offset = sequence_row_offset(seqlen.offset_q, batch_idx, self.pack_factor, self.tile_m)
                mLSE_cur = cute.domain_offset((row_offset,), mLSE[None, head_idx])
                mdPsum_cur = cute.domain_offset((row_offset,), mdPsum[None, head_idx])

            if const_expr(self.use_2cta_instrs):
                if const_expr(not seqlen.has_cu_seqlens_q):
                    mQt_cur = mQt[None, None, head_idx, batch_idx]
                    mdOt_cur = mdOt[None, None, head_idx, batch_idx]
                else:
                    mQt_cur = cute.domain_offset((0, q_off, 0), mQt)[None, None, head_idx]
                    mdOt_cur = cute.domain_offset((q_off, 0, 0), mdOt)[None, None, head_idx]
                if const_expr(not seqlen.has_cu_seqlens_k):
                    mKt_cur = mKt[None, None, head_idx_kv, batch_idx]
                else:
                    mKt_cur = cute.domain_offset((0, seqlen.offset_k, 0), mKt)[None, None, head_idx_kv]

            # (1) S.T = K @ Q.T
            gK = cute.local_tile(mK_cur, cute.select(self.mma_tiler_kq, mode=[0, 2]), (kv_block, 0))
            tSgK = thr_mma_S.partition_A(gK)

            gQ = cute.local_tile(mQ_cur, cute.select(self.mma_tiler_kq, mode=[1, 2]), (None, 0))
            tSgQ = thr_mma_S.partition_B(gQ)
            gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (None,))
            gMAX = cute.local_tile(aux_data.tensors[0][batch_idx, head_idx, None], (self.tile_m,), (None,))
            sMAX = cute.make_tensor(sLSE.iterator + self.tile_m, sLSE.layout)
            gdPsum = cute.local_tile(mdPsum_cur, (self.tile_m,), (None,))
            gdO = cute.local_tile(mdO_cur, cute.select(self.mma_tiler_pdo, mode=[1, 2]), (0, None))
            tdPgdO = thr_mma_dV.partition_B(gdO)

            a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
            load_K, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_K,
                block_in_cluster_coord_vmnk[2],
                a_cta_layout,
                tSgK,
                sK,
                single_stage=True,
                **({"tma_desc_ptr": desc_ptrs[0]} if const_expr(self.ragged_kv_tma) else {}),
            )

            b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
            load_Q, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q,
                cta_coord=block_in_cluster_coord_vmnk[1],
                cta_layout=b_cta_layout,
                src_tensor=tSgQ,
                dst_tensor=sQ,
                mcast_mask=q_do_mcast_mask,
            )
            load_Q = copy_utils.tma_producer_copy_fn(load_Q, pipeline_Q)

            # (2) dP = V @ dO.T
            gV = cute.local_tile(mV_cur, cute.select(self.mma_tiler_vdo, mode=[0, 2]), (kv_block, 0))
            tdPgV = thr_mma_dP.partition_A(gV)

            load_V, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_V,
                0,
                cute.make_layout(1),
                tdPgV,
                sV,
                single_stage=True,
                **({"tma_desc_ptr": desc_ptrs[1]} if const_expr(self.ragged_kv_tma) else {}),
            )

            load_dOt = None
            if const_expr(tma_atom_dOt is not None):
                gdOt = cute.local_tile(mdOt_cur, cute.select(self.mma_tiler_vdo, mode=[1, 2]), (None, 0))
                tdPgdO = thr_mma_dP.partition_B(gdOt)
                load_dOt, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_dOt,
                    cta_coord=block_in_cluster_coord_vmnk[1],
                    cta_layout=b_cta_layout,
                    src_tensor=tdPgdO,
                    dst_tensor=sdOt,
                    mcast_mask=q_do_mcast_mask,
                )
                load_dOt = copy_utils.tma_producer_copy_fn(load_dOt, pipeline_dO)

            # (3) dV += P.T @ dO
            gdO = cute.local_tile(mdO_cur, cute.select(self.mma_tiler_pdo, mode=[1, 2]), (0, None))
            tdVgdO = thr_mma_dV.partition_B(gdO)
            load_dO, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_dO,
                cta_coord=block_in_cluster_coord_vmnk[1],
                cta_layout=b_cta_layout,
                src_tensor=tdVgdO,
                dst_tensor=sdO,
                mcast_mask=q_do_mcast_mask,
            )
            load_dO = copy_utils.tma_producer_copy_fn(load_dO, pipeline_dO)

            # (4) dK += dS.T @ Q (2-CTA: needs separate Qt load)
            load_Qt = None
            if const_expr(tma_atom_Qt is not None):
                gQt = cute.local_tile(mQt_cur, cute.select(self.mma_tiler_dsq, mode=[1, 2]), (0, None))
                tdKgQt = thr_mma_dK.partition_B(gQt)
                load_Qt, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_Qt,
                    cta_coord=block_in_cluster_coord_vmnk[1],
                    cta_layout=b_cta_layout,
                    src_tensor=tdKgQt,
                    dst_tensor=sQt,
                    mcast_mask=q_do_mcast_mask,
                )
                load_Qt = copy_utils.tma_producer_copy_fn(load_Qt, pipeline_Qt)

            # (5) dQ = dS @ K
            load_Kt = None
            if const_expr(self.use_2cta_instrs):
                gKt = cute.local_tile(mKt_cur, cute.select(self.mma_tiler_dsk, mode=[1, 2]), (0, kv_block))
                tdQgK = thr_mma_dQ.partition_B(gKt)

                load_Kt, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_Kt,
                    block_in_cluster_coord_vmnk[1],
                    b_cta_layout,
                    tdQgK,
                    sKt,
                    single_stage=True,
                    **({"tma_desc_ptr": desc_ptrs[2]} if const_expr(self.ragged_kv_tma) else {}),
                )

            copy_atom_stats = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), Float32)
            copy_stats = partial(cute.copy, copy_atom_stats)
            # copy_atom_stats = cute.make_copy_atom(cpasync.CopyBulkG2SMulticastOp(), Float32)
            # sLSE = cute.logical_divide(sLSE, (64,))[(None, block_in_cluster_coord_vmnk[1]), None]
            # gLSE = cute.logical_divide(gLSE, (64,))[(None, block_in_cluster_coord_vmnk[1]), None]
            # sdPsum = cute.logical_divide(sdPsum, (64,))[(None, block_in_cluster_coord_vmnk[1]), None]
            # gdPsum = cute.logical_divide(gdPsum, (64,))[(None, block_in_cluster_coord_vmnk[1]), None]
            # copy_stats = partial(cute.copy, copy_atom_stats, mcast_mask=q_do_mcast_mask)

            if const_expr(not self.use_block_sparsity):
                process_tile = const_expr(not self.is_local and not self.is_varlen_q) or m_block_min < m_block_max

                if process_tile:
                    first_m_block = m_block_min
                    if const_expr(self.cta_schedule == CtaSchedule.TWO_CTA_SERIAL):
                        #### Prologue ####
                        assert should_load_Q and should_load_dO
                        assert load_dOt is not None and load_Qt is not None
                        # K & Q (for S)
                        pipeline_Q.producer_acquire(
                            producer_state_Q_Qt,
                            extra_tx_count=self.tma_copy_bytes["K"],
                        )
                        load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q_Qt))
                        load_Q(first_m_block, producer_state=producer_state_Q_Qt)
                        pipeline_Q.producer_commit(producer_state_Q_Qt)
                        producer_state_Q_Qt.advance()
                        # LSE
                        pipeline_LSE.producer_acquire(producer_state_LSE)
                        with cute.arch.elect_one():
                            copy_stats(
                                gLSE[None, first_m_block],
                                sLSE[None, producer_state_LSE.index],
                                mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_LSE),
                            )
                            copy_stats(
                                gMAX[None, first_m_block],
                                sMAX[None, producer_state_LSE.index],
                                mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_LSE),
                            )
                        producer_state_LSE.advance()

                        # dOt + V, for dP.T = V @ dO.T
                        pipeline_dO.producer_acquire(
                            producer_state_O_Ot,
                            extra_tx_count=self.tma_copy_bytes["V"],
                        )
                        load_V(tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state_O_Ot))
                        load_dOt(first_m_block, producer_state=producer_state_O_Ot)
                        pipeline_dO.producer_commit(producer_state_O_Ot)
                        producer_state_O_Ot.advance()
                        # dPsum
                        pipeline_dPsum.producer_acquire(producer_state_dPsum)
                        with cute.arch.elect_one():
                            copy_stats(
                                gdPsum[None, first_m_block],
                                sdPsum[None, producer_state_dPsum.index],
                                mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dPsum),
                            )
                        producer_state_dPsum.advance()

                        # Qt, for dK = dS.T @ Q
                        pipeline_Qt.producer_acquire(
                            producer_state_Q_Qt,
                            extra_tx_count=self.tma_copy_bytes["K"],
                        )
                        load_Qt(first_m_block, producer_state=producer_state_Q_Qt)
                        load_Kt(tma_bar_ptr=pipeline_Qt.producer_get_barrier(producer_state_Q_Qt))
                        pipeline_Qt.producer_commit(producer_state_Q_Qt)
                        producer_state_Q_Qt.advance()

                        # dO, for dV = P.T @ dO
                        pipeline_dO.producer_acquire(producer_state_O_Ot)
                        load_dO(first_m_block, producer_state=producer_state_O_Ot)
                        pipeline_dO.producer_commit(producer_state_O_Ot)
                        producer_state_O_Ot.advance()

                        #### Mainloop ####
                        # 2CTA: [lse | Q | dOt | dPsum | Qt | dO]
                        for m_block in cutlass.range(m_block_min + 1, m_block_max, unroll=1):
                            # LSE
                            pipeline_LSE.producer_acquire(producer_state_LSE)
                            self.lse_handoff.arrive_and_wait()
                            with cute.arch.elect_one():
                                copy_stats(
                                    gLSE[None, m_block],
                                    sLSE[None, producer_state_LSE.index],
                                    mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_LSE),
                                )
                                copy_stats(
                                    gMAX[None, m_block],
                                    sMAX[None, producer_state_LSE.index],
                                    mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_LSE),
                                )
                            producer_state_LSE.advance()

                            # Q
                            pipeline_Q.producer_acquire(producer_state_Q_Qt)
                            load_Q(m_block, producer_state=producer_state_Q_Qt)
                            pipeline_Q.producer_commit(producer_state_Q_Qt)
                            producer_state_Q_Qt.advance()

                            # dPsum
                            pipeline_dPsum.producer_acquire(producer_state_dPsum)
                            self.delta_handoff.arrive_and_wait()
                            with cute.arch.elect_one():
                                copy_stats(
                                    gdPsum[None, m_block],
                                    sdPsum[None, producer_state_dPsum.index],
                                    mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dPsum),
                                )
                            producer_state_dPsum.advance()

                            # dOt, for dP.T = V @ dO.T
                            pipeline_dO.producer_acquire(producer_state_O_Ot)
                            load_dOt(m_block, producer_state=producer_state_O_Ot)
                            pipeline_dO.producer_commit(producer_state_O_Ot)
                            producer_state_O_Ot.advance()

                            # Qt, for dK = dS.T @ Q
                            pipeline_Qt.producer_acquire(producer_state_Q_Qt)
                            load_Qt(m_block, producer_state=producer_state_Q_Qt)
                            pipeline_Qt.producer_commit(producer_state_Q_Qt)
                            producer_state_Q_Qt.advance()

                            # dO, for dV = P.T @ dO
                            pipeline_dO.producer_acquire(producer_state_O_Ot)
                            load_dO(m_block, producer_state=producer_state_O_Ot)
                            pipeline_dO.producer_commit(producer_state_O_Ot)
                            producer_state_O_Ot.advance()

                        pipeline_Q.producer_tail(producer_state_Q_Qt)
                        pipeline_LSE.producer_tail(producer_state_LSE)
                        pipeline_dO.producer_tail(producer_state_O_Ot)
                        pipeline_dPsum.producer_tail(producer_state_dPsum)
                    else:
                        #### Prologue ####
                        if const_expr(should_load_Q):
                            # K & Q (for S)
                            pipeline_Q.producer_acquire(producer_state_Q_LSE, extra_tx_count=self.tma_copy_bytes["K"])
                            load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q_LSE))
                            load_Q(first_m_block, producer_state=producer_state_Q_LSE)
                            pipeline_Q.producer_commit(producer_state_Q_LSE)

                            # LSE
                            pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                            with cute.arch.elect_one():
                                copy_stats(
                                    gLSE[None, first_m_block],
                                    sLSE[None, producer_state_Q_LSE.index],
                                    mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_Q_LSE),
                                )
                                copy_stats(
                                    gMAX[None, first_m_block],
                                    sMAX[None, producer_state_Q_LSE.index],
                                    mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_Q_LSE),
                                )
                            producer_state_Q_LSE.advance()

                        if const_expr(should_load_dO):
                            pipeline_dO.producer_acquire(
                                producer_state_dO_dPsum,
                                extra_tx_count=(
                                    self.tma_copy_bytes["V"] + self.tma_copy_bytes["dO"]
                                    if const_expr(load_dOt is not None)
                                    else self.tma_copy_bytes["V"]
                                ),
                            )
                            load_V(tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state_dO_dPsum))
                            load_dO(first_m_block, producer_state=producer_state_dO_dPsum)
                            if const_expr(load_dOt is not None):
                                load_dOt(first_m_block, producer_state=producer_state_dO_dPsum)
                            pipeline_dO.producer_commit(producer_state_dO_dPsum)

                            # dPsum
                            pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                            with cute.arch.elect_one():
                                copy_stats(
                                    gdPsum[None, first_m_block],
                                    sdPsum[None, producer_state_dO_dPsum.index],
                                    mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dO_dPsum),
                                )
                            producer_state_dO_dPsum.advance()

                        if const_expr(self.use_2cta_instrs):
                            pipeline_Kt.producer_acquire(producer_state_Kt)
                            load_Kt(tma_bar_ptr=pipeline_Kt.producer_get_barrier(producer_state_Kt))
                            pipeline_Kt.producer_commit(producer_state_Kt)
                            producer_state_Kt.advance()
                        #### Main Loop ####
                        for m_block in cutlass.range(m_block_min + 1, m_block_max, unroll=1):
                            if const_expr(should_load_Q):
                                if const_expr(load_Qt is not None):
                                    pipeline_Qt.producer_acquire(producer_state_Qt)
                                    load_Qt(m_block - 1, producer_state=producer_state_Qt)
                                    pipeline_Qt.producer_commit(producer_state_Qt)
                                    producer_state_Qt.advance()

                                # Q (for S)
                                pipeline_Q.producer_acquire(producer_state_Q_LSE)
                                load_Q(m_block, producer_state=producer_state_Q_LSE)
                                pipeline_Q.producer_commit(producer_state_Q_LSE)

                                # LSE
                                pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                                self.lse_handoff.arrive_and_wait()
                                with cute.arch.elect_one():
                                    copy_stats(
                                        gLSE[None, m_block],
                                        sLSE[None, producer_state_Q_LSE.index],
                                        mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_Q_LSE),
                                    )
                                    copy_stats(
                                        gMAX[None, m_block],
                                        sMAX[None, producer_state_Q_LSE.index],
                                        mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_Q_LSE),
                                    )
                                producer_state_Q_LSE.advance()

                            if const_expr(should_load_dO):
                                pipeline_dO.producer_acquire(
                                    producer_state_dO_dPsum,
                                    extra_tx_count=self.tma_copy_bytes["dO"] if const_expr(load_dOt is not None) else 0,
                                )
                                load_dO(m_block, producer_state=producer_state_dO_dPsum)
                                if const_expr(load_dOt is not None):
                                    load_dOt(m_block, producer_state=producer_state_dO_dPsum)
                                pipeline_dO.producer_commit(producer_state_dO_dPsum)

                                # dPsum
                                pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                                self.delta_handoff.arrive_and_wait()
                                with cute.arch.elect_one():
                                    copy_stats(
                                        gdPsum[None, m_block],
                                        sdPsum[None, producer_state_dO_dPsum.index],
                                        mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dO_dPsum),
                                    )
                                producer_state_dO_dPsum.advance()

                        #### Tail ####
                        if const_expr(should_load_Q):
                            if const_expr(load_Qt is not None):
                                pipeline_Qt.producer_acquire(producer_state_Qt)
                                load_Qt(m_block_max - 1, producer_state=producer_state_Qt)
                                pipeline_Qt.producer_commit(producer_state_Qt)
                                producer_state_Qt.advance()

                            pipeline_Q.producer_tail(producer_state_Q_LSE.clone())
                            pipeline_LSE.producer_tail(producer_state_Q_LSE)
                            if const_expr(load_Qt is not None):
                                pipeline_Qt.producer_tail(producer_state_Qt)
                        if const_expr(should_load_dO):
                            pipeline_dO.producer_tail(producer_state_dO_dPsum.clone())
                            pipeline_dPsum.producer_tail(producer_state_dO_dPsum)

            else:
                assert blocksparse_tensors is not None
                if const_expr(self.cta_schedule == CtaSchedule.TWO_CTA_SERIAL):
                    assert should_load_Q and should_load_dO
                    assert load_dOt is not None and load_Qt is not None
                    assert load_Kt is not None and pipeline_Qt is not None
                    (
                        producer_state_Q_Qt,
                        producer_state_O_Ot,
                        producer_state_LSE,
                        producer_state_dPsum,
                    ) = produce_block_sparse_q_loads_bwd_sm100_2cta_hdim192(
                        blocksparse_tensors,
                        batch_idx,
                        head_idx,
                        n_block_sparse,
                        producer_state_Q_Qt,
                        producer_state_O_Ot,
                        producer_state_LSE,
                        producer_state_dPsum,
                        pipeline_Q,
                        pipeline_LSE,
                        pipeline_dO,
                        pipeline_dPsum,
                        pipeline_Qt,
                        load_K,
                        load_V,
                        load_Q,
                        load_dO,
                        load_Qt,
                        load_Kt,
                        load_dOt,
                        copy_stats,
                        gLSE,
                        sLSE,
                        gdPsum,
                        sdPsum,
                        self.tma_copy_bytes["K"],
                        self.tma_copy_bytes["V"],
                        q_subtile_factor=self.q_subtile_factor,
                        m_block_max=m_block_max,
                    )
                else:
                    (
                        producer_state_Q_LSE,
                        producer_state_dO_dPsum,
                        producer_state_Qt,
                        producer_state_Kt,
                    ) = produce_block_sparse_q_loads_bwd_sm100_default(
                        blocksparse_tensors,
                        batch_idx,
                        head_idx,
                        n_block_sparse,
                        producer_state_Q_LSE,
                        producer_state_dO_dPsum,
                        pipeline_Q,
                        pipeline_LSE,
                        pipeline_dO,
                        pipeline_dPsum,
                        load_K,
                        load_V,
                        load_Q,
                        load_dO,
                        copy_stats,
                        gLSE,
                        sLSE,
                        gdPsum,
                        sdPsum,
                        self.tma_copy_bytes["K"],
                        self.tma_copy_bytes["V"],
                        should_load_Q,
                        should_load_dO,
                        q_subtile_factor=self.q_subtile_factor,
                        m_block_max=m_block_max,
                        use_2cta_instrs=self.use_2cta_instrs,
                        producer_state_Qt=producer_state_Qt,
                        producer_state_Kt=producer_state_Kt,
                        pipeline_Qt=pipeline_Qt,
                        pipeline_Kt=pipeline_Kt,
                        load_Qt=load_Qt,
                        load_Kt=load_Kt,
                        load_dOt=load_dOt,
                        tma_copy_bytes_dO=self.tma_copy_bytes["dO"],
                    )
            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def compute_loop(
        self,
        thr_mma_S: cute.ThrMma,
        thr_mma_dP: cute.ThrMma,
        thr_mma_dV: cute.ThrMma,
        thr_mma_dK: cute.ThrMma,
        tStS: cute.Tensor,
        tdPtdP: cute.Tensor,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        sdS: cute.Tensor,
        sdS_xchg: Optional[cute.Tensor],
        sP: Optional[cute.Tensor],
        sdS_dK: Optional[cute.Tensor],
        pipeline_LSE: PipelineAsync,
        pipeline_dPsum: PipelineAsync,
        pipeline_S_P: PipelineAsync,
        pipeline_dS: PipelineAsync,
        pipeline_dKV: PipelineAsync,
        pipeline_dP: PipelineAsync,
        pipeline_P: Optional[PipelineAsync],
        dS_cluster_empty_mbar_ptr: cute.Pointer,
        dS_cluster_full_mbar_ptr: cute.Pointer,
        dQaccum_empty_mbar_ptr: Optional[cute.Pointer],
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        sdV: Optional[cute.Tensor],
        sdK: Optional[cute.Tensor],
        mdV_tma_tensor: Optional[cute.Tensor],
        mdK_tma_tensor: Optional[cute.Tensor],
        tma_atom_dV: Optional[cute.CopyAtom],
        tma_atom_dK: Optional[cute.CopyAtom],
        tiled_copy_r2s_dKV: Optional[cute.TiledCopy],
        mdK_semaphore: Optional[cute.Tensor],
        mdV_semaphore: Optional[cute.Tensor],
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
    ):
        sLSE_2D = cute.make_tensor(
            sLSE.iterator,
            cute.make_layout(
                (self.tile_m, self.cluster_tile_n if self.acc_m64 else self.tile_n, self.Q_stage),
                stride=(1, 0, cute.round_up(self.tile_m, 64) * 2),
            ),
        )
        sdPsum_2D = cute.make_tensor(
            sdPsum.iterator,
            cute.make_layout(
                (self.tile_m, self.cluster_tile_n if self.acc_m64 else self.tile_n, self.dO_stage),
                stride=(1, 0, cute.round_up(self.tile_m, 64)),
            ),
        )
        # if const_expr(self.SdP_swapAB):
        if const_expr(True):
            sLSE_2D = layout_utils.transpose_view(sLSE_2D)
            sdPsum_2D = layout_utils.transpose_view(sdPsum_2D)

        # tix: [128...384]  8 warps
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())  # 4-11
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.compute_warp_ids))
        # tidx = cute.arch.thread_idx()[0] - (cute.arch.WARP_SIZE * self.compute_warp_ids[0])
        dp_idx = tidx % 128
        num_wg = len(self.compute_warp_ids) // 4  # 2
        # wg_idx:
        # 0: [256...384]
        # 1: [128...256]

        if const_expr(not self.acc_m64):
            tileP_f32_like = self.cta_tiler[1] // 32 * self.v_dtype.width
            # tStS has shape ((128, 128), 1, 1), tStP has shape ((128, 64), 1, 1)
            tStP = cute.composition(tStS, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))
            # Rebase the iterator explicitly (P may alias S), otherwise the tmem address is wrong
            tStP = cute.make_tensor(tStS.iterator + (self.tmem_P_offset - self.tmem_S_offset), tStP.layout)
            tScS = thr_mma_S.partition_C(cute.make_identity_tensor(self.mma_tiler_kq[:2]))
            tScP = cute.composition(tScS, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))
            tdPtdS = cute.composition(tdPtdP, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))
            tdPtdS = cute.make_tensor(tdPtdP.iterator + (self.tmem_dS_offset - self.tmem_dP_offset), tdPtdS.layout)
            tdPcdP = thr_mma_dP.partition_C(cute.make_identity_tensor(self.mma_tiler_vdo[:2]))
            tdPcdS = cute.composition(tdPcdP, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))

            # 2-CTA assumes: repetiton should always be 32 & 16
            tmem_load_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32)
            tmem_store_atom = cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), Float32)

            # tmem -> rmem
            thr_copy_t2r = copy_utils.make_tmem_copy(tmem_load_atom, num_wg).get_slice(tidx)
            tStS_t2r = thr_copy_t2r.partition_S(tStS)  # (((32, 32), 1), 2, 1, 1)
            tdPtdP_t2r = thr_copy_t2r.partition_S(tdPtdP)
            tScS_t2r = thr_copy_t2r.partition_D(tScS)  # ((32, 1), 2, 1, 1)
            t0ScS_t2r = thr_copy_t2r.get_slice(0).partition_D(tScS)  # ((32, 1), 2, 1, 1)
            # ((32, 1), 2, 1, 1, STAGE)
            tSsLSE = thr_copy_t2r.partition_D(thr_mma_S.partition_C(sLSE_2D))
            tSsdPsum = thr_copy_t2r.partition_D(thr_mma_dP.partition_C(sdPsum_2D))
            # rmem -> tmem
            thr_copy_r2t = copy_utils.make_tmem_copy(tmem_store_atom, num_wg).get_slice(tidx)
            tScP_r2t = thr_copy_r2t.partition_S(tScP)
            tStP_r2t = thr_copy_r2t.partition_D(tStP)
            tdPcdS_r2t = thr_copy_r2t.partition_S(tdPcdS)
            tdPtdS_r2t = thr_copy_r2t.partition_D(tdPtdS)
            # rmem -> smem
            # This part is a bit iffy, we might be making a lot of assumptions here
            copy_atom_r2s = sm100_utils_basic.get_smem_store_op(LayoutEnum.ROW_MAJOR, self.ds_dtype, Float32, thr_copy_t2r)
            thr_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, thr_copy_t2r).get_slice(tidx)

            # We assume the swizzle (i.e. layout.inner) stays the same
            sdS_epi_layout = sm100_utils_basic.make_smem_layout_epi(
                self.ds_dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_m), 1
            )
            sdS_layout = cute.slice_(sdS_epi_layout.outer, (None, None, 0))  # ((8,16), (64,2))
            # Need to group into 1 mode to be compatible w thr_copy_r2s
            sdS_layout = cute.make_layout((sdS_layout.shape,), stride=(sdS_layout.stride,))
            sdS_epi = cute.make_tensor(sdS.iterator, sdS_layout)
            tRS_sdS = thr_copy_r2s.partition_D(sdS_epi)

            if const_expr(sdS_xchg is not None):
                sdS_xchg_epi = cute.make_tensor(cute.recast_ptr(sdS_xchg.iterator, sdS_epi_layout.inner), sdS_layout)
                tRS_sdS_xchg = thr_copy_r2s.partition_D(sdS_xchg_epi)
        else:
            # NOTE [M=64 accumulator layout]: the layout-aware tiled copy gives each of the 128
            # lanes its 64 columns as two 32-column chunks (lanes 0-63 hold columns [0, 64),
            # lanes 64-127 columns [64, 128) of the same rows); warpgroup g takes chunk g, so
            # every thread owns one (row, 32-column) piece of S^T / dP^T and the per-thread
            # register vector is a single stage.
            wg_idx = tidx // 128
            tmem_load_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32)
            tiled_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tStS)
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx % 128)
            tScS = thr_mma_S.partition_C(cute.make_identity_tensor(self.mma_tiler_kq[:2]))
            tdPcdP = thr_mma_dP.partition_C(cute.make_identity_tensor(self.mma_tiler_vdo[:2]))
            tStS_t2r = self.split_wg(thr_copy_t2r.partition_S(tStS), wg_idx, num_wg)
            tdPtdP_t2r = self.split_wg(thr_copy_t2r.partition_S(tdPtdP), wg_idx, num_wg)
            tScS_t2r = self.split_wg(thr_copy_t2r.partition_D(tScS), wg_idx, num_wg)
            t0ScS_t2r = self.split_wg(tiled_copy_t2r.get_slice(0).partition_D(tScS), 0, num_wg)
            tSsLSE = self.split_wg(thr_copy_t2r.partition_D(thr_mma_S.partition_C(sLSE_2D)), wg_idx, num_wg)
            tSsdPsum = self.split_wg(thr_copy_t2r.partition_D(thr_mma_dP.partition_C(sdPsum_2D)), wg_idx, num_wg)
            # P^T and dS^T are staged in SMEM (the A operands of the dV / dK MMAs) with the same
            # store op the dS exchange uses; the epilogue layout and the MMA A layout share the
            # 128B swizzle, so the MMA reads what the compute warps wrote.
            thr_copy_r2t = None
            tScP_r2t = tStP_r2t = tdPcdS_r2t = tdPtdS_r2t = None
            copy_atom_r2s = sm100_utils_basic.get_smem_store_op(LayoutEnum.ROW_MAJOR, self.ds_dtype, Float32, thr_copy_t2r)
            thr_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, thr_copy_t2r).get_slice(tidx % 128)
            sdS_epi_layout = sm100_utils_basic.make_smem_layout_epi(
                self.ds_dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_m), 1
            )
            sdS_layout_2d = cute.slice_(sdS_epi_layout.outer, (None, None, 0))
            # Rank 3 like the TMEM fragment, ((rows, cols), 1, 1), so the layout-aware r2s
            # tiled copy can partition it.
            sdS_layout = cute.make_layout((sdS_layout_2d.shape, 1, 1), stride=(sdS_layout_2d.stride, 0, 0))
            sdS_epi = cute.make_tensor(sdS.iterator, sdS_layout)
            tRS_sdS = self.split_wg(thr_copy_r2s.partition_D(sdS_epi), wg_idx, num_wg)
            sP_epi = cute.make_tensor(sP.iterator, sdS_layout)
            tRS_sP = self.split_wg(thr_copy_r2s.partition_D(sP_epi), wg_idx, num_wg)
            sdS_dK_epi = cute.make_tensor(sdS_dK.iterator, sdS_layout)
            tRS_sdS_dK = self.split_wg(thr_copy_r2s.partition_D(sdS_dK_epi), wg_idx, num_wg)
            # Q-column half this thread's chunk belongs to (from its partition coordinates); the
            # CTA keeps the half it owns for dQ (the peer's half reaches it from sdS_dK, see the
            # exchange copy below).
            q_half = tScS_t2r[0][1] // (self.tile_m // 2)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        dS_cluster_empty_phase = Int32(1)
        # 2-CTA: CTA 0 exchanges stage 1 (bottom half), CTA 1 exchanges stage 0 (top half)
        exchange_stage = cta_rank_in_cluster ^ 1 if const_expr(self.use_2cta_instrs) else Int32(0)
        if const_expr(self.use_2cta_instrs):
            peer_cta_rank_in_cluster = cta_rank_in_cluster ^ 1
            sdS_xchg_src = self._dS_exchange_source(sdS_xchg, sdS_dK, peer_cta_rank_in_cluster)

        consumer_state_S_P_dP = pipeline.make_pipeline_state(  # Our impl has shortcut for stage==1
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )
        # consumer_phase_S_P_dP = Int32(0)
        consumer_state_S = pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 1)
        producer_state_P = pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 1)
        producer_state_dS = pipeline.make_pipeline_state(  # Our impl has shortcut for stage==1
            cutlass.pipeline.PipelineUserType.Producer, 1
        )
        consumer_state_dKV = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 2)
        consumer_state_LSE = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage)
        consumer_state_dPsum = pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)

            recompute_fastdiv_mods_q = const_expr(
                aux_data.tensors is not None and (seqlen.has_cu_seqlens_q or seqlen.has_seqused_q)
            )
            recompute_fastdiv_mods_k = const_expr(
                aux_data.tensors is not None and (seqlen.has_cu_seqlens_k or seqlen.has_seqused_k)
            )

            if const_expr(fastdiv_mods is not None and fastdiv_mods[0] is not None):
                seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
                fastdiv_mods = (
                    seqlen_q_divmod if not recompute_fastdiv_mods_q else FastDivmodDivisorV2(seqlen.seqlen_q),
                    seqlen_k_divmod if not recompute_fastdiv_mods_k else FastDivmodDivisorV2(seqlen.seqlen_k),
                )
            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block // self.cluster_shape_mnk[0])
            mask = AttentionMaskCls(seqlen)
            cluster_n_block = n_block // self.cta_group_size
            # TODO: condition mask_seqlen
            mask_fn = partial(
                mask.apply_mask_sm100_transposed,
                tScS_t2r=tScS_t2r,
                t0ScS_t2r=t0ScS_t2r,
                n_block=cluster_n_block,
                mask_seqlen=True,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                mask_mod=self.mask_mod,
                batch_idx=batch_idx,
                head_idx=head_idx,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
            )

            # prefetch_LSE = not self.is_causal
            prefetch_LSE = False

            curr_q_cnt = Int32(0)
            curr_q_idx = None
            curr_full_cnt = Int32(0)
            curr_full_idx = None
            loop_count = m_block_max - m_block_min
            process_tile = const_expr(not self.is_local and not self.is_varlen_q) or m_block_min < m_block_max
            if const_expr(self.use_block_sparsity):
                assert blocksparse_tensors is not None
                (
                    curr_q_cnt,
                    curr_q_idx,
                    curr_full_cnt,
                    curr_full_idx,
                    loop_count,
                ) = get_block_sparse_iteration_info_bwd(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    n_block // self.kv_subtile_factor,
                    q_subtile_factor=self.q_subtile_factor,
                    m_block_max=m_block_max,
                )
                process_tile = loop_count > Int32(0)

            # Mainloop
            # Block sparsity: iterate over sparse m_block count and derive actual m_block
            # from Q_IDX/FULL_Q_IDX tensors. Dense: iterate m_block_min..m_block_max directly.
            for iter_idx in cutlass.range(loop_count, unroll=1):
                m_block = m_block_min + iter_idx
                m_block_oob = False
                is_full_block = False
                if const_expr(self.use_block_sparsity):
                    m_block, is_full_block = get_m_block_from_iter_bwd(
                        iter_idx,
                        curr_q_cnt,
                        curr_q_idx,
                        curr_full_cnt,
                        curr_full_idx,
                        q_subtile_factor=self.q_subtile_factor,
                        m_block_max=m_block_max,
                    )
                    m_block_oob = m_block >= m_block_max
                # Prefetch 1 stage of LSE
                pipeline_LSE.consumer_wait(consumer_state_LSE)
                tSrLSE_s2r = cute.make_rmem_tensor(tScS_t2r[None, 0, 0, 0].shape, Float32)
                if const_expr(prefetch_LSE and not self.shuffle_LSE):
                    cute.autovec_copy(tSsLSE[None, 0, 0, 0, consumer_state_LSE.index], tSrLSE_s2r)

                if const_expr(self.split_P_dS):
                    pipeline_S_P.consumer_wait(consumer_state_S)
                else:
                    pipeline_S_P.consumer_wait(consumer_state_S_P_dP)
                # pipeline_S_P.sync_object_full.wait(0, consumer_phase_S_P_dP)
                #### TMEM->RMEM (Load S from TMEM)
                tSrS_t2r = cute.make_rmem_tensor(tScS_t2r.shape, Float32)
                cute.copy(thr_copy_t2r, tStS_t2r, tSrS_t2r)

                if const_expr(self.split_P_dS):
                    # S is in registers: release it before the softmax so the MMA warp can
                    # issue the next QK into the slot
                    cute.arch.fence_view_async_tmem_load()
                    cute.arch.sync_warp()
                    with cute.arch.elect_one():
                        pipeline_S_P.consumer_release(consumer_state_S)
                    consumer_state_S.advance()
                elif const_expr(self.cta_schedule == CtaSchedule.TWO_CTA_SERIAL):
                    # Signal S tmem load completion using pipeline_S_P: dP overlaps S
                    cute.arch.fence_view_async_tmem_load()
                    with cute.arch.elect_one():
                        pipeline_S_P.consumer_release(consumer_state_S_P_dP)
                elif const_expr(self.cta_schedule == CtaSchedule.TWO_CTA_PIPELINED):
                    # Signal S tmem load completion using pipeline_dS: dQ overlaps S
                    if iter_idx > 0:
                        cute.arch.fence_view_async_tmem_load()
                        with cute.arch.elect_one():
                            pipeline_dS.producer_commit(producer_state_dS)
                        producer_state_dS.advance()

                if const_expr(self.score_mod_bwd is not None):
                    tSrS_pre = cute.make_fragment_like(tSrS_t2r)
                    cute.autovec_copy(tSrS_t2r, tSrS_pre)

                if const_expr(self.score_mod is not None):
                    # Apply score_mod FIRST -> matches forward
                    self.apply_score_mod(
                        tSrS_t2r,
                        thr_copy_t2r,
                        thr_mma_S,
                        batch_idx,
                        head_idx,
                        m_block,
                        n_block,
                        softmax_scale,
                        seqlen,
                        aux_data,
                        fastdiv_mods,
                        wg_split=(wg_idx, num_wg) if const_expr(self.acc_m64) else None,
                    )

                #### APPLY MASK (after score_mod, matching forward pass order)
                check_m_boundary = (m_block + 1) * self.tile_m > seqlen.seqlen_q
                mask_fn(
                    tSrS_t2r,
                    m_block=m_block,
                    is_full_block=is_full_block,
                    check_m_boundary=check_m_boundary,
                )
                num_stages = cute.size(tScS_t2r, mode=[1])
                # ---------------------------------------------
                #### P = exp(S - LSE)
                # ---------------------------------------------
                lane_idx = cute.arch.lane_idx()
                if const_expr(not self.acc_m64):
                    tSrP_r2t_f32 = cute.make_rmem_tensor(tScP_r2t.shape, Float32)  # 64
                    tSrP_r2t = cute.recast_tensor(tSrP_r2t_f32, self.q_dtype)
                for stage in cutlass.range_constexpr(num_stages):
                    tSrS_cur = tSrS_t2r[None, stage, 0, 0]
                    tSsLSE_cur = tSsLSE[None, stage, 0, 0, consumer_state_LSE.index]
                    if const_expr(not self.shuffle_LSE):
                        if const_expr(stage > 0 or not prefetch_LSE):
                            tSsMAX_cur = cute.make_tensor(tSsLSE_cur.iterator + self.tile_m, tSsLSE_cur.layout)
                            cute.autovec_copy(tSsMAX_cur, tSrLSE_s2r)
                            for index in cutlass.range_constexpr(cute.size(tSrLSE_s2r) // 2):
                                tSrS_cur[2 * index], tSrS_cur[2 * index + 1] = cute.arch.sub_packed_f32x2(
                                    (tSrS_cur[2 * index], tSrS_cur[2 * index + 1]),
                                    (tSrLSE_s2r[2 * index], tSrLSE_s2r[2 * index + 1]),
                                )
                            cute.autovec_copy(tSsLSE_cur, tSrLSE_s2r)
                        tSrLSE = tSrLSE_s2r
                    else:
                        tSrLSE = tSsLSE_cur[lane_idx]
                    for v in cutlass.range_constexpr(cute.size(tSrS_t2r, mode=[0]) // 2):
                        if const_expr(not self.shuffle_LSE):
                            lse_pair = (tSrLSE[2 * v], tSrLSE[2 * v + 1])
                        else:
                            lse_pair = (
                                utils.shuffle_sync(tSrLSE, offset=2 * v),
                                utils.shuffle_sync(tSrLSE, offset=2 * v + 1),
                            )
                        tSrS_cur[2 * v], tSrS_cur[2 * v + 1] = cute.arch.fma_packed_f32x2(
                            ((tSrS_cur[2 * v], tSrS_cur[2 * v + 1])),
                            (softmax_scale_log2, softmax_scale_log2),
                            (-lse_pair[0], -lse_pair[1]),
                        )
                        tSrS_cur[2 * v] = cute.math.exp2(tSrS_cur[2 * v], fastmath=True)
                        tSrS_cur[2 * v + 1] = cute.math.exp2(tSrS_cur[2 * v + 1], fastmath=True)
                    if const_expr(self.acc_m64):
                        # P^T goes to SMEM; S stays untouched in TMEM until the dS pipeline
                        # releases it, so no cross-warp barrier is needed before the store.
                        tSrP_cvt = cute.make_fragment_like(tSrS_cur, self.q_dtype)
                        utils.cvt_f16(tSrS_cur, tSrP_cvt)
                        cute.autovec_copy(tSrP_cvt, tRS_sP[None, 0, 0, 0])
                    else:
                        utils.cvt_f16(tSrS_cur, tSrP_r2t[None, stage, 0, 0])
                        if const_expr(stage == 0):
                            if const_expr(self.split_P_dS):
                                # wait until the previous pdo UMMA has consumed the P slot
                                pipeline_P.producer_acquire(producer_state_P)
                            else:
                                cute.arch.fence_view_async_tmem_load()
                                # P overwrites S lanes another warp may still be loading, so
                                # every warp must have loaded S before any stores P.
                                self.compute_sync_barrier.arrive_and_wait()
                        cute.copy(
                            thr_copy_r2t,
                            tSrP_r2t_f32[None, stage, None, None],
                            tStP_r2t[None, stage, None, None],
                        )

                if const_expr(not self.acc_m64):
                    cute.arch.fence_view_async_tmem_store()
                cute.arch.fence_view_async_shared()
                # The P (or S_P) and LSE mbarriers expect one arrival per compute warp
                if const_expr(self.split_P_dS):
                    cute.arch.sync_warp()
                else:
                    self.compute_sync_barrier.arrive_and_wait()
                if const_expr(self.split_P_dS):
                    with cute.arch.elect_one():
                        pipeline_P.producer_commit(producer_state_P)
                    producer_state_P.advance()
                elif const_expr(self.cta_schedule != CtaSchedule.TWO_CTA_SERIAL):
                    # Signal tmem store P completion with pipeline_S_P
                    with cute.arch.elect_one():
                        pipeline_S_P.consumer_release(consumer_state_S_P_dP)
                        # pipeline_S_P.sync_object_empty.arrive(0, pipeline_S_P.consumer_mask)
                # Normally we'd need syncwarp here since only 1 thread will signal in
                # consumer_release, but we already have the barrier or sync_warp before this
                if iter_idx + 1 < loop_count:
                    self.lse_handoff.arrive()
                pipeline_LSE.consumer_release(consumer_state_LSE)
                consumer_state_LSE.advance()
                # ---------------------------------------------
                # dS.T = P.T * (dP.T - D)
                # ---------------------------------------------
                pipeline_dPsum.consumer_wait(consumer_state_dPsum)
                pipeline_dP.consumer_wait(consumer_state_S_P_dP)
                # pipeline_dP.sync_object_full.wait(0, consumer_phase_S_P_dP)
                ### Now delayed to after loop
                # consumer_state_S_P_dP.advance()
                # consumer_phase_S_P_dP ^= 1

                ##### dS.T = P.T * (dP.T - Psum)
                for stage in cutlass.range_constexpr(num_stages):
                    tdPrdP_t2r = cute.make_rmem_tensor(tScS_t2r[None, 0, None, None].shape, Float32)
                    cute.copy(thr_copy_t2r, tdPtdP_t2r[None, stage, None, None], tdPrdP_t2r)
                    cute.arch.fence_view_async_tmem_load()
                    if const_expr(not self.split_P_dS):
                        # dS overwrites dP with the same cross-warp lane sharing as P over S
                        self.compute_sync_barrier.arrive_and_wait()
                    tdPrdP_cur = tdPrdP_t2r[None, 0, 0]
                    tSrS_cur = cute.make_rmem_tensor(tdPrdP_cur.shape, Float32)
                    if const_expr(self.acc_m64):
                        tSrS_cur.store(tSrP_cvt.load().to(Float32))
                    else:
                        tSrS_cur.store(tSrP_r2t[None, stage, 0, 0].load().to(Float32))
                    tSsdPsum_cur = tSsdPsum[None, stage, 0, 0, consumer_state_dPsum.index]
                    if const_expr(not self.shuffle_dPsum):
                        tSrdPsum = cute.make_fragment_like(tSsdPsum_cur, Float32)
                        cute.autovec_copy(tSsdPsum_cur, tSrdPsum)
                    else:
                        tSrdPsum = tSsdPsum_cur[lane_idx]
                    for v in cutlass.range_constexpr(cute.size(tdPrdP_t2r, mode=[0]) // 2):
                        if const_expr(not self.shuffle_dPsum):
                            dPsum_pair = (tSrdPsum[2 * v], tSrdPsum[2 * v + 1])
                        else:
                            dPsum_pair = (
                                utils.shuffle_sync(tSrdPsum, offset=2 * v),
                                utils.shuffle_sync(tSrdPsum, offset=2 * v + 1),
                            )
                        tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1] = cute.arch.sub_packed_f32x2(
                            (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]), dPsum_pair
                        )
                        tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1] = cute.arch.mul_packed_f32x2(
                            (tSrS_cur[2 * v], tSrS_cur[2 * v + 1]),
                            (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]),
                        )

                    if const_expr(self.score_mod_bwd is not None):
                        tSrS_pre_cur = tSrS_pre[None, stage, 0, 0]
                        cluster_tile_n = self.tile_n * self.cta_group_size
                        cluster_n_block = n_block // self.cta_group_size
                        cS_bwd = cute.make_identity_tensor((cluster_tile_n, self.tile_m))
                        cS_bwd = cute.domain_offset(
                            (cluster_n_block * cluster_tile_n, m_block * self.tile_m),
                            cS_bwd,
                        )
                        tScS_bwd = thr_mma_S.partition_C(cS_bwd)
                        tScS_idx_bwd = thr_copy_t2r.partition_D(tScS_bwd)
                        if const_expr(self.acc_m64):
                            tScS_idx_bwd = self.split_wg(tScS_idx_bwd, wg_idx, num_wg)
                        tScS_idx_cur = tScS_idx_bwd[None, stage, 0, 0]
                        self.apply_score_mod_bwd(
                            tdPrdP_cur,
                            tSrS_pre_cur,
                            tScS_idx_cur,
                            batch_idx,
                            head_idx,
                            softmax_scale,
                            seqlen,
                            aux_data,
                            fastdiv_mods,
                        )
                        # Zero out OOB positions (kv_idx >= seqlen_k) after score_mod_bwd
                        for i in cutlass.range(cute.size(tdPrdP_cur), unroll_full=True):
                            kv_idx = tScS_idx_cur[i][0]
                            tdPrdP_cur[i] = 0.0 if kv_idx >= seqlen.seqlen_k else tdPrdP_cur[i]

                    tdPrdS_cvt = cute.make_fragment_like(tdPrdP_cur, self.ds_dtype)
                    utils.cvt_f16(tdPrdP_cur, tdPrdS_cvt)
                    if const_expr(stage == 0):
                        pipeline_dS.producer_acquire(producer_state_dS)
                        if const_expr(self.use_2cta_instrs and not self.acc_m64):
                            tdPrdS_xchg = cute.make_fragment_like(tdPrdS_cvt, self.ds_dtype)

                    if const_expr(self.acc_m64):
                        # dS^T for the dK MMA: this CTA's rows, both Q-column halves; the half
                        # the peer needs for its dQ MMA is exchanged straight from here.
                        cute.autovec_copy(tdPrdS_cvt, tRS_sdS_dK[None, 0, 0, 0])
                        # dS for this CTA's own dQ MMA: the Q-column half it owns.
                        if q_half == cta_rank_in_cluster:
                            cute.autovec_copy(tdPrdS_cvt, tRS_sdS[None, 0, 0, 0])
                    else:
                        # RMEM->TMEM: always write to TMEM for MMA
                        if const_expr(not self.use_smem_dS_for_mma_dK or self.use_2cta_instrs):
                            tdPrdS_r2t_f32 = cute.recast_tensor(tdPrdS_cvt, Float32)
                            cute.copy(thr_copy_r2t, tdPrdS_r2t_f32, tdPtdS_r2t[None, stage, 0, 0])

                        # RMEM->SMEM: For 2-CTA, keep exchange stage in registers, write
                        # non-exchange to sdS
                        if const_expr(self.use_2cta_instrs):
                            if exchange_stage == stage:
                                cute.autovec_copy(tdPrdS_cvt, tdPrdS_xchg)
                            else:
                                cute.autovec_copy(tdPrdS_cvt, tRS_sdS[None, stage])
                        else:
                            cute.autovec_copy(tdPrdS_cvt, tRS_sdS[None, stage])

                if const_expr(self.acc_m64):
                    # dS^T is in SMEM: make this warp's stores visible to the async proxy before
                    # the per-warp arrival the MMA warp waits on.
                    cute.arch.fence_view_async_shared()
                elif const_expr(not self.use_smem_dS_for_mma_dK):
                    cute.arch.fence_view_async_tmem_store()

                if const_expr(self.use_2cta_instrs):
                    # use pipeline_dP to signal the dS store (TMEM, or SMEM with acc_m64)
                    with cute.arch.elect_one():
                        pipeline_dP.consumer_release(consumer_state_S_P_dP)
                consumer_state_S_P_dP.advance()

                # After the loop: copy exchange registers to sdS_xchg buffer
                if const_expr(sdS_xchg is not None):
                    # sdS_xchg aliases sdQaccum: wait until the dQ reduce has read it
                    if const_expr(dQaccum_empty_mbar_ptr is not None):
                        cute.arch.mbarrier_wait(dQaccum_empty_mbar_ptr, phase=producer_state_dS.phase)
                    cute.autovec_copy(tdPrdS_xchg, tRS_sdS_xchg[None, 0])

                cute.arch.fence_view_async_shared()
                # The dS and dPsum mbarriers expect one arrival per compute warp
                if const_expr(self.split_P_dS):
                    cute.arch.sync_warp()
                else:
                    self.compute_sync_barrier.arrive_and_wait()
                # Normally we'd need syncwarp here since only 1 thread will signal in
                # consumer_release, but we already have the barrier or sync_warp before this
                if iter_idx + 1 < loop_count:
                    self.delta_handoff.arrive()
                pipeline_dPsum.consumer_release(consumer_state_dPsum)
                consumer_state_dPsum.advance()
                # pipelined 2CTA schedule: pipeline_dS also signals S tmem load completion so is deferred
                if const_expr(self.cta_schedule != CtaSchedule.TWO_CTA_PIPELINED):
                    with cute.arch.elect_one():
                        pipeline_dS.producer_commit(producer_state_dS)
                    producer_state_dS.advance()

                # 2-CTA: DSMEM copy of the exchanged dS half to the peer's sdS buffer
                if const_expr(self.use_2cta_instrs):
                    stage_copy_bytes = const_expr(self.tma_copy_bytes["dS"] // 2)
                    stage_copy_elems = const_expr(stage_copy_bytes // (self.ds_dtype.width // 8))
                    if tidx == 0:
                        smem_src_ptr = sdS_xchg_src.iterator
                        # Destination is peer's sdS at our CTA's offset (exchange_stage position)
                        smem_dst_ptr = sdS.iterator + cta_rank_in_cluster * stage_copy_elems
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            dS_cluster_full_mbar_ptr,
                            stage_copy_bytes,
                            peer_cta_rank_in_cluster=peer_cta_rank_in_cluster,
                        )
                        copy_utils.cpasync_bulk_s2cluster(
                            smem_src_ptr,
                            smem_dst_ptr,
                            dS_cluster_full_mbar_ptr,
                            stage_copy_bytes,
                            peer_cta_rank_in_cluster=peer_cta_rank_in_cluster,
                        )

            # Final signal for dS smem store completion
            if const_expr(self.cta_schedule == CtaSchedule.TWO_CTA_PIPELINED):
                if process_tile:
                    with cute.arch.elect_one():
                        pipeline_dS.producer_commit(producer_state_dS)
                    producer_state_dS.advance()

            # Epilogue
            # Run epilogue if we processed any m_blocks for this n_block
            if process_tile:
                if const_expr(not self.use_tma_store):
                    consumer_state_dKV = self.epilogue_dKV(
                        dp_idx,
                        warp_idx,
                        batch_idx,
                        head_idx,
                        n_block,
                        seqlen,
                        thr_mma_dV,
                        thr_mma_dK,
                        tdVtdV,
                        tdKtdK,
                        mdV,
                        mdK,
                        pipeline_dKV,
                        consumer_state_dKV,
                        softmax_scale,
                    )
                else:
                    thr_copy_r2s_dKV = tiled_copy_r2s_dKV.get_slice(dp_idx)
                    #### STORE dV
                    consumer_state_dKV = self.epilogue_dK_or_dV_tma(
                        dp_idx,
                        batch_idx,
                        head_idx,
                        n_block,
                        seqlen,
                        thr_mma_dV,
                        tdVtdV,
                        mdV_tma_tensor,
                        sdV,
                        tma_atom_dV,
                        thr_copy_r2s_dKV,
                        pipeline_dKV,
                        consumer_state_dKV,
                        None,  # Don't scale
                        int(NamedBarrierBwdSm100.EpilogueWG1),  # barrier_id
                        mdV_semaphore,
                        "V",
                    )
                    #### STORE dK
                    consumer_state_dKV = self.epilogue_dK_or_dV_tma(
                        dp_idx,
                        batch_idx,
                        head_idx,
                        n_block,
                        seqlen,
                        thr_mma_dK,
                        tdKtdK,
                        mdK_tma_tensor,
                        sdK,
                        tma_atom_dK,
                        thr_copy_r2s_dKV,
                        pipeline_dKV,
                        consumer_state_dKV,
                        softmax_scale if const_expr(not self.dKV_postprocess) else None,
                        int(NamedBarrierBwdSm100.EpilogueWG1),  # barrier_id
                        mdK_semaphore,
                        "K",
                    )
            # Zero dK/dV for empty tiles (local attention or block sparsity)
            # When total_m_block_cnt == 0 for block sparsity, no Q tiles contribute to this KV tile
            if const_expr(not self.dKV_postprocess):
                should_zero_dKV = False
                if const_expr(self.is_local or self.is_varlen_q):
                    should_zero_dKV = m_block_min >= m_block_max
                if const_expr(self.use_block_sparsity):
                    # For block sparsity, zero when no m_blocks contribute to this n_block
                    if not process_tile:
                        should_zero_dKV = True

                if should_zero_dKV:
                    # For 2-CTA: use cluster-wide tile size (cta_group_size * tile_n)
                    cluster_tile_n = self.tile_n * self.cta_group_size
                    n_block_for_tile = n_block // self.cta_group_size
                    gmem_tiled_copy_zero_dK = copy_utils.tiled_copy_2d(
                        self.dk_dtype,
                        math.gcd(64, self.tile_hdim),
                        128,  # num_threads
                    )
                    gmem_tiled_copy_zero_dV = copy_utils.tiled_copy_2d(
                        self.dv_dtype,
                        math.gcd(64, self.tile_hdimv),
                        128,  # num_threads
                    )
                    gmem_thr_copy_zero_dK = gmem_tiled_copy_zero_dK.get_slice(dp_idx)
                    gmem_thr_copy_zero_dV = gmem_tiled_copy_zero_dV.get_slice(dp_idx)
                    mdV_cur = seqlen.offset_batch_K(mdV, batch_idx, dim=3)[None, None, head_idx]
                    mdK_cur = seqlen.offset_batch_K(mdK, batch_idx, dim=3)[None, None, head_idx]
                    gdK = cute.local_tile(mdK_cur, (cluster_tile_n, self.tile_hdim), (n_block_for_tile, 0))
                    gdV = cute.local_tile(mdV_cur, (cluster_tile_n, self.tile_hdimv), (n_block_for_tile, 0))
                    tdKgdK = gmem_thr_copy_zero_dK.partition_D(gdK)
                    tdVgdV = gmem_thr_copy_zero_dV.partition_D(gdV)
                    cdK = cute.make_identity_tensor((cluster_tile_n, self.tile_hdim))
                    cdV = cute.make_identity_tensor((cluster_tile_n, self.tile_hdimv))
                    tdKcdK = gmem_thr_copy_zero_dK.partition_D(cdK)
                    tdVcdV = gmem_thr_copy_zero_dV.partition_D(cdV)
                    assert cute.size(tdKgdK[None, 0, 0]) == cute.size(tdVgdV[None, 0, 0])
                    zero = cute.make_fragment_like(tdKgdK[None, 0, 0])
                    zero.fill(0.0)
                    if tidx < 128:
                        for i in cutlass.range_constexpr(tdKgdK.shape[1]):
                            row_idx = tdKcdK[0, i, 0][0]
                            if row_idx < seqlen.seqlen_k - cluster_tile_n * n_block_for_tile:
                                for j in cutlass.range_constexpr(tdKgdK.shape[2]):
                                    cute.copy(gmem_tiled_copy_zero_dK, zero, tdKgdK[None, i, j])
                    else:
                        for i in cutlass.range_constexpr(tdVgdV.shape[1]):
                            row_idx = tdVcdV[0, i, 0][0]
                            if row_idx < seqlen.seqlen_k - cluster_tile_n * n_block_for_tile:
                                for j in cutlass.range_constexpr(tdVgdV.shape[2]):
                                    cute.copy(gmem_tiled_copy_zero_dV, zero, tdVgdV[None, i, j])

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta_handoff = _RoleBarrier(barrier_id=6, num_threads=288)
        self.lse_handoff = _RoleBarrier(barrier_id=7, num_threads=288)
