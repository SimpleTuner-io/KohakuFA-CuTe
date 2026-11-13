"""Explicit unqualified ordinary-store control for TMA initialization tracking.

The caller has already applied the attention scale to FP32 dK. Each accumulator
is converted once to its output dtype. This diagnostic directly stores its
owned C coordinates and does not reuse any shared-memory output staging.
"""

from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr
from cutlass.cute import FastDivmodDivisorV2
from flash_attn.cute.seqlen_info import SeqlenInfoQK


class OrdinaryGradientStores:
    ordinary_gradient_stores = True

    @cute.jit
    def epilogue_dKV(
        self,
        acc_dV: cute.Tensor,
        mdV: cute.Tensor,
        sV: cute.Tensor,
        acc_dK: cute.Tensor,
        mdK: cute.Tensor,
        sK: cute.Tensor,
        seqlen: SeqlenInfoQK,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tidx: Int32,
        n_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
        qhead_per_kvhead_divmod: Optional[FastDivmodDivisorV2] = None,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV_semaphore: Optional[cute.Tensor] = None,
    ):
        assert const_expr(self.qhead_per_kvhead == 1 and not self.dKV_swapAB)
        mdK_cur = seqlen.offset_batch_K(mdK, batch_idx, dim=3, ragged=self.varlen_k)[None, None, head_idx]
        mdV_cur = seqlen.offset_batch_K(mdV, batch_idx, dim=3, ragged=self.varlen_k)[None, None, head_idx]
        gdK = cute.local_tile(mdK_cur, (self.tile_n, self.tile_hdim), (n_block, 0))
        gdV = cute.local_tile(mdV_cur, (self.tile_n, self.tile_hdimv), (n_block, 0))
        thr_K = tiled_mma_dK.get_slice(tidx)
        thr_V = tiled_mma_dV.get_slice(tidx)
        target_K = thr_K.partition_C(gdK)
        target_V = thr_V.partition_C(gdV)
        coord_K = thr_K.partition_C(cute.make_identity_tensor((self.tile_n, self.tile_hdim)))
        coord_V = thr_V.partition_C(cute.make_identity_tensor((self.tile_n, self.tile_hdimv)))
        target_K = cute.make_tensor(target_K.iterator, target_K.layout)
        target_V = cute.make_tensor(target_V.iterator, target_V.layout)
        # Materialize before the predicates. TensorSSA values can lazily cache
        # their first extraction inside a branch, which then cannot dominate
        # subsequent predicate regions on this compiler.
        value_K = cute.make_fragment_like(acc_dK, self.dtype)
        value_V = cute.make_fragment_like(acc_dV, self.dtype)
        value_K.store(acc_dK.load().to(self.dtype))
        value_V.store(acc_dV.load().to(self.dtype))
        assert const_expr(not self.check_hdim_oob and not self.check_hdim_v_oob)
        for i in cutlass.range_constexpr(cute.size(acc_dK)):
            if n_block * self.tile_n + coord_K[i][0] < seqlen.seqlen_k:
                target_K[i] = value_K[i]
        for i in cutlass.range_constexpr(cute.size(acc_dV)):
            if n_block * self.tile_n + coord_V[i][0] < seqlen.seqlen_k:
                target_V[i] = value_V[i]
