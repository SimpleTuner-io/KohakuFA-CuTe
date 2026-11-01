"""Flat host entry for the qualified SM80 GPU methods.

Compilation specializes metadata; the native launcher supplies fresh non-owning
tensor descriptors and the current stream on every invocation.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
from flash_attn.cute.flash_bwd_postprocess import FlashAttentionBackwardPostprocess
from flash_attn.cute.utils import AuxData

from ._attention import Schedule
from ._backward import FlashAttentionBackwardSm80
from ._forward import FlashAttentionForwardSm80
from ._preprocess import FlashAttentionBackwardPreprocess


@cute.jit
def _bshd(tensor):
    return cute.make_tensor(tensor.iterator, cute.select(tensor.layout, mode=[0, 2, 1, 3]))


class FlatForward:
    def __init__(self, dtype, dimension, causal, capacity, schedule):
        self.forward = FlashAttentionForwardSm80(
            dtype,
            dimension,
            dimension,
            is_causal=causal,
            pack_gqa=False,
            tile_m=schedule.forward_m,
            tile_n=schedule.forward_n,
            num_stages=schedule.forward_stages,
            num_threads=128,
            Q_in_regs=schedule.query_in_registers,
            has_aux_tensors=True,
        )
        self.forward.shared_capacity = capacity

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
        unused_partial: cute.Tensor,
        unused_split: cute.Tensor,
        unused_dq: cute.Tensor,
        unused_dk: cute.Tensor,
        unused_dv: cute.Tensor,
        unused_blocks: cute.Tensor,
        unused_base: cute.Tensor,
        scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        self.forward(
            _bshd(q),
            _bshd(k),
            _bshd(v),
            _bshd(out),
            logsum,
            scale,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            AuxData([maximum], None),
            None,
            None,
            stream,
        )


class FlatBackward:
    def __init__(self, dtype, dimension, causal, capacity, schedule):
        self.prepare = FlashAttentionBackwardPreprocess(dtype, dimension, dimension, schedule.backward_m)
        self.main = FlashAttentionBackwardSm80(
            dtype,
            dimension,
            dimension,
            1,
            schedule.backward_m,
            schedule.backward_n,
            schedule.query_stages,
            schedule.dout_stages,
            schedule.threads,
            False,
            causal,
            False,
            False,
            False,
            False,
            schedule.score_m_warps,
            schedule.kv_m_warps,
            schedule.dq_m_warps,
            V_in_regs=False,
        )
        self.main.shared_capacity = capacity
        self.convert = FlashAttentionBackwardPostprocess(
            dtype,
            dimension,
            80,
            schedule.backward_m,
            schedule.threads,
            schedule.dq_m_warps,
            False,
        )

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
        delta: cute.Tensor,
        dq: cute.Tensor,
        dk: cute.Tensor,
        dv: cute.Tensor,
        scratch: cute.Tensor,
        lognorm: cute.Tensor,
        scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        self.prepare(
            _bshd(out),
            _bshd(dout),
            delta,
            logsum,
            lognorm,
            scratch,
            None,
            None,
            None,
            None,
            None,
            cutlass.Float32(1.0),
            None,
            None,
            stream,
        )
        self.main(
            _bshd(q),
            _bshd(k),
            _bshd(v),
            _bshd(dout),
            lognorm,
            delta,
            scratch,
            _bshd(dk),
            _bshd(dv),
            scale,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            AuxData([maximum], None),
            None,
            None,
            stream,
        )
        self.convert(scratch, _bshd(dq), scale, None, None, None, None, stream)


def compile_kernel(mode, tensors, causal):
    """Export exactly fourteen tensor arguments and one runtime scale."""
    from . import available

    if not available():
        raise ValueError("The native SM80 launcher requires the qualified optional runtime.")
    q = tensors[0]
    if torch.cuda.get_device_capability(q.device) != (8, 0):
        raise ValueError("The native SM80 launcher requires an SM80 device.")
    dtypes = {torch.float16: cutlass.Float16, torch.bfloat16: cutlass.BFloat16, torch.float32: cutlass.Float32}
    # Preserve dynamic 32-bit dimensions and the actual stride divisibility.
    # Preparation needs aligned row strides for vector copies, including when
    # metadata is built inside this combined host entry. Cache misses alone
    # construct these descriptors; hot calls use native non-owning DLTensors.
    arguments = [
        make_fake_tensor(
            dtypes[tensor.dtype],
            tuple(cute.sym_int() for _ in tensor.shape),
            stride=tuple(
                (
                    stride
                    if stride <= 1
                    else cute.sym_int64(divisibility=min(16 // tensor.element_size(), stride & -stride))
                )
                for stride in tensor.stride()
            ),
            memspace=cute.AddressSpace.gmem,
            assumed_align=16,
        )
        for tensor in tensors
    ]
    schedule = Schedule()
    capacity = torch.cuda.get_device_properties(q.device).shared_memory_per_block_optin
    program = (FlatBackward if mode == "backward" else FlatForward)(
        dtypes[q.dtype],
        q.shape[-1],
        causal,
        capacity,
        schedule,
    )
    with torch.cuda.device(q.device):
        return cute.compile(
            program,
            *arguments,
            cutlass.Float32(1.0),
            make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
