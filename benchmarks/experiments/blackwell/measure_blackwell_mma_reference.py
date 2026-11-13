"""Measure a resident-operand reference for D512 backward matrix throughput.

This is a diagnostic, not attention. It retains the production grid, allocation
footprint and SS/TS product counts, while removing TMA and pointwise work. Shared
initialization and a small output read remain timed. Reused zero operands and a
different dependency schedule mean this is not a hard lower bound.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import torch
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05, warpgroup
from cutlass.cute.runtime import from_dlpack
from cutlass.memory import SmemAllocator, TmemAllocator

from kohakufa_cute.blackwell_pipeline import Backward


class Resident(Backward):
    def __init__(self, commits):
        super().__init__("q", False, False, columns=128, width=128, warps=12, tma=True)
        self.commits = commits

    @cute.jit
    def __call__(self, output, stream):
        self.kernel(output).launch(grid=(128, 16, 1), block=(384, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, output):
        tid, _, _ = cute.arch.thread_idx()
        tile, bh, _ = cute.arch.block_idx()
        allocator = SmemAllocator()
        bars = allocator.allocate_array(cutlass.Int64, 20)
        # Reserve production's ring-barrier footprint, even though operands
        # are resident and no transfer barriers are needed by this diagnostic.
        allocator.allocate_array(cutlass.Int64, 6)
        allocator.allocate_array(cutlass.Int64, 6)
        holding = allocator.allocate_array(cutlass.Int32, 1)
        atom = warpgroup.make_smem_layout_atom(warpgroup.SmemLayoutAtomKind.K_SW128, output.element_type)
        layout = cute.tile_to_shape(atom, (128, 64, 6), (0, 1, 2))
        a = self.shared(allocator, output.element_type, layout)
        b = self.shared(allocator, output.element_type, layout)
        allocator.allocate_tensor(cutlass.Float32, cute.make_layout((4, 128, 2)), byte_alignment=128)
        tmem = TmemAllocator(
            alloc_result_dst_smem_ptr=holding, barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=384)
        )
        tmem.allocate(512)
        tmem.wait_for_alloc()
        pointer = tmem.retrieve_ptr(cutlass.Float32)
        tmem.relinquish_alloc_permit()
        if tid == 0:
            cute.arch.mbarrier_init(bars, 1)
        for index in cutlass.range(cute.ceil_div(128 * 64 * 6, 384), unroll=1):
            linear = index * 384 + tid
            if linear < 128 * 64 * 6:
                row = linear % 128
                col = (linear // 128) % 64
                slot = linear // (128 * 64)
                a[row, col, slot] = cutlass.BFloat16(0)
                b[row, col, slot] = cutlass.BFloat16(0)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        cute.arch.fence_view_async_shared()
        mma = cute.make_tiled_mma(
            tcgen05.MmaF16BF16Op(
                output.element_type,
                cutlass.Float32,
                (128, 128, 16),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.SMEM,
                OperandMajorMode.K,
                OperandMajorMode.K,
            )
        )
        frag = mma.make_fragment_C(mma.get_slice(0).partition_shape_C((128, 128)))
        score = cute.make_tensor(pointer, frag.layout)
        dp = cute.make_tensor(pointer + 128, frag.layout)
        if tid >= 256 and tid < 288:
            phase = cutlass.Int32(0)
            for step in cutlass.range(32, unroll=1):
                for c in cutlass.range(8, unroll_full=True):
                    self.issue(mma, a[None, None, c % 6], b[None, None, c % 6], score, c > 0)
                    if cutlass.const_expr(self.commits):
                        with cute.arch.elect_one():
                            tcgen05.commit(bars)
                        cute.arch.mbarrier_wait(bars, phase)
                        phase = phase ^ 1
                for c in cutlass.range(8, unroll_full=True):
                    self.issue(mma, a[None, None, c % 6], b[None, None, c % 6], dp, c > 0)
                    if cutlass.const_expr(self.commits):
                        with cute.arch.elect_one():
                            tcgen05.commit(bars)
                        cute.arch.mbarrier_wait(bars, phase)
                        phase = phase ^ 1
                for c in cutlass.range(2, unroll_full=True):
                    self.issue_pv(output.element_type, pointer, b[None, None, c], pointer + 256 + c * 64, step > 0)
                    if cutlass.const_expr(self.commits):
                        with cute.arch.elect_one():
                            tcgen05.commit(bars)
                        cute.arch.mbarrier_wait(bars, phase)
                        phase = phase ^ 1
                for c in cutlass.range(2, unroll_full=True):
                    self.issue_pv(output.element_type, pointer + 128, b[None, None, c], pointer + 384 + c * 64, step > 0)
                    if cutlass.const_expr(self.commits):
                        with cute.arch.elect_one():
                            tcgen05.commit(bars)
                        cute.arch.mbarrier_wait(bars, phase)
                        phase = phase ^ 1
            with cute.arch.elect_one():
                tcgen05.commit(bars)
            cute.arch.mbarrier_wait(bars, phase)
        cute.arch.sync_threads()
        if tid < 128:
            value = self.read_matrix(pointer + 256, 32, tid)
            output[bh, tile, tid] = value[0].to(output.element_type)
        cute.arch.sync_threads()
        tmem.free(pointer)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        p.error("Use Blackwell sm100 or sm103.")
    output = torch.empty((16, 128, 128), device="cuda", dtype=torch.bfloat16)
    args = [from_dlpack(output, assumed_align=16)]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    rows = []
    for commits in (False, True):
        print("compiling", commits, flush=True)
        fn = cute.compile(Resident(commits), *args, stream)
        print("compiled", flush=True)

        def call():
            fn(*args, cuda.CUstream(torch.cuda.current_stream().cuda_stream))

        call()
        torch.cuda.synchronize()
        assert bool((output == 0).all())
        for _ in range(2):
            call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        graph.replay()
        torch.cuda.synchronize()
        assert bool((output == 0).all())
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        samples = []
        for _ in range(5):
            ev[0].record()
            for _ in range(20):
                graph.replay()
            ev[1].record()
            torch.cuda.synchronize()
            samples.append(ev[0].elapsed_time(ev[1]) / 20)
        row = {
            "serialized_each_operand_chunk": commits,
            "ms": samples,
            "median_ms": statistics.median(samples),
            "product_flops": 10 * 2 * 16 * 4096**2 * 512,
            "zero_output_checked": True,
        }
        print(json.dumps(row), flush=True)
        rows.append(row)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    properties = torch.cuda.get_device_properties(0)
    a.output.write_text(
        json.dumps(
            {
                "device": properties.name,
                "multiprocessors": properties.multi_processor_count,
                "capability": list(torch.cuda.get_device_capability()),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "scope": "Diagnostic only: 2048 CTAs, 384 threads, 512 TMEM columns, 201728 shared bytes, same 32-step 8+8+2+2 SS/TS MMA counts as one production backward pass. All operands are resident zeroes; no TMA, pointwise, or attention correctness claim. Serialized variant commits/waits every 64-wide score or 128-wide product chunk.",
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
