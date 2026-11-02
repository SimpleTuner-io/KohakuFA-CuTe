# Hopper and Blackwell

The standard attention APIs support SM90 (H100), SM100 (B200), and SM103 (B300)
under [the default precision contract](../precision.md). Generic support and
focused tuned qualification have different scope.

## H100 PCIe and B300

The [H100 PCIe report](../../results/h100.json) and [B300 report](../../results/b300.json)
retain 16 BF16 cases each: B1/H16, S128/S4096, D64/128/256/512, dense and causal.
They measure full devices: H100 PCIe has 114 SMs and an 80-GB allocation at 350 W;
B300 SXM6 AC has 148 SMs at 1100 W. The stack is PyTorch 2.15.0+cu132 and CuTe
DSL 4.8.1. Five rotated 100-ms graph windows exclude compilation and profiling.

These grids include both SDPA wins and losses. Smaller-head native calls often
favor SDPA; B300 D512 joint calls strongly favor CuTe in the measured grid.
Earlier H100 SXM and B300 MIG results do not substitute for these devices.

## B200 wide backward

The [B200 report](../../results/b200.json) covers BF16 D512 self-attention,
B1/H16/S512–S1024 or B1/H8/S1024, dense or causal. The selected backward owner
uses 256-column output slices and the existing aligned TMA pipeline. Runtime
length checks preserve fallback behavior across cache boundaries.

The full-device graph comparison includes preceding CuTe and actual SDPA
memory-efficient CUTLASS kernels. H16 complete-call latency falls 19–43% against
the preceding owner on its measured cases. H8/S512 remains a preceding-owner
control because forcing the wider kernel regressed causal execution.
The tested FA4 D512 configuration failed setup; it has no successful timing.

Initialization reports remain [unresolved](../diagnostics/blackwell-allocator.md).
The standalone allocator diagnostic does not waive B200 or B300 attention findings.

## Model and experimental limits

Captured H100 Clef attention improvements did not pass whole-model logit
compatibility. Those model results remain [experimental evidence](../../benchmarks/experiments/README.md),
not an end-to-end performance claim. The explicit [H200 backend](../experimental/h200.md)
uses a separate experimental path and is excluded from the supported-target chart.
