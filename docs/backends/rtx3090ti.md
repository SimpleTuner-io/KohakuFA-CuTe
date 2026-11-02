# RTX 3090 Ti

The standard attention APIs accept SM86 FP16/BF16 calls and select device-specific
schedules on a full RTX 3090 Ti. This target is separate from the ordinary
[RTX 3090 opt-in](rtx3090.md). Generic support does not qualify other SM86 cards
for the Ti's tuning.

## Evidence

[The RTX 3090 Ti report](../../results/rtx3090ti.json) is the retained 60-case
default-arithmetic grid: B1/H16, S128/S1024/S4096, D16/64/128/256/512,
FP16/BF16, dense and causal. It records a full 84-SM device at 450 W,
PyTorch 2.15.0+cu132, and CUTLASS DSL 4.8.1.

CUDA events measure warmed graph replay for forward, backward, and joint calls.
Compilation and profiling are excluded. Each case retains output/gradient
accuracy, samples, actual selected SDPA kernel symbols, and launch resources.
The previous-policy backend is an arithmetic comparison, rather than an isolated
scheduling control.

D512 long joint calls favor CuTe in this grid; several long smaller-head calls
favor SDPA. See [the common chart](../../results/performance.svg) for all measured
cells and [benchmarking](../benchmarking.md) to evaluate an application workload.
