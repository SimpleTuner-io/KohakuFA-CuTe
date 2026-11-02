# Determinism

Enable `torch.use_deterministic_algorithms(True)` before forward through the
public attention API to avoid optional atomic-backward schedules. The supported
native fallback retains the project's existing rounding and accuracy contract.
Determinism does not require different attention backends or revisions to return
bitwise-identical values.

Shared warp and experimental Hopper backward kernels accumulate dQ with
unordered FP32 atomics. Their native and Python backward entry points check the
live PyTorch policy on every invocation, including warm caches and a policy
enabled after forward. Strict mode raises before launching atomic backward;
`warn_only=True` warns and permits execution. Forward-only execution and
owned-gradient kernels are exempt from this restriction.

CUDA graph replay executes the recorded kernels without re-entering these host
checks. Enable the intended policy before warmup and capture, and recapture when
changing it. An existing atomic graph cannot become deterministic by changing a
host flag after capture.

## Validation

Run `python -m unittest tests.test_determinism -v`. GPU methods require a
qualified optional runtime and retain independent per-head output/gradient norm
limits of 1% BF16 and 0.5% FP16. Different arithmetic revisions can return
different stable values even when each revision repeats bitwise.
