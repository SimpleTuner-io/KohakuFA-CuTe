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

## Measured scope

An RTX 4090 audit of FP16/BF16 on the preceding revision observed dQ variation
in selected shared schedules and in default SDPA's bundled FlashAttention
kernels. Output, dK and dV repeated bitwise in those controls. Both public APIs
repeated bitwise with deterministic mode enabled before forward. The explicit
tuned native runner previously ignored strict and warning-only policies; this
change enforces them at backward execution.

Synthetic MOSS-shaped masked cases with 24 heads, D128 and 129/4353 tokens
repeated bitwise in current and locally vendored kernels. Their forward and dV
hashes matched across revisions while dQ/dK differed consistently. These are
separate arithmetic revisions, not evidence of run-to-run variation. This audit
does not reproduce a full-model loss or sampling discrepancy from
[issue #50](https://github.com/SimpleTuner-io/KohakuFA-CuTe/issues/50), which remains
open for a matching runtime/configuration and captured inputs.

Run the policy regression tests with:

```sh
python -m unittest tests.test_determinism -v
```

GPU methods require a qualified optional runtime. Their output and independent
per-head gradient checks retain the existing 1% BF16 and 0.5% FP16 norm limits.
