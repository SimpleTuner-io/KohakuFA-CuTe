# Explicit warp-MMA schedule controls

The shared backend keeps `Schedule()` stable. Device-specific experiments select
an explicit schedule; adding a control does not change automatic dispatch or
broaden support to another architecture. The dependency and arithmetic contracts
remain those in [the A100 backend notes](a100-sm80-fa4.md) and
[precision.md](precision.md).

`Schedule.v_in_regs` defaults to `False`. When enabled, the backward prologue
copies V from shared memory into its existing register fragment before Q reuses
the same shared allocation. The dP GEMM must consume that register operand through
`B_in_regs=True`; reloading its shared-memory view after Q arrives reads the wrong
operand. The aliased storage class contains a single allocation sized for both
Q and V, rather than reserving a separate unused V allocation. The actual device
shared-memory capacity check remains in place.

This control preserves separate raw maxima and logarithmic normalization, the
input-dtype probability operand reused by dS and dV, actual rounded O in delta,
and late dQ/dK scaling. It changes operand ownership and resource requirements.
Higher register pressure can outweigh the shared-memory and reload reduction.

The explicit A100 controls can be reproduced with:

```sh
python -m benchmarks.sm80_schedule_candidates \
  --preset d64-do1-kv4dq4-resident --output directed.json
python -m benchmarks.validate_sm80_residency \
  --preset d64-do1-kv4dq4-resident --output changed-graphs.json
```

The first command runs the established 74 directed, statistic, and graph cases.
The second exercises D64 with both input dtypes, dense and causal visibility,
lengths 129/256/257, changed operands, and poisoned output/gradient storage.
Its `--smoke` option retains both dtypes and visibility modes at length 129 for
memory, race, synchronization, and initialization checks.

Qualification on A100 does not establish another device's eligibility. Consumers
must requalify the exact selected tile, warp layouts, stage counts, storage, and
actual compilation target. In particular, backward tile changes require matching
dQ scratch allocation, preprocessing, and postprocessing layouts in every host
entry, including the experimental native launch bridge.

The A100 D64 M64N128, Q2/dO1, MMA 2/4/4 resident control passed all 74
cases, all 24 additional graph cases, and all four sanitizer checks with zero
device errors or race warnings. The exact source hashes and check scope are in
[the compact qualification record](../results/a100-sm80-resident-control.json).
This validates the shared ownership control; it does not select that schedule
automatically or establish it as the fastest A100 schedule.
