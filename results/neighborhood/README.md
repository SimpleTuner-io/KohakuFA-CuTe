# Neighborhood evidence

Start with the [current neighborhood guide](../../docs/neighborhood/README.md)
for usage and the [precision contract](../../docs/precision.md#neighborhood-default).
The combined normalized dQ and reconstructed dK method with column-staged
statistics is now the public default on every supported GPU. Its original
campaigns called it an explicit experiment; their `default_changed: false`
fields describe those campaigns, not current dispatch.

## Campaign catalog

Use the reports for readable findings and the data for original samples,
component errors, source hashes and acceptance details. “Former baseline” means
the neighborhood arithmetic before the corrected method became default.

| Campaign / devices | Measured policy and current role | Acceptance and retained limits | Report / data |
| --- | --- | --- | --- |
| Initial qualification / H100 | Former baseline; historical correctness evidence | Scoped neighborhood checks passed; ordinary-attention regression failures remain unresolved | [Report](../../docs/neighborhood/reports/initial.md) / [qualification](qualification-h100.json) |
| Initial performance / H100, B200, RTX 4090, RTX 5090, L40S | Former baseline versus FlexAttention and NATTEN; historical complete-implementation comparison | Per-component accuracy and campaign validation; B300 unavailable, no model speedup claim | [Report](../../docs/neighborhood/reports/performance.md) / [timings](performance.json), [validation](performance-validation.json), [original figure](performance.svg) |
| Schedule tuning / H100 | Former baseline arithmetic; bounded schedule retained by current default | Selection and held-out evaluation; qualified shapes only, preserved broader-suite failures | [Report](../../docs/neighborhood/reports/tuning.md) / [timings](tuning.json), [validation](tuning-validation.json) |
| Baseline accuracy / retained H100 operands and five-GPU campaign results | Former baseline; explains rounding and cancellation | Mathematical FP64 and rounded-equation references remain separate; stronger-method results must not inherit baseline failures | [Report](../../docs/neighborhood/reports/accuracy.md) / [diagnostics](accuracy.json) |
| dQ precision / H100 | Normalized dQ correction; intermediate experiment | Formula and component gates; mathematical limitations retained | [Report](../../docs/neighborhood/reports/dq-precision.md) / [evidence](dq-precision.json) |
| dQ+dK precision / H100 | Combined corrections and column-staged statistics; method now used by default | 0.5% FP16 / 1% BF16 component norm gates on historical fixtures; separate zero/constant-dP gates; held-out cost exceeds 5% aim | [Report](../../docs/neighborhood/reports/dk-precision.md) / [evidence](dk-precision.json) |
| Combined precision / B200, B300, RTX 4090, RTX 5090, L40S | Same combined method; cross-device qualification supporting current default | Independent component gates and complete-call costs; window-13 and some tails favor other implementations | [Report](../../docs/neighborhood/reports/multigpu-precision.md) / [evidence](multigpu-precision.json) |
| Default promotion / H100 | Public default versus explicit combined method; final organized source | 29 retained inputs byte-identical with compiled-kernel reuse; 16 GPU methods and four scoped sanitizers pass; no new multi-GPU timing campaign | [Current guide](../../docs/neighborhood/README.md#default-precision) / [compact receipt](default-promotion.json) |

## Reading and preserving the evidence

The original JSON and SVG describe frozen campaigns. Source paths and hashes
resolve against their recorded source snapshots, not necessarily today's file
layout. The default-promotion receipt separately identifies the tested snapshot
and its matching committed files. It records recovered checks, not a new run.

Large precision reports retain per-fixture, per-component and timing records;
use the linked reports before opening the complete JSON. In particular,
`multigpu-precision.json` groups records under `nodes`, and `dk-precision.json`
separates directed checks, timing, held-out cases and negative sweeps. Historical
failures and unsuccessful controls remain evidence; they are not current timing
cells or reasons to discard the campaign.

New measurements need a new source snapshot and explicit arithmetic, hardware,
timing boundary and acceptance criteria. Full raw captures and operands remain
outside the repository. Archive hashes identify retained evidence without
publishing its storage location.
