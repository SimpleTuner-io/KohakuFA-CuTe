# Neighborhood evidence

Start with the [current neighborhood guide](../../docs/neighborhood/README.md).
This directory holds recorded data and the original performance figure;
explanations live under `docs/neighborhood/reports/`.

| Data | Report and scope |
| --- | --- |
| [qualification-h100.json](qualification-h100.json) | [Initial correctness qualification](../../docs/neighborhood/reports/initial.md), including unresolved ordinary-attention regressions |
| [performance.json](performance.json), [performance-validation.json](performance-validation.json), [figure](performance.svg) | [Initial five-GPU baseline](../../docs/neighborhood/reports/performance.md); B300 was unavailable for this campaign |
| [tuning.json](tuning.json), [tuning-validation.json](tuning-validation.json) | [Bounded H100 schedule](../../docs/neighborhood/reports/tuning.md), selection and held-out evaluation |
| [accuracy.json](accuracy.json) | [Baseline mathematical and rounded accuracy](../../docs/neighborhood/reports/accuracy.md) |
| [dq-precision.json](dq-precision.json) | [Explicit dQ precision experiment](../../docs/neighborhood/reports/dq-precision.md) |
| [dk-precision.json](dk-precision.json) | [Explicit combined dQ+dK H100 experiment](../../docs/neighborhood/reports/dk-precision.md) |
| [multigpu-precision.json](multigpu-precision.json) | [Combined method on B200, B300, RTX 4090, RTX 5090 and L40S](../../docs/neighborhood/reports/multigpu-precision.md) |

Recorded JSON and SVG bytes are unchanged by the file reorganization. Source
paths, hashes, versions, acceptance criteria and timings describe the original
frozen campaign. They are not manifests of the current checkout. A new source
snapshot is required for new measurements. Full raw captures and operands are
retained outside the repository.
