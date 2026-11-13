# Repository layout

| Location | Responsibility |
| --- | --- |
| [`src/kohakufa_cute/`](source-map.md) | Installed APIs, dispatch, kernels and native launcher; follow the source map |
| [`tests/`](../../tests/README.md) | Unittest families and their CPU/GPU requirements |
| [`benchmarks/`](../../benchmarks/README.md) | Shared benchmark entry points and references |
| `benchmarks/neighborhood/` | Neighborhood comparison, qualification and accuracy tooling |
| `benchmarks/experiments/` | Retained controls grouped by GPU family, with negative-result evidence |
| `benchmarks/validation/`, `benchmarks/profiling/`, `benchmarks/reporting/` | Specialized diagnostic and presentation tools |
| `docs/` | Usage, contracts, qualification guidance and implementation notes |
| `docs/neighborhood/` | Current neighborhood guide and scoped study reports |
| `results/` | Ordinary-attention per-GPU data and figure |
| `results/neighborhood/` | Neighborhood study data and figure |
| `tools/`, `.githooks/` | Repository and publication checks |
| [`controls/`](../../controls/README.md) | Frozen historical source controls and their consumers |

Keep the main README and documentation index focused on entry points. Explain
current behavior in a guide, link the relevant evidence, and label older
reports by the method and campaign they measured. Put generated tables inside
their report rather than creating an additional navigation document. Raw
operands, compiler outputs, traces, environments and lifecycle records belong
in persistent storage outside the tracked tree.

Historical result metadata keeps the paths and source hashes of its own snapshot.
Exact historical reruns use that snapshot; current commands use the
[benchmark index](../../benchmarks/README.md).

Current guides own API behavior and numerical contracts. Study reports explain
what was measured at a particular revision and why it remains relevant. Retain
their measured policies and failures even when a method is later selected by
default. The [experiment catalog](../../benchmarks/experiments/README.md) connects
historical controls to the studies that consume them; an unimported CLI factory
or a duplicate frozen snapshot is not sufficient evidence that a file is unused.

For contribution and worktree procedures, see [repository instructions](../../AGENTS.md).

After moving tools or documentation, run `python -m tools.check_repository`.
It checks Python syntax, local imports, literal source paths and local Markdown
link targets without importing Torch or running numerical calculations. It
does not replace runtime tests or GPU qualification.
