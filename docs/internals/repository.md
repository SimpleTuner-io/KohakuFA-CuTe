# Repository layout

| Location | Responsibility |
| --- | --- |
| `src/kohakufa_cute/` | Installed APIs, dispatch, kernels and native launcher |
| `tests/` | Unittest regressions; common fixtures keep their stable module names |
| `benchmarks/` | Shared benchmark entry points and references |
| `benchmarks/neighborhood/` | Neighborhood comparison, qualification and accuracy tooling |
| `benchmarks/experiments/` | Retained controls grouped by GPU family, with negative-result evidence |
| `benchmarks/validation/`, `benchmarks/profiling/`, `benchmarks/reporting/` | Specialized diagnostic and presentation tools |
| `docs/` | Usage, contracts, qualification guidance and implementation notes |
| `docs/neighborhood/` | Current neighborhood guide and scoped study reports |
| `results/` | Ordinary-attention per-GPU data and figure |
| `results/neighborhood/` | Neighborhood study data and figure |
| `tools/`, `.githooks/` | Repository and publication checks |
| `controls/` | Small retained historical source controls |

Keep the main README and documentation index focused on entry points. Explain
current behavior in a guide, link the relevant evidence, and label older
reports by the method and campaign they measured. Put generated tables inside
their report rather than creating an additional navigation document. Raw
operands, compiler outputs, traces, environments and lifecycle records belong
in persistent storage outside the tracked tree.

The neighborhood effort continues on one feature branch and checkout. A new
benchmark or accuracy phase within that effort does not need another worktree.
Independent tasks still follow the repository's worktree instructions; do not
discard unfinished or untracked work when cleaning up.

The organization change relocates benchmark imports and commands; it does not
rename the installed package APIs or change kernel arithmetic. Historical
result metadata keeps the paths and source hashes of its own snapshot.
Exact historical reruns use that snapshot; current commands use the
[benchmark index](../../benchmarks/README.md).

After moving tools or documentation, run `python -m tools.check_repository`.
It checks Python syntax, local imports, literal source paths and local Markdown
link targets without importing Torch or running numerical calculations. It
does not replace runtime tests or GPU qualification.
