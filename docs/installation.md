# Installation

## Standard CUDA installation

Install Python 3.12+, CUDA-enabled PyTorch 2.11+, and a C++17 compiler. From the
repository root:

```bash
python -m pip install '.[cuda]'
```

The CUDA extra installs CUTLASS DSL 4.8, TVM FFI 0.1.14+, Triton 3.6+, and Ninja
within the bounds in [pyproject.toml](../pyproject.toml). The native launcher builds
on first use; compilation is excluded from published warmed timings.

`pip install .` installs the API without CUDA extras. The automatic API can use
native SDPA on CPU/MPS and for unsupported CUDA calls. Strict CuTe APIs require
the CUDA runtime and propagate build or execution failures.

## Qualified tuned runtime

The optional shared warp-MMA schedules require the measured stack:

| Component | Qualified version |
| --- | --- |
| Python | 3.13 |
| PyTorch / CUDA | 2.15.0 / 13.2 |
| CUTLASS DSL | 4.8.1 |
| TVM FFI | 0.1.15.post0 |
| Quack | 0.6.5 |
| torch-c-dlpack-ext | 0.1.5 |
| FlashAttention CuTe source | `4616f5116d33e3ba3cbb71d582caffcfcb49649c` |

Install the matching PyTorch CUDA build, then:

```bash
python -m pip install '.[cuda,sm80-fa4]'
```

The exact dependency gate protects private compiler interfaces and measured
schedule qualification. These pins describe the qualified stack; they are not a
claim about the newest available framework release. An incompatible stack uses
ordinary CuTe dispatch. Failure after selecting an eligible tuned kernel propagates.

The `sm80-fa4` extra also supplies selected Ada, RTX 5090, and GB10 shared entries.
See the [GPU guides](README.md#gpu-guides) for automatic versus opt-in selection.
Standalone FA2/FA3 comparisons require their own installations and arithmetic policies.

## GB10 on Linux ARM64

Use an ARM64 CUDA-enabled PyTorch build and CUDA 13 DSL libraries:

```bash
python -m pip install '.[cuda]' 'nvidia-cutlass-dsl[cu13]>=4.8,<4.9'
export CUTE_DSL_ARCH=sm_121a
```

The generic GB10 stack was validated with Python 3.13, PyTorch 2.11/CUDA 13, and
CUTLASS DSL 4.8.1. Its opt-in shared schedules require the pinned runtime above.

## Neighborhood comparison runtime

The neighborhood campaigns use a separately pinned stack: Python 3.13.18,
PyTorch 2.13.0+cu130, CUTLASS DSL 4.8.1, TVM FFI 0.1.15.post0,
CUDA bindings 13.0.3, Triton 3.7.1 and NATTEN 0.21.7+torch2130cu130.
NATTEN is a separately installed comparator, not a dependency of the API.
The optional `sm80-fa4`/`hopper-fa4` extras above select another framework stack;
do not use them to reproduce the recorded neighborhood comparison.
See the [neighborhood guide](neighborhood/README.md#reproduce-and-inspect) for
scope and entry points. These versions identify the measured runtime, not the
newest available releases.

## Development

Use Python 3.13 with `pip install -e '.[dev,plots]'`. The project uses unittest,
Black 24.8, and isort 5.13. Follow [AGENTS.md](../AGENTS.md) for worktrees,
qualification, and publication checks.
