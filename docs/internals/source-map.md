# Source and dispatch map

Start with the [public API](../api.md) and [precision contract](../precision.md).
This map explains where behavior lives; exact runtime predicates in the linked
source take precedence over a family name. A GPU capability alone does not imply
that a tuned specialization accepts every shape, layout or runtime.

## From a public call to a kernel

| Layer | Source | Responsibility |
| --- | --- | --- |
| Exports | [package initializer](../../src/kohakufa_cute/__init__.py) | Public attention APIs and the explicit RTX3090 API |
| API | [api.py](../../src/kohakufa_cute/api.py) | Input validation, optional key centering, strict versus automatic SDPA, eager specialization selection |
| Compiler/autograd | [_ops.py](../../src/kohakufa_cute/_ops.py) | Custom operators, fake implementations and saved backward operands for compiled calls |
| Native host | [native.py](../../src/kohakufa_cute/native.py), [native.cpp](../../src/kohakufa_cute/native.cpp) | Extension build identity, compiler callbacks, runner families, cache, native dispatch, autograd and FFI launches |
| Shared selection | [kernels.py](../../src/kohakufa_cute/kernels.py) | Shape/layout descriptors, device configurations, compilation cache and kernel construction |
| Neighborhood | [neighborhood_ops.py](../../src/kohakufa_cute/neighborhood_ops.py), [neighborhood_kernels.py](../../src/kohakufa_cute/neighborhood_kernels.py) | Shared 2D/3D spatial geometry/autograd path and corrected neighborhood arithmetic |

For eager ordinary calls, `api.py` tries the bounded RTX4090 S1024 path and
qualified `fa4_sm80` selector before native generic dispatch. Compiled calls use
`_ops.py`. Automatic SDPA supports a native PyTorch fallback for unsupported
calls; strict APIs validate their supported computation. Inspect both API and
native host predicates when changing selection: a Python API name alone does not
identify the launched kernel.

`native.cpp` calls `native.compile_kernel` with mode strings. That Python callback
lazily imports the corresponding family. The exported `rtx3090_attention`, for
example, selects configured `AttentionRunner` families that reach
[short_reciprocal.py](../../src/kohakufa_cute/short_reciprocal.py),
[short_key_stats_owned.py](../../src/kohakufa_cute/short_key_stats_owned.py) and
[short_key32_owned.py](../../src/kohakufa_cute/short_key32_owned.py). These are live
implementations even though few Python files import them directly.

## Families and status

| Family | Entry and implementation | Selection / reading guide |
| --- | --- | --- |
| Qualified warp-MMA paths | [fa4_sm80 selector](../../src/kohakufa_cute/fa4_sm80/__init__.py), [_schedules.py](../../src/kohakufa_cute/fa4_sm80/_schedules.py), [_host.py](../../src/kohakufa_cute/fa4_sm80/_host.py), [_native.py](../../src/kohakufa_cute/fa4_sm80/_native.py) | Automatically selected only inside measured predicates and the qualified optional runtime. The package name is historical; the selector also delegates supported device-specific families. |
| Warp implementations | [_forward.py](../../src/kohakufa_cute/fa4_sm80/_forward.py), [_backward.py](../../src/kohakufa_cute/fa4_sm80/_backward.py), [_preprocess.py](../../src/kohakufa_cute/fa4_sm80/_preprocess.py) | Kernel computation beneath the selector; [_private_api.py](../../src/kohakufa_cute/fa4_sm80/_private_api.py) checks the external backend interface. |
| Generic and specialized shared tiles | [kernels.py](../../src/kohakufa_cute/kernels.py), [sm86.py](../../src/kohakufa_cute/sm86.py), [l40s.py](../../src/kohakufa_cute/l40s.py), [sm120_schedule.py](../../src/kohakufa_cute/sm120_schedule.py) | Public generic route and bounded specializations; selection remains in the compiler and policy modules. |
| Hopper shared kernels | [hopper_forward.py](../../src/kohakufa_cute/hopper_forward.py) | Used through shared-kernel selection; distinct from the explicit H200 backend below. |
| Blackwell | [blackwell_pipeline.py](../../src/kohakufa_cute/blackwell_pipeline.py), [sm103.py](../../src/kohakufa_cute/sm103.py), [blackwell_register_aux.py](../../src/kohakufa_cute/blackwell_register_aux.py) | Production paths within their bounded selectors. [Backend guide](../backends/hopper-blackwell.md) distinguishes qualified devices and partitions. |
| RTX3090 opt-in | [rtx3090.py](../../src/kohakufa_cute/rtx3090.py), [short_owned.py](../../src/kohakufa_cute/short_owned.py), [short_key_quad_skip_owned.py](../../src/kohakufa_cute/short_key_quad_skip_owned.py) | Explicit exported API; see the [RTX3090 guide](../backends/rtx3090.md), including eligibility and fallbacks. |
| H200 experimental | [fa4_hopper](../../src/kohakufa_cute/fa4_hopper/__init__.py), [_persistent.py](../../src/kohakufa_cute/fa4_hopper/_persistent.py) | Explicit experiment, not ordinary automatic selection; [known limits](../experimental/h200.md). |
| Neighborhood default | [neighborhood_kernels.py](../../src/kohakufa_cute/neighborhood_kernels.py) | Normalized dQ and reconstructed dK are default on every supported GPU. [Guide](../neighborhood/README.md). |
| Historical compensated ordinary precision | [precision.py](../../src/kohakufa_cute/precision.py), [weight_precision.py](../../src/kohakufa_cute/weight_precision.py), [medium_backward.py](../../src/kohakufa_cute/medium_backward.py), [wide_backward.py](../../src/kohakufa_cute/wide_backward.py) | Retained explicit stronger-accuracy controls, not ordinary default policy. [Experiment owner](../../benchmarks/experimental_precision_control.py). |

## Navigating the large modules

`kernels.py` contains several layers in source order: layout and arithmetic
primitives; `Attention` setup/load/store/math helpers and its main kernel;
`ForwardLoadSpecialized`; descriptors and architecture configuration tables;
`kernel_configuration` / `backward_query_configuration`; compilation and schedule
selection; `PackMask` / `OutputDelta`; and `JointBackward` / `JointForward`.
Search these symbols before reading the whole file. Its subclasses share storage
and synchronization assumptions, so a file split requires more than moving text.

`blackwell_pipeline.py` groups `Forward`, `Backward` and the `Query`/`KeyValue`
mode subclasses. The `fa4_sm80` and `fa4_hopper` directories already separate
forward, backward, preprocessing, native integration and API checks. Preserve
those ownership boundaries when navigating or proposing a refactor.

## Evidence and validation

The [experiment catalogs](../../benchmarks/experiments/README.md) identify CLI
runners, factories and historical controls. The [snapshot guide](../../controls/README.md)
explains package-context loaders and intentional duplicate source. The
[test map](../../tests/README.md) separates lightweight inspection from numerical
and GPU qualification. See the [GPU catalog](../../results/README.md) for measured
scope; the presence of a source module does not qualify an unmeasured workload.
