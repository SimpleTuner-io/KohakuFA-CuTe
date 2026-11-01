# Experimental SM80 host launcher

The shared SM80 GPU methods have an explicit native host entry for measuring
complete eager calls. C++ owns autograd, buffer allocation, current-device and
current-stream scopes, and non-owning tensor descriptors. A compiled flat host
program launches the same forward, preparation, joint main and dQ conversion
methods. Python compilation runs on cache misses rather than every call.

The entry is currently an experiment. The existing public selector still uses
the qualified Python wrapper; automatic short-call selection needs its own
complete-API comparison. The direct experiment requires the qualified optional
runtime documented in [the A100 schedule](a100-sm80-fa4.md), aligned contiguous
square FP16/BF16 MHA, D64/D128, and positive scales. It propagates unsupported
inputs and execution failures.

## What changed

Each cached function accepts fourteen tensor descriptors and a runtime scale.
The native cache distinguishes the optional family, device, dtype, visibility,
and exact runtime sizes and strides. It retains compiled functions rather than
operand pointers. Every call supplies current operands and newly owned output
or scratch buffers. Preparation initializes the padded dQ accumulator before
atomic updates, including graph replay.

Layout permutations are metadata operations inside the compiled host program.
The maximum remains unscaled, the normalization remains shifted log2, and delta
uses the actual rounded output. The launcher preserves the GPU methods' P/dS
conversions and FP32 accumulation. Expanded output gradients become contiguous;
an already-contiguous gradient with an unaligned storage offset is cloned into
aligned storage. The explicit entry rejects unaligned Q/K/V.

## Measurements

Measurements used the full 108-SM A100-SXM4-40GB and the pinned runtime from the
parent schedule. The eager matrix contains 48 rows: FP16/BF16, H16,
S128/1024/4096, D64/D128, dense/causal, and forward/combined phases. Seven
rotating backend rounds contain twenty complete calls per window, with
synchronization at the boundaries. Compilation and initial warming are separate.

Examples in microseconds, BF16:

| S | D | Visibility | Phase | Generic CuTe | Python shared wrapper | Native shared wrapper | SDPA |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| 128 | 64 | dense | forward | 11.39 | 55.92 | 9.81 | 11.71 |
| 128 | 64 | dense | forward + backward | 77.03 | 213.05 | 80.30 | 101.22 |
| 128 | 128 | dense | forward | 16.34 | 57.42 | 14.14 | 15.39 |
| 4096 | 128 | dense | forward + backward | 5069.99 | 2880.48 | 2893.03 | 2982.57 |

Reducing wrapper overhead does not make every short combined call faster than
generic CuTe. Long calls mostly retain the GPU path's existing performance.
The [compact results](../results/a100-native-host.json) retain every eager row
and its samples. These measurements qualify the explicit experiment, not a
new public dispatch policy or a model-level speedup.

### Metadata changes can change GPU performance

The first flat entry used fully static tensor metadata. A paired graph control
isolated metadata from Python dispatch: BF16 H16/S4096, seven rotating rounds
of one hundred replays.

| D | Python wrapper | Native static metadata | Native dynamic metadata |
| ---: | ---: | ---: | ---: |
| 64 | 1.684 ms | 1.786 ms | 1.682 ms |
| 128 | 2.939 ms | 3.066 ms | 2.934 ms |

The static specialization regressed by 4–6%. Dynamic 32-bit dimensions and
actual stride-divisibility constraints restored the previous GPU performance.
A dynamic variant without the preparation kernel's stride-alignment guarantees
failed compiler IR verification; that failed experiment is retained separately.

Actual graph CUfunction queries report 235 registers for the dynamic D64 main
and 246 for its static control, with equal shared memory and no local storage.
D128 reports 255 registers and 24 local bytes for both forms. Resource counts
alone therefore do not explain the regression. The control retains symbols and
resource metadata; its compiler-artifact requests returned unavailable values.
It does not establish independently captured loaded-image or ISA identity.

## Qualification and limitations

The final dynamic GPU methods passed independent per-head output/dQ/dK/dV
checks, literal saved-statistic checks, tails, negative common offsets, rounded
cancellation and underflow fixtures, runtime-scale/cache reuse, and changed
poisoned graphs. Final alignment-guard tests exercise expanded and unaligned
output gradients, rejected unaligned inputs, and selected graph symbols.

Memcheck, racecheck, synccheck and initcheck report zero device errors/hazards
on the tail/changed-graph/alignment smoke set with host API reporting disabled.
The original dynamic memcheck remains explicit: forty host API negotiation
diagnostics, comprising thirteen version-13030 probes and seven version-13041
probes against driver API 13020, plus twenty matching address-lookup failures.
Suppressing those reports does not suppress device checking. The earlier static
metadata run additionally reported unresolved internal-helper lookup diagnostics;
it is not used to qualify the final dynamic entry.

The final C++ alignment guard adds no GPU arithmetic. Its targeted tests and
memory tools were repeated after the change; the numerical methods and dynamic
metadata are unchanged from the full correctness and timing matrices.

## Reproduction

```sh
python -m unittest tests.test_sm80_native_host -v
python -m benchmarks.compare_sm80_host --output artifacts/a100-native-host.json
```

The benchmark calls the native generic control directly and compares the
explicit Python and native shared wrappers with default SDPA. It does not
substitute another backend when the experimental entry fails.
