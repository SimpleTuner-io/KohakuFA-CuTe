# A40 D128 shared attention

For the current public entry, see the [causal D64/D128 scheduling and host
overhead follow-up](a40-causal-grouped.md). The measurements below describe the
initial D128 selection before causal grouping and immutable-device caching.

The explicit A40 entry uses the shared warp-MMA forward and joint backward through
cached native C++ runners. It keeps the [standard arithmetic](precision.md): P and
dS each convert once to the input dtype, products and delta accumulate in FP32,
and attention scaling follows gradient accumulation.

## Use and limits

Install `.[cuda,sm80-fa4]` with Python 3.13 and Torch 2.15.0/CUDA 13.2, then:

```python
from kohakufa_cute.fa4_sm80.a40 import scaled_dot_product_attention

output = scaled_dot_product_attention(q, k, v, is_causal=False)
```

The measured selection is a full NVIDIA A40 with 84 SMs and a 101376-byte
per-block shared-memory limit; contiguous, aligned BHSD FP16/BF16 inputs; B1/H16;
D128; and square S1024 or S4096. Dense and causal calls use separate backward
schedules. Default scaling or finite scales from `2**-126` through `2e38` select
this path; custom-scale cache reuse is checked separately.

Other calls delegate to the ordinary strict CuTe API, preserving its validation
and argument handling. This includes D16/D64/D256/D512, other head counts and
lengths, masks, GQA, noncontiguous views, compiled calls, deterministic mode and
signed/zero scales. A delegated call can still raise when the strict API does not
support it. Selecting this entry does not modify the default APIs or patch SDPA.
First-use compilation is excluded from the measurements below.

## Scheduling change

Dense backward uses M32/N128, Q1/dO1, eight warps, a 2/2/2 MMA layout and V in
shared memory. Causal backward keeps M64/N64, Q1/dO1, 4/2/2 and resident V. Both
use M64/N64, one stage and shared Q in forward.

The dense layout trades shared storage for a wider key tile while keeping the
query fragment small. At S4096 the main grid has 32 key blocks per head, versus
64 in the loaded FA2 schedule. That reduces the number of CTAs independently
reading Q/dO and contributing dQ partials. It is a work-distribution change; the
number of attention products and the rounding policy stay the same. The actual
main kernel uses 99328 shared bytes, 255 registers and 8 local bytes per thread.
The measured FA2 main uses 81920 shared bytes, 255 registers and 16 local bytes.
These are resource measurements, not stall or memory-traffic percentages.

An independent native M128/N64-forward control passes the timing matrix but regresses
causal forward substantially and does not improve the complete dense call
consistently across dtypes. M64 remains selected. The wider dense backward is not
transferred to causal attention.

## Measurements

[Compact results](../results/a40-d128-shared.json) retain samples, source hashes,
qualification summaries, controls and exact loaded-code identities. All rows use
B1/H16/D128 on the same full 300 W A40 and qualified software stack. Standalone
FA2 2.8.4 uses its own arithmetic policy; this is a complete implementation
comparison, not an assertion that its arithmetic matches CuTe's.

Five rotating CUDA-graph rounds, with approximately 80 ms per measurement
window; forward plus all three gradients, milliseconds:

| Dtype | S | Visibility | Ordinary CuTe | A40 entry | FA2 |
| --- | ---: | --- | ---: | ---: | ---: |
| BF16 | 1024 | dense | 0.561 | 0.415 | 0.422 |
| BF16 | 4096 | dense | 6.936 | 5.023 | 5.070 |
| FP16 | 1024 | dense | 0.554 | 0.413 | 0.431 |
| FP16 | 4096 | dense | 6.904 | 4.994 | 5.115 |
| BF16 | 1024 | causal | 0.380 | 0.284 | 0.281 |
| BF16 | 4096 | causal | 3.725 | 2.767 | 2.760 |
| FP16 | 1024 | causal | 0.385 | 0.289 | 0.282 |
| FP16 | 4096 | causal | 3.845 | 2.842 | 2.827 |

Dense graph latency is 25–28% below ordinary CuTe in these four cases. Its small
lead over FA2 is specific to this matrix and timing boundary; causal is close to
FA2 rather than a consistent win. The result file also separates the native
runner from the Python shared-kernel preparation control. Similar captured GPU
time does not imply similar eager host overhead.

An additional same-process test measures the actual public wrapper, including
its compatibility predicate, separately from a cached direct runner. It uses
seven rotating rounds, a 300 ms warmup and windows of at least approximately
60 ms. BF16 complete eager forward/backward wall time, milliseconds:

| S | Visibility | Ordinary CuTe | Public entry | Direct runner | SDPA | FA2 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1024 | dense | 0.561 | 0.439 | 0.419 | 0.466 | 0.863 |
| 1024 | causal | 0.386 | 0.388 | 0.313 | 0.401 | 0.770 |
| 4096 | dense | 6.866 | 5.010 | 5.000 | 5.307 | 5.081 |
| 4096 | causal | 3.810 | 2.789 | 2.784 | 2.921 | 2.790 |

**S1024 causal eager calls do not improve over ordinary CuTe in this test.**
Their graph replay improves, but the public predicate and host preparation erase
that gain in complete eager execution. FP16 has the same short-causal limitation;
both dtypes and every timing sample are retained in the result file. FA2's eager
numbers include its Python/autograd preparation and must not be read as its
kernel latency. Choose this opt-in entry using the application's timing boundary.

## Qualification and code identity

All 53 directed cases pass: both dtypes, partial tiles, negative score offsets,
signed/zero-scale delegation, rounded key/value cancellation and FP16 underflow,
model head counts, saved statistics and changing/poisoned graph replays. The
public-entry GPU tests also exercise changing positive scales through cached
runners and native fallbacks for D16/D64/D256/D512.

Random cases retain independent per-head FP64 output/dQ/dK/dV norm limits of 1%
for BF16 and 0.5% for FP16. Directed cases check the declared rounded equations
using the actual rounded O. Saved maxima and shifted log2 normalizers are checked
independently. No precision policy or acceptance threshold changes here.

Compute Sanitizer memcheck, racecheck, synccheck and initcheck each pass the
focused dense/causal tail and changed-graph cases. Host API diagnostics are
disabled with `--report-api-errors no`; device memory and synchronization checks
remain enabled.

A separate CUPTI capture retains the actually loaded dense backward ELF and
matches the observed graph symbol to its exported SASS and PTX. The CuTe loaded
ELF hash also matches the compiler-returned cubin. FA2 is checked against its
mapped extension and packaged PTX. The observed binaries target SM86 for CuTe and
SM80 for the FA2 build, both running on the actual SM86 A40. Raw captures stay
outside the repository; the compact report retains their hashes and resources.

Reproduce from the qualified installation, writing captures outside the checkout:

```sh
python -m benchmarks.validate_a40_shared --source "$FA_SOURCE" \
  --dependencies "$FA_DEPENDENCIES" --output "$CAPTURE_DIR/directed.json"
python -m unittest tests.test_a40_shared -v
python -m benchmarks.compare_flashattention --backends cute fa2 sdpa \
  --seq 1024 4096 --dim 128 --dtype bf16 fp16 --modes dense causal \
  --repeats 5 --window-ms 80 \
  --candidate benchmarks.a40_shared_candidates:make_attention \
  --output "$CAPTURE_DIR/a40.json" --export-code "$CAPTURE_DIR/code"
```

`FA_SOURCE` is the pinned official FlashAttention checkout; `FA_DEPENDENCIES` is
its optional dependency prefix. Neither is needed by the public entry when the
qualified packages are installed normally. The comparison adapter accepts them
for isolated source-based environments.
