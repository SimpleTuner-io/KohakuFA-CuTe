# Neighborhood accuracy against FP64

The retained ordinary-workload results do **not** demonstrate a general
mathematical-accuracy gain over FlexAttention or NATTEN. On the 50 matched
device/workload cases, the candidate's worst-head sampled dQ error was
1.056–1.474 times FlexAttention's, and dK error was 1.138–1.685 times
FlexAttention's. Both gradient errors were larger in every matched case.
Output and dV errors were close. NATTEN comparisons varied by case.

These compare each implementation's worst head within the same workload;
the identity of that head can differ. They are descriptive norm-error
comparisons, without statistical or model-quality conclusions.

The newer native-backend probes below show a defined large-offset benefit
for output/dK/dV, while dQ remains worse than FlexAttention and can have
very large mathematical error. There is **no overall accuracy winner**.

## What the reference means

`benchmarks/reference.py` and the sparse equations in
`benchmarks/benchmark_neighborhood.py` use real `torch.float64` arithmetic.
The performance matrix used CPU FP64 references. The initial small H100
qualification also required the reference gates with CUDA FP64. No
double-from-two-floats emulation was used. FP64 remains finite precision
and is not exact real arithmetic or an arbitrary-precision proof.

The operands are the identical **already-rounded FP16/BF16** Q/K/V/dO
values used by each implementation on that GPU. Their conversion to FP64
does not recover pre-quantization values, so these measurements isolate
operator arithmetic from initial input-quantization error.

The metric is `norm(actual - reference) / norm(reference)` per batch/head,
over sampled boundary/interior token positions and all 64 channels. dK/dV
references include every query contributing to each sampled key. This is
not a universal per-coordinate error bound or an exhaustive large-image
comparison. Separate small directed fixtures check full tensors.

`rounded_reference` answers a different question: agreement with the
declared P/dS operand conversions, FP32 QK/dP, and delta from actual rounded
O. Its final products use FP64 to isolate operand-rounding effects. Close
agreement with that reference does not imply the closest mathematical
FP64 result. The candidate retains the [declared precision contract](../precision.md).

## Representative ordinary workload

H100 SXM, B1/H8/D64, 64 x 64 image, 7 x 7 window, BF16, scale 0.125.
Values are worst-head sampled relative L2 errors **in percent against
mathematical FP64**:

| implementation | output | dQ | dK | dV |
| --- | ---: | ---: | ---: | ---: |
| Candidate | 0.2203% | 0.3502% | 0.3203% | 0.2538% |
| FlexAttention | 0.2209% | 0.2929% | 0.2507% | 0.2538% |
| NATTEN | 0.2201% | 0.2893% | 0.2547% | 0.2538% |

NATTEN used its native Hopper automatic/default preset in this matrix.
These complete implementations retain their own arithmetic policies.
Different operand-rounding policies are consistent with these differences;
the results alone do not identify every contributing error source.

## Maximum observed random-workload errors

Maximum worst-head sampled relative L2 error, in percent, across 5 GPUs
and the retained geometries per dtype. A different case may produce each
maximum; these maxima do not establish a paired accuracy winner.

| dtype | implementation | output | dQ | dK | dV |
| --- | --- | ---: | ---: | ---: | ---: |
| fp16 | Candidate | 0.0285% | 0.0519% | 0.0493% | 0.0343% |
| fp16 | FlexAttention | 0.0287% | 0.0385% | 0.0344% | 0.0343% |
| fp16 | NATTEN | 0.0284% | 0.0515% | 0.0463% | 0.0343% |
| bf16 | Candidate | 0.2210% | 0.3691% | 0.3566% | 0.2975% |
| bf16 | FlexAttention | 0.2210% | 0.2956% | 0.2989% | 0.2975% |
| bf16 | NATTEN | 0.2281% | 0.3716% | 0.3931% | 0.2975% |

## Directed common-offset fixture

The earlier H100 PCIe test used a 5 x 9 image, 3 x 3 window, scale 0.125,
and a large shared score offset. Its NATTEN comparator was explicitly
`cutlass-fna`. FlexAttention was not measured on this fixture, nor were
the later native Hopper/Blackwell NATTEN presets. Full-tensor worst-head
dQ relative errors were:

| dtype | candidate / FP64 | NATTEN / FP64 | relative error reduction | candidate / rounded | NATTEN / rounded |
| --- | ---: | ---: | ---: | ---: | ---: |
| float16 | 1.6092% | 1.7979% | 10.50% | 0.0280% | 1.6494% |
| bfloat16 | 12.1567% | 12.6729% | 4.07% | 0.1744% | 7.2507% |

The mathematical FP64 improvement on this fixture is about 10.5% lower
dQ error in FP16 and 4.1% lower in BF16. The much larger improvement
against the rounded reference must not be described as an equivalent
FP64-accuracy gain. Its BF16 mathematical dQ error is still about 12%,
showing a substantial rounding floor on this cancellation-sensitive case.

The demonstrated benefit is robust probability reconstruction and closer
agreement with the declared rounded computation on defined difficult
inputs. There is no demonstrated universal accuracy gain over FlexAttention,
and the initial qualification did not measure FlexAttention on this fixture.
The separate current native-backend probe below fills that comparison.

The [machine-readable audit](neighborhood-accuracy.json) retains source-report
hashes, all component maxima and their locations, matched error-ratio ranges,
and both reference types. The original
[performance summary](neighborhood-performance.json) and
[initial H100 qualification](neighborhood-h100.json) retain the source measurements.

Regenerate from the repository root:
Set `ACCURACY_REPORT_n` and `ACCURACY_RECEIPT_n` to each retained
off-tree raw report and terminal receipt first.
Raw reports and operands remain outside the repository.

```sh
python3 -m benchmarks.summarize_neighborhood_accuracy \
    --performance docs/experimental/neighborhood-performance.json \
    --qualification docs/experimental/neighborhood-h100.json \
    --probe H100-initial-C32 "${ACCURACY_REPORT_1}" \
    --probe H100-v2-exactmixed "${ACCURACY_REPORT_2}" \
    --probe H100-v4-production-candidate "${ACCURACY_REPORT_3}" \
    --probe H100-v6-production "${ACCURACY_REPORT_4}" \
    --probe-receipt H100-initial-C32 "${ACCURACY_RECEIPT_1}" \
    --probe-receipt H100-v2-exactmixed "${ACCURACY_RECEIPT_2}" \
    --probe-receipt H100-v4-production-candidate "${ACCURACY_RECEIPT_3}" \
    --probe-receipt H100-v6-production "${ACCURACY_RECEIPT_4}" \
    --output docs/experimental/neighborhood-accuracy.json \
    --markdown docs/experimental/neighborhood-accuracy.md
```

## Current native-backend probe: H100-initial-C32

This separate full-tensor diagnostic uses native CPU FP64 equations on
identical already-rounded operands. It measures no latency or model quality.
Device: NVIDIA H100 80GB HBM3, 132 SMs. Qualified stack: PyTorch 2.13.0+cu130, NATTEN 0.21.7+torch2130cu130.
The retained PyTorch 2.13 stack matches the newest NATTEN prebuilt pairing
and prior baseline; these results do not evaluate PyTorch 2.15 FlexAttention.
See the [NATTEN installation matrix](https://natten.org/install/) and
[current FlexAttention documentation](https://docs.pytorch.org/docs/2.15/nn.attention.flex_attention.html).
Candidate variant: `no_stat_fills`, columns `[32, 32, 32]`, materialize output gradients `False`.
Exact contiguous traversal flags: `False`.
Probe status: `execution-error`; candidate failed rows: 0; execution errors: 3; comparator rounded-policy differences: 12.
Retained stage exit status: `1`.

11/11 candidate rows passed their declared gates. Three NATTEN rows
could not execute because its public API rejects a (1, 3) window; these
two uniform-boundary cases and one underflow case remain explicit errors.
Negative-scale rows are compatibility probes; NATTEN produced nonfinite
results there. Exact-zero references have undefined relative norm error
and are labeled explicitly. The JSON retains absolute error and finite flags.
The overall probe exit status therefore indicates external execution errors,
not a candidate numerical failure. Other comparator numerical differences
measure compatibility with the candidate rounded policy, not universal defects.

Full-tensor worst-head mathematical FP64 relative L2 error in percent:

| fixture | dtype | scale | implementation | output | dQ | dK | dV | status |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| random | float16 | 0.125 | Candidate | 0.0238% | 0.0388% | 0.0381% | 0.0296% | pass |
| random | float16 | 0.125 | FlexAttention | 0.0239% | 0.0355% | 0.0355% | 0.0296% | pass |
| random | float16 | 0.125 | NATTEN | 0.0239% | 0.0394% | 0.0380% | 0.0296% | pass |
| common-offset | float16 | 0.125 | Candidate | 0.0257% | 1.6092% | 0.0402% | 0.0310% | pass |
| common-offset | float16 | 0.125 | FlexAttention | 0.0265% | 1.2435% | 0.0342% | 0.0310% | numerical-difference |
| common-offset | float16 | 0.125 | NATTEN | 0.0265% | 1.8494% | 0.0363% | 0.0314% | numerical-difference |
| common-offset | float16 | -0.125 | Candidate | 0.0256% | 1.3546% | 0.0433% | 0.0304% | pass |
| common-offset | float16 | -0.125 | FlexAttention | 0.0251% | 1.0889% | 0.0350% | 0.0304% | numerical-difference |
| common-offset | float16 | -0.125 | NATTEN | nonfinite | nonfinite | nonfinite | nonfinite | numerical-difference |
| large-common-offset | float16 | 0.125 | Candidate | 0.0257% | 25.6798% | 0.0402% | 0.0310% | pass |
| large-common-offset | float16 | 0.125 | FlexAttention | 0.3023% | 24.0523% | 0.4827% | 0.4783% | numerical-difference |
| large-common-offset | float16 | 0.125 | NATTEN | 0.0303% | 27.6104% | 0.5604% | 0.5666% | numerical-difference |
| uniform-inverse-boundary | float16 | 0.125 | Candidate | exact zero | exact zero | exact zero | 0.0290% | pass |
| uniform-inverse-boundary | float16 | 0.125 | FlexAttention | exact zero | exact zero | exact zero | 0.0290% | pass |
| uniform-inverse-boundary | float16 | 0.125 | NATTEN | unavailable | unavailable | unavailable | unavailable | execution-error |
| random | bfloat16 | 0.125 | Candidate | 0.1955% | 0.3607% | 0.3652% | 0.2557% | pass |
| random | bfloat16 | 0.125 | FlexAttention | 0.1951% | 0.2833% | 0.2800% | 0.2557% | pass |
| random | bfloat16 | 0.125 | NATTEN | 0.1958% | 0.3091% | 0.2980% | 0.2557% | pass |
| common-offset | bfloat16 | 0.125 | Candidate | 0.1956% | 12.1567% | 0.3644% | 0.2382% | pass |
| common-offset | bfloat16 | 0.125 | FlexAttention | 0.1952% | 8.9227% | 0.2445% | 0.2393% | numerical-difference |
| common-offset | bfloat16 | 0.125 | NATTEN | 0.1980% | 10.0708% | 0.3084% | 0.2393% | numerical-difference |
| common-offset | bfloat16 | -0.125 | Candidate | 0.1861% | 11.6883% | 0.2970% | 0.2402% | pass |
| common-offset | bfloat16 | -0.125 | FlexAttention | 0.1851% | 9.7408% | 0.2584% | 0.2405% | numerical-difference |
| common-offset | bfloat16 | -0.125 | NATTEN | nonfinite | nonfinite | nonfinite | nonfinite | numerical-difference |
| large-common-offset | bfloat16 | 0.125 | Candidate | 0.1956% | 194.0947% | 0.3644% | 0.2382% | pass |
| large-common-offset | bfloat16 | 0.125 | FlexAttention | 0.3135% | 163.3302% | 0.5297% | 0.5154% | numerical-difference |
| large-common-offset | bfloat16 | 0.125 | NATTEN | 0.2839% | 213.4885% | 0.6049% | 0.6572% | numerical-difference |
| uniform-inverse-boundary | bfloat16 | 0.125 | Candidate | exact zero | exact zero | exact zero | 0.2321% | pass |
| uniform-inverse-boundary | bfloat16 | 0.125 | FlexAttention | exact zero | exact zero | exact zero | 0.2321% | pass |
| uniform-inverse-boundary | bfloat16 | 0.125 | NATTEN | unavailable | unavailable | unavailable | unavailable | execution-error |
| probability-underflow | float16 | 1.0 | Candidate | exact zero | exact zero | exact zero | 0.0135% | pass |
| probability-underflow | float16 | 1.0 | FlexAttention | exact zero | exact zero | exact zero | 0.0135% | pass |
| probability-underflow | float16 | 1.0 | NATTEN | unavailable | unavailable | unavailable | unavailable | execution-error |

The large-offset fixture sets Q[0] = K[0] = 1024, Q[1] = 1 and
K[1] = token index modulo 7. Its raw scores are exactly representable
as FP32 (2^20 plus integer gaps 0 through 6). The shared offset changes
normalization magnitude while preserving the mathematical probabilities.
This is a defined reconstruction diagnostic, not a typical-workload claim.

Directed rows are gated against the declared rounded equations; random
rows are gated against mathematical FP64. Passing the directed rounded
gate does not imply a small mathematical error. In particular the large
K component can amplify the intrinsic P/dS rounding floor in dQ.

On this large-offset fixture, candidate FP16 dK/dV errors are approximately
12–18 times smaller than FlexAttention/native NATTEN, and output error is
also lower. BF16 output/dK/dV errors are lower by smaller factors. This
does not extend to dQ: candidate BF16 dQ error is about 194%, compared
with FlexAttention 163% and native NATTEN 213%. Those large errors make
a general mathematical-gradient accuracy claim inappropriate.

Backend selection matters. On the positive small-offset BF16 fixture,
native Hopper NATTEN dQ error is 10.07%, below the candidate 12.16%.
The earlier 4.1% relative dQ improvement above compared the candidate
with CUTLASS NATTEN, and does not describe this native Hopper result.
The JSON retains both references, every head/component, operand hashes,
actual launched kernels and source hashes, including failed diagnostics.

## Current native-backend probe: H100-v2-exactmixed

This separate full-tensor diagnostic uses native CPU FP64 equations on
identical already-rounded operands. It measures no latency or model quality.
Device: NVIDIA H100 80GB HBM3, 132 SMs. Qualified stack: PyTorch 2.13.0+cu130, NATTEN 0.21.7+torch2130cu130.
The retained PyTorch 2.13 stack matches the newest NATTEN prebuilt pairing
and prior baseline; these results do not evaluate PyTorch 2.15 FlexAttention.
See the [NATTEN installation matrix](https://natten.org/install/) and
[current FlexAttention documentation](https://docs.pytorch.org/docs/2.15/nn.attention.flex_attention.html).
Candidate variant: `exactmixed_no_stat_fills`, columns `[32, 32, 64]`, materialize output gradients `False`.
Exact contiguous traversal flags: `True`.
Probe status: `execution-error`; candidate failed rows: 0; execution errors: 3; comparator rounded-policy differences: 12.
Retained stage exit status: `1`.

11/11 candidate rows passed their declared gates. Three NATTEN rows
could not execute because its public API rejects a (1, 3) window; these
two uniform-boundary cases and one underflow case remain explicit errors.
Negative-scale rows are compatibility probes; NATTEN produced nonfinite
results there. Exact-zero references have undefined relative norm error
and are labeled explicitly. The JSON retains absolute error and finite flags.
The overall probe exit status therefore indicates external execution errors,
not a candidate numerical failure. Other comparator numerical differences
measure compatibility with the candidate rounded policy, not universal defects.

Full-tensor worst-head mathematical FP64 relative L2 error in percent:

| fixture | dtype | scale | implementation | output | dQ | dK | dV | status |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| random | float16 | 0.125 | Candidate | 0.0238% | 0.0388% | 0.0381% | 0.0296% | pass |
| random | float16 | 0.125 | FlexAttention | 0.0236% | 0.0355% | 0.0345% | 0.0296% | pass |
| random | float16 | 0.125 | NATTEN | 0.0239% | 0.0394% | 0.0380% | 0.0296% | pass |
| common-offset | float16 | 0.125 | Candidate | 0.0257% | 1.6092% | 0.0402% | 0.0310% | pass |
| common-offset | float16 | 0.125 | FlexAttention | 0.0257% | 1.2770% | 0.0354% | 0.0310% | numerical-difference |
| common-offset | float16 | 0.125 | NATTEN | 0.0265% | 1.8494% | 0.0363% | 0.0314% | numerical-difference |
| common-offset | float16 | -0.125 | Candidate | 0.0256% | 1.3546% | 0.0433% | 0.0304% | pass |
| common-offset | float16 | -0.125 | FlexAttention | 0.0251% | 1.0889% | 0.0350% | 0.0304% | numerical-difference |
| common-offset | float16 | -0.125 | NATTEN | nonfinite | nonfinite | nonfinite | nonfinite | numerical-difference |
| large-common-offset | float16 | 0.125 | Candidate | 0.0257% | 25.6798% | 0.0402% | 0.0310% | pass |
| large-common-offset | float16 | 0.125 | FlexAttention | 0.3002% | 21.4250% | 0.4865% | 0.4783% | numerical-difference |
| large-common-offset | float16 | 0.125 | NATTEN | 0.0303% | 27.6104% | 0.5604% | 0.5666% | numerical-difference |
| uniform-inverse-boundary | float16 | 0.125 | Candidate | exact zero | exact zero | exact zero | 0.0290% | pass |
| uniform-inverse-boundary | float16 | 0.125 | FlexAttention | exact zero | exact zero | exact zero | 0.0290% | pass |
| uniform-inverse-boundary | float16 | 0.125 | NATTEN | unavailable | unavailable | unavailable | unavailable | execution-error |
| random | bfloat16 | 0.125 | Candidate | 0.1955% | 0.3607% | 0.3652% | 0.2557% | pass |
| random | bfloat16 | 0.125 | FlexAttention | 0.1951% | 0.2833% | 0.2800% | 0.2557% | pass |
| random | bfloat16 | 0.125 | NATTEN | 0.1958% | 0.3091% | 0.2980% | 0.2557% | pass |
| common-offset | bfloat16 | 0.125 | Candidate | 0.1956% | 12.1567% | 0.3644% | 0.2382% | pass |
| common-offset | bfloat16 | 0.125 | FlexAttention | 0.1952% | 8.9227% | 0.2445% | 0.2393% | numerical-difference |
| common-offset | bfloat16 | 0.125 | NATTEN | 0.1980% | 10.0708% | 0.3084% | 0.2393% | numerical-difference |
| common-offset | bfloat16 | -0.125 | Candidate | 0.1861% | 11.6883% | 0.2970% | 0.2402% | pass |
| common-offset | bfloat16 | -0.125 | FlexAttention | 0.1851% | 9.7408% | 0.2584% | 0.2405% | numerical-difference |
| common-offset | bfloat16 | -0.125 | NATTEN | nonfinite | nonfinite | nonfinite | nonfinite | numerical-difference |
| large-common-offset | bfloat16 | 0.125 | Candidate | 0.1956% | 194.0947% | 0.3644% | 0.2382% | pass |
| large-common-offset | bfloat16 | 0.125 | FlexAttention | 0.3135% | 163.3302% | 0.5297% | 0.5154% | numerical-difference |
| large-common-offset | bfloat16 | 0.125 | NATTEN | 0.2839% | 213.4885% | 0.6049% | 0.6572% | numerical-difference |
| uniform-inverse-boundary | bfloat16 | 0.125 | Candidate | exact zero | exact zero | exact zero | 0.2321% | pass |
| uniform-inverse-boundary | bfloat16 | 0.125 | FlexAttention | exact zero | exact zero | exact zero | 0.2321% | pass |
| uniform-inverse-boundary | bfloat16 | 0.125 | NATTEN | unavailable | unavailable | unavailable | unavailable | execution-error |
| probability-underflow | float16 | 1.0 | Candidate | exact zero | exact zero | exact zero | 0.0135% | pass |
| probability-underflow | float16 | 1.0 | FlexAttention | exact zero | exact zero | exact zero | 0.0135% | pass |
| probability-underflow | float16 | 1.0 | NATTEN | unavailable | unavailable | unavailable | unavailable | execution-error |

The large-offset fixture sets Q[0] = K[0] = 1024, Q[1] = 1 and
K[1] = token index modulo 7. Its raw scores are exactly representable
as FP32 (2^20 plus integer gaps 0 through 6). The shared offset changes
normalization magnitude while preserving the mathematical probabilities.
This is a defined reconstruction diagnostic, not a typical-workload claim.

Directed rows are gated against the declared rounded equations; random
rows are gated against mathematical FP64. Passing the directed rounded
gate does not imply a small mathematical error. In particular the large
K component can amplify the intrinsic P/dS rounding floor in dQ.

On this large-offset fixture, candidate FP16 dK/dV errors are approximately
12–18 times smaller than FlexAttention/native NATTEN, and output error is
also lower. BF16 output/dK/dV errors are lower by smaller factors. This
does not extend to dQ: candidate BF16 dQ error is about 194%, compared
with FlexAttention 163% and native NATTEN 213%. Those large errors make
a general mathematical-gradient accuracy claim inappropriate.

Backend selection matters. On the positive small-offset BF16 fixture,
native Hopper NATTEN dQ error is 10.07%, below the candidate 12.16%.
The earlier 4.1% relative dQ improvement above compared the candidate
with CUTLASS NATTEN, and does not describe this native Hopper result.
The JSON retains both references, every head/component, operand hashes,
actual launched kernels and source hashes, including failed diagnostics.

## Current native-backend probe: H100-v4-production-candidate

This separate full-tensor diagnostic uses native CPU FP64 equations on
identical already-rounded operands. It measures no latency or model quality.
Device: NVIDIA H100 80GB HBM3, 132 SMs. Qualified stack: PyTorch 2.13.0+cu130, NATTEN 0.21.7+torch2130cu130.
The retained PyTorch 2.13 stack matches the newest NATTEN prebuilt pairing
and prior baseline; these results do not evaluate PyTorch 2.15 FlexAttention.
See the [NATTEN installation matrix](https://natten.org/install/) and
[current FlexAttention documentation](https://docs.pytorch.org/docs/2.15/nn.attention.flex_attention.html).
Candidate variant: `production`, columns `automatic per shape`, materialize output gradients `False`.
Exact contiguous traversal flags: `automatic per shape`.
These tiny probe tensors have at most 45 tokens and exercise the
production C32 fallback. The bounded H100 KV64 preset requires at
least 1024 tokens and is qualified separately on the timing matrix
and held-out workloads; the production label does not imply KV64
executes for these small directed probes.
Probe status: `execution-error`; candidate failed rows: 0; execution errors: 3; comparator rounded-policy differences: 12.
Retained stage exit status: `1`.

11/11 candidate rows passed their declared gates. Three NATTEN rows
could not execute because its public API rejects a (1, 3) window; these
two uniform-boundary cases and one underflow case remain explicit errors.
Negative-scale rows are compatibility probes; NATTEN produced nonfinite
results there. Exact-zero references have undefined relative norm error
and are labeled explicitly. The JSON retains absolute error and finite flags.
The overall probe exit status therefore indicates external execution errors,
not a candidate numerical failure. Other comparator numerical differences
measure compatibility with the candidate rounded policy, not universal defects.

Full-tensor worst-head mathematical FP64 relative L2 error in percent:

| fixture | dtype | scale | implementation | output | dQ | dK | dV | status |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| random | float16 | 0.125 | Candidate | 0.0238% | 0.0388% | 0.0381% | 0.0296% | pass |
| random | float16 | 0.125 | FlexAttention | 0.0236% | 0.0355% | 0.0345% | 0.0296% | pass |
| random | float16 | 0.125 | NATTEN | 0.0239% | 0.0394% | 0.0380% | 0.0296% | pass |
| common-offset | float16 | 0.125 | Candidate | 0.0257% | 1.6092% | 0.0402% | 0.0310% | pass |
| common-offset | float16 | 0.125 | FlexAttention | 0.0257% | 1.2770% | 0.0354% | 0.0310% | numerical-difference |
| common-offset | float16 | 0.125 | NATTEN | 0.0265% | 1.8494% | 0.0363% | 0.0314% | numerical-difference |
| common-offset | float16 | -0.125 | Candidate | 0.0256% | 1.3546% | 0.0433% | 0.0304% | pass |
| common-offset | float16 | -0.125 | FlexAttention | 0.0251% | 1.0889% | 0.0350% | 0.0304% | numerical-difference |
| common-offset | float16 | -0.125 | NATTEN | nonfinite | nonfinite | nonfinite | nonfinite | numerical-difference |
| large-common-offset | float16 | 0.125 | Candidate | 0.0257% | 25.6798% | 0.0402% | 0.0310% | pass |
| large-common-offset | float16 | 0.125 | FlexAttention | 0.3002% | 21.4250% | 0.4865% | 0.4783% | numerical-difference |
| large-common-offset | float16 | 0.125 | NATTEN | 0.0303% | 27.6104% | 0.5604% | 0.5666% | numerical-difference |
| uniform-inverse-boundary | float16 | 0.125 | Candidate | exact zero | exact zero | exact zero | 0.0290% | pass |
| uniform-inverse-boundary | float16 | 0.125 | FlexAttention | exact zero | exact zero | exact zero | 0.0290% | pass |
| uniform-inverse-boundary | float16 | 0.125 | NATTEN | unavailable | unavailable | unavailable | unavailable | execution-error |
| random | bfloat16 | 0.125 | Candidate | 0.1955% | 0.3607% | 0.3652% | 0.2557% | pass |
| random | bfloat16 | 0.125 | FlexAttention | 0.1951% | 0.2833% | 0.2800% | 0.2557% | pass |
| random | bfloat16 | 0.125 | NATTEN | 0.1958% | 0.3091% | 0.2980% | 0.2557% | pass |
| common-offset | bfloat16 | 0.125 | Candidate | 0.1956% | 12.1567% | 0.3644% | 0.2382% | pass |
| common-offset | bfloat16 | 0.125 | FlexAttention | 0.1952% | 8.9227% | 0.2445% | 0.2393% | numerical-difference |
| common-offset | bfloat16 | 0.125 | NATTEN | 0.1980% | 10.0708% | 0.3084% | 0.2393% | numerical-difference |
| common-offset | bfloat16 | -0.125 | Candidate | 0.1861% | 11.6883% | 0.2970% | 0.2402% | pass |
| common-offset | bfloat16 | -0.125 | FlexAttention | 0.1851% | 9.7408% | 0.2584% | 0.2405% | numerical-difference |
| common-offset | bfloat16 | -0.125 | NATTEN | nonfinite | nonfinite | nonfinite | nonfinite | numerical-difference |
| large-common-offset | bfloat16 | 0.125 | Candidate | 0.1956% | 194.0947% | 0.3644% | 0.2382% | pass |
| large-common-offset | bfloat16 | 0.125 | FlexAttention | 0.3135% | 163.3302% | 0.5297% | 0.5154% | numerical-difference |
| large-common-offset | bfloat16 | 0.125 | NATTEN | 0.2839% | 213.4885% | 0.6049% | 0.6572% | numerical-difference |
| uniform-inverse-boundary | bfloat16 | 0.125 | Candidate | exact zero | exact zero | exact zero | 0.2321% | pass |
| uniform-inverse-boundary | bfloat16 | 0.125 | FlexAttention | exact zero | exact zero | exact zero | 0.2321% | pass |
| uniform-inverse-boundary | bfloat16 | 0.125 | NATTEN | unavailable | unavailable | unavailable | unavailable | execution-error |
| probability-underflow | float16 | 1.0 | Candidate | exact zero | exact zero | exact zero | 0.0135% | pass |
| probability-underflow | float16 | 1.0 | FlexAttention | exact zero | exact zero | exact zero | 0.0135% | pass |
| probability-underflow | float16 | 1.0 | NATTEN | unavailable | unavailable | unavailable | unavailable | execution-error |

The large-offset fixture sets Q[0] = K[0] = 1024, Q[1] = 1 and
K[1] = token index modulo 7. Its raw scores are exactly representable
as FP32 (2^20 plus integer gaps 0 through 6). The shared offset changes
normalization magnitude while preserving the mathematical probabilities.
This is a defined reconstruction diagnostic, not a typical-workload claim.

Directed rows are gated against the declared rounded equations; random
rows are gated against mathematical FP64. Passing the directed rounded
gate does not imply a small mathematical error. In particular the large
K component can amplify the intrinsic P/dS rounding floor in dQ.

On this large-offset fixture, candidate FP16 dK/dV errors are approximately
12–18 times smaller than FlexAttention/native NATTEN, and output error is
also lower. BF16 output/dK/dV errors are lower by smaller factors. This
does not extend to dQ: candidate BF16 dQ error is about 194%, compared
with FlexAttention 163% and native NATTEN 213%. Those large errors make
a general mathematical-gradient accuracy claim inappropriate.

Backend selection matters. On the positive small-offset BF16 fixture,
native Hopper NATTEN dQ error is 10.07%, below the candidate 12.16%.
The earlier 4.1% relative dQ improvement above compared the candidate
with CUTLASS NATTEN, and does not describe this native Hopper result.
The JSON retains both references, every head/component, operand hashes,
actual launched kernels and source hashes, including failed diagnostics.

## Current native-backend probe: H100-v6-production

This separate full-tensor diagnostic uses native CPU FP64 equations on
identical already-rounded operands. It measures no latency or model quality.
Device: NVIDIA H100 80GB HBM3, 132 SMs. Qualified stack: PyTorch 2.13.0+cu130, NATTEN 0.21.7+torch2130cu130.
The retained PyTorch 2.13 stack matches the newest NATTEN prebuilt pairing
and prior baseline; these results do not evaluate PyTorch 2.15 FlexAttention.
See the [NATTEN installation matrix](https://natten.org/install/) and
[current FlexAttention documentation](https://docs.pytorch.org/docs/2.15/nn.attention.flex_attention.html).
Candidate variant: `production`, columns `automatic per shape`, materialize output gradients `False`.
Exact contiguous traversal flags: `automatic per shape`.
These tiny probe tensors have at most 45 tokens and exercise the
production C32 fallback. The bounded H100 KV64 preset requires at
least 1024 tokens and is qualified separately on the timing matrix
and held-out workloads; the production label does not imply KV64
executes for these small directed probes.
Probe status: `execution-error`; candidate failed rows: 0; execution errors: 3; comparator rounded-policy differences: 12.
Retained stage exit status: `1`.

11/11 candidate rows passed their declared gates. Three NATTEN rows
could not execute because its public API rejects a (1, 3) window; these
two uniform-boundary cases and one underflow case remain explicit errors.
Negative-scale rows are compatibility probes; NATTEN produced nonfinite
results there. Exact-zero references have undefined relative norm error
and are labeled explicitly. The JSON retains absolute error and finite flags.
The overall probe exit status therefore indicates external execution errors,
not a candidate numerical failure. Other comparator numerical differences
measure compatibility with the candidate rounded policy, not universal defects.

Full-tensor worst-head mathematical FP64 relative L2 error in percent:

| fixture | dtype | scale | implementation | output | dQ | dK | dV | status |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| random | float16 | 0.125 | Candidate | 0.0238% | 0.0388% | 0.0381% | 0.0296% | pass |
| random | float16 | 0.125 | FlexAttention | 0.0236% | 0.0355% | 0.0345% | 0.0296% | pass |
| random | float16 | 0.125 | NATTEN | 0.0239% | 0.0394% | 0.0380% | 0.0296% | pass |
| common-offset | float16 | 0.125 | Candidate | 0.0257% | 1.6092% | 0.0402% | 0.0310% | pass |
| common-offset | float16 | 0.125 | FlexAttention | 0.0257% | 1.2770% | 0.0354% | 0.0310% | numerical-difference |
| common-offset | float16 | 0.125 | NATTEN | 0.0265% | 1.8494% | 0.0363% | 0.0314% | numerical-difference |
| common-offset | float16 | -0.125 | Candidate | 0.0256% | 1.3546% | 0.0433% | 0.0304% | pass |
| common-offset | float16 | -0.125 | FlexAttention | 0.0251% | 1.0889% | 0.0350% | 0.0304% | numerical-difference |
| common-offset | float16 | -0.125 | NATTEN | nonfinite | nonfinite | nonfinite | nonfinite | numerical-difference |
| large-common-offset | float16 | 0.125 | Candidate | 0.0257% | 25.6798% | 0.0402% | 0.0310% | pass |
| large-common-offset | float16 | 0.125 | FlexAttention | 0.3002% | 21.4250% | 0.4865% | 0.4783% | numerical-difference |
| large-common-offset | float16 | 0.125 | NATTEN | 0.0303% | 27.6104% | 0.5604% | 0.5666% | numerical-difference |
| uniform-inverse-boundary | float16 | 0.125 | Candidate | exact zero | exact zero | exact zero | 0.0290% | pass |
| uniform-inverse-boundary | float16 | 0.125 | FlexAttention | exact zero | exact zero | exact zero | 0.0290% | pass |
| uniform-inverse-boundary | float16 | 0.125 | NATTEN | unavailable | unavailable | unavailable | unavailable | execution-error |
| random | bfloat16 | 0.125 | Candidate | 0.1955% | 0.3607% | 0.3652% | 0.2557% | pass |
| random | bfloat16 | 0.125 | FlexAttention | 0.1951% | 0.2833% | 0.2800% | 0.2557% | pass |
| random | bfloat16 | 0.125 | NATTEN | 0.1958% | 0.3091% | 0.2980% | 0.2557% | pass |
| common-offset | bfloat16 | 0.125 | Candidate | 0.1956% | 12.1567% | 0.3644% | 0.2382% | pass |
| common-offset | bfloat16 | 0.125 | FlexAttention | 0.1952% | 8.9227% | 0.2445% | 0.2393% | numerical-difference |
| common-offset | bfloat16 | 0.125 | NATTEN | 0.1980% | 10.0708% | 0.3084% | 0.2393% | numerical-difference |
| common-offset | bfloat16 | -0.125 | Candidate | 0.1861% | 11.6883% | 0.2970% | 0.2402% | pass |
| common-offset | bfloat16 | -0.125 | FlexAttention | 0.1851% | 9.7408% | 0.2584% | 0.2405% | numerical-difference |
| common-offset | bfloat16 | -0.125 | NATTEN | nonfinite | nonfinite | nonfinite | nonfinite | numerical-difference |
| large-common-offset | bfloat16 | 0.125 | Candidate | 0.1956% | 194.0947% | 0.3644% | 0.2382% | pass |
| large-common-offset | bfloat16 | 0.125 | FlexAttention | 0.3135% | 163.3302% | 0.5297% | 0.5154% | numerical-difference |
| large-common-offset | bfloat16 | 0.125 | NATTEN | 0.2839% | 213.4885% | 0.6049% | 0.6572% | numerical-difference |
| uniform-inverse-boundary | bfloat16 | 0.125 | Candidate | exact zero | exact zero | exact zero | 0.2321% | pass |
| uniform-inverse-boundary | bfloat16 | 0.125 | FlexAttention | exact zero | exact zero | exact zero | 0.2321% | pass |
| uniform-inverse-boundary | bfloat16 | 0.125 | NATTEN | unavailable | unavailable | unavailable | unavailable | execution-error |
| probability-underflow | float16 | 1.0 | Candidate | exact zero | exact zero | exact zero | 0.0135% | pass |
| probability-underflow | float16 | 1.0 | FlexAttention | exact zero | exact zero | exact zero | 0.0135% | pass |
| probability-underflow | float16 | 1.0 | NATTEN | unavailable | unavailable | unavailable | unavailable | execution-error |

The large-offset fixture sets Q[0] = K[0] = 1024, Q[1] = 1 and
K[1] = token index modulo 7. Its raw scores are exactly representable
as FP32 (2^20 plus integer gaps 0 through 6). The shared offset changes
normalization magnitude while preserving the mathematical probabilities.
This is a defined reconstruction diagnostic, not a typical-workload claim.

Directed rows are gated against the declared rounded equations; random
rows are gated against mathematical FP64. Passing the directed rounded
gate does not imply a small mathematical error. In particular the large
K component can amplify the intrinsic P/dS rounding floor in dQ.

On this large-offset fixture, candidate FP16 dK/dV errors are approximately
12–18 times smaller than FlexAttention/native NATTEN, and output error is
also lower. BF16 output/dK/dV errors are lower by smaller factors. This
does not extend to dQ: candidate BF16 dQ error is about 194%, compared
with FlexAttention 163% and native NATTEN 213%. Those large errors make
a general mathematical-gradient accuracy claim inappropriate.

Backend selection matters. On the positive small-offset BF16 fixture,
native Hopper NATTEN dQ error is 10.07%, below the candidate 12.16%.
The earlier 4.1% relative dQ improvement above compared the candidate
with CUTLASS NATTEN, and does not describe this native Hopper result.
The JSON retains both references, every head/component, operand hashes,
actual launched kernels and source hashes, including failed diagnostics.
