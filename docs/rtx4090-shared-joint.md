# RTX 4090 shared joint backward

The optional entry below uses measured forward and joint-backward schedules on a
full RTX 4090. It retains the existing CuTe API for calls outside its qualified
family. Install the [pinned optional runtime](a100-sm80-fa4.md): Python 3.13,
PyTorch 2.15.0/CUDA 13.2 and the `cuda,sm80-fa4` extras.

The same entry also supports [S128/D16 full-row training](rtx4090-short.md)
under its separately qualified short-call conditions.

```python
import torch
from kohakufa_cute.fa4_sm80.rtx4090 import scaled_dot_product_attention

# Contiguous BHSD FP16/BF16 tensors, B1/H16, S1024 or S4096, D64 or D128.
out = scaled_dot_product_attention(q, k, v, is_causal=True)
grads = torch.autograd.grad(out, (q, k, v), dout)
```

The measured family requires matching square Q/K/V, 16-byte alignment, a full
128-SM device with 101,376 bytes of opt-in shared memory, and a positive normal
FP32 attention scale. Masks, GQA, other head widths, short lengths, tails,
strided inputs, signed/zero scales and other supported CuTe calls use the ordinary
CuTe path. Selected-kernel errors propagate. The standard top-level APIs keep
their existing selection; this entry is opt-in.

## Measurements

Milliseconds, lower is better. These are warmed CUDA-graph forward plus backward
calls including all three gradients, GPU preparation, scratch initialization and
gradient conversion. Compilation, capture and host allocation are excluded.
All implementations used identical rounded inputs in one process, with seven
rotated 100 ms sampling windows. The full RTX 4090 ran at a 400 W power limit.
Standalone FA2 and SDPA retain their native arithmetic policies.

| Dtype | Length | Head dim | Visibility | Previous CuTe F+B | Shared F+B | FA2 F+B | SDPA F+B |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| bf16 | 1024 | 64 | dense | 0.137 | 0.105 | 0.107 | 0.111 |
| bf16 | 1024 | 64 | causal | 0.134 | 0.099 | 0.105 | 0.113 |
| bf16 | 1024 | 128 | dense | 0.251 | 0.197 | 0.208 | 0.217 |
| bf16 | 1024 | 128 | causal | 0.245 | 0.180 | 0.177 | 0.187 |
| bf16 | 4096 | 64 | dense | 1.941 | 1.511 | 1.489 | 1.532 |
| bf16 | 4096 | 64 | causal | 1.222 | 0.965 | 1.016 | 1.053 |
| bf16 | 4096 | 128 | dense | 3.876 | 3.005 | 3.046 | 3.119 |
| bf16 | 4096 | 128 | causal | 2.305 | 1.839 | 1.771 | 1.846 |
| fp16 | 1024 | 64 | dense | 0.141 | 0.104 | 0.107 | 0.112 |
| fp16 | 1024 | 64 | causal | 0.134 | 0.098 | 0.106 | 0.113 |
| fp16 | 1024 | 128 | dense | 0.255 | 0.197 | 0.209 | 0.219 |
| fp16 | 1024 | 128 | causal | 0.251 | 0.182 | 0.178 | 0.187 |
| fp16 | 4096 | 64 | dense | 1.974 | 1.515 | 1.515 | 1.551 |
| fp16 | 4096 | 64 | causal | 1.229 | 0.957 | 1.017 | 1.051 |
| fp16 | 4096 | 128 | dense | 3.910 | 3.054 | 3.072 | 3.133 |
| fp16 | 4096 | 128 | causal | 2.371 | 1.862 | 1.799 | 1.872 |

The shared path reduced graph F+B latency in 16/16 calls against the previous CuTe path, 10/16 against FA2 and 16/16 against SDPA. Counts use the measured medians; the record retains every sample.

The [compact record](../results/rtx4090-shared-joint.json) also includes separate
forward/backward samples, synchronized eager timings, no-grad graph and eager
forward timings, actual kernel resources, software versions and source hashes.
Eager calls include normal allocation and launch behavior. Their FA2 rankings
can differ from graph replay; graph results alone do not establish an eager
speedup. These are attention-call measurements rather than model-training times.

## Design and arithmetic

The backward preparation computes delta from the actual rounded output,
normalizes saved statistics and clears FP32 dQ scratch. The joint kernel reconstructs P once and computes dP once. P feeds
dV and dS; dS feeds dK and dQ. A final pass converts dQ. All passes and required allocation behavior are included in the
appropriate timing boundary.

| Family | Forward M×N | Backward M×N | Q/dO stages | Score/KV/dQ M warps | V storage |
| --- | --- | --- | --- | --- | --- |
| D64 dense | 128×64 | 64×128 | 1/1 | 2/4/4 | registers |
| D64 causal S1024 | 64×64 | 64×64 | 1/1 | 4/2/2 | registers |
| D64 causal S4096 | 64×64 | 64×128 | 1/1 | 2/4/4 | registers |
| D128 dense | 64×64 | 32×128 | 1/1 | 2/2/2 | shared |
| D128 causal | 64×64 | 64×64 | 1/1 | 4/2/2 | registers |

D128 dense uses a wider key tile to reduce dQ partial updates. Moving V to shared
memory avoids the long-lived register operand that made the earlier M32 layout
spill inside its main loop. The retained layout uses 99,328 shared bytes on the
measured device. D128 causal keeps its M64N64 ownership; the dense D128 layout regressed
causal calls. Long D64 causal uses its independently qualified M64N128 path. D64 causal uses a smaller forward query tile,
which improved the complete call as well as forward latency.

The [precision contract](precision.md) is unchanged: separate unscaled maximum
and logarithmic normalization; P rounded once to the input dtype before dS; dS
rounded once; FP32 products/accumulation; delta from the actual rounded O; and
late dQ/dK scaling. No compensation or higher-precision correction is selected.
A redundant-wait experiment removed a barrier without a consistent latency gain;
the selected path retains its original synchronization.

## Qualification and reproduction

The native entry passed 96 directed cases and all four sanitizer tools with zero
device errors, including partial tiles, sharp/common-offset logits, declared
rounding fixtures, model head counts and changed/poisoned graph replay. Sixteen
fallback cases checked masks, all-hidden rows, GQA, rectangles, strides, batch
changes and zero scale. A broader 60-row matrix covered D16/64/128/256/512,
both dtypes and dense/causal forward, backward and joint calls. Direct FA2 D512
is unsupported; SDPA and the existing CuTe path supplied those comparisons.
Independent per-head FP64 norm limits remained 1% for BF16 and 0.5% for FP16;
directed fixtures retained their existing rounded-equation criteria.

Systems traces identify the launched preparation/main/conversion sequence.
CUPTI-loaded ELF hashes were matched to compiler-returned code for the selected
native kernels, with exact-symbol PTX and SASS retained separately. Compute
counters were denied by device permissions; no stall or utilization percentages
are inferred from static instruction counts.

```bash
python -m benchmarks.validate_rtx4090_shared \
  --source path/to/flash-attention --output directed.json
python -m benchmarks.compare_flashattention --backends cute fa2 sdpa \
  --candidate benchmarks.rtx4090_shared_candidates:make_attention \
  --candidate-kwargs '{"source":"path/to/flash-attention"}' \
  --heads 16 --seq 1024 4096 --dim 64 128 --dtype bf16 fp16 \
  --modes dense causal --phases forward backward joint \
  --repeats 7 --window-ms 100 --output comparison.json
```

The optional dependency notes specify the pinned FlashAttention revision. Source
and qualifier hashes in the compact record identify the measured implementation.

The [causal head-grouping follow-up](rtx4090-causal-head-groups.md) applies the measured G16 ordering to the same bounded causal calls. Its newer record includes separate graph and eager results.

The [long D64 causal follow-up](rtx4090-long-causal.md) uses N128 at S4096 and retains N64 at S1024. Its held-out current-policy record supersedes the earlier long D64 causal timings above.
