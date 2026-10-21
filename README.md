# KohakuFA-CuTe

CuTe DSL attention for **RTX 5090, B200/B300, H100, and L40S**, with stable backward gradients when attention becomes sharp. It brings [KohakuFA's](https://github.com/KohakuBlueleaf/KohakuFA) precision approach to Ada, Hopper, and Blackwell, and adds a centered FP32 correction that retains tiny saturated gradients.

## When to use it

- **Training with sharp attention or large logits:** forward outputs can look correct while query/key gradients are wrong. The kernels preserve separate softmax statistics and a centered backward correction. See [precision](docs/precision.md).
- **RTX 5090 with 512-wide heads:** forward plus backward is **1.60–3.86× faster than default PyTorch SDPA** across all eight tested FP16/BF16, dense/causal, S1024/S4096 cases.
- **RTX 5090 with explicit boolean masks:** the tested BF16 causal masks are **1.48–4.72× faster than default SDPA**, including mask preparation. Arbitrary masks need their own benchmarks.
- **Other workloads:** choose it for its gradient calculation, then benchmark your shapes. Smaller unmasked RTX heads still have slower backward than FlashAttention-2. H100/L40S also trade throughput for precision in the measured training cases.

Automatic selection checks compatibility; it does **not** choose whichever backend is fastest for a shape.

## Performance

Attention-call latency, **forward + backward**, in milliseconds; lower is better. B=1, H=16, BF16, S=4096. These are matched comparisons within each row, with compilation excluded.

| GPU | Head dim | Visibility | CuTe | Default SDPA | SDPA / CuTe |
| --- | ---: | --- | ---: | ---: | ---: |
| RTX 5090 | 512 | Dense | **27.802** | 48.211 | **1.73×** |
| RTX 5090 | 512 | Causal | **13.194** | 29.634 | **2.25×** |
| RTX 5090 | 256 | Boolean causal mask | **6.052** | 24.294 | **4.01×** |
| RTX 5090 | 128 | Boolean causal mask | **2.318** | 10.050 | **4.33×** |
| RTX 5090 | 128 | Dense | 3.608 | **2.601** | 0.72× |
| H100 | 128 | Dense | 1.300 | **0.927** | 0.71× |
| L40S | 128 | Dense | 3.547 | **2.721** | 0.77× |

On the **B300 MIG 1g.34gb (18 SMs)**, BF16 D512 causal forward + backward is approximately tied with upstream Gluon at S4096 (**13.649 vs 13.703 ms**) and slightly faster at S8192 (**50.766 vs 51.276 ms**). Smaller cases still favor Gluon. These timings describe the partition; current full-device B200/B300 performance has not been remeasured.

[Performance details](results/README.md) cover both dtypes, forward-only results, actual SDPA backends, measurement conditions, and reproduction. [Machine-readable measurements](results/performance.json) retain timing samples where available. These are attention benchmarks, not end-to-end training speedups.

## Install

Install a CUDA-enabled **PyTorch 2.11+**, then:

```bash
git clone https://github.com/SimpleTuner-io/KohakuFA-CuTe.git
cd KohakuFA-CuTe
python -m pip install '.[cuda]'
```

Requires Python 3.12+, CUTLASS DSL 4.8, TVM FFI 0.1.14+, Triton 3.6+, Ninja, and a C++17 compiler. The native launcher builds on first use. `pip install .` installs the API without CUDA extras; automatic selection can then use native SDPA on CPU/MPS.

## Use

```python
import torch
from kohakufa_cute import attention, automatic_scaled_dot_product_attention

# Q/K/V: [batch, heads, sequence, head_dim]
q = torch.randn(1, 16, 1024, 512, device="cuda", dtype=torch.bfloat16,
                requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)

out = attention(q, k, v, causal=True)
out.square().mean().backward()

# SDPA-compatible arguments and native SDPA for unsupported inputs.
out = automatic_scaled_dot_product_attention(q, k, v, is_causal=True)
compiled = torch.compile(automatic_scaled_dot_product_attention, fullgraph=True)
out = compiled(q, k, v)
```

| API | Behavior |
| --- | --- |
| `attention(q, k, v, *, mask=None, causal=False, scale=None, center_keys=False)` | Strict CuTe; implicit GQA; optional key centering |
| `scaled_dot_product_attention(...)` | Strict CuTe with PyTorch SDPA arguments |
| `automatic_scaled_dot_product_attention(...)` | CuTe for supported calls, native SDPA otherwise |

Strict APIs propagate validation, build, and kernel errors. Automatic selection never patches PyTorch globally; errors from a selected CuTe kernel propagate too.

Supported CuTe inputs:

- CUDA SM89/90/100/103/120; matching FP16 or BF16 Q/K/V; head dimensions **1–512**.
- Four-dimensional BHSD tensors, including supported strided layouts; equal K/V shapes; query heads divisible by K/V heads.
- Dense, token-causal, or broadcastable boolean masks (`True` means visible). All-hidden rows return zero.
- GQA, eager/compiled autograd, CUDA graphs, and signed or zero scales.

The SDPA wrappers require `enable_gqa=True` for different head counts. With an explicit mask, fold causal visibility into that mask; `attention` also accepts `mask` and `causal=True` together. FP32, dropout, additive masks, unsupported devices/shapes, and missing CUDA dependencies select native SDPA through the automatic API. There is no packed variable-length API.

`attention(..., center_keys=True)` subtracts each batch/head's mean key in FP32 before rounding once back to the input dtype. This can reduce errors from a shared key component, at the cost of a reduction and temporary storage. It defaults to `False` and is separate from the centered backward calculation.

## Validate and benchmark

```bash
python -m unittest discover -v -f
python -m benchmarks.speed --dtype bf16 --seq 1024 4096 --dim 64 128 256 512 \
  --backends sdpa cute --iterations 100 --repeats 6 --output speed.json
python -m benchmarks.precision --dtype bf16 --seq 256 --dim 128 \
  --backends auto kohakufa-cute --output precision.json
```

Tests cover FP64 gradient comparisons, tiny gradients, masks, GQA, head tails, strides, compiled backward, checkpointed LoRA updates, streams, and changed-input graph replay. The [benchmark guide](results/README.md#reproduce) includes causal and upstream comparisons.

## License and attribution

Developed in [SimpleTuner](https://github.com/bghira/SimpleTuner/pull/3256), under [AGPL-3.0-or-later](LICENSE). [KohakuFA](https://github.com/KohakuBlueleaf/KohakuFA) originated the precision investigation and arithmetic approach; its Gluon kernels are not bundled. NVIDIA CUTLASS and FlashAttention helper adaptations retain their BSD notices in [NOTICE](NOTICE) and [LICENSES/BSD-3-Clause.txt](LICENSES/BSD-3-Clause.txt).
