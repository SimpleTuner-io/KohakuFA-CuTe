# KohakuFA-CuTe

CuTe DSL attention for NVIDIA Ampere, Ada, Hopper, and Blackwell GPUs, including
GB10. It uses [KohakuFA's](https://github.com/KohakuBlueleaf/KohakuFA) shifted-score
probability reconstruction and gradient scaling after FP32 accumulation to address
large-logit reconstruction errors and premature gradient underflow.
Tensor-core operands are FP16/BF16; see the [precision contract](docs/precision.md).

## Install

Install CUDA-enabled PyTorch 2.11+ and a C++17 compiler, then:

```bash
git clone https://github.com/SimpleTuner-io/KohakuFA-CuTe.git
cd KohakuFA-CuTe
python -m pip install '.[cuda]'
```

Requires Python 3.12+. The native launcher builds on first use. See
[installation](docs/installation.md) for the pinned tuned runtime and GB10 ARM64.

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

# SDPA-compatible arguments; native SDPA for unsupported inputs.
out = automatic_scaled_dot_product_attention(q, k, v, is_causal=True)
```

| API | Behavior |
| --- | --- |
| `attention(...)` | Strict CuTe, implicit GQA, optional key centering |
| `scaled_dot_product_attention(...)` | Strict CuTe with SDPA arguments |
| `automatic_scaled_dot_product_attention(...)` | CuTe for supported calls; native SDPA otherwise |

CuTe accepts FP16/BF16 BHSD tensors, head dimensions 1–512, GQA, boolean masks,
and dense or causal visibility on SM80/86/89/90/100/103/120/121. It supports
autograd, `torch.compile`, and CUDA graphs. Dropout, additive masks, FP32 inputs,
and packed variable-length attention are outside the CuTe API.

Automatic selection checks compatibility; benchmark your workload to choose the
fastest backend. Errors from a selected CuTe kernel propagate. See the
[API guide](docs/api.md) for masks, strides, scales, and fallback rules.

## Performance

Wide-head attention and selected GPU-specific schedules can outperform default
SDPA. Other shapes can favor SDPA. The [performance chart](results/performance.svg)
and [per-GPU reports](results/README.md) show measured wins, losses, and coverage
across twelve GPU targets, with historical and experimental runs labeled.
These are attention-call measurements; graph replay,
eager execution, inference, and training have distinct timing boundaries.

## Documentation

- [Installation and optional runtime](docs/installation.md)
- [API and supported inputs](docs/api.md)
- [GPU-specific selection and evidence](docs/README.md#gpu-guides)
- [Precision](docs/precision.md) and [determinism](docs/determinism.md)
- [Validation](docs/validation.md) and [benchmarking](docs/benchmarking.md)

## License and attribution

Developed in [SimpleTuner](https://github.com/bghira/SimpleTuner/pull/3256), under
[AGPL-3.0-or-later](LICENSE). [KohakuFA](https://github.com/KohakuBlueleaf/KohakuFA)
originated the precision investigation and arithmetic approach; its Gluon kernels
are not bundled. NVIDIA CUTLASS and FlashAttention helper adaptations retain their
BSD notices in [NOTICE](NOTICE) and [LICENSES/BSD-3-Clause.txt](LICENSES/BSD-3-Clause.txt).
