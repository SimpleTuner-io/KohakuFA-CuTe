# KohakuFA-CuTe

CuTe DSL attention for **RTX 3090 Ti, RTX 4090, RTX 5090, B200/B300, GB10, H100, and L40S**, with stable backward gradients when attention becomes sharp. It brings [KohakuFA's](https://github.com/KohakuBlueleaf/KohakuFA) precision approach to Ampere, Ada, Hopper, and Blackwell, and adds a centered FP32 correction that retains tiny saturated gradients.

## When to use it

- **Training with sharp attention or large logits:** forward outputs can look correct while query/key gradients are wrong. The kernels preserve separate softmax statistics and a centered backward correction. See [precision](docs/precision.md).
- **RTX cards with 512-wide heads:** forward plus backward beats default PyTorch SDPA across all eight tested FP16/BF16, dense/causal, S1024/S4096 cases per card: **1.84–4.41× on RTX 5090**, **2.09–4.32× on RTX 4090**, and **1.67–2.53× on RTX 3090 Ti** (285 W).
- **RTX 4090 short D128 and long D16:** forward plus backward beats SDPA in both dtypes and visibility modes at S128/D128 and S1024/S4096/D16. All 20 short cases across D16/64/128/256/512 win; see the [4090 comparison](docs/rtx4090-tuning.md) for timings and sustained-load checks.
- **L40S with 512-wide heads:** forward plus backward is **2.50–4.20× faster than default SDPA** across the same eight tested FP16/BF16, dense/causal, S1024/S4096 cases.
- **L40S text inference with Clef-Flash:** the measured BF16 D64 schema attention calls are **1.05–1.59× faster than SDPA**. D256 causal GQA is **1.13× faster at 787 tokens** and approximately tied at 9,835 tokens. These are attention-call gains; see the [model comparisons](results/README.md#clef-flash-text-inference).
- **H100 text inference with Clef 27B:** captured BF16 D256 backbone calls beat cuDNN SDPA by **1.24× at 787 tokens and 1.03× at 9,835 tokens**; schema calls beat it by **1.61–4.28×**. Small shapes and whole-model numerical compatibility still need work; see the [27B comparison](results/README.md#clef-27b-text-inference).
- **RTX 5090 with explicit boolean masks:** the tested BF16 causal masks are **1.48–4.72× faster than default SDPA**, including mask preparation. Arbitrary masks need their own benchmarks.
- **Other workloads:** choose it for its gradient calculation, then benchmark your shapes. Long D64/D128/D256 complete calls still trail SDPA on RTX 4090; smaller heads on the other RTX cards also have gaps. The measured H100 training cases trade throughput for precision; L40S comparisons are below.

Automatic selection checks compatibility; it does **not** choose whichever backend is fastest for a shape.

## Performance

Attention-call latency, **forward + backward**, in milliseconds; lower is better. B=1, H=16, BF16, S=4096. These are matched comparisons within each row, with compilation excluded.

| GPU | Head dim | Visibility | CuTe | Default SDPA | SDPA / CuTe |
| --- | ---: | --- | ---: | ---: | ---: |
| RTX 5090 | 512 | Dense | 23.175 | 48.215 | 2.08× |
| RTX 5090 | 512 | Causal | 11.449 | 29.683 | 2.59× |
| RTX 4090 | 512 | Dense | 32.045 | 67.136 | 2.10× |
| RTX 4090 | 512 | Causal | 16.225 | 39.596 | 2.44× |
| RTX 4090 | 16 | Dense | 0.718 | 0.877 | 1.22× |
| RTX 4090 | 16 | Causal | 0.558 | 0.605 | 1.08× |
| RTX 3090 Ti | 512 | Dense | 68.507 | 114.648 | 1.67× |
| RTX 3090 Ti | 512 | Causal | 33.843 | 63.758 | 1.88× |
| L40S | 512 | Dense | 26.523 | 67.186 | 2.53× |
| L40S | 512 | Causal | 14.014 | 37.417 | 2.67× |
| RTX 5090 | 256 | Boolean causal mask | 6.052 | 24.294 | 4.01× |
| RTX 5090 | 128 | Boolean causal mask | 2.318 | 10.050 | 4.33× |
| RTX 5090 | 128 | Dense | 3.669 | 2.657 | 0.72× |
| L40S | 128 | Dense | 4.648 | 3.363 | 0.72× |
| H100 | 128 | Dense | 1.300 | 0.927 | 0.71× |

On the **B300 MIG 1g.34gb (18 SMs)**, BF16 D512 causal forward + backward is approximately tied with upstream Gluon at S4096 (**13.649 vs 13.703 ms**) and slightly faster at S8192 (**50.766 vs 51.276 ms**). Smaller cases still favor Gluon. These timings describe the partition; current full-device B200/B300 performance has not been remeasured.

**GB10 (DGX / RTX Spark, SM121)** has tuned D512 forward/backward and D256/D512 short-sequence forward paths. See the [GB10 comparison](results/README.md#gb10-dgx--rtx-spark) for all tested head dimensions, dense/causal results, and remaining limitations.

[Performance details and charts](results/README.md) cover both dtypes, forward-only results, actual SDPA backends, measurement conditions, and reproduction. Machine-readable measurements retain the [October 30 RTX 4090 run](results/rtx4090.json) and [other device runs](results/performance.json), including their different stacks and dates. These are attention benchmarks, not end-to-end training speedups.

## Install

Install a CUDA-enabled **PyTorch 2.11+**, then:

```bash
git clone https://github.com/SimpleTuner-io/KohakuFA-CuTe.git
cd KohakuFA-CuTe
python -m pip install '.[cuda]'
```

Requires Python 3.12+, CUTLASS DSL 4.8, TVM FFI 0.1.14+, Triton 3.6+, Ninja, and a C++17 compiler. The native launcher builds on first use. `pip install .` installs the API without CUDA extras; automatic selection can then use native SDPA on CPU/MPS.

**GB10 (SM121, DGX / RTX Spark)** is validated on Linux ARM64. Install an ARM64
CUDA-enabled PyTorch build and the CUDA 13 DSL libraries:

```bash
python -m pip install '.[cuda]' 'nvidia-cutlass-dsl[cu13]>=4.8,<4.9'
export CUTE_DSL_ARCH=sm_121a
```

Validation used Python 3.13, PyTorch 2.11/CUDA 13, and CUTLASS DSL 4.8.1.

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

- CUDA SM86/89/90/100/103/120/121; matching FP16 or BF16 Q/K/V; head dimensions **1–512**.
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

Directed [cuDNN regression cases](benchmarks/adversarial.py) cover negative row
maxima/LSE, partial query/key tiles, the reported single-query and tiny-gradient
inputs, masked rows, GQA, and signed/zero scales. Compare each forced backend
with centered FP64 equations on identical rounded inputs:

```bash
python -m unittest tests.test_adversarial -v
python -m benchmarks.cudnn_regressions --backends kohakufa-cute cudnn flash math \
  --output adversarial.json
```

The comparator checks first/last rows separately and uses norm-relative gradient
errors so small gradients cannot disappear under a loose absolute tolerance.
Optional `--backends frontend frontend-padding` additionally tests cuDNN Frontend
directly and its full-length padding-mask control; install `nvidia-cudnn-frontend`
for those modes. Backend execution/support errors are recorded separately from
numerical failures, and any CuTe failure makes the command exit unsuccessfully.

Additional tracker regressions exercise large batch/head launch grids, warmed
cache reuse across lengths and strides, NaN-poisoned storage outside live KV
views, masked nonpositive scales, and every head's saved GQA statistics:

```bash
python -m benchmarks.tracker_regressions --backends cute cudnn --frontend-plan \
  --output tracker-regressions.json
```

Each test runs in an isolated process with a timeout so native backend crashes
are retained as results. The optional Frontend control builds only the reported
single-head/single-query backward plan; it does not establish numerical parity.
Dense view/cache cases adapt the reported contracts without adding a ragged or
execute-time shape-override API to CuTe.
Run `python -m unittest tests.test_blackwell_launch_limits -v` on B200/B300
to force native TMEM, fused, wide-backward, and split-forward launch paths
across the batch/head boundary.
The cases derive from NVIDIA/cudnn-frontend issues
[1573](https://github.com/NVIDIA/cudnn-frontend/issues/1573) (launch limits),
[1567](https://github.com/NVIDIA/cudnn-frontend/issues/1567),
[1570](https://github.com/NVIDIA/cudnn-frontend/issues/1570), and
[624](https://github.com/NVIDIA/cudnn-frontend/issues/624) (live views/cache storage),
[1428](https://github.com/NVIDIA/cudnn-frontend/issues/1428) and
[1435](https://github.com/NVIDIA/cudnn-frontend/issues/1435) (nonpositive scales),
[1071](https://github.com/NVIDIA/cudnn-frontend/issues/1071) (single query),
[1083](https://github.com/NVIDIA/cudnn-frontend/issues/1083) (GQA statistics), and
[100](https://github.com/NVIDIA/cudnn-frontend/issues/100) (gradient strides).
CuTe flattens batch/head products above 65,535 onto grid-x for attention,
mask packing, and backward delta preparation; smaller launches keep their
existing grid layout.

## License and attribution

Developed in [SimpleTuner](https://github.com/bghira/SimpleTuner/pull/3256), under [AGPL-3.0-or-later](LICENSE). [KohakuFA](https://github.com/KohakuBlueleaf/KohakuFA) originated the precision investigation and arithmetic approach; its Gluon kernels are not bundled. NVIDIA CUTLASS and FlashAttention helper adaptations retain their BSD notices in [NOTICE](NOTICE) and [LICENSES/BSD-3-Clause.txt](LICENSES/BSD-3-Clause.txt).
