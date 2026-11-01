# KohakuFA-CuTe

CuTe DSL attention for **A100, A40, RTX 3090 Ti, RTX 4090, RTX 5090, B200/B300, GB10, H100, and L40S**, using [KohakuFA's](https://github.com/KohakuBlueleaf/KohakuFA) shifted-score reconstruction and gradient-scale placement. Tensor-core products use FP16/BF16 operands with FP32 accumulation; see the [arithmetic contract](docs/precision.md).

## When to use it

- **Sharp attention or large logits:** separate unscaled softmax statistics, shifted-score reconstruction and gradient scaling after accumulation address lost normalization information and premature underflow. The default uses single-rounded FP16/BF16 operands; see the [precision contract](docs/precision.md).
- **512-wide heads:** forward plus backward beats default SDPA in all eight tested long FP16/BF16 dense/causal calls per card: RTX 5090 **1.96–4.71×**, RTX 4090 **2.17–4.57×**, RTX 3090 Ti **1.69–2.85×**, L40S **2.80–4.80×**.
- **Other attention shapes:** benchmark the complete call. Short shapes can benefit; long smaller-head calls still have substantial gaps against SDPA. The [full grid](results/README.md#rtx-cards-and-l40s) shows both wins and losses.
- **H100 Clef 27B text inference:** retained D256 backbone measurements beat cuDNN SDPA by 1.24× at 787 tokens and 1.03× at 9,835 tokens. Those forward-only measurements and their model-compatibility limits are [documented separately](results/README.md#clef-27b-text-inference).

Automatic selection checks compatibility; it does **not** predict the fastest backend.

## Performance

**Forward + backward**, milliseconds; lower is better. B1/H16, BF16, S4096,
dense attention. The RTX 5090 D128 row uses the [pinned schedule](docs/rtx5090-configured-host.md);
other rows retain the existing baseline. All use the default rounded arithmetic.
Compilation is excluded.

| GPU | Head dim | CuTe | Default SDPA | SDPA / CuTe |
| --- | ---: | ---: | ---: | ---: |
| RTX 5090 | 512 | 22.354 | 50.147 | 2.24× |
| RTX 4090 | 512 | 31.159 | 67.562 | 2.17× |
| RTX 3090 Ti | 512 | 61.916 | 104.654 | 1.69× |
| L40S | 512 | 24.781 | 71.017 | 2.87× |
| RTX 5090 (pinned schedule) | 128 | 2.493 | 2.664 | 1.07× |
| L40S | 128 | 3.132 | 2.663 | 0.85× |

The L40S D128 row predates the bounded [shared warp training update](docs/l40s-warp-training.md),
which retains its separate paired native-CuTe/FA2 measurements.

[Performance details and charts](results/README.md) cover both dtypes, dense/causal
visibility, all five head widths, forward/backward phases and actual SDPA kernels.
The [precision-removal report](docs/baseline-precision.md) includes the full H100
PCIe/B300 checks, previous-policy comparison and model-shape audit. These are
attention benchmarks, not end-to-end training speedups.

The explicit [experimental H200 WGMMA backend](docs/h200-experimental.md) includes
joint-backward, native-host and FA3 comparisons, with its validation limits.

**GB10 (DGX / RTX Spark, SM121)** has D512 and short-forward schedules; its
[recorded comparison](results/README.md#gb10-dgx--rtx-spark) predates this arithmetic
change and was not requalified in this campaign.

## Install

Install a CUDA-enabled **PyTorch 2.11+**, then:

```bash
git clone https://github.com/SimpleTuner-io/KohakuFA-CuTe.git
cd KohakuFA-CuTe
python -m pip install '.[cuda]'
```

Requires Python 3.12+, CUTLASS DSL 4.8, TVM FFI 0.1.14+, Triton 3.6+, Ninja, and a C++17 compiler. The native launcher builds on first use. `pip install .` installs the API without CUDA extras; automatic selection can then use native SDPA on CPU/MPS.

**Tuned A100 and L40S training:** use Python 3.13 and PyTorch 2.15.0 with CUDA 13.2,
then install `python -m pip install '.[cuda,sm80-fa4]'`. This enables the bounded
[A100](docs/a100-sm80-fa4.md) and [L40S](docs/l40s-warp-training.md) D64/D128 schedules.
Other CUDA calls retain the standard CuTe dispatch.

**A40 opt-in:** the same qualified stack supports an explicit
[A40 entry for dense D128 and causal D64/D128](docs/a40-native-predicate.md),
with separate graph and complete eager measurements.

The same pinned installation provides an opt-in [RTX 4090 joint-backward path](docs/rtx4090-shared-joint.md), with measured D64/D128 schedules and ordinary CuTe fallbacks.

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

- CUDA SM80/86/89/90/100/103/120/121; matching FP16 or BF16 Q/K/V; head dimensions **1–512**.
- Four-dimensional BHSD tensors, including supported strided layouts; equal K/V shapes; query heads divisible by K/V heads.
- Dense, token-causal, or broadcastable boolean masks (`True` means visible). All-hidden rows return zero.
- GQA, eager/compiled autograd, CUDA graphs, and signed or zero scales.

The SDPA wrappers require `enable_gqa=True` for different head counts. With an explicit mask, fold causal visibility into that mask; `attention` also accepts `mask` and `causal=True` together. FP32, dropout, additive masks, unsupported devices/shapes, and missing CUDA dependencies select native SDPA through the automatic API. There is no packed variable-length API.

The optional tuned installation selects bounded [A100](docs/a100-sm80-fa4.md),
[L40S training](docs/l40s-warp-training.md), and [RTX 5090 training](docs/rtx5090-configured-host.md)
D64/D128 schedules. Their shape and software bounds are documented; other CUDA calls
retain standard CuTe dispatch.

GB10 has an [explicit D128 shared entry](docs/gb10-shared.md) with measured shape/dtype schedules and ordinary CuTe fallbacks. It is opt-in.

`attention(..., center_keys=True)` subtracts each batch/head's mean key in FP32 before rounding once back to the input dtype. This can reduce errors from a shared key component, at the cost of a reduction and temporary storage. It defaults to `False` and is separate from the default output/dO delta calculation.

## Validate and benchmark

```bash
python -m unittest discover -v -f
python -m benchmarks.speed --dtype bf16 --seq 1024 4096 --dim 64 128 256 512 \
  --backends sdpa cute --iterations 100 --repeats 6 --output speed.json
python -m benchmarks.precision --dtype bf16 --seq 256 --dim 128 \
  --backends auto kohakufa-cute --output precision.json
```

Tests cover FP64 and declared-operand gradient comparisons, tiny gradients, masks, GQA, head tails, strides, compiled backward, checkpointed LoRA updates, streams, and changed-input graph replay. The [benchmark guide](results/README.md#reproduce) includes causal and upstream comparisons.

Directed [cuDNN regression cases](benchmarks/adversarial.py) cover negative row
maxima/LSE, partial query/key tiles, the reported single-query and tiny-gradient
inputs, masked rows, GQA, and signed/zero scales. Compare each forced backend
with centered FP64 equations on identical rounded inputs:

```bash
python -m unittest tests.test_adversarial -v
python -m benchmarks.cudnn_regressions --reference fp64 --backends kohakufa-cute cudnn flash math \
  --output adversarial.json
```

The comparator checks first/last rows separately and uses norm-relative gradient
errors so small gradients cannot disappear under a loose absolute tolerance.
`--reference rounded` checks the default operand contract instead; reports retain both assessments. The [precision report](docs/baseline-precision.md) documents synthetic cases that fail mathematical checks after removing compensation.
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
