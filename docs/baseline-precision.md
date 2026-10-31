# Default arithmetic and precision-removal validation

Default attention now uses a single FP16/BF16 probability operand, forms dS from
that rounded P and the FP32 output/dO delta, and converts dS once for FP32 gradient
accumulation. Attention scaling follows the gradient products. Separate unscaled
maxima and logarithmic normalization, shifted-score reconstruction, masking, and
tail predicates remain part of the computation. See the [equations](precision.md).

## What changed

The previous default precision policy added P/dS residual products, independent
high/low gradient accumulators, scaled FP16 residuals, and one compensated forward
normalizer. Query owners also computed a P-weighted-key product and a correction
based on the complete row. These operations pursued a stronger accuracy target
than single-rounded operand attention. They also replaced some tuned KV owners
with geometry chosen to fit the extra products.

The automatic residual-product replacement previously ran on RTX 3090 Ti, L40S
and RTX 5090, but not RTX 4090. The earlier relative performance comparisons
therefore included different arithmetic policies. The query correction was also
present on other architectures.

Default dispatch now retains the architecture schedules without that final
precision replacement. Query products omit the extra weighted-key GEMM and its
epilogue correction, KV products omit residual corrections/products, and the
unneeded correction-statistics slot is initialized without loading stale workspace.
Existing producer/consumer synchronization is retained when readers are removed.
The internal compensated owners remain available to the benchmark control.

The reference arithmetic was checked against the current Gluon implementation at
revision `0c5ede6859307bedc0c6089523c16b1ca51b7da9`. Its backward source is unchanged
from the earlier audited revision: P is converted to the input dtype before dS,
dS is converted once, delta is computed from rounded O/dO in FP32, and gradient
scaling follows FP32 accumulation.

## Accuracy boundaries

The original mathematical fixtures and FP64 references are retained. Tests of the
default now distinguish implementation errors from operand rounding. A second reference
implements the intended P/dS conversions, accepts the actual O to isolate backward,
uses IEEE FP32 score/dP products, and uses FP64 final gradient products to expose
the separate FP32 accumulation floor. Reduction order can differ from the deployed
tensor-core instructions.

Two retained D128 fixtures illustrate why unrounded coordinate gates demanded more
work. The values below describe the equations, before final output rounding:

| Fixture | Unrounded FP64 dV coordinate | Single-rounded-P coordinate |
| --- | ---: | ---: |
| BF16 cancellation, 1,024 square tokens | −0.01824 | +0.37164 (final BF16: +0.37109) |
| FP16 cancellation, 1,024 square tokens | −0.01824 | +0.00871 |
| FP16 probability underflow, 1,024 square tokens | +0.67997 | 0 |

The underflow fixture uses near-maximum finite FP16 dO. A representable final
unrounded gradient does not imply that its intermediate probability is
representable. The sign changes and zero above are expected under the declared
operand conversion; residual products recover a different, more accurate result.
They are useful accuracy experiments, with a separately measured cost.

The representative D16/64/128/256/512 timing matrix retains independent per-head
FP64 norm-error limits of 1% for BF16 and 0.5% for FP16. Dedicated directed
fixtures retain coordinate checks against the declared rounded equations.
Architecture checks retain their existing component norm limits. Large common
offsets and sharp signed scales use those limits because FP32 reduction order
before P/dS conversion can change a cancelled coordinate. Mathematical FP64 errors
remain separate diagnostics. Masked/hidden rows, tails, changed graph inputs,
cache reuse and poisoned storage retain their own checks.

A BF16 B2/H16/D128 rectangle with 1,023 or 1,027 queries and 17 keys also demonstrates
the distinction on ordinary random operands: the previous pointwise FP64 check
reports two coordinate failures, while the rounded reference has none. Independent
FP64 component norm errors remain approximately 0.20–0.30%. This is an operand
rounding effect, rather than evidence that the query/key boundary lost data.

## SimpleTuner model-shape audit

The source audit used [SimpleTuner revision
`7ffd66dd5f4c170992a394c86c03784490314899`](https://github.com/bghira/SimpleTuner/tree/7ffd66dd5f4c170992a394c86c03784490314899/simpletuner/helpers/models). Paths below are relative to its repository.
These are constructor defaults and source-derived sequence examples, not captured
activations or checkpoint quality measurements.

| Model sources under `simpletuner/helpers/models` | Default heads / head dimension | Relevant geometry |
| --- | --- | --- |
| `flux/transformer.py`, `moss_image/transformer.py`, `qwen_image/transformer.py` | 24 / 128 | Matches the D128 cancellation family; the model-head-count test repeats it with H24. FLUX and MOSS concatenate image/text tokens into attention sequences. |
| `hunyuanvideo/transformer.py`, `anima/transformer.py` | 16 / 128 | Matches both H16 and D128 in the earlier fixtures and comparison grid. Video/image and text lengths vary. |
| `qwen_image/transformer_21.py` | 32 / 128 | Matches D128; the model-head-count test also exercises H32. |
| `ltxvideo/transformer.py` | 32 / 64 | Matches D64; the model-head-count test exercises H32. Video token count varies with the spatial/temporal grid. |
| `sd3/transformer.py` | 18 / 64 | Matches D64; the model-head-count test exercises H18. The 128×128 latent sample and 2×2 patches give 4,096 image tokens; 77 CLIP plus 256 T5 text tokens give a 4,429-token joint example. |
| `auraflow/transformer.py` | 12 / 256 | Matches D256, with H12 rather than the H16 timing preset. A 64×64 latent sample, 2×2 patches, eight registers and 256 text tokens give a 1,288-token joint example. |
| `ltxvideo2/transformer.py` | Video 32 / 128; audio 32 / 64 | Both tested head-width families occur; text and video/audio sequence lengths differ. |

The FLUX packing recipe gives 1,024 image tokens plus 512 text tokens at 512-pixel
square input, and 4,096 plus 512 at 1,024 pixels: joint lengths 1,536 and 4,608.
Those lengths differ from an image-only 1,024/4,096-token benchmark. Variable text,
reference tokens, and masks can also introduce partial tiles.

A packed training sequence contains many query rows; it is not the single-query
reproducer. Packing does not remove the per-row probability reconstruction or
operand-rounding rules. The head/dimension overlaps above establish geometric
relevance, without establishing that a model produces the synthetic cancellation
values. No D512 default is asserted for these constructors.

## Measurements and verification

The [four refreshed RTX/L40S grids](../results/README.md#rtx-cards-and-l40s)
cover 240 FP16/BF16 dense/causal cases. The full H100 PCIe and B300 additionally
cover 16 BF16 cases each: B1/H16, S128/S4096, D64/128/256/512, both visibility
modes. See [H100 PCIe](../results/h100-pcie.json) and [B300](../results/b300.json).

The [native-device table](../results/README.md#native-h100-pcie-and-b300) records
representative latencies and both complete-call baselines.

The H100 is a full 114-SM PCIe/80-GB device at 350 W; B300 is a full 148-SM
SXM6 AC device at 1,100 W. These are not the earlier H100 SXM or B300 MIG hosts.
Both use Python 3.13.18, PyTorch 2.15.0+cu132, CuTe DSL 4.8.1 and Triton 3.9.0.
Five rotating rounds use at least 16 warmed CUDA-graph replays per phase,
targeting 100 ms. Source hashes, samples, selected SDPA graph kernels, resources
and loaded library names are retained. A deletion can still be a slowdown:
B300 D64 backward regresses approximately 5–8% at S4096 in this run.

The benchmark-only previous-policy reconstruction was checked for exact numerical
equality against actual pre-change outputs/gradients from revision `183f4f5`, in separate processes
on the same machine/compiler. Native devices check nine fixtures, including the
wide BF16 D512 owner. The consumer control checks span all five head widths and
both dtypes. These compare different arithmetic policies, not an isolated
pipeline improvement.

Matched H100 PTX/cubins reproduce the exact hashes retained during timing.
The BF16 D128 query kernel retains grid `[64,16,1]` and 99,328 dynamic shared
bytes, while registers/thread fall from 245 to 185. The emitted body retains 24
64×64×16 HGMMA instructions and reduces 64×128×16 product instructions from eight
to four. The latter removes the weighted-key product. These static counts are
not percentages of runtime. Systems traces independently show the forward,
query and KV launch sequence, with the same per-kernel register allocations.
Compute counters were denied on H100 (`ERR_NVGPUCTRPERM`); no stall, cache-hit,
HFU or achieved-occupancy percentages are inferred.

RTX 3090 Ti, RTX 4090, RTX 5090, L40S and H100 pass focused device memcheck,
racecheck and synccheck probes. These include generic masked/tail and wide paths;
additional probes cover the retained 3090 Ti query-major and 5090 D64 owners.
The broad core suites, model-head-count checks, changed/poisoned graphs and
directed arithmetic checks also pass on these devices.
Device sanitizer checks use `--report-api-errors no`; they do not qualify
unrelated driver API probes. B300 device memcheck passes, but racecheck reports
hazards in its register-aux KV and pipeline KV owners and an unchanged forward
path. The preceding checkout reproduces the same backward hazard classes
(33 grouped reports before, 24 after), and three grouped forward reports.
Those observations do not establish false positives or race-clean native
Blackwell execution. They remain a separate synchronization investigation;
numerical timing and graph checks do not prove race freedom.

B200/SM100 and GB10/SM121 were unavailable for direct requalification. Shared
source changes reach them; historical timings do not qualify their new default.
No checkpoint-level quality or end-to-end model-speedup claim is made here.

Reproduce the arithmetic comparison with:

```sh
python -m unittest tests.test_baseline_precision tests.test_key_precision tests.test_value_precision -v
python -m benchmarks.qualify_baseline_precision --seq 128 4096 \
  --dim 16 64 128 256 512 --heads 16 --dtype bf16 \
  --compare-previous --output precision-comparison.json
```

`--compare-previous` is a benchmark-only control. It restores the preceding
precision policy on the same device and software stack. Its outputs must first
be checked against the preceding checkout; the retained campaign performs that
check in separate processes before using same-process timing comparisons. These
are complete implementation comparisons with different arithmetic policies,
rather than an isolated scheduling speedup.

Check the preceding-policy control before interpreting its ratios. From this
source checkout, with a separate checkout of `183f4f5` at `../previous`:

```sh
PYTHONPATH=../previous/src:. python -m benchmarks.verify_previous_precision \
  --fixtures ../precision-control
python -m benchmarks.verify_previous_precision --fixtures ../precision-control --previous
```

The [original-Gluon comparison](../benchmarks/compare_upstream_precision.py)
records both FP64 and rounded errors and actual launched kernels. It requires
an installed checkout of the pinned upstream revision. The qualification uses
Triton 3.7.0 in a separate import environment because Triton 3.9 removed the
`float2` API used by that source; these are arithmetic controls, not timings
across different stacks. The D8 tracker fixture is explicitly zero-embedded to
D16 with its original scale to meet the original TMA kernel's width requirement.

```sh
python -m benchmarks.compare_upstream_precision --upstream-checkout ../KohakuFA \
  --output ../upstream-precision.json
python -m benchmarks.cudnn_regressions --quick --reference rounded \
  --backends kohakufa-cute --output ../rounded-regressions.json
python -m benchmarks.cudnn_regressions --quick --reference fp64 \
  --backends kohakufa-cute --output ../mathematical-diagnostics.json
```

The regression command applies the selected reference to every backend and
retains both assessments. It exits unsuccessfully when the selected check fails.
On H100, all 24 quick cases match the declared contract, while ten fail the former
mathematical/edge-row criterion. BF16 issue160, for example, has about 31% dQ
norm error against FP64. This is an explicit accuracy tradeoff, not evidence
that those gradients became FP64-accurate. The compensated computation can
remain useful where that stronger mathematical target is required.

The [retained direct control](../results/precision-contract.json) contains 30
comparisons with actual Gluon forward/backward launches. It reproduces the same
BF16 cancellation coordinate (+0.37109), FP16 underflow zero, and approximately
31% BF16 issue160 dQ error in both implementations. The latter uses the documented
D8-to-D16 zero embedding and original scale. Equal arithmetic expectations do not
imply byte-identical outputs on every shape or FP64-level gradient accuracy.
