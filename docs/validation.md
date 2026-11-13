# Validation

The [precision contract](precision.md) defines the computation on every GPU.
Validate output, dQ, dK, and dV independently. The representative timing matrix
uses per-head FP64 norm-error limits of 1% for BF16 and 0.5% for FP16. These
limits are not universal pointwise or bitwise-equality requirements.

`benchmarks.reference.rounded_reference` diagnoses the specified P/dS rounding
and uses the actual rounded O for backward delta. Its final FP64 gradient
products separate operand rounding from FP32 reduction error; they do not replay
tensor-core instruction order. Independent mathematical FP64 forward expectations
remain part of validation.

## Run checks

From the repository root, using Python 3.13:

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -v -f
python -m unittest tests.test_public_text -v
# On a supported GPU:
python -m unittest tests.test_kernels tests.test_baseline_precision \
  tests.test_key_precision tests.test_value_precision -v
python -m benchmarks.precision --dtype bf16 --seq 256 --dim 128 \
  --backends auto kohakufa-cute --output "$CAPTURE_DIR/precision.json"
```

Set `CAPTURE_DIR` to storage outside the checkout. Run relevant device tests and
sanitizers when changing kernels or memory ownership. CPU skips do not establish
GPU qualification.

For neighborhood attention, the [current guide](neighborhood/README.md) links
the independent geometry/reference checks and six-GPU results. The combined
dQ+dK experiment has its own prescribed equations and does not redefine the
default policy. GPU tests include `tests.test_neighborhood`,
`tests.test_neighborhood_precision` and `tests.test_neighborhood_key_precision`.
Run numerical CPU references and GPU work on the remote qualification host.

## Directed regressions

Preserve the [baseline](../tests/test_baseline_precision.py),
[key](../tests/test_key_precision.py), and [value](../tests/test_value_precision.py)
fixtures. They distinguish expected low-precision underflow and cancellation
from reconstruction, mask, indexing, or synchronization defects.

```bash
python -m unittest tests.test_adversarial -v
python -m benchmarks.validation.cudnn_regressions --reference rounded \
  --backends kohakufa-cute cudnn flash math --output "$CAPTURE_DIR/adversarial.json"
python -m benchmarks.validation.tracker_regressions --backends cute cudnn --frontend-plan \
  --output "$CAPTURE_DIR/tracker.json"
```

The cuDNN driver also accepts `--reference fp64` for mathematical diagnostics and
optional `frontend frontend-padding` backends with `nvidia-cudnn-frontend` installed.
Backend support/execution errors are separate from numerical failures.

Coverage includes negative maxima/LSE, partial tiles, full/key-padding masks,
all-hidden rows, single queries, GQA statistics, signed/zero scales, tiny
gradients, strides, cache transitions, and changed or poisoned graph storage.
On B200/B300, `tests.test_blackwell_launch_limits` checks batch/head launch-grid
boundaries. Finite loss alone is insufficient.

## Qualification limits

Each [GPU report](../results/README.md) retains its original accuracy gates,
source hashes, tool outcomes, and known failures. A focused pass does not
supersede a recorded broader-suite failure. In particular:

- GB10 query-owner studies retain full-suite failures separately from focused passes.
- B200 allocator and B300 attention initialization reports remain unresolved;
  see [the diagnostic](diagnostics/blackwell-allocator.md).
- The [H200 backend](experimental/h200.md) remains experimental with unresolved initcheck.
- Captured H100 Clef attention improvements did not pass the whole-model logit
  compatibility gate. They do not establish model throughput or task accuracy.
- Initial neighborhood qualification reproduced six ordinary-attention failing
  subcases on both the candidate and untouched base; see the
  [initial report](neighborhood/reports/initial.md). Later neighborhood passes
  do not establish a full ordinary-attention GPU-suite pass.

Previous compensated arithmetic is an explicit accuracy experiment. Its stricter
coordinate checks do not redefine the default contract. The retained
[evidence](../benchmarks/experiments/README.md) includes precision tradeoffs and
the model-shape audit; shape overlap alone does not qualify a model.
