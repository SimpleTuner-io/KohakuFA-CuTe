# Measurements

This directory contains source measurements and summaries for the README tables.

Upstream KohakuFA values are attributed to its published B300 results at revision
`efd5b72622ce04fcafbe6ca3e368649d8b7b9527`. H100/L40S measurements use this project's
CuTe implementation and compare with the native PyTorch implementations on each GPU.

```bash
python -m benchmarks.precision --dtype bf16 --seq 256 --dim 128 \
  --backends auto cudnn kohakufa-cute --output results/precision.json
python -m benchmarks.speed --dtype bf16 --seq 1024 4096 --dim 64 128 256 \
  --iterations 100 --repeats 6 --output results/speed.json
```

`forward_backward_ms` includes the forward pass. CUDA-graph timing excludes compilation
and rotates backend order each repetition. Precision references use identical rounded
inputs and an analytically centered FP64 backward. Relative errors near saturation
must be read with the reference norm and cosine, which are retained in the JSON.

The SimpleTuner block table measures complete compiled, checkpointed LoRA optimizer
steps on H100. Its source is retained in `simpletuner-training-blocks.json`. These
measurements use production block dimensions, random weights, and learning rate zero.

Generate the precision figure from these measurements:

```bash
python -m pip install '.[plots]'
python -m benchmarks.plot
```

`validation.json` records the kernel regression suite and the API/reference/training
suite separately. All 37 tests are covered on H100 and L40S, with the MPS-only test
skipped on CUDA machines. Local Apple validation also covers MPS native output and
gradient parity.
