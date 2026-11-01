# Explicit native launch schedules

`AttentionRunner` retains a warp-attention schedule once and supplies fresh tensor descriptors and the current CUDA stream from C++ on each call. Its combined host entry launches forward, or backward preparation, the joint gradient kernel and gradient conversion. This removes repeated Python layout preparation while retaining the shared GPU methods and the arithmetic in [the precision policy](precision.md).

The explicit interface accepts aligned, contiguous, square FP16/BF16 MHA with D64 or D128 and a positive normal FP32 scale. It validates device family, configuration fields, pipeline geometry and alignment. Its cache distinguishes tensor layouts and schedule values; autograd retains the configuration even after the runner object is released. Backward allocation rounds scratch rows to the selected query tile, including M128 tails, and makes an aligned copy of an unaligned output gradient.

```python
from kohakufa_cute.native import extension

runner = extension().AttentionRunner("warp", {
    "forward_m": 64,
    "backward_n": 64,
    "query_stages": 1,
    "dout_stages": 1,
    "score_m_warps": 4,
    "kv_m_warps": 2,
    "dq_m_warps": 2,
    "v_in_regs": True,
})
output = runner(q, k, v, causal=True)
```

This experimental entry separates schedule qualification from automatic public selection. Each target and schedule needs its own correctness and complete-call measurements. Optional dependency versions remain those qualified by the shared package.

## A40 evidence

The retained comparison uses a full 84-SM A40, Python 3.13.18, Torch 2.15.0+cu132, CuTe DSL 4.8.1, TVM FFI 0.1.15.post0 and Quack 0.6.5. It retains the campaign stack to compare matching runtime and code-generation policies. Standalone FA2 2.8.4 is built from revision `4616f5116d33e3ba3cbb71d582caffcfcb49649c`; its actual loaded extension is identified in the result metadata.

Four GPU test methods cover 32 configuration frames, eight M128 tail cases, independent saved maxima/log-normalizers, outputs and every gradient, changed/poisoned graphs, cache isolation, unaligned output gradients and runner lifetime. Replays cover D64 Q2/N128 and D128 Q1/N64 in both dtypes, and custom positive scales are changed through the same cache entry. Focused memcheck, racecheck, synccheck and initcheck runs report zero device errors with Compute Sanitizer 2026.1.0.0. They include both graph geometries, M128 D64 tails and the scale/lifetime test; host API diagnostics are disabled with `--report-api-errors no`.

The benchmark retains five samples per backend, alternates backend order, warms compilation/allocation before a one-second sustained warmup, and sizes each measurement window to at least approximately 100 ms. Eager timing includes Python and autograd preparation. The graph control times synchronized replay windows. Forward uses inference mode through `torch.no_grad()`; joint includes forward and all three gradients. Profiling is separate from these uninstrumented samples.

[The compact results](../results/a40-configured-host.json) retain configuration values, source hashes and every timing sample. The selected D64 schedule uses N128, Q2/dO1 and a 2/4/4 MMA layout with resident V; D128 uses N64, Q1/dO1 and 4/2/2. Both use M64 forward tiles.

B1, H16, S4096, BF16; median synchronized graph replay windows:

| D | Visibility | Existing CuTe joint (ms) | Selected runner joint (ms) | SDPA (ms) | FA2 (ms) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 64 | dense | 3.848 | 2.690 | 2.651 | 2.551 |
| 64 | causal | 2.048 | 1.522 | 1.539 | 1.482 |
| 128 | dense | 6.873 | 5.074 | 5.289 | 5.023 |
| 128 | causal | 3.768 | 2.794 | 2.926 | 2.787 |

The long D128 joint comparison is close to FA2; D64 retains a larger backward gap. Host preparation matters most for short eager calls, while long graph comparisons expose GPU scheduling costs. Use both timing boundaries when assessing an application.

Reproduce an explicit schedule comparison from a qualified environment:

```sh
PYTHONPATH=src python3 -m benchmarks.compare_sm80_host \
  --seq 128 1024 4096 --dim 128 --dtype bf16 --rounds 5 \
  --configuration '{"forward_m":64,"backward_n":64,"query_stages":1,"dout_stages":1,"score_m_warps":4,"kv_m_warps":2,"dq_m_warps":2,"v_in_regs":true}' \
  --fa2 --output a40-host.json
```

Add `--graph` for the matched replay control. Large captures remain outside the tracked tree.
