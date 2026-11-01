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

The optional [causal head-group control](causal-head-groups.md) changes backward
CTA traversal with identity group 1 as the default. Its native A40 graph/eager
measurements and full-device qualification are separate from automatic selection.

## Selected API validation and cold loader state

A selected native runner loads through `native.extension()`. This can leave the
strict API's separate `_extension()` loader cache cold indefinitely. Checking
that cache before selected dispatch therefore repeated generic Python validation
on every call, even though the bounded selector and native runner validate the
selected computation themselves.

The strict `scaled_dot_product_attention` entry now tries qualified selection
before that generic-cache check. Compilation and unselected calls retain their
existing validation. A selected execution failure still propagates.

Using the qualified stack in [the L40S report](l40s-warp-training.md), a
full-device L40S control used four independent copies of the exact old/new API
body, each with its own cold or explicitly warmed loader cache. All four used
the same shared GPU methods. The cold copies remained cold, the warmed copies
remained warm, and graph queries matched launched symbols and resources for all
32 rows. Nine rotating rounds used 100 ms measurement windows after 500 ms
warmup. Forward retained gradients; joint included forward and all gradients.

At B1/H16/S1024/D64, complete eager joint wall medians were 101.45 to 95.54 us
for BF16 and 101.54 to 96.27 us for FP16. The warmed old API measured 95.70 and
96.36 us respectively, isolating the avoidable cold-cache validation overhead.
Other families were largely device-bound or tied within variability. This is a
host-path improvement, with no claim of a GPU pipeline or model-level speedup.
[All samples and cache-state proof](../results/selected-validation.json) are
retained separately from the architecture timing grid.

Reproduce on a qualified selected device from a checkout containing the prior
API revision:

```sh
python -m benchmarks.compare_attention_variants \
  --operations '[{"label":"old-cold","factory":"benchmarks.selected_validation_controls:make_attention","kwargs":{"previous":true}},{"label":"new-cold","factory":"benchmarks.selected_validation_controls:make_attention"},{"label":"old-warm","factory":"benchmarks.selected_validation_controls:make_attention","kwargs":{"previous":true,"warm":true}},{"label":"new-warm","factory":"benchmarks.selected_validation_controls:make_attention","kwargs":{"warm":true}}]' \
  --seq 1024 4096 --dim 64 128 --visibility dense --phases forward joint \
  --warmup-ms 500 --window-ms 100 --repeats 9 --output ../selected-validation.json
```

The control defaults to prior API revision `969f1f9`. An explicit `control_file`
factory argument accepts a retained prior API source file for source-only
installations. Set `KOH_VALIDATION_CACHE_PROOF` to an output file to retain the
independent loader states and API source hashes at process exit.
