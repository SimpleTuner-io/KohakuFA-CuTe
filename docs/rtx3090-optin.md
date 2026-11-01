# Ordinary RTX 3090 opt-in

`rtx3090_attention` exposes the qualified native-host warp schedules without
changing ordinary attention dispatch:

```python
from kohakufa_cute import rtx3090_attention

out = rtx3090_attention(q, k, v, is_causal=True)
```

Use the `sm80-fa4` extra with Python 3.13, PyTorch 2.15.0/CUDA 13.2, and the
pinned optional runtime described in the [A100 installation](a100-sm80-fa4.md).
The opt-in selects a full 82-SM GeForce RTX 3090 for contiguous, aligned square
FP16/BF16 attention with batch 1, 16 equal heads, D64/D128, and 1024–4096 tokens.
It supports dense and causal training. Other calls retain ordinary CuTe
dispatch, including masks, GQA layouts, signed/zero scales and compiled calls.

The schedule uses M64/N64 forward and Q1/dO1 backward. Dense D64 uses a
128-key backward tile; causal D64 and D128 use 64 keys. V stays in registers.
The input-dtype P/dS conversions, actual rounded-output delta, FP32 products,
unscaled maximum/lognorm reconstruction and late dQ/dK scale follow the normal
[precision contract](precision.md).

This entry is for an ordinary RTX 3090. RTX 3090 Ti, other SM86 devices and GPU
partitions require their own qualification. The faster Ada M32 shared-V preset
regressed long backward on this card and is not selected.

Reproduce the public-entry correctness and matched comparisons with:

```sh
capture_dir=$(mktemp -d)
python -m benchmarks.validate_rtx3090_optin --compare-reference-devices --output "$capture_dir/qualification.json"
python -m benchmarks.compare_rtx3090_optin --source-revision "$(git rev-parse HEAD)" --output "$capture_dir/comparison/report.json"
```

The validator checks independent per-head FP64 norms, saved statistics,
partial tiles, CPU/CUDA reference agreement, and changed/poisoned graph replay.
Sanitizer reference calculations may run on CPU only after the ungated
CPU/CUDA agreement check passes; the same acceptance limits apply. Comparison
reports retain graph and eager samples, exact source/library identities,
selected kernel resources and compiler-returned code. Raw captures are kept
outside the tracked tree.

## Measured comparison

The retained [16-case public-entry comparison](../results/rtx3090-optin.json)
uses one full 82-SM RTX 3090, batch 1, 16 heads, S1024/4096, D64/128, both
dtypes and both visibility modes. It compares the actual public entry, ordinary
CuTe, standalone FA2 and the selected SDPA backend in the same process. Seven
rotating 100-ms graph and synchronized eager windows retain all samples;
profiling is separate. The source, loaded-library hashes and device metadata
are included in the data.

Ratios below are baseline time divided by public-entry time; geometric means
cover all 16 cases. Wins count measured medians rather than statistical
significance.

| Timing | Phase | Ordinary CuTe wins / ratio | FA2 wins / ratio | SDPA wins / ratio |
| --- | --- | ---: | ---: | ---: |
| Graph | Forward | 16/16 / 1.149× | 14/16 / 1.089× | 16/16 / 1.090× |
| Graph | Backward | 16/16 / 1.388× | 3/16 / 0.989× | 11/16 / 1.044× |
| Graph | Combined | 16/16 / 1.321× | 7/16 / 1.019× | 14/16 / 1.064× |
| Eager | Forward | 16/16 / 1.140× | 14/16 / 1.150× | 16/16 / 1.078× |
| Eager | Backward | 16/16 / 1.349× | 7/16 / 1.178× | 11/16 / 1.053× |
| Eager | Combined | 14/16 / 1.218× | 11/16 / 1.202× | 13/16 / 1.037× |

FA2 backward remains faster on most graph cases. For long causal D128, public
combined graph time is 3.801/3.846 ms (BF16/FP16), versus FA2 3.652/3.680 ms
and SDPA 3.764/3.794 ms. Short causal D64 is stronger: 0.193/0.194 ms, versus
FA2 0.237/0.240 ms. These measurements qualify this explicit entry; they do not
establish a win for other SM86 cards or change default dispatch.

Actual public-entry validation passed 24 cases, including partial tiles,
saved statistics, CPU/CUDA reference agreement and changed/poisoned replay.
Each of memcheck, racecheck, synccheck and initcheck passed 16 selected cases
with zero reported errors. The original racecheck timed out after 15 cases;
its partial run is retained as incomplete. An unchanged-source recovery reran
racecheck with a longer time bound, completed the remaining tools and measured
the comparison. No acceptance limit or attention arithmetic changed. CPU
checks ran 272 tests with 156 GPU skips; the 24 publication-guard tests passed.
