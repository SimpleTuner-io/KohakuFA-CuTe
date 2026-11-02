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
FP16/BF16 attention with batch 1, 16 equal heads, D64/D128, and 1024–4096 tokens. It also selects a
[full-row backward kernel for S128 D16](rtx3090-short.md) under the same
batch/head, dtype and device restrictions.
It supports dense and causal training. Other calls retain ordinary CuTe
dispatch, including masks, GQA layouts, signed/zero scales and compiled calls.

The schedule uses M64/N64 forward and Q1/dO1 backward. Dense D64 uses a
128-key backward tile; causal D64 and D128 use 64 keys. V stays in registers.
Causal backward orders key tiles across groups of 16 heads at exactly 1024
tokens and groups of 4 at longer eligible lengths.
Dense ordering remains unchanged. Earlier key tiles require more query-block
iterations in causal backward; grouping changes their launch order across
heads. The selected group sizes come from measured complete calls. This
changes CTA assignment, without changing operand conversions, shared-memory
ownership or barriers.
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
selected kernel resources and the availability of compiler-returned code.
Raw captures are kept outside the tracked tree.

## Measured comparison

The retained [32-case public-entry comparison](../results/rtx3090-optin.json)
uses one full 82-SM RTX 3090, batch 1, 16 heads, S1024/1025/2048/4096,
D64/128, both dtypes and both visibility modes. It compares the actual public
entry, the previously published group-1 entry, ordinary CuTe, standalone FA2
and SDPA in the same process. SDPA selected PyTorch Flash Attention in all
32 cases, verified from the launched `pytorch_flash` kernel symbols. FA2
and SDPA measurements include their own preparation and arithmetic; the
group-1 comparison isolates scheduling under the same CuTe contract. The
group-1 control loads its exact published Python source once, with no
per-call patching. Seven
rotating 100-ms graph and synchronized eager windows retain every sample.
Evaluation uses a different seed from schedule selection; S1025 and S2048 were
held out from the schedule sweep.

The table covers the 16 causal cases. Each cell gives candidate wins over the
baseline, followed by the geometric mean of baseline time / candidate time.
Wins count measured medians, rather than statistical significance.

| Timing | Phase | Group 1 | FA2 | SDPA |
| --- | --- | ---: | ---: | ---: |
| Graph | Backward | 16/16 / 1.130× | 14/16 / 1.153× | 16/16 / 1.242× |
| Graph | Combined | 16/16 / 1.085× | 14/16 / 1.142× | 16/16 / 1.210× |
| Eager | Backward | 14/16 / 1.097× | 14/16 / 1.394× | 16/16 / 1.222× |
| Eager | Combined | 15/16 / 1.071× | 14/16 / 1.341× | 16/16 / 1.148× |

Causal combined graph time improves by 2.4–12.1% over group 1. All 32 dense
and causal combined graph medians beat this run's SDPA backend; 22/32 beat FA2.
Dense code and forward geometry stay unchanged. Their small timing changes
are controls, rather than evidence of a new forward or dense improvement.

Long causal D128 still trails FA2: public combined graph time is
3.713/3.763 ms (BF16/FP16), versus FA2 3.655/3.681 ms and SDPA
3.790/3.804 ms. At S1024 causal D64, public takes 0.170/0.171 ms,
versus FA2 0.238/0.239 ms. Eager launch gaps can limit the device gain:
FP16 causal S1025 D64 improves from 0.207 to 0.186 ms in graphs but its
synchronized eager median increases from 0.357 to 0.364 ms. Graph speedups
must not be presented as guaranteed serving or eager speedups.

Actual public-entry validation passed 32 cases, including saved statistics,
partial tiles, independent per-head FP64 gates, CPU/CUDA reference agreement
and changed/poisoned graph replay. Each of memcheck, racecheck, synccheck and
initcheck passed 16 selected public calls with zero reported errors; CPU
references use unchanged equations after their agreement check. The maintained
limits remain 1% BF16 and 0.5% FP16 for output and each gradient. Arithmetic
and acceptance criteria did not change.

Revalidation on the current shared runtime passed the same 32 cases, a
24-case paired comparison, and eight negative-view eager/changed-graph cases.
All 12 causal combined graph medians improved in that second comparison
(1.074× geometric mean over group 1); eager improved in 10/12 (1.050×).
Three fallback-delegation checks and 28 routing/publication-guard tests passed.
The CPU suite ran 343 tests: 166 passed and 177 were skipped.

A separate capture bound all 24 CuTe compilation units from 12 training cases
to actual loaded module images. Qualified and final source produced identical
PTX and cubin bytes, with the same graph symbols, grids and resources as the
primary measurements. This supports retaining the completed sanitizer checks
for unchanged device code and ownership. Exact artifact/report hashes are in
the dataset; raw binaries and captures remain outside the tracked tree.

To compare the exact earlier entry under the current shared runtime:

```sh
capture_dir=$(mktemp -d)
git show f5f3a76e077c38e9984d3eff652e234a93587bd1:src/kohakufa_cute/rtx3090.py > "$capture_dir/group1.py"
python -m benchmarks.validate_rtx3090_optin --reference-device cpu --compare-reference-devices --seq 1024 1025 2048 4096 --seed 62407 --output "$capture_dir/qualification.json"
python -m benchmarks.compare_rtx3090_optin --source-revision "$(git rev-parse HEAD)" --baseline-source "$capture_dir/group1.py" --seq 1024 1025 2048 4096 --seed 61103 --output "$capture_dir/comparison/report.json"
```
