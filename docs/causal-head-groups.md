# Explicit causal head grouping

`Schedule.causal_head_group` changes the CTA traversal of the shared warp-MMA
backward kernel. The default is **1**. No automatic GPU selector changes here.
The native C++ runner accepts the same optional configuration key; omitted keys
retain group 1, and compiled/cache entries include the group value.

Grouping interleaves key tiles across flattened batch/head coordinates. Causal
key tiles have unequal numbers of query tiles to visit, so launch order can
change the completion tail without changing attention products. The remap is a
bijection, including a final incomplete head group and groups crossing batches.
It preserves each CTA's query-loop order, buffer lifetimes, barriers, operand
rounding and late scaling. Global FP32 dQ atomic additions can occur in a
different order; bitwise equality across orders is not the acceptance goal.

The legal experimental values are 1, 2, 4, 8 and 16. Actual native math and
sanitizer qualification below covers **groups 4 and 16 on a full A40**. Other
values, targets and workload families need their own qualification. The Hopper
runner rejects nonidentity grouping; its default group is removed before calling
the Hopper compiler, preserving that family's existing configuration.

## Explicit native use

For the qualified A40 D64 causal control:

```python
from kohakufa_cute.native import extension

runner = extension().AttentionRunner("warp", {
    "forward_m": 64,
    "backward_n": 128,
    "query_stages": 2,
    "dout_stages": 1,
    "score_m_warps": 2,
    "kv_m_warps": 4,
    "dq_m_warps": 4,
    "v_in_regs": True,
    "causal_head_group": 16,
})
out = runner(q, k, v, causal=True)
```

This is a schedule experiment, not a replacement for a bounded public device
selector. D128 uses N64, Q1/dO1 and 4/2/2 instead. Group choice matters: larger
is not consistently faster.

## A40 complete-call evidence

[Compact measurements](../results/a40-causal-head-groups.json) retain all samples,
per-component error summaries, runtime source hashes and loaded-code identities.
The stack is the qualified Python 3.13/Torch 2.15.0+cu132/CuTe 4.8.1 environment.
The native comparison has seven rotating rounds, 300 ms warmup and approximately
100 ms measurement windows; compilation and profiling are excluded.

B1/H16, BF16 causal forward plus all three gradients, graph device-event medians
in milliseconds:

| S | D | Group 1 | Group 4 | Group 16 | FA2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 64 | 0.1802 | 0.1653 | 0.1464 | 0.1751 |
| 4096 | 64 | 1.5210 | 1.4245 | 1.4345 | 1.5174 |
| 1024 | 128 | 0.2861 | 0.2735 | 0.2779 | 0.2851 |
| 4096 | 128 | 2.7866 | 2.7600 | 2.8596 | 2.8370 |

The corresponding **complete eager native-call wall times** are separate:

| S | D | Group 1 | Group 4 | Group 16 | FA2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 64 | 0.3041 | 0.3035 | 0.3053 | 0.8353 |
| 4096 | 64 | 1.5218 | 1.4313 | 1.4394 | 1.5253 |
| 1024 | 128 | 0.3048 | 0.3040 | 0.3047 | 0.7514 |
| 4096 | 128 | 2.7986 | 2.7655 | 2.8758 | 2.8538 |

Short graph wins do not become eager wins over group 1 in these measurements;
host/autograd overhead hides the device improvement. Long D64/D128 group-4 gains
remain visible in eager calls. These native controls include their own thin
Python factory and runner lookup, not a public device compatibility predicate.
FA2 retains its own arithmetic and Python/autograd preparation; its eager time
is not a kernel latency. Both dtypes, all phases, SDPA and every sample remain in
the result file.

## Qualification and deployed code

Each group passes 48 native residual-head/batch cases plus 48 changed/poisoned
replays, with independent output/dQ/dK/dV norm gates and saved literal maxima and
shifted log2 normalizers. Cases cover both dtypes, D64/D128, S129/S257, head counts
1/3/7/18/24 and batch sizes 1/2. References use identical rounded operands on CPU;
CUDA candidate kernels and graph replays execute normally. The existing 1% BF16
and 0.5% FP16 per-head limits and rounded diagnostics are unchanged. Group-4 math
is reused from its first native run after checking runtime identity: 33 source
files are byte-identical and two Python files differ only in formatting/AST
locations. Group-16 math runs on the final snapshot.

Memcheck, racecheck, synccheck and initcheck all pass separately for both groups,
with all 12 smoke cases retained. `--report-api-errors no` suppresses host API
negotiation diagnostics; device memory and synchronization errors remain tracked.
Constructor tests reject invalid groups and nonidentity Hopper grouping.

Nine separate-process captures bind the selected native/FA2 graph symbols to
actually loaded ELFs and PTX; native cubins also match compiler-returned hashes.
Each group is captured separately because symbols can be identical across
variants. No symbol-only association across group binaries is assumed.

The selected CuTe query-loop backedge retains the same heavy instruction counts
within each D: D64 has 160 HMMA, 80 LDSM, 6 async-copy and 4 barrier instructions;
D128 has 160/80/10/5 respectively. Mapping adds mostly prologue integer work; total
loop counts change from 829 to 832 instructions for D64 and 626 to 625 for D128
with register-allocation differences. D128 grouping also adds 8 local bytes per
thread, while shared storage and block geometry stay fixed. Static counts are
not stall percentages, cache-hit rates or executed-work measurements, and
predicated residual/padding paths must not be counted as always executed. FA2's
separate backedges are retained without equating them to the CuTe query loop.

Reproduce the native checks and same-process comparisons from a qualified A40:

```sh
python -m benchmarks.validate_a40_head_groups --group 4 \
  --reference-device cpu --output "$CAPTURE_DIR/g4.json"
python -m benchmarks.validate_a40_head_groups --group 16 \
  --reference-device cpu --output "$CAPTURE_DIR/g16.json"
python -m unittest tests.test_causal_head_groups -v
python -m benchmarks.compare_attention_variants \
  --operations '[{"label":"g1","factory":"benchmarks.a40_native_head_groups:make_attention","kwargs":{"head_group":1}},{"label":"g4","factory":"benchmarks.a40_native_head_groups:make_attention","kwargs":{"head_group":4}}]' \
  --seq 1024 4096 --dim 64 128 --dtype bf16 fp16 --visibility causal \
  --phases backward joint --repeats 7 --output "$CAPTURE_DIR/groups.json"
```

Use `--smoke` under each sanitizer. `--compare-reference-devices` performs the
additional CPU/CUDA reference sanity check outside instrumentation. Large raw
captures and failed-harness records stay outside the tracked tree.
