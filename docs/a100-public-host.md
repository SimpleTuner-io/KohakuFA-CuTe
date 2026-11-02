# A100 cached public host entry

The eligible A100 public attention path calls a cached native
`AttentionRunner` with the already selected schedule. C++ allocates owned
outputs and saved statistics, constructs tensor views, and prepares the flat
launch arguments. Each invocation receives its current tensors, scale, device
and stream. The GPU method sources and arithmetic are unchanged.

Eligibility remains the full qualified A100 with 108 SMs, batch 1, 8–16 heads,
sequence lengths 2048–4096, D64/D128, aligned contiguous FP16/BF16, dense or
causal attention and a finite FP32-normal positive scale. The medium H8/D128
causal exclusion remains. Masks, GQA, dropout, unsupported scales, views and
compiled calls retain their qualified native CuTe dispatch. Installation still
uses the pinned `sm80-fa4` extra.

## Actual public API comparison

[Retained measurements](../results/a100-public-host.json) compare the exact same
public SDPA API body with isolated previous/new selected-module sources and
independent generic-loader caches. Both caches remain cold throughout; no
separate fast-validation improvement is included. Each control executes its
own exact module, without mutating shared modules per call. The final runtime
source hashes match the measured candidate exactly.

The full A100-SXM4-40GB had 108 SMs and a 350 W limit, using Python 3.13.18,
Torch 2.15 CUDA 13.2, CUTLASS DSL 4.8.1 and FA2 2.8.4. The matrix covers H16,
S2048/S4096, D64/D128, FP16/BF16 and dense/causal visibility. Seven rotating
windows target 50 ms each, after compilation and a one-second warmup.
Forward uses NoGrad; joint includes forward and all three gradients.

| Boundary | Rows | Candidate faster | Geometric speedup over previous public API |
| --- | ---: | ---: | ---: |
| Complete eager forward | 16 | 13 | 1.427× |
| Complete eager forward + backward | 16 | 14 | 1.531× |
| Graph forward | 16 | 9 | 0.991× |
| Graph forward + backward | 16 | 10 | 1.002× |

This is a host-preparation improvement. Captured device times do not show a
broad pipeline gain. Complete eager joint beats SDPA in all 16 measured cases
(1.051× geometric speedup); graph joint also wins 16/16 (1.062×). Against FA2,
graph joint wins only 2/16 (0.967× geometric speedup), so the remaining device
kernel gap is still measurable. Eager FA2 comparisons include that backend's
host implementation and must not be substituted for graph performance.

There are regressions: S4096/D128 causal forward loses about 3.1% BF16 and
4.1% FP16 in complete eager time. D128 causal graph-forward losses are about
4–5% across both lengths. Joint regressions against the previous public API
are at most about 1%. The retained samples include all cases; this change does
not claim uniform wins. H8 remains within earlier qualified eligibility but is
not included in this new actual API timing matrix.

Reproduce the matched boundaries with:

```sh
python -m benchmarks.compare_attention_variants \
  --operations '[{"label":"previous_public","factory":"benchmarks.a100_public_api_controls:make_attention","kwargs":{"previous":true}},{"label":"candidate_public","factory":"benchmarks.a100_public_api_controls:make_attention"},{"label":"fa2","backend":"fa2"},{"label":"sdpa","backend":"sdpa"}]' \
  --seq 2048 4096 --dim 64 128 --heads 16 --dtype bf16 fp16 \
  --phases forward joint --visibility dense causal --no-grad-forward \
  --warmup-ms 1000 --window-ms 50 --repeats 7 --output artifacts/public-host.json
```

## Correctness and ownership

Six actual-GPU public methods cover repeated API calls, selected boundaries,
native fallbacks, tiny scales, unaligned inputs/output gradients, changing
scales, literal saved maxima/log normalization and poisoned graph replay.
The full 74-case directed suite passed. Trace checks identify forward,
preparation, joint main and dQ conversion; compiled calls retain native CuTe.
Every backend in the 64-row matrix independently passed the per-head output
and dQ/dK/dV FP64 norm limits (1% BF16, 0.5% FP16), with rounded diagnostics.

All four sanitizer tools passed. The original reference-heavy racecheck was
interrupted with status143 while instrumenting diagnostic FP64 GEMMs. The
passed public, directed and memcheck stages were preserved. Both original
CUDA and bit-preserving CPU reference gates then passed outside sanitizers on
the same actual operands, including rounded O. Only missing race/sync/init
checks continued with unchanged CPU diagnostic equations; no gate changed.
The `benchmarks.cpu_reference_unittest` wrapper retains this reproducible
sanitizer boundary.

The cache forwards all 14 current schedule fields. New scheduling controls need
explicit runner support and qualification; silently dropping unknown controls
is unsupported. See the [host ABI](sm80-host-abi.md).

The subsequent [long causal ordering](a100-causal-order.md) update selects
measured head interleaving for S4096/H8 and H16. Its actual API evidence is
separate from the host-launch comparison above.
