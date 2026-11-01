# A40 causal scheduling and public-entry overhead

The explicit `kohakufa_cute.fa4_sm80.a40.scaled_dot_product_attention` entry
extends its measured family to causal D64 and groups causal backward CTAs across
heads. It also caches immutable device classification. The default attention API
and the [arithmetic contract](precision.md) retain their existing behavior.

## Selected cases

Use Python 3.13, Torch 2.15.0/CUDA 13.2 and `.[cuda,sm80-fa4]`, as in the
[D128 qualification](a40-d128-shared.md). Selection requires a full NVIDIA A40:
SM86, 84 SMs and 101376 bytes of per-block shared capacity; aligned contiguous
BHSD FP16/BF16 inputs, B1/H16, square S1024 or S4096. D128 supports dense and
causal calls; the additional D64 selection is causal only. Existing shape,
dtype, scale, mask, GQA, compilation and deterministic-mode guards remain live.
Other calls delegate to the ordinary strict CuTe API.

| D | Visibility | S | Backward M/N | Q/dO stages | Score/KV/dQ warp layout | Resident V | Head group |
| ---: | --- | ---: | --- | --- | --- | --- | ---: |
| 64 | causal | 1024 | 64/128 | 2/1 | 2/4/4 | yes | 16 |
| 64 | causal | 4096 | 64/128 | 2/1 | 2/4/4 | yes | 4 |
| 128 | causal | 1024, 4096 | 64/64 | 1/1 | 4/2/2 | yes | 4 |
| 128 | dense | 1024, 4096 | 32/128 | 1/1 | 2/2/2 | no | 1 |

All use eight backward warps and M64/N64, one-stage, shared-Q forward. Grouping
uses the [shared default-one control](causal-head-groups.md): it changes the
assignment of key tiles to head/batch CTAs, including residual head groups.
Products, per-CTA query traversal, shared-memory ownership and barriers stay the
same. The order in which CTAs contribute FP32 dQ partials can change.

## Host preparation

A CPU profile on warmed resident tensors measured the old compatibility
predicate at 13.49 microseconds. Device-capability and device-property queries
accounted for about 2.90 and 2.51 microseconds per call respectively. Caching
the immutable device classification reduced the predicate to 7.25 microseconds.
Tensor properties, the current device, deterministic mode, compiler state,
alignment, optional-runtime availability and scale eligibility are still checked
on every call. The cache is keyed by CUDA device index.

An isolated same-process cache control with otherwise identical native runners
reduced S1024/D128 causal complete eager F+B from 0.377 to 0.347 ms in both
dtypes. This is an API measurement, not an assertion that all of its improvement
is explained by the isolated predicate microbenchmark. The final paired matrix
below measures cache and scheduling together, with an ungrouped public control.

## Actual public API measurements

See [compact results](../results/a40-causal-grouped.json) for every timing
sample, graph/eager separation, independent numerical checks, source hashes and
exact loaded-code bindings. Comparisons use identical rounded inputs in one
process, seven rotating rounds, a 300 ms warmup and approximately 80 ms windows.
The final evaluation seed is separate from schedule selection. Graph timings
measure replayed device work; eager wall times include the advertised public
entry, autograd preparation and launch gaps. Compilation and profiling are
excluded. FA2 and SDPA are complete implementations with their own arithmetic.

Causal graph F+B, milliseconds:

| Dtype | S | D | Before | Public G1 | Public grouped | Direct grouped | Ordinary CuTe | SDPA | FA2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 1024 | 64 | 0.227 | 0.179 | 0.146 | 0.146 | 0.226 | 0.193 | 0.176 |
| FP16 | 1024 | 64 | 0.226 | 0.179 | 0.147 | 0.147 | 0.225 | 0.193 | 0.176 |
| BF16 | 1024 | 128 | 0.285 | 0.285 | 0.271 | 0.273 | 0.382 | 0.315 | 0.282 |
| FP16 | 1024 | 128 | 0.288 | 0.288 | 0.274 | 0.274 | 0.389 | 0.316 | 0.284 |
| BF16 | 4096 | 64 | 2.060 | 1.531 | 1.434 | 1.428 | 2.078 | 1.560 | 1.494 |
| FP16 | 4096 | 64 | 2.103 | 1.546 | 1.467 | 1.467 | 2.112 | 1.579 | 1.516 |
| BF16 | 4096 | 128 | 2.828 | 2.803 | 2.772 | 2.778 | 3.817 | 2.918 | 2.786 |
| FP16 | 4096 | 128 | 2.889 | 2.872 | 2.821 | 2.830 | 3.930 | 2.969 | 2.844 |

Causal complete eager F+B wall time, milliseconds:

| Dtype | S | D | Before | Public G1 | Public grouped | Direct grouped | Ordinary CuTe | SDPA | FA2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 1024 | 64 | 0.273 | 0.271 | 0.270 | 0.237 | 0.257 | 0.274 | 0.538 |
| FP16 | 1024 | 64 | 0.346 | 0.346 | 0.345 | 0.312 | 0.331 | 0.386 | 0.753 |
| BF16 | 1024 | 128 | 0.377 | 0.354 | 0.351 | 0.318 | 0.385 | 0.391 | 0.760 |
| FP16 | 1024 | 128 | 0.375 | 0.351 | 0.348 | 0.314 | 0.390 | 0.387 | 0.758 |
| BF16 | 4096 | 64 | 2.066 | 1.540 | 1.438 | 1.435 | 2.084 | 1.565 | 1.502 |
| FP16 | 4096 | 64 | 2.110 | 1.555 | 1.467 | 1.466 | 2.109 | 1.590 | 1.523 |
| BF16 | 4096 | 128 | 2.827 | 2.825 | 2.787 | 2.797 | 3.830 | 2.932 | 2.795 |
| FP16 | 4096 | 128 | 2.890 | 2.885 | 2.829 | 2.830 | 3.942 | 2.991 | 2.856 |

`Before` is the preceding A40 public entry: D128 uses its original ungrouped
schedule/predicate, while D64 delegates to ordinary CuTe. `Public G1` uses the
new public predicate and identical geometry with head group one, isolating the
work-order effect. `Direct grouped` omits the public predicate and is a diagnostic
control; its timing is not substituted for the public API. Both dtypes and dense
D128 forward/F+B preservation are retained in the compact report.


In this matrix, causal D64 graph F+B is approximately 3–17% below FA2 latency.
Short D128 improves by approximately 3–4%; long D128 is within 1% and should be
read as near parity. These are measured shape families and timing boundaries,
not a claim that every attention call is faster.

**Short D64 eager calls remain slower than ordinary CuTe in this matrix.** The
public entry adds about 33 microseconds over its direct native runner, enough
to hide the scheduling improvement at S1024. The preceding A40 wrapper also
paid eligibility and delegation costs for D64; a small gain against that wrapper
does not establish a gain over calling ordinary CuTe directly. Use the graph
and eager boundaries that match the application. Long D64 retains a complete
public-call gain; D128 short eager calls benefit mainly from the device cache.

## Qualification and limitations

The final runtime passes 106 directed cases across D64/D128, plus 53 forced-G16
D64 cases. Coverage includes both dtypes, partial tiles, model head counts,
negative/common-offset statistics, signed/zero-scale delegation, rounded
cancellation, FP16 underflow and changed/poisoned graphs. Actual-public GPU tests
cover positive-scale cache reuse, both selected lengths/dtypes, and native
D16/D64/D256/D512 fallback behavior. Separate G4/G16 native qualification includes
residual flattened batch/head groups and all four sanitizer tools.

The public memcheck passed its original long graph-replay probe. Its original
racecheck did not finish within 900 seconds; that timeout remains an incomplete
result. The continuation passes racecheck, synccheck and initcheck with one
actual selected S4096 F+B for BF16 D64/G4 and FP16 D128/G4, plus changed/poisoned
graph replays at S1024 for BF16 D64/G16 and FP16 D128/G4. Identical rounded
operands and outputs are copied to CPU for the independent FP64, rounded-policy
and saved-statistics oracles, avoiding instrumentation of the reference itself.
No accuracy gate or GPU memory/synchronization check was disabled. Host API
negotiation diagnostics alone use `--report-api-errors no`.

Output, dQ, dK and dV retain independent per-head norm limits of 1% for BF16 and
0.5% for FP16. Directed cases retain their declared rounded equations, using
actual rounded O for delta. Literal unscaled maxima and shifted log2 normalizers
are checked separately. No compensation or additional precision is introduced.

Actual loaded backward ELF, compiler cubin, PTX and graph symbols are bound in
separate processes for each selected schedule; symbols alone are insufficient
because different groups can share a name. The shared-control report retains
the FA2 reference bindings. Static code confirms a work-order change and does
not establish runtime stall percentages. A separate short scheduling probe
exceeded its predefined 5% perturbation limit and supplies qualitative evidence
only; it contributes no timing or quantitative attribution here.

Reproduce the public and directed checks with:

```sh
python -m unittest tests.test_a40_shared tests.test_causal_head_groups -v
python -m benchmarks.validate_a40_shared --source "$FA_SOURCE" \
  --dependencies "$FA_DEPENDENCIES" --dimensions 64 128 \
  --output "$CAPTURE_DIR/directed.json"
python -m benchmarks.validate_a40_shared --source "$FA_SOURCE" \
  --dependencies "$FA_DEPENDENCIES" --dimensions 64 --head-group 16 \
  --output "$CAPTURE_DIR/g16.json"
python -m benchmarks.compare_attention_variants --seq 1024 4096 --dim 64 128 \
  --dtype bf16 fp16 --visibility causal --phases backward joint \
  --warmup-ms 300 --window-ms 80 --repeats 7 --seed 1380 \
  --operations '[{"label":"public","factory":"benchmarks.a40_shared_candidates:make_attention"},{"label":"g1","factory":"benchmarks.a40_shared_candidates:make_attention","kwargs":{"head_group":1}},{"label":"fa2","backend":"fa2"},{"label":"sdpa","backend":"sdpa"}]' \
  --output "$CAPTURE_DIR/paired.json"
```

The optional source/dependency paths are only for the source-based benchmark
adapter. Public use needs the qualified installed packages. Raw captures and
frozen commands are retained outside the repository; the compact report records
their content hashes and the exact measured scope.
