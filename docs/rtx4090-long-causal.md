# RTX 4090 long D64 causal attention

The explicit RTX 4090 entry selects a 128-key backward tile for batch 1, 16 heads, 4,096-token causal D64 attention in FP16/BF16. It retains the 64-key tile at 1,024 tokens: the wider tile regressed that shorter workload. Other head dimensions, dense schedules and fallback boundaries retain their preceding choices. The top-level dispatcher is unchanged.

```python
from kohakufa_cute.fa4_sm80.rtx4090 import scaled_dot_product_attention
out = scaled_dot_product_attention(q, k, v, is_causal=True)
```

## Complete-call comparison

Milliseconds on a full 128-SM RTX 4090 at a 400 W limit. Python 3.13.18, PyTorch 2.15.0+cu132, CuTe DSL 4.8.1 and Triton 3.9.0 retain the qualified stack. FA2 is the actual loaded official 2.8.4 CUDA extension. Its native precision policy differs from CuTe's, so FA2/SDPA are complete implementation comparisons.

Both CuTe policies use the same public compatibility checks and cached native runner. Independent seed 9631 evaluates the choice made on the earlier tuning inputs. Nine rotated windows of at least 100 ms follow a 500 ms warmup. Eager event intervals include launch gaps; host wall samples and graph replay are recorded separately. Compilation and graph capture are outside these timings.

| Dtype | Mode | Previous backward | N128 backward | Previous F+B | N128 F+B | FA2 F+B | SDPA F+B |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | graph | 0.6434 | 0.5994 | 0.8971 | 0.8630 | 1.0286 | 1.0552 |
| bf16 | eager | 0.6451 | 0.6014 | 0.8958 | 0.8638 | 1.0300 | 1.0593 |
| fp16 | graph | 0.6385 | 0.5915 | 0.8948 | 0.8554 | 1.0319 | 1.0578 |
| fp16 | eager | 0.6414 | 0.5916 | 0.8960 | 0.8583 | 1.0351 | 1.0593 |

The improvement is about 7% for backward and 4% for F+B on these held-out long calls, including preparation and all three gradients. Forward arithmetic and layout are unchanged. The [compact record](../results/rtx4090-long-causal.json) retains the full 16-shape forward/backward/F+B matrix in graph and eager mode, no-grad forward checks, samples and source/code hashes.

## Design and validation

The selected backward remains M64, Q1/dO1, V in registers and head group 16. N128 uses score/KV/dQ warp factors 2/4/4. It halves the key-block grid from 64×16 to 32×16, reducing dQ partial contributions and repeated query/output-gradient staging. It also increases register and shared-memory pressure; the choice is measured rather than inferred from occupancy. Saved maxima/log normalization, P/dS input-dtype rounding, actual-O delta, FP32 products and late scaling are unchanged.

The typed N128 body retains its 96-case directed qualification and four sanitizer passes. Final current-host public validation checks saved statistics, independent per-head output/dQ/dK/dV errors, CPU/GPU reference agreement, cache reuse and changing/poisoned graph replay. All four sanitizers additionally exercise the actual long public path in both dtypes. Twenty-eight public fallbacks include D16/256/512, masks, GQA, rectangles and tails. Per-head FP64 limits remain 1% BF16 and 0.5% FP16; directed cases keep their existing rounded-reference criteria.

CUPTI binds selected and preceding kernels to actual loaded cubins and compiler exports, with exact PTX/SASS retained. Both binaries and disassemblies are byte-identical to their corresponding qualified tuning controls. Systems captures cover graph and eager F+B. Counter access remains denied; no stall, cache-hit or HFU percentage is inferred. Earlier probability-store and barrier-elision variants were slower and are not selected. Input fixtures regenerated from the recorded evaluation seed are archived with tensor/file hashes and explicitly labelled as reconstructions; they were not dumped inside the timed windows.

Reproduce the same public policies with `benchmarks.rtx4090_causal_controls:make_attention`, using `policy="selected"` or `policy="previous"`, through `benchmarks.compare_attention_variants`. Keep independent evaluation seeds, both dtypes and the short-sequence boundary in the comparison.
