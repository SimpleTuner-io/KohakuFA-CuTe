# RTX 4090 causal head grouping

The explicit RTX 4090 entry uses a group of 16 heads for its qualified causal calls: batch 1, 16 heads, 1,024 or 4,096 tokens, D64/D128, FP16/BF16. Dense schedules and eligibility boundaries retain the [shared joint path](rtx4090-shared-joint.md). The default package dispatcher is unchanged.

```python
from kohakufa_cute.fa4_sm80.rtx4090 import scaled_dot_product_attention
```

Grouping changes the mapping from CUDA block coordinates to key blocks and heads. It interleaves heads around causal key tiles instead of exhausting one head's key tiles first. The computation, saved statistics, rounded P/dS operands, actual-O delta and late gradient scaling remain unchanged. The shared control and its partial-head/batch mapping are described in [causal head groups](causal-head-groups.md).

## Measured complete calls

Full RTX 4090, 128 SMs, 400 W limit; Python 3.13.18, PyTorch 2.15.0+cu132, CuTe DSL 4.8.1 and Triton 3.9.0. FA2 is the actual loaded official 2.8.4 CUDA extension from revision `4616f5116d33e3ba3cbb71d582caffcfcb49649c`; its native arithmetic is a complete implementation comparison.

The following warmed CUDA-graph medians include every backward stage. Nine windows of at least 100 ms rotate backend order in the same process on identical rounded inputs. Times are milliseconds; all samples are in [the compact record](../results/rtx4090-causal-head-groups.json).

| Dtype | Tokens | Head dim | G1 backward | G16 backward | G1 F+B | G16 F+B | FA2 F+B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 1024 | 64 | 0.0702 | 0.0477 | 0.0989 | 0.0765 | 0.1055 |
| bf16 | 1024 | 128 | 0.1274 | 0.0883 | 0.1810 | 0.1416 | 0.1774 |
| bf16 | 4096 | 64 | 0.6986 | 0.6493 | 0.9641 | 0.9081 | 1.0149 |
| bf16 | 4096 | 128 | 1.3263 | 1.2365 | 1.8395 | 1.7329 | 1.7779 |
| fp16 | 1024 | 64 | 0.0689 | 0.0476 | 0.0979 | 0.0766 | 0.1057 |
| fp16 | 1024 | 128 | 0.1285 | 0.0895 | 0.1820 | 0.1430 | 0.1776 |
| fp16 | 4096 | 64 | 0.6887 | 0.6435 | 0.9607 | 0.9099 | 1.0173 |
| fp16 | 4096 | 128 | 1.3490 | 1.2508 | 1.8684 | 1.7626 | 1.8079 |

G16 reduces causal backward latency by 30–32% at 1,024 tokens and 6–8% at 4,096 tokens relative to G1. Graph F+B is lower than FA2 in all eight measured causal calls. This is a bounded selection, with no claim for other GPUs, head counts or lengths.

Eager calls have a different boundary. At 4,096 tokens, BF16 G16 F+B measured 0.873 ms at D64 and 1.714 ms at D128, versus FA2 1.028 and 1.753 ms; FP16 results follow the same direction. At 1,024 tokens, the public entry can trail the default CuTe entry despite its faster graph. The Python compatibility check and ordinary launch costs matter there. The G1 native benchmark control bypasses that public check, so its eager numbers do not isolate the scheduling change. Eager and no-grad samples are retained separately; graph wins are not universal eager wins.

## Validation and deployed code

G2/G4/G16 each pass 96 directed cases and 48 partial-head/batch cases, including changing and poisoned graph inputs, with independent CPU/GPU references. Each passes memcheck, racecheck, synccheck and initcheck. Final-source G16 repeats the directed and sanitizer checks, validates 28 actual public fallbacks (including D16/256/512, dense/causal, masks, GQA and rectangular shapes), and checks all 16 dense/causal timing rows. FP64 per-head limits remain 1% BF16 and 0.5% FP16; directed cases retain the existing rounded-reference criteria.

CUPTI binds the actual graph functions to loaded cubins and compiler exports; PTX and SASS are retained outside the checkout. At D128, G1/G16 have the same 160 tensor instructions, 80 matrix loads, five barriers and two dependency waits per query-tile loop. Instruction sites change from 626 to 625. At D64, both have 80 tensor instructions and 52 matrix loads, while address/iterator sites change from 548 to 516. These are static counts over equal query/key work, not runtime stall percentages. D64 register allocation rises from 198 to 208 with a resource-derived limit of one 256-thread CTA per SM in either case; D128 remains at 255 registers and acquires an 8-byte stack slot outside the steady loop. Faster timing therefore does not establish a simple occupancy explanation.

Systems captures verify the complete graph and eager sequence. Compute counters are denied by the host, so no cache-hit, stall or measured utilization percentage is asserted. The record retains source and loaded-code hashes.
