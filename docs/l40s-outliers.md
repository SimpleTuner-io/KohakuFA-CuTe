# L40S attention schedules and gradient precision

The L40S work targets a full 142-SM, 48-GB device at its 350 W board limit. It adds a D16 logical-width schedule, compact short attention, and a larger forward query tile for aligned long D64/D128 calls. The final backward path also retains low components of probability and softmax-derivative operands. Those arithmetic corrections have a measurable cost; faster implementations that lose small gradients are diagnostic controls.

The comparison uses Python 3.13.18, PyTorch 2.15.0+cu132, CUDA 13.2, CuTe DSL 4.8.1 and driver 615.71.09. The loaded driver API reports 13040. All four controls use this same qualified stack; loaded library basenames and complete runtime source hashes are retained in the report.

The canonical measurements are in [the L40S report](../results/l40s.json), with the cross-device comparison in [the performance grid](../results/performance.svg). The final four-backend grid is described below; numerical examples and controlled experiments are separate evidence.

## Measurement provenance and this device split

The retained JSON is the original combined-development measurement snapshot, published under the `measurement-source/gpu-outliers-20261031` source reference. Its canonical samples, gates, source hashes and benchmark hashes are unchanged. The later integration smoke is recorded separately. Shared kernel owners and `l40s.py` retain their qualified bytes; changes to other devices' selectors do not replace those measurements. The completed stack restores the original benchmark control imports now that all selector modules are present. Publication packaging received CPU/static checks and does not claim a new L40S GPU run or a byte-identical whole-repository measurement snapshot.

## What changed

- D16 keeps physical input padding at 32 features but performs logical 16-feature work in the measured long schedule. Both runtime sequence lengths participate in dispatch, including cached calls.
- Short D128/D256/D512 forward uses a compact query tile. Short square D128 backward uses a separate compact owner, with independent high/low key and value sums.
- Long D64/D128 forward uses 128 query rows, 128 key columns and eight warps. Both sequence axes must be at least 1,024 and divisible by 128. Shifted tails use the preceding forward implementation. Masked, grouped-head and unsupported layouts retain their qualified forward fallbacks.
- Precision selection runs after architecture scheduling, preserving query ownership and runtime guards. It memoizes shared objects so cached wrapper branches retain consistent state. The generic D16/D64/D128 policy covers the qualified low-precision key/value owners; wide dimensions use separately qualified owners.

Wide precision qualification here covers unmasked D256 with equal query/key-value head counts and unmasked D512. The D512 owner also supports the separately checked grouped-head path. Other wide configurations retain their preceding fallback; these are distinct from the generic D16/D64/D128 masked/GQA checks.

The forward predicate is deliberately stricter than “long sequence.” A measured D64 1,025-query/1,031-key case regressed with the larger query tile; restoring the stock tail path avoids that result. Shape-cache and compiled-call tests cross both length thresholds and the aligned/shifted boundary.

## Measured results

All 60 final matrix cases passed the eager and captured-graph component gates for all four backends. CuTe had a lower measured median than native SDPA in 33/60 forward, 28/60 backward and 28/60 complete calls. These counts describe this matrix, not a universal workload distribution.

The L40S selector changes forward work in 36 rows. All 36 had lower forward and complete medians than the precision-matched preceding schedule: 1.09–3.05× for forward and 1.02–1.68× for complete calls. Its 12 changed backward rows were 1.33–1.86× faster. Unchanged schedules also vary between samples; small median differences are not separate kernel improvements.

Representative BF16 dense timings, in microseconds:

| Sequence / head width | Boundary | Precision-matched preceding | Selected | Native SDPA |
| --- | --- | ---: | ---: | ---: |
| 128 / 128 | Complete | 24.02 | 15.41 | 25.10 |
| 128 / 256 | Complete | 38.15 | 26.65 | 34.20 |
| 128 / 512 | Complete | 50.82 | 44.63 | 175.11 |
| 4,096 / 16 | Complete | 1,511.54 | 916.17 | 756.42 |
| 4,096 / 64 | Forward | 485.64 | 391.92 | 351.37 |
| 4,096 / 128 | Forward | 852.42 | 670.99 | 654.67 |
| 4,096 / 128 | Complete | 4,491.58 | 4,284.12 | 2,665.93 |

At S1024/S4096, D512 complete calls remain 2.45–4.35× faster than native SDPA across both dtypes and visibility modes. Long D64/D128 training still trails native SDPA. The new forward schedule reduces that gap, while the corrected backward arithmetic adds work relative to the released implementation. The JSON retains all seven samples rather than only these rounded medians.

The selected long forward also survived a separate sustained check. BF16 S4096/D128 dense ran four alternating 20-second passes in previous/selected/selected/previous order, excluding the first five seconds of each. Across 143 retained event windows per backend, pooled medians were 852.33 µs and 678.96 µs: 20.3% lower latency. Sampled board power had a roughly 350 W median for both, temperature ranged from 53°C to 65°C, and only the benchmark process was present. Sampled clocks differed; these observations do not provide cycle normalization or establish a hardware ceiling.

## Precision failures and their repair

The original BF16 rectangular failure used B2/H16, 1,023 queries, 17 keys and D128. One dV coordinate was approximately −0.15918 instead of the centered FP64 result −0.18469. Its whole-gradient relative error was only 0.236%, so a norm-only gate missed it. The corrected probability product returns approximately −0.18457 and passes the existing component-wise checks.

A compact cancellation example makes the issue clearer: a single rounded probability operand gives BF16 dV **+0.37109** where FP64 gives **−0.0182377**. FP16 gives **+0.00870514**. A second probability component and its own FP32 accumulator preserve the negative result. Embedding the same finite inputs in square attention demonstrates that rectangular shape is not a mathematical protection.

The corresponding key-gradient failure comes from rounding dS before multiplying by Q. Capturing the actual GPU FP32 dS isolated the loss: its FP64 product with Q agreed closely with the centered reference, while the rounded operand reproduced the released kernel’s wrong-sign result. The low dS product repairs that rounding boundary. An arbitrary query-anchor correction was not substituted for the actual residual product.

High and low products have independent FP32 sums. Interleaving both products into one accumulator was rejected because FP32 accumulation rounding can consume the small residual during cancellation. The low sum is combined only in the epilogue.

The qualified generic and D512 FP16 residual paths use additional scaling:

- dS residuals are multiplied by 1,024 before packing and their independent output sum is divided by 1,024.
- Probability residuals are multiplied by 2²⁴ before packing and their independent value sum is divided by 2²⁴. Probability-rounding residuals are small enough for this scaling to remain representable.

An FP16 underflow fixture produces a dV coordinate of about **0.67997**. Unscaled probability components both disappear and produce zero. Scaled residual storage gives about **0.67969**, with all output/dQ/dK/dV gates passing. A short-square variant also fails without scaling. The shared regression factories preserve the exact rounded operands and check these coordinates in addition to whole tensors. The D256 owner retains its separately qualified derivative-packing policy; its probability residual is also scaled.

The reference is centered FP64 attention on identical rounded inputs. Native SDPA is a comparator. Component-wise tolerances remain atol=rtol 0.02 for BF16 and 0.003 for FP16, with relative-norm limits of 1% and 0.5%. Key/value cancellation fixtures additionally require the nonzero coordinate’s sign and a zero-atol 2% relative bound. The FP16 probability-underflow coordinate retains its stricter zero-atol 0.5% bound.

## Resource choices and rejected controls

For BF16 dense S4096 forward, the larger query tile halves query CTAs from 64 to 32 per head and doubles threads per CTA from 128 to 256. D64 shared memory rises from 24 to 48 KiB and registers/thread from 168 to 197. D128 rises from 48 to 96 KiB and from 216 to 247 registers. Both selected forward kernels have zero local memory. This trades larger per-CTA state for fewer query owners that each reuse the key/value stages across more query rows. Native D64 still has lower forward latency with 128 local bytes/thread, demonstrating why resource counts alone cannot choose the winner.

Representative uninstrumented CUDA graph resources for the frozen generic correction were:

| Key/value owner | BF16 registers/thread | FP16 registers/thread | Local bytes/thread | Dynamic shared memory |
| --- | ---: | ---: | ---: | ---: |
| Long logical D16 | 148 | 168 | 0 | 16 KiB |
| D64, one output-column warp | 255 | 255 | 40 / 32 | 24 KiB |
| Long D128, two output-column warps | 255 | 252 | 0 | 40 KiB |
| Short causal D128, four narrow output-column warps | 156 | 156 | 0 | 28 KiB |
| Dense D256, frozen medium portability control | 250 | 250 | 0 | 96 KiB |
| Dense D512, frozen wide portability control | 254 | 254 | 0 | 99 KiB |

Resources vary with the instantiated layout, visibility and dtype. The final report retains actual functions and resources for every measured graph phase.

The first full-width D128 residual layout was rejected after a severe allocation failure: 40 allocated registers and roughly 3,000 bytes of local memory per thread, with extensive local loads/stores. Its PTX was register-only; the local traffic appeared in assembled SASS. A smaller output fragment removed this spill cliff. This differs from D64’s small, measured spill tradeoff: the final BF16 export has seven local-load and ten local-store sites across the whole function, while FP16 has four and six. These are static sites, not executed counts or runtime percentages. The selected owner still beat the spill-free alternative on complete calls.

A smaller 16-key-row/four-column-warp layout reduced register use substantially but increased complete latency at long D64/D128. The controlled intervention repeats query/gradient staging over more key tiles; lower registers did not establish a speedup. It remains a rejected long-sequence control, while the compact short layout is retained.

Reusing query-column statistics also showed why cold medians are insufficient. FP16 pilot runs favored it by roughly 3–6%, while BF16 often regressed. Four alternating 20-second sustained passes, excluding the first five seconds of each, reduced the aggregate FP16 advantage to about 1.65%. Temperature rose from roughly 55°C to 65°C and sampled board power approached 350 W. The final generic policy keeps the simpler original statistic schedule rather than adding a small dtype-specific preset. Sampled clock values are telemetry, not cycle-normalized timings.

The additional products carry a real execution cost. In one matched BF16 S4096/D64 dense pilot, the corrected complete call took about 2.240 ms versus 1.912 ms for the preceding arithmetic. A long D128 square pilot also became slower after both low products were retained. These are implementation costs, not numerical lower bounds or hardware ceilings. Final released/precision-matched/selected comparisons must be read from the four-backend report rather than splicing these pilots into a new benchmark row.

## Profiling and validation

Systems captures established actual forward/prepass/query/key-value launch order. The large-query forward intervention was measured at the public autograd-retaining forward boundary and again in the complete forward+backward call. Eager activity verification prepares operands before entering the trace so benchmark-clone copies are not attributed to attention.

Sixteen final-source Systems captures cover eager and graph execution for four BF16 cases: short causal D128 and long dense D128/D256/D512. They are representative activity checks, not traces of every matrix row. Short D128 launches three CuTe kernels per complete call versus six native kernels. Long D128 launches four versus six. Across three long D128 graph calls, CuTe query and key/value owners occupy approximately 3.985 ms and 6.802 ms; the native combined gradient loop occupies 6.026 ms. The CuTe between-kernel intervals total only about 0.0033 ms. The remaining long-case concentration lies inside gradient work. These instrumented intervals explain placement and order; they do not replace the canonical event medians.

Loaded native SDPA cubins were captured separately from the actual launched functions. The retained dense D64 native forward used a 128×128 query/key tile; dense D128 used 128×32. Those deployed native modules target SM86 and execute on SM89. They are distinct from a source reconstruction. Where the library contained no embedded PTX, the evidence is deployed SASS and resources, not claimed native PTX.

CuTe compiler exports were checked for real PTX entries and nonempty ELF bytes, then associated with the captured backward bundle and disassembled. Forward-warmup modules are excluded from these associations: different modules can reuse a mangled entry name. The exported backward ELF supplies the code hash; graph function names and resources are recorded separately. Loaded CuTe module bytes were not independently dumped in this export process. PTX ISA versions come from emitted headers; CUDA function virtual-target attributes are not interpreted as PTX ISA versions.

Compute counters were unavailable: a corrected capture attempt returned `ERR_NVGPUCTRPERM`. An earlier export-harness failure is retained separately. Timing, graph identities, deployed code, resources and controlled changes support the conclusions; no cache-hit, stall or utilization percentages were inferred.

The frozen generic correction passed 64 focused cases, including both dtypes, exact key/value/underflow factories, the original rectangular operands, three negative-max key fixtures, signed and zero scales, strided masked GQA and 12 graph replays with changed inputs and poisoned outputs. Separate existing regression methods cover the same public frontends and runtime boundaries. Full GPU discovery passed 194 tests with 56 skips on revision `671a8a2`. The final `2806bcc` runtime then passed all ten shared key/value test methods and a public four-backend dispatch check. These are separate validation runs; the final matrix records its own complete runtime hashes. A later `26a1061` integration check passed eight D64/D128 cases across both dtypes and visibility modes, checking eager outputs/gradients and all three captured phases. Its only runtime differences are the nonselected RTX 5090 schedule and new backward module. The capability guard bypasses that selector on L40S; sixteen direct predicate checks on actual normalized compilation arguments were false, and the new module was never imported.

Twenty focused CPU-FP64-reference cases passed memcheck and racecheck with CUDA API-status reporting suppressed. Unfiltered memcheck reported seven pairs of API-version lookup diagnostics: a 13041 request against driver API 13040 and `cuGetProcAddress_v2` invalid-value status. They were retained separately from kernel-memory findings. Only API-status reporting was suppressed; kernels were not name-filtered. The same memcheck flags detected an intentional invalid global write and returned the configured failure exit code, confirming that memory instrumentation remained active.

Twelve additional final-source wide cases passed both tools, covering D256/D512 strided B2 equal-head tensors, D512 grouped heads, shifted tails, negative maxima and both dtypes/visibility modes. Those wide checks use the same API-status suppression and retain their own source hashes and functional component results.

## Reading and reproducing the final grid

The report uses four explicitly labelled backends:

| Backend | Computation |
| --- | --- |
| `sdpa` | Default native PyTorch SDPA |
| `released` | New architecture selectors and final precision hook disabled; the preceding computation compiled through the current stack |
| `previous` | Final shared precision retained, with only L40S scheduling disabled |
| `cute` | Final L40S scheduling and shared precision |

`released` is not a claim of a byte-identical historical binary. Comparing `previous` with `cute` isolates the architecture schedule; comparing `released` with the corrected paths also exposes arithmetic costs. Failed native/released numerical gates leave diagnostic timings and are not qualified speedup claims.

Each case uses B1/H16 contiguous BHSD tensors, sequence lengths 128/1,024/4,096, head dimensions 16/64/128/256/512, both dtypes and dense/causal visibility. Seven rotating rounds target 200 ms per phase with at least 16 graph replays; long calls can exceed that target. Compilation, initial API dispatch and graph capture are outside timing. Event spans can include host graph-replay submission gaps. Profilers and sustained telemetry run in separate processes.

```sh
python -m benchmarks.compare_l40s --expect-device 'NVIDIA L40S' \
  --output results/l40s.json
```

The JSON records rounded-input correctness, paired samples, memory headroom, source hashes, compiler/runtime metadata, loaded library basenames and actual graph functions/resources. Raw machine captures remain outside the tracked repository.
