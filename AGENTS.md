# Repository instructions

Use Python 3.13, unittest, Black 24.8, and isort 5.13. Keep changes focused and preserve untracked files.

## Worktree workflow

Before editing files for a task, create its branch and dedicated worktree from the current upstream base. Perform all task edits, tests, commits, and publication preparation inside that worktree. Keep the primary checkout clean; do not edit it first and then copy changes into a worktree, or leave duplicate uncommitted task files there after publishing.

Periodically, especially before starting new work and after a PR merges, fetch upstream and inspect `git worktree list`. Clean up completed task worktrees and their local branches once their changes are confirmed merged upstream. For squash or rebase merges, verify the PR state and resulting content rather than relying only on commit ancestry.

Before removing a worktree, check its index, working files, and untracked files, and confirm the task is finished. Preserve unfinished or untracked work and leave active or unrelated worktrees alone. Do not force-remove worktrees or use broad cleanup commands that discard files. After merged work is cleaned up, fast-forward the primary checkout when possible and verify that it is clean.

## Kernel optimization workflow

Optimize the user's complete computation under a declared low-precision arithmetic and accuracy contract. The goal is fast attention with correct probability reconstruction, gradients, masking, and memory ownership. Use measured evidence to choose and validate changes; language choice, fusion, occupancy, and pipeline depth do not establish a speedup.

### Numerical goal and scope

- Preserve separate unscaled score maxima and logarithmic normalization, shift scores before scaling, and apply the attention scale after FP32 gradient accumulation. Fix probability-reconstruction and gradient errors caused by losing normalization information at large logits or scaling derivatives before their low-precision conversion.
- Maintain the default arithmetic in [docs/precision.md](docs/precision.md): reconstruct P from the saved unscaled statistics, convert P once to the input FP16/BF16 dtype before forming dS, and convert dS once for the gradient products. QK, dP, and gradient products accumulate in FP32; apply the attention scale after dQ/dK accumulation and before the final input-dtype output conversion.
- Compute `delta = sum(FP32(O) * FP32(dO))` from the actual rounded forward output. An ideal FP64 O defines a different backward reference. Converting P back to FP32 storage after its input-dtype conversion does not restore discarded bits. Expected underflow, cancellation and sign changes from these conversions are part of the baseline; agreement with unrounded FP64 at every small coordinate is not required.
- Use this same arithmetic contract in default dispatch on every GPU. Tile, layout, pipeline and ownership tuning must preserve the operand conversions and FP32 accumulation. Keep the former compensated-policy helpers and previous-policy benchmark control confined to explicit experiments; do not reactivate them automatically on selected devices.
- Fix incorrect masks, partial-tile predicates, stale data, dispatch, indexing, synchronization, and omitted operations required by that computation. Distinguish those implementation defects from expected operand rounding and from requests for a stronger numerical method.
- Define the arithmetic and acceptance criteria before tuning. Use FP64 to diagnose mathematical error and establish the expected rounding floor. Preserve the original sharp-attention fixes while comparing implementations under matching precision policies.
- Extra residual products, compensated sums, higher accumulator precision, and additional centering or correction terms are separate accuracy experiments unless explicitly requested for the default computation. Retain their failing inputs, accuracy results, and complete-call costs; a stricter diagnostic must not silently redefine the default optimization target.

### Establish the computation and baseline

- Read the relevant upstream implementation and official backend support before designing a replacement. Reuse available source and inspect existing experiments; do not repeatedly rediscover already rejected approaches. Check current supported framework releases and record why any older qualified stack is retained.
- Capture representative application inputs and inventory shapes, strides, masks, dtypes, head counts, quantization coverage, and required transforms. Include scaling, rotations, correction biases, normalization, activation rounding, retained floating-point layers, and output heads. A baseline that omits required math is a different computation and cannot establish a speedup for the complete path.
- Compare identical rounded inputs, weights, and output contracts in the same process and software stack where possible. For isolated scheduling or pipeline comparisons, also match operand conversions, accumulation policy, and correction terms. Separate different precision policies, checkpoints, calibration recipes, layer coverage, runtime stacks, and graph policies; label those as complete implementation comparisons rather than isolated kernel improvements.
- Define the timing boundary before measuring: kernel only, preparation plus kernel, forward, backward, combined training step, complete model, or serving latency. Distinguish prefill from decoding and resident GPU time from tokenization, transfers, HTTP scheduling, loading, compilation, and graph capture. Include costs required by the advertised API.
- Record source revisions, framework/compiler/runtime versions, actual loaded libraries, device capability, SM count, partitioning, memory limits, and power limit. A partition or another GPU sharing the same compute capability is not a substitute for full-device performance or qualification.

### Keep GPU campaigns resumable

- Before launching work on an ephemeral machine, save the complete source snapshot, including task patches, exact commands and toolchain/runtime configuration on persistent storage outside that machine. Hashes alone cannot recover missing source. Record owned resources, job identities, budget/deadline and continuation steps in a durable handoff.
- Preflight the frozen launch scripts and benchmark entrypoints: check shell/Python syntax, required modules, supported options and source-revision arguments from that snapshot before queueing GPU work.
- Checkpoint stage receipts, failing operands, numerical results and timing samples throughout the run. Copy loaded binaries, PTX/SASS and profiler captures off-host as they become available, outside timed windows where possible. Keep raw captures outside the tracked tree.
- Before deleting an owned resource or its storage, verify a completed final off-host archive against its expected file inventory and hashes. Starting a copy or retaining only a partial checkpoint does not establish completion. Serialize manual and automatic cleanup through one lifecycle lock. If archival fails, stop compute and retain storage for recovery. Leave unrelated resources alone.
- Record each stage's actual exit status. Mark intentional interruptions explicitly and resume only missing work under a new immutable job identity. Preserve passed checks when the relevant implementation is unchanged; a controller failure does not erase those passes or turn incomplete stages into successes.
- Keep expensive reference calculations outside sanitizer instrumentation when possible. A CPU reference may evaluate identical rounded operands, actual outputs and gradients using unchanged equations and acceptance criteria; first check its agreement with the device reference outside sanitizers. Continue instrumenting the implementation's GPU kernels, and preserve reference failures and tool limitations.

### Verify that the intended path executes

- Identify the selected backend and actual launched kernels; an API name or installed package is insufficient. Count expected modules, preparations, GEMMs, normalizations, copies, and fallback calls. Trace both the measured graph path and eager execution when their preparation differs.
- Check dispatch predicates against real runtime objects, including parametrized modules, subclasses, wrappers, compiled calls, tails, and cached shapes. Prefer appropriate type or capability checks over exact class-name strings. A skipped fusion can dominate performance even when the selected GEMM is efficient.
- Make initialization failures of required transforms or adapters stop execution rather than silently serving a different computation. Distinguish an explicitly supported fallback from a failed installation or unsupported selected kernel.
- Inspect repeated casts, reductions, pointwise passes, layout conversions, dequantization, materialized intermediates, and duplicated input preparation before assuming tensor-core GEMM is the bottleneck. Cache immutable inference preparation when valid, and verify that graph replay still reads changing inputs.
- Check GPU memory and other running work before interpreting OOM or timing results. Serialize independent benchmark campaigns on one GPU unless concurrency is the workload being measured. Clean up owned test processes while honoring the user's resource-retention instructions.

### Profile the application and deployed machine code

- Measure the complete workload first, rank concentrations of device time and launch gaps, then isolate suspected stages on captured operands. After a local improvement, remeasure the original workload. A microbenchmark win alone is insufficient to retain a dispatch preset or claim an application speedup.
- Use Systems timelines to establish launch order, copies, synchronization, preparation, overlap, and actual kernel selection. Account for warmup calls and host-thread/autograd correlation before deriving per-call kernel counts or interpreting NVTX projections. Check profiler record schemas before treating identifiers as handles for another API. Use Compute counters when available to test specific hypotheses. If counters are denied or unavailable, record the limitation and use timing, traces, loaded code, resource metadata, and controlled interventions; do not invent stall, cache-hit, bank-conflict, or utilization measurements.
- Inspect PTX and SASS for both implementations. Match the exact profiled kernel symbol, loaded cubin, architecture, and compile configuration. Extract packaged PTX when available. Clearly distinguish deployed code from a source rebuild, a similar library symbol, or an empty disassembly result.
- Analyze prologue, steady main loop, drain, and epilogue separately. Normalize instruction counts by equal K work, warp geometry, and output work. Separate inner GEMM backedges from persistent tile-scheduler or stream-K loops. Exclude predicated-off padding from claims about executed work.
- Compare matrix-load width, shared-memory layout, address arithmetic, carried iterators, async-copy issue placement, register prefetch, tensor instructions, barriers, and output-store width. Fewer matrix-load instructions need not mean fewer transferred bytes; equal barrier counts can still have different scheduling. Static instruction counts are evidence, not runtime percentages.
- Inspect actual register allocation, spills, static and dynamic shared memory, alignment, and resident-CTA limits. Higher occupancy or fewer registers is not automatically faster. Compute resource requirements from the instantiated kernel and target device, including padding and epilogue storage.

### Test hypotheses about pipelines and layouts

- State the suspected bottleneck, supporting evidence, proposed intervention, and expected observation before a sweep. Change one factor at a time or use an explicit small factorial comparison. Record negative results so the same experiment is not repeated under another name.
- Retain promising fast experiments and their failing operands. Establish whether a failure violates the declared computation, reflects its expected rounding floor, or belongs to a stronger precision experiment. Implement and measure repairs for implementation defects before discarding the design.
- Treat pipeline depth, tile size, register layout, and CTA ordering as coupled design choices. Matching another kernel's stage count does not reproduce its iterator, register pipeline, load schedule, or work distribution. A K64 stage and a K128 stage contain different amounts of work and storage.
- Verify producer/consumer ownership and buffer lifetimes when moving prefetches, reducing barriers, reusing shared memory, or fusing epilogues. Preserve tail predicates and zero filling. Reuse staging only after every reader has finished; validate asynchronous paths for races, uninitialized reads, and deadlocks.
- Evaluate projection merging and shared preparation separately from the GEMM loop. Fewer launches and activation reads can help, but a merge can also discard useful normalization or activation fusion, add copies, or increase resource pressure. Include those costs in the comparison.
- Requalify on each target GPU and important workload family. Transfer arithmetic and design lessons across architectures; do not assume previously tuned tiles, memory budgets, instruction paths, or pipeline schedules remain optimal.
- Do not stop at a list of plausible gaps. While optimization work is authorized, implement and test the highest-impact supported intervention. Distinguish disproven hypotheses, validated improvements, unresolved causes, and measured hardware limits. A profiler permission failure or an unsuccessful tuning knob is not itself a performance ceiling.

### Measure sustained performance fairly

- Warm compilation, allocation, caches, and execution separately. Use repeated device-event measurements or appropriately scoped graph replay, alternate baseline/candidate order, and retain samples and variability. Keep profiling runs separate from uninstrumented benchmark medians.
- Observe sustained power, clocks, temperature, and competing activity where available. Kernel design can change power limiting and reverse the ranking between short and long workloads. Do not substitute cold single-pass trace durations for sustained latency or treat coarse post-call clock samples as precise cycle normalization.
- Distinguish theoretical peak throughput, resource-derived occupancy, sampled device utilization, and measured accelerator work. Do not report HFU from a generic utilization percentage or infer a bandwidth/power ceiling without evidence.
- Cover the user's actual cases and important dispatch boundaries. For attention this includes short sequences, multiple head dimensions including 16/64/128/256/512, dense and causal visibility, rectangular shapes, GQA, key-padding and full masks, partial tiles, and backward paths. Use a focused matrix during diagnosis and broaden relevant correctness/performance checks before accepting a dispatch change.

### Validate the declared computation and report honestly

- Separate mathematical accuracy, the declared low-precision computation, compatibility with an execution order, and task-level quality. SDPA or another optimized backend is not a numerical oracle. Use stable FP64 equations on identical rounded inputs for diagnosis, and account for the specified operand rounding and accumulation floor when deciding whether the implementation is correct.
- Use `benchmarks.reference.rounded_reference` with the actual O to diagnose backward rounding, while retaining its independent forward expectation and the mathematical FP64 diagnostics. Its final FP64 products isolate operand rounding; its reduction order is not an exact replay of tensor-core instructions. Preserve the directed rounding fixtures in `tests/test_baseline_precision.py`, `tests/test_key_precision.py` and `tests/test_value_precision.py`.
- Retain the representative timing matrix's independent per-head FP64 norm-error limits of 1% for BF16 and 0.5% for FP16, separately for output and each gradient. Directed fixtures check the declared rounded equations; cancellation-sensitive common-offset and sharp signed-scale checks retain their qualified component norm limits. These checks do not impose universal pointwise or bitwise equality across FP32 reduction orders. Preserve recorded acceptance criteria with each result; derive new-shape checks from the same arithmetic and its rounding floor.
- Check outputs and dQ/dK/dV independently, including tails, all-hidden rows, negative maxima/LSE, signed or zero scales, strided views, and stale or poisoned storage. Finite loss alone is insufficient. Distinguish zeros expected from specified rounding from gradients lost through incorrect reconstruction or implementation errors. Use relative and absolute metrics appropriate to the declared accuracy contract.
- Audit FMA contraction, reduction/shuffle order, rounding boundaries, saturation, overflow, and accumulator types against the intended computation. Keep acceptance criteria stable within each comparison. Preserve failures and investigate their cause; neither widen tolerances to hide an implementation defect nor add precision work merely to satisfy a newly invented gate. Report stronger accuracy requirements and their results separately.
- Qualify reference-sensitive fused operations on actual intermediate tensors as well as synthetic inputs. A framework upgrade can change a native reduction policy. Identify the intended rounding and reduction behavior before making compatibility fixes, and restrict them to the versions, layouts, and devices actually checked.
- Validate graph replay with changed data, supported shape/stride changes, cache reuse, and selected fallbacks. Run relevant GPU correctness checks and sanitizer checks for changed memory ownership or asynchronous synchronization. Revalidate final source after meaningful changes to the implementation or toolchain.
- Keep calibration, selection, and evaluation data disjoint. For quality comparisons, retain paired decisions and resample independent source units; multiple questions from the same problem are not independent observations. Small score differences or agreement with a floating-point baseline do not establish an accuracy winner.
- Report measured scope, samples, precision contract, selected backend, and limitations alongside gains. Execution errors, OOM, numerical failures, and unsupported paths are not successful timing results. Distinguish released code from experimental candidates and keep the main README concise, with detailed evidence in the appropriate performance documentation.
- Preserve compact reproducible summaries, configuration/source hashes, and retained experiment code. Keep raw machine captures outside the tracked tree and apply the publication workflow below to every outgoing artifact and its metadata.

## Publication workflow

Commit and publish only within the user's authorized scope. Use repository-relative references and commands in publication text and generated metadata. Keep raw machine captures outside the tracked tree.

Install local hooks with `git config core.hooksPath .githooks`.

Before each commit, inspect the staged diff and exact commit message, then run:

```sh
python3 tools/check_public_text.py --staged --metadata
python3 tools/check_public_text.py --text-file commit-message.txt
```

Before each push, run `python3 tools/check_public_text.py --history --metadata`.

Before creating or editing a PR, comment, issue, release, or other publication, place the exact outgoing text in repository-relative temporary files and scan those files with `--text-file`. Send precisely the checked text. Never publish unreviewed shell output or attribution trailers. If a check cannot run, stop before publishing.

The guard blocks with a generic message; do not print matching content. Additional local aliases may be configured with repeated `git config --add privacy.identity VALUE` entries. Hooks supplement this workflow; do not bypass them.

Run guard tests with `python3 -m unittest tests.test_public_text -v`. Run the project's CPU checks with `CUDA_VISIBLE_DEVICES='' python3 -m unittest discover -v -f`; run relevant GPU checks for kernel changes.
