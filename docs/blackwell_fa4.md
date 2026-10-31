# Experimental Blackwell joint attention: unresolved initcheck reports

The optional Blackwell schedule retains the project's probability and gradient
arithmetic while reusing FlashAttention-4's joint backward ownership. On the
B300 qualification matrix, it reduces complete forward plus backward time by
1.50–2.03× relative to the previous native CuTe path. The candidate takes
2.4–16.7% more time than standalone FA4 in these combined calls, under FA4’s
own arithmetic policy.

Memcheck, racecheck and synccheck pass the final six-case source checks.
Initcheck reports remain under investigation, so these results identify a
candidate and do not qualify automatic dispatch.

The compact measurements, round samples, source hashes, component norm errors
and directed checks are in [the retained comparison](../results/b300-fa4.json).
Raw profiler captures and machine code are kept separately.

## Scope and arithmetic

The current qualification covers a full B300 SXM6 AC: SM103, 148 SMs and an
1100 W power limit. The timing matrix uses B=1, H=16, FP16/BF16, D64/D128,
2048/4096/8192 tokens, and dense or square causal visibility. Directed checks
also cover H1/H3/H4/H5/H24, partial tiles, BSHD storage views, changed-input
CUDA graphs, negative maxima, cancellation and expected FP16 probability
underflow. SM100 requires its own qualification before dispatch is enabled.

QK, dP and gradient products accumulate in FP32. Forward saves a literal
unscaled score maximum separately from its logarithmic normalization. Backward
subtracts the maximum before scaling, reconstructs P, converts it once to the
input dtype before dS, and converts dS once for gradient products. Delta uses
FP32 products of the actual rounded O and dO. dQ/dK scaling follows FP32
accumulation. There are no residual products or automatic compensation.

Standalone FA4 uses combined scaled LSE and unrounded FP32 P when forming dS.
Its forward also permits a lazy normalization anchor with rescale threshold 8;
the retained schedule currently uses threshold 0. These measurements compare
complete implementations. They do not attribute every difference to scheduling.
Late dQ/dK scaling is already present in FA4 and is not a newly removed cost.

Random timing cases retain independent per-head FP64 norm limits of 1% for
BF16 and 0.5% for FP16, separately for output and each gradient. Directed
rounding fixtures use the declared rounded reference with actual O. FP32
reduction ordering and specified operand underflow remain part of that contract.

## What changed

The prior query plus key/value schedule recomputed score and dP products in
separate owners. The joint schedule computes those once alongside the three
gradient products. Key/value tiles own dK/dV; dQ accumulates into FP32 storage
before a postprocess applies its scale and output conversion. The atomic dQ
accumulation does not provide deterministic reduction ordering.

The first direct adaptation was still slow. A four-policy diagnostic held the
native forward and actual O fixed while separating combined versus shifted
statistics and unrounded versus rounded P. Most of the added time was inside
the main kernel. Systems showed nearly equal preprocess/postprocess costs and
an unchanged work grid; extra API preparation did not explain the gap.

Three source changes removed substantial overhead:

- dS reuses P that has already been packed to the input dtype for its gradient
  product. Converting that fragment to FP32 storage does not recover discarded
  bits. It avoids a second scalar low-precision conversion and reduces live
  probability storage.
- The positive-scale path subtracts the unscaled maximum first, then uses a
  paired FP32 FMA for scale-log2 and logarithmic normalization. This preserves
  the shift and operand-rounding boundaries while allowing ordinary FP32
  contraction.
- The producer stages the maximum with logarithmic normalization. They occupy
  separate halves of the small shared statistic slot and complete one doubled
  byte transaction. Compute warps consume both before releasing that stage.
  Padding is allocated with the maximum and initialized in the existing
  forward statistic epilogue, including zero filling required tail rows.
  This adds no separate preparation launch.

A register-vector maximum experiment was slower: its broadcast layout emitted
128 scalar global loads over the inspected unrolled work, versus 2 lane loads
and 64 shuffles in the lane-load version. The retained producer path removes
that dependent global-load/shuffle work from compute warps. Instruction counts
are static evidence over matched work, not runtime percentages.

Wider forward tiles, earlier unpacking of rounded P, register redistribution,
and wider output-correction transactions were evaluated separately. A smaller
instruction count or higher register allowance did not establish a retained win.
Promising numerical and synchronization failures remain reproducible experiments.

## Runtime and fallback boundary

The retained methods are static licensed subclasses, with private interface
function globals and compilation caches. The retained static schedule does not edit
installed source or mutate the standalone FA4 module's class selection.
Unchanged scheduling helpers use FA4 revision
`4616f5116d33e3ba3cbb71d582caffcfcb49649c`; source checks reject a changed helper
interface. The BSD license and contributor list are included under `LICENSES`.

Measurements used Python 3.13.18, Torch 2.15.0+cu132, CuTe DSL 4.8.1,
TVM FFI 0.1.15.post0, quack-kernels 0.6.5 and torch-c-dlpack-ext 0.1.5. Install
an already configured CUDA 13.2 Torch environment before adding the optional
source-pinned dependency:

```sh
python -m pip install '.[cuda,blackwell-fa4]'
```

Automatic optional dispatch is disabled while initialization classification remains
unresolved. The explicit `make_static_attention` factory remains available for
review and profiling. The separately tested experimental boundary is eager SM103, B=1, H≤32, square MHA,
FP16/BF16 D64/D128, positive finite scale ≤2e38, no mask/dropout, contiguous final
dimension, and 2048–8192 tokens. Other calls and missing or unqualified optional
runtimes retain the existing CuTe implementation. Compiled calls and deterministic
execution also retain that implementation. An actual failure of an explicitly selected
experimental backend propagates; it does not silently change the computation.

D512 was probed through the real FA4 forward and backward APIs independently.
They raise ValueError and AssertionError, respectively, on the checked square
D512 case. There is no standalone FA4 D512 speed comparison or implicit SDPA
fallback in this evidence.

## Profiling and memory ownership

Systems captures record the complete forward, preprocess, joint backward and
postprocess order. Compute performance counters were denied by the host; no
stall percentages, cache-hit rates or HFU are inferred from that denial.
Register metadata alone does not identify executed spills: selected SASS
loads/stores and role-specific register allocation were inspected separately.
Timing medians use warmed uninstrumented graph rounds with rotated backend
order. Instrumented sanitizer or profiler durations are not performance samples.

Memory checks are kept separate from host API symbol-probe diagnostics. The
installed sanitizer's extended API reporting counted unsupported future-version
`cuGetProcAddress` lookups. Disabling API-error reporting leaves memory checking
active; the original diagnostic report is retained.

Sanitizer 2026.3 reproduces shared-statistic WARs in original FA4 and the first
adaptation. Matched SASS identifies a bulk-copy writer and shared readers of
the delta and paired maximum/normalization slots. Separate handoffs now join
all 256 compute threads and the 32 load threads before either slot is reused.
First and last pipeline generations avoid unmatched arrivals. A compute-only
fence did not repair the report.

Synccheck then exposed a distinct issue at the inherited tensor-memory barrier:
416 threads from the MMA, compute and dQ-reduction roles enter waits at different
instruction sites. The SDK's allocator reconstruction replaces a customized
barrier with the base type. The local schedule therefore performs the three
allocation waits through its local role-aware barrier before retrieving the
pointer, preserving allocation, retrieval, final joining and free ownership.
Cross-role waits use explicit `barrier.sync` without `.aligned`, with a compiler
memory clobber; arrivals use the matching non-aligned instruction. The 416-thread
allocator barrier and 288-thread statistic handoffs keep their original counts.
The corrected prototype passes six BF16/FP16 synccheck cases, including dense
257- and 513-token inputs that execute repeated handoff generations. Final
static-source validation is recorded separately.

The distinction between legacy aligned `bar.sync` and role-aware `barrier.sync`
is specified by the [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar).

Tool semantics and version changes are documented in [NVIDIA’s manual](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) and [release notes](https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html).

## Reproduce the retained comparison

The explicit control keeps the prior native CuTe implementation selected even
when the optional schedule is installed. The candidate factory bypasses that
public dispatch guard; standalone FA4 remains unchanged.

```sh
python -m benchmarks.compare_flashattention --backends cute fa4 \
  --context benchmarks.blackwell_fa4_controls:previous_native_dispatch \
  --candidate benchmarks.blackwell_fa4_experiments:make_static_attention \
  --seq 2048 4096 8192 --dim 64 128 --dtype bf16 fp16 \
  --modes dense causal --repeats 5 --window-ms 100 \
  --flash-revision 4616f5116d33e3ba3cbb71d582caffcfcb49649c \
  --output b300-fa4.json
python -m benchmarks.qualify_blackwell_fa4 --static --output directed.json
python -m unittest tests.test_fa4_blackwell -v
python -m benchmarks.qualify_blackwell_experimental_public --output public.json
```

These checks reproduce the recorded experimental scope; initialization
qualification remains open. Previously
recorded KV/forward racecheck reports in the native fallback remain separate,
unresolved evidence. No repository-wide sanitizer-clean claim is made.

The final source passes memcheck, racecheck and synccheck for both dtypes at
129/D64 causal, 257/D128 dense and 513/D64 dense, including changed-input graphs.
The all-address-space initcheck run reports shared and downstream global reads;
it was bounded after reproducing the failures. A focused, unmeasured diagnostic
poisoned the allocator holding word with an invalid integer before allocation.
The allocator replaced it with valid pointer value 0 in all three recorded CTA
slots, and independent FP64 forward checks passed. Original FA4 reproduces the
shared-shadow report. This proves the allocation producer executes correctly. The shared read report
persists under the pointer-poison intervention; exact loaded-code association
for that read is incomplete. Both shared and downstream reports remain
unresolved, and neither is waived as a tool false positive. No blanket initialization-clean or
production-ready claim follows from the pointer control. Additional stores,
barriers and output poisoning are diagnostic-only and remain outside the default
and measured kernels.

Experimental public verification also covers all three exported APIs on H32/S2049/D64,
changed-input graph replay, and a fresh fullgraph compile. Runtime correlation
inside each eager NVTX range identifies exactly forward, preprocess, joint
backward and postprocess kernels. Captured public graphs select the same static
schedule; the compiled graph selects the existing native CuTe pipeline. A stable
`run_attention` package entry prevents Python’s implementation-module import
from replacing its callable. A repeated-import regression and the GPU partial-tile
test pass. Runtime-entrypoint hashes are recorded separately from the earlier
matrix snapshot; the six numerical implementation modules are unchanged.
These traces used the experimental selection path before the final host gate
was disabled. They do not qualify automatic/default dispatch. CPU tests verify
that the final gate returns before runtime loading or GPU geometry checks;
the GPU regression opts into the experimental path explicitly.

The retained host factory and checkers omit abandoned tuning prototypes. Their
preparation cleanup leaves all six numerical implementation modules unchanged;
the original timing and trace source hashes remain in the retained results.

The focused pointer control and bounded initialization reproducer are explicit
experiments, with no latency claim:

```sh
python -m benchmarks.check_blackwell_forward_pointer --output pointer.json
compute-sanitizer --tool initcheck --initcheck-address-space all \
  --report-api-errors no --error-exitcode 5 \
  python -m benchmarks.check_blackwell_handoff_memory \
  --factory benchmarks.blackwell_fa4_experiments:make_static_attention \
  --quick --no-graphs --output initialization.json
```

The bounded command is expected to retain an initialization failure while the
reports are unresolved. Its nonzero exit is not a successful qualification.
