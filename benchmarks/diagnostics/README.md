# Standalone rank-4 TMA initialization diagnostic

On the recorded RTX 5090 stack, initcheck does not recognize writes from this
standalone `UTMASTG.4D` producer even though every output byte has the expected
value. Ordinary stores pass; poisoning the TMA destination removes the reports;
waiting for full destination completion does not. The
[compact evidence](../../results/tma4d-store-summary.json) retains all 33 executed
cases, status codes, source/binary hashes and code-binding scope.

This is a constant-store diagnostic with no attention arithmetic or performance
measurement. It does not qualify the experimental H200 attention implementation
or waive any of its unresolved initialization reports.

## Measured result

The device was a full RTX 5090, SM120, with 170 SMs. The standalone program used
NVCC 13.2.51, CUDA driver/runtime API version 13.2, CUPTI and Compute Sanitizer
**2026.1.0.0**. It does not use PyTorch or CuTe. All runs use the same executable.

| Producer | Initial output | 256 rows: initcheck status / errors | 257 rows: initcheck status / errors |
| --- | --- | ---: | ---: |
| TMA + read wait | untouched | 99 / 36,864 | 99 / 37,008 |
| TMA + full wait | untouched | 99 / 36,864 | 99 / 37,008 |
| Ordinary stores | untouched | 0 / 0 | 0 / 0 |
| TMA + read wait | BF16 NaN poison | 0 / 0 | 0 / 0 |
| TMA + full wait | BF16 NaN poison | 0 / 0 | 0 / 0 |
| Ordinary stores | BF16 NaN poison | 0 / 0 | 0 / 0 |

All 12 uninstrumented controls pass actual loaded-code binding and physical
storage verification. All 12 initcheck runs also pass physical storage
verification, including the four that return 99. The nine focused memcheck,
racecheck and synccheck runs pass, covering all three producers with untouched
257-row output. Collection took 57.81 seconds and retained terminal status **99**.
Error counts are sanitizer reports, not counts of incorrectly stored elements.

The first reported consumer is `tma4d_consume+0x140`, reading 16 bytes. Retained
reported addresses map into the logical output allocation. The uninstrumented
loaded consumer has an `LDG.E.128` at that offset. The loaded TMA producer has an
actual `UTMASTG.4D`; the ordinary control has ordinary stores. Raw addresses,
loaded ELF/SASS and complete logs are retained outside Git.

## Memory ownership and independent oracle

The global tensor has fastest-to-slowest dimensions `[64, rows, 2, 2]`. Each
higher stride includes a physical gap, so this is a genuinely noncontiguous
rank-4 descriptor. Its BF16 box is `[64, 128, 1, 1]`, with unit traversal strides
and no swizzle or interleave. The 257-row case includes a tile with one valid row.

The untouched mode initializes only gaps and guards. It leaves every logical
output byte unwritten until the selected producer. A separate diagnostic fills
logical output with BF16 NaN before each overwrite. Every physical word is then
checked: logical elements must equal the current constant, while gaps and both
guards must retain their distinct pattern.

An ordinary GPU consumer reads every physical 16-byte chunk. The independent
host oracle constructs its expected mask by forward enumeration of batch, head,
row and column; it does not reuse the initializer/consumer's inverse address
mapping. The eager call writes BF16 1. Two replays of the same graph read a
changing device scalar and must write BF16 2 and 4, respectively. The host checks
against its own expected generation, preventing a shared stale scalar from
hiding an error in both GPU producer and consumer.

Every shared-memory writer executes `fence.proxy.async.shared::cta`, followed by
CTA synchronization. A single initiating thread issues and commits the TMA copy,
waits for either source-read or full completion, then rejoins the CTA before
shared storage expires. The consumer runs later in the same stream without PDL.
The immutable aligned descriptor is a `const __grid_constant__` argument. Output,
payload, descriptor, graph and stream owners survive until all calls complete.

## Captured code and sanitizer controls

CUPTI captures the actual uninstrumented loaded ELF and eager launch symbol.
The runner checks only the selected producer function's SASS, rejecting a nearby
function, collapsed rank or empty disassembly. All 12 captured controls pass.

The initial attempt to run CUPTI capture under Compute Sanitizer failed before
the first instrumented API call, including for ordinary stores. Those failures
remain execution limitations, not initialization results.

The final comparison explicitly disables CUPTI subscription **only** in the
sanitizer process. The runner requires an uninstrumented control matching the
exact source, executable and PTX hashes, producer, geometry and initialization,
with successful physical verification and direct loaded-code binding. Its
instrumented record says `unavailable_under_sanitizer` and associates the
uninstrumented control separately. It never claims directly captured
instrumented ELF. All sanitizer flags and raw statuses are retained.

These controls support an initialization-tracking failure for this standalone
rank-4 producer on the recorded SM120 stack. They do not prove that every
attention initialization report is a tool defect. In particular, the H200
attention implementation remains experimental with unresolved unmodified
initialization checking. Its prior actual-attention 4D controls and independent
constant 3D diagnostic remain separate evidence.

## Checker-version control

A separate [checker-only comparison](../../results/tma4d-checker-comparison.json)
reused the identical NVCC 13.2 executable, source/PTX and previously captured
loaded-code controls. Compute Sanitizer **2026.3.0.0** reproduces the results
from **2026.1.0.0**: untouched read/full-wait TMA output reports 36,864 errors
at 256 rows and 37,008 at 257 rows; ordinary untouched output and poisoned
full-wait TMA output report zero at 257 rows. All twelve physical checks pass.

Only the checker changes. The newer executable comes from the official CUDA
13.4.3 redistribution, verified against its published SHA256. Compiler, CUDA
runtime, attention precision and diagnostic executable are unchanged. NVIDIA's
[release notes](https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html)
describe newer shared-memory initialization-tracking fixes; this controlled
comparison shows they do not remove this rank-4 global-store report. No H200
or B300 attention finding is waived.

## Reproduce

Use Python 3.13, CUDA Toolkit 13.2 with NVCC/CUPTI, and Compute Sanitizer. The
matrix is intentionally scoped to SM120. Run from the repository root, with a
fresh output directory outside the checkout:

```bash
python benchmarks/diagnostics/tma4d_matrix.py --output "$ARTIFACT_DIR/tma4d"
```

The matrix builds once, runs 12 captured controls, then 12 matching initcheck
controls with capture disabled. It runs nine focused memory/race/sync controls
when physical checks and same-executable associations pass. Its overall limit
is 240 seconds with a 45-second per-case limit. The recorded result returns 99;
inspect `matrix.json` rather than treating collection as an initialization pass.

Individual controls can also be selected explicitly:

```bash
python benchmarks/diagnostics/tma4d_store.py build \
  --arch sm_120 --output "$ARTIFACT_DIR/build"
python benchmarks/diagnostics/tma4d_store.py run \
  --build "$ARTIFACT_DIR/build" --mode read --initialization uninitialized \
  --rows 257 --output "$ARTIFACT_DIR/control"
python benchmarks/diagnostics/tma4d_store.py run \
  --build "$ARTIFACT_DIR/build" --mode read --initialization uninitialized \
  --rows 257 --tool initcheck --capture disabled \
  --control-result "$ARTIFACT_DIR/control/result.json" \
  --output "$ARTIFACT_DIR/instrumented"
```

The runner retains source/build hashes, the raw encoded descriptor, physical
allocation ranges, changing-data checks, code bindings and logs. It preserves an
inherited GPU-lock descriptor 9 and cleans up the whole owned process group on
timeout, TERM or INT, including sanitizer target children. Serialize independent
GPU campaigns. A timeout, capture failure, physical mismatch or unrelated
sanitizer report remains a failure requiring investigation.

## Official protocol references

- The [CUDA Programming Guide's asynchronous-copy section](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)
  documents immutable grid-constant descriptors, per-writer shared proxy fences,
  CTA synchronization, S2G completion and out-of-bounds tiles.
- The [PTX bulk-tensor specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor)
  defines generic rank-4 S2G copying on `sm_90` or higher. The
  [bulk-group wait specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-wait-group)
  distinguishes source-read from destination-write completion.
- The [driver tensor-map API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__TENSOR__MEMORY.html)
  specifies descriptor alignment, base/stride alignment, rank and box limits.
- [Compute Sanitizer's known limitations](https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html#known-limitations)
  describe incomplete global tracking for some tensormap instructions. That
  general limitation supplies context; the controls above establish the scope
  of this particular diagnostic.

## Standalone Blackwell allocator control

The [B200 allocator diagnostic](../../docs/blackwell-allocator-initcheck.md)
isolates an initialization report from attention and tensor-memory data
operations. It includes native-code build tooling, captured-code association,
and paired old/new initcheck evidence. The allocator report and the separate
B300 attention report remain unresolved; this control does not qualify either
attention backend.
