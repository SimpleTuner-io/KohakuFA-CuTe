# B200 allocator initialization diagnostic

The same standalone allocator executable passes the recorded physical checks
and old initcheck, but newer initcheck reports reads inside its allocation
lowering. This is an unresolved compiler/runtime/checker initialization report.
It does not qualify attention, automatic dispatch, or B300.

The compact [B200 evidence](../results/blackwell-allocator-initcheck.json) records
statuses, hashes, code association and limits. Raw loaded ELF files, sanitizer
logs and source/build snapshots remain outside Git.

## Observed result

Full B200, SM100a, 148 SMs; CUDA driver API 13.2 and runtime 13.2. NVCC 13.2.51 built
one executable with native SM100a code and no embedded PTX fallback. The two
producers share the same source, executable, CUDA libraries and launch protocol.

| Run | Allocation | Ordinary shared write |
| --- | --- | --- |
| Uninstrumented physical control | pass | pass |
| Initcheck 2026.1.0, build 37182542 | 0 errors, exit 0 | 0 errors, exit 0 |
| Initcheck 2026.3.0, build 38855100 | 102 errors, exit 99 | 0 errors, exit 0 |

All six processes verify one eager call and two graph replays with a changing
device generation. Each generation has 416 participating pointer readers and
96 dormant-thread guards; every output and guard matches. Collection took six
seconds, excluding provisioning, transfer and dependency checks. This is a
small diagnostic collection time, not an attention latency measurement.

The newer tool printed six reports because `--print-limit 6` was selected.
They describe one-byte shared reads at `0x5c`, beginning with thread 384. The
summary counts 102 total errors and 96 unprinted errors. The other 96 instances were not individually classified in that initial
collection; the unlimited-report follow-up below classifies all reports.

## Same-executable controls

A follow-up full-B200 collection ran 18 processes in 22 seconds. Its eight
physical captures all match the same compiler ELF above. Each process verifies
one eager generation and two changed-data graph replays with zero mismatches.
The old checker passes the untouched/roles allocation and ordinary pair. The
new checker reports the following complete factorial:

| User word | Post-allocation barrier | Readers | Allocation | Ordinary write |
| --- | --- | --- | --- | --- |
| Untouched | 416 roles | 416 | exit 99, 102 errors | exit 0, 0 errors |
| Sentinel | 416 roles | 416 | exit 99, 102 errors | exit 0, 0 errors |
| Untouched | Full CTA | 512 | exit 99, 102 errors | exit 0, 0 errors |
| Sentinel | Full CTA | 512 | exit 99, 102 errors | exit 0, 0 errors |

All four allocation logs retain all 102 reports, with no unprinted errors.
The analyzer checks every reported PC against the same retained SASS and inline
lineage. Each control has these classes:

| PC | Shared address | Width | SASS | Reports |
| --- | --- | --- | --- | --- |
| `0x190` | `0x5c` | 1 byte | `LDS.U8 R3, [R2+0x1c]` | 96 |
| `0x300` | `0x54` | 4 bytes | `ATOMS.OR RZ, [R2+0x14], R5` | 3 |
| `0x310` | `0x58` | 4 bytes | `ATOMS.OR RZ, [R2+0x18], R7` | 3 |

Every report names source line 75, inlined at kernel line 95. The byte load
reports threads 384–415 three times each; both atomic sites report thread 384
three times. The three ranges fit the ELF's explicit 96-byte reserved shared
section. The user pointer remains at `0x400`. The cluster-adjusted register
expression requires source/SASS review; the analyzer deliberately refuses to
infer a constant across its `LEA`.

The [compact controls evidence](../results/blackwell-allocator-controls.json)
records all 18 statuses, input/log hashes, physical equality and report classes.
Its final archive contains 108 entries, including 88 files; all recorded sizes
and hashes were independently rechecked. The measured CUDA source, executable
and compiler ELF are unchanged from the initial collection. Updated analyzer
code runs on retained files after collection, without another CUDA execution.

These controls disprove user-word sentinel initialization and the full-CTA
barrier as fixes for this executable's reports. They leave initialization of
compiler-reserved allocator state unresolved. The atomic sites show that
`ATOMS.OR` at shared `0x54` can occur inside standalone SM100a allocation. This
does not establish the original SM103 attention report's loaded instruction,
initialization provenance or checker behavior. No memory-qualification waiver
or attention dispatch change follows.

## Code and ABI association

Both uninstrumented producers capture a loaded ELF byte-identical to the
compiler cubin:
`caecd40f64aed3008db8fd8eac4cccbb91b21fd94fdcbc9d5c3f5bf2e631f87d`.
Runtime callbacks identify the actual `tmem_init_allocation` and
`tmem_init_ordinary` launch symbols. No instrumented ELF was captured: CUPTI
capture was explicitly disabled for both sanitizer versions because a previous
controlled experiment reproduced a CUPTI/sanitizer conflict. Matching the
uninstrumented ELF strengthens source association; it does not recover the
checker's patched image.

The first new report names inline `inspect<true>+0x190`, source line 75, with
containing frame `tmem_init_allocation+0x20`, line 95. `nvdisasm -gi` places the
following instruction at kernel offset `0x190`, annotated line 75 inlined at 95:

```text
LDS.U8 R3, [R2+0x1c]
```

The ELF has no out-of-line `inspect` function body. The containing-frame offset
is inline-callsite metadata; adding it to `0x190` would select an unrelated
instruction. Source PTX independently assigns the same inline lineage to the
`tcgen05.alloc` instruction. This establishes the association in the
uninstrumented loaded ELF, while the instrumented-image limitation remains.

Immediately before that load, `R2` is initialized to `0x40` and adjusted by a
cluster-CTA address term. Its local address is consistent with `0x5c` for the
recorded single-CTA launch. The ELF explicitly declares
`.nv.shared.reserved.0`, size `0x60`; the reported byte lies in that reserved
region. The application's four-byte pointer word is at `0x400`. The runtime
reports four user shared bytes; the ELF function shared section is `0x404`
bytes. These are distinct views; their difference does not establish that
every padding byte is reserved system state.

The byte is tested before allocation; a nonzero value takes an error path.
The lowering of `relinquish_alloc_permit` later writes 1 to the same byte. This
supports an **inference** that it tracks allocation-permit state. The exact
runtime initialization and checker shadow-state provenance remain unverified.
[NVIDIA's PTX specification](https://docs.nvidia.com/cuda/parallel-thread-execution/)
separates system-reserved shared storage from user storage; direct user access
to the reserved region is undefined behavior. Zeroing `0x5c` is not a valid fix.

The old clean result is not proof that the newer report is false. It could
reflect changed coverage, a checker defect, or compiler/runtime initialization.
The [sanitizer release notes](https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html)
mention shared-memory and `atom.exch` fixes, but neither identifies this
`LDS.U8` report. No blanket exception follows.

## Reproduce

The source has no attention, TMA copy, GEMM, tensor-memory data load/store,
framework allocator, or floating-point reduction. One fully active warp 12
allocates 512 TMEM columns; the 416-role barrier joins warps 0–12. All readers
finish before warp 12 deallocates. The ordinary control publishes zero to the
same legitimate user word. Sentinel and full-CTA options exist in the
unchanged measured source and were exercised by the follow-up above.

Use Python 3.13 for the CPU-only builder, a configured Linux CUDA/CUPTI toolkit,
and an independently checked newer sanitizer directory. Reproduce the recorded
compiler with NVCC 13.2.51; a changed compiler is a separate experiment.

```sh
python3 -m benchmarks.diagnostics.tmem_init --arch sm_100a \
  --sanitizer-dir "$NEW_SANITIZER_DIR" --output "$ARTIFACT_DIR/build"
```

The builder retains source, PTX, cubin, inline SASS, resources and hashes. It
uses explicit virtual/real architecture flags: shorthand `-arch=sm_100a`
can add a generic fallback that rejects architecture-specific instructions.
Its PTX detector distinguishes actual entries from `No PTX file found` text.
CUDART/CUPTI are bundled with `$ORIGIN/lib`; a CPU loader trace checks their
resolution without entering application main or initializing CUDA.

The public builder also passed a CPU-only build and loader check on the
recorded compiler. That build has a different cubin/executable hash from the
measured B200 bundle and was not executed on B200; its successful build does
not extend the physical qualification. The measured CUDA source is byte-identical.

The runner requires the caller's common GPU lock as inherited descriptor 9,
an already installed old sanitizer, and fresh external output storage:

```sh
cd "$ARTIFACT_DIR/build"
timeout --signal=TERM --kill-after=2s 110s \
  bash run_tmem_init.sh "$ARTIFACT_DIR/results"
```

It performs two captured physical controls and four same-executable old/new
tool controls. Every status, including 99 and timeout, is retained. Its final
collection marker is not a qualification pass. Raw outputs must remain outside
Git. Use `sm_103a` only for a separately authorized, separately qualified B300
experiment; this B200 result does not substitute for it.

For retained source association:

```sh
python3 -m benchmarks.diagnostics.blackwell_init_association \
  --log "$INIT_LOG" --sass "$SASS" --inline-sass "$INLINE_SASS" \
  --cubin "$CUBIN" --loaded-cubin "$LOADED_CUBIN" \
  --symbol tmem_init_allocation --output "$ARTIFACT_DIR/association.json"
```

## Separate failures and next control

The original B300 attention report is distinct: four-byte shared read at
`0x54`, reported PC `0xee0`. Its retained compiler artifact contains
`ATOMS.OR`, not this B200 byte load, and lacks the same direct loaded-image
binding. It remains unresolved. The allocator diagnostic cannot waive that
attention report or downstream output reports.

The completed follow-up retains all newer reports with `--print-limit 0`,
compares legitimate user-word sentinel initialization and the full 512-thread
barrier, and preserves paired same-executable physical controls. Neither
intervention removes the allocator reports. Focused memcheck/synccheck could
independently test source ownership, but those checks were not performed in
this collection. Do not write reserved bytes. Any further GPU collection
requires its own authorized setup, storage, transfer and archival budget.

## Sanitizer API review and optional controls

The reviewed NVIDIA SDK is 13.4.92 (Compute Sanitizer 2026.3.0). Its documented
interfaces do not establish a supported export of initcheck's patched ELF:

| Interface | Available evidence | Missing evidence |
| --- | --- | --- |
| `RESOURCE_MODULE_LOADED` | Associated cubin bytes | No guarantee they contain initcheck patches |
| `RESOURCE_FUNCTIONS_LAZY_PATCHED` | Context, module, function handles | No byte buffer or reported-PC mapping |
| `sanitizerGetFunctionPcAndSize` | Function PC and size | No specified original/patched distinction |
| Custom patch callbacks | PCs for the caller's own patches | Not initcheck's instrumentation |

The [module payload](https://docs.nvidia.com/compute-sanitizer/api/struct_sanitizer___resource_module_data.html),
[lazy-function payload](https://docs.nvidia.com/compute-sanitizer/api/struct_sanitizer___resource_functions_lazy_loaded_data.html)
and [patching API](https://docs.nvidia.com/compute-sanitizer/api/group___s_a_n_i_t_i_z_e_r___p_a_t_c_h_i_n_g___a_p_i.html)
define these boundaries. The
[API guide](https://docs.nvidia.com/compute-sanitizer/SanitizerApiGuide/index.html)
also permits only one active subscriber. An additional observer cannot be
assumed to coexist with the built-in checker. Replacing its subscriber or
interposing private calls would change the experiment. No such collector was
built, and no instrumented-image capture is claimed. Code PCs are not assumed
to be readable CUDA data pointers. This is a limitation of the reviewed public
interfaces, not proof that no internal extraction mechanism exists.

The optional `controls` runner mode reproduces the independent controls. It
keeps the measured CUDA source and executable unchanged and selects existing
runtime branches: allocation/ordinary producer, untouched/sentinel user word,
and 416-role/512-CTA barrier. Eight physical captures must pass and match the
provided compiler ELF before any instrumentation. New initcheck runs all eight
cases with unlimited reports; old initcheck retains the untouched/roles pair.
This is 18 processes. The measured follow-up above uses this mode. CPU tests
use synthetic tools and validate collection behavior, not CUDA ownership or
real lock exclusion.
The default mode retains the original six-process comparison and print limit.

```sh
cd "$ARTIFACT_DIR/build"
timeout --signal=TERM --kill-after=2s 110s \
  bash run_tmem_init.sh "$ARTIFACT_DIR/controls" controls
python3 blackwell_init_association.py \
  --log "$ARTIFACT_DIR/controls/new-allocation-untouched-roles.log" \
  --counts-only --output "$ARTIFACT_DIR/counts.json"
```

Use inherited lock descriptor 9, checked input hashes, the same native ELF,
and the existing portable CUDA libraries. Retain every process status and
all reports. The count parser requires a complete unique error summary;
unprinted-error subtotals cannot replace total errors. Count consistency alone
is not a clean-memory result. Full source association additionally requires
`--sass`, `--inline-sass`, `--cubin`, `--loaded-cubin` and the selected symbol;
it groups read reports by address, PC, size and instruction. Association still
uses the uninstrumented artifact. Changed outcomes from sentinel or CTA barriers
need instruction-level review; neither permits writing reserved shared state
or waiving the distinct B300 report.
