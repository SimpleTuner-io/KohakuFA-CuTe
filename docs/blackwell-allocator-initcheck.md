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
summary counts 102 total errors and 96 unprinted errors. The other 96 instances
have not been individually classified.

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
same legitimate user word. Sentinel and full-CTA options already exist in the
unchanged measured source, but were not run in this B200 collection.

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

A future bounded session should first retain **all** newer reports with
`--print-limit 0`, then compare legitimate user-word sentinel initialization
and the full 512-thread barrier against the current 416-role control. Keep the
same executable and paired physical controls; do not write reserved bytes.
The reported byte load precedes the post-allocation role barrier and addresses
a different word from the sentinel. If either control changes the report, inspect
its executed code before attributing the change to repaired ownership. Focused
memcheck/synccheck can test the source ownership independently. These
are proposed tests, not completed results. Before another rental, verify that setup, storage, transfer and archival
fit its authorized budget.
