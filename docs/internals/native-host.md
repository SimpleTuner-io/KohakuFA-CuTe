# Native host ownership

The shared warp-MMA GPU methods live in `src/kohakufa_cute/fa4_sm80`.
Qualified public routes use cached native `AttentionRunner` objects; the Python
wrapper remains an explicit implementation control. A runner caches configuration
and compiled handles while receiving fresh tensors, scale, device, and current
stream on every call. See [installation](../installation.md) for the private-API pins.

## Buffers and lifetime

For square BHSD inputs, `R = ceil(S / backward_m) * backward_m`:

| Buffer | Shape / dtype | Producer and last consumer |
| --- | --- | --- |
| Output | `[B,H,S,D]`, input dtype | Forward; backward delta preparation |
| Maximum and log normalization | `[B,H,S]`, FP32 | Forward; backward reconstruction |
| Delta and padded log normalization | `[B,H,R]`, FP32 | Preparation; joint backward |
| dQ accumulator | `[B,H,R*D]`, FP32 | Preparation zero-fill; atomic accumulation; conversion |
| dQ/dK/dV | `[B,H,S,D]`, input dtype | Conversion or joint epilogue; caller |

Autograd retains Q/K/V, the actual rounded owned output, and its statistics.
Preparation zeroes dQ scratch every invocation and computes delta from that output
and dO. It copies shifted log2 normalization without a second log-base conversion.
Allocation must use the selected backward tile, including M128 tails.

BSHD launch views change metadata without copying. Expanded dO becomes contiguous;
an unaligned starting address requires an aligned clone before vectorized loads.
Owners survive every asynchronous consumer and graph replay. Compiled/layout caches
must never retain stale operand pointers or replace live metadata checks.

## Configuration and execution

`AttentionRunner("warp", configuration)` validates schedule fields, architecture,
shared-memory capacity, alignment, and pipeline geometry. The adapter forwards all
14 [schedule fields](scheduling.md); unknown controls must fail visibly.
The native entry owns allocation, autograd, non-owning DLTensor descriptors, and
current-device/current-stream scopes. Autograd retains configuration after the
Python runner object is released.

Forward launches its selected GPU method. Backward launches preparation, joint
main, and dQ conversion with scale applied after FP32 gradient accumulation.
Forward-only GB10 uses a separately qualified inference entry. Exported FFI
schemas are authoritative: Python-specialized `None` arguments need not survive
in the packed ABI. Inspect the actual schema before calling raw functions.

Cache keys distinguish device, dtype, layout/alignment, causal mode, schedule,
thread/warp geometry, and relevant shape boundaries. Scale remains a live scalar
within a qualified family. Validate changed inputs/scales, actual statistics,
poisoned replay, tails, output-gradient alignment, and runner lifetime after changes.

Atomic backward checks the live [determinism policy](../determinism.md) on every
host invocation. Captured graphs require the policy to be set before capture.
