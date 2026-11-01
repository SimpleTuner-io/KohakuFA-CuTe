# Shared SM80 host launch contract

The static kernels live in `src/kohakufa_cute/fa4_sm80`. This document records
the current callable arguments for a cached host launcher. The existing Python
wrapper in `_attention.py` owns allocation, layout views, preparation, and
autograd. Its short-call wall overhead motivates moving those operations into
the native extension while retaining the same GPU methods.

## Buffers and ownership

The API receives contiguous, 16-byte-aligned `[B,H,S,D]` Q/K/V, with equal shapes
and input dtype FP16 or BF16. The current selected schedule uses `B=1`, D64/D128, square
attention, and positive scale. A BSHD view has strides
`(H*S*D, D, S*D, 1)`; it changes metadata without moving data.

| Buffer | Shape | Dtype | Producer / last consumer |
| --- | --- | --- | --- |
| `out` | `[B,H,S,D]`, BSHD view at launch | Input dtype | Forward / delta preparation |
| `maximum` | `[B,H,S]` | FP32 | Forward / joint main |
| `logsum` | `[B,H,S]` | FP32 | Forward / preparation |
| `delta` | `[B,H,R]` | FP32 | Preparation / joint main |
| `lognorm` | `[B,H,R]` | FP32 | Preparation / joint main |
| `dq_accum` | `[B,H,R*D]` | FP32 | Preparation zero-fill, joint atomic accumulation / dQ conversion |
| `dq`, `dk`, `dv` | `[B,H,S,D]`, BSHD views at launch | Input dtype | Conversion or joint epilogue / caller |

`R = ceil(S / backward_m) * backward_m`. The default `backward_m` is 64.
Preparation zero-fills `dq_accum` on every invocation. `maximum` is the literal
unscaled maximum and `logsum` is the shifted log2 normalization. Preparation
copies that normalization into padded `lognorm`, without another log-base
conversion, and reduces actual rounded `out` with `dout` in FP32.

Forward saves Q, K, V, actual owned `out`, `maximum`, and `logsum` for autograd.
Fresh operands, statistics, output, and gradient pointers must reach each call
and graph capture. Cache compilation and layout classes; preserve tensor buffer
ownership and initialization lifetimes. Expanded `dout` currently converts to a
contiguous BHSD tensor before the BSHD view. A contiguous dO whose starting
address is not 16-byte aligned requires an aligned clone before the vectorized
backward loads. Unaligned Q/K/V use the native CuTe path.

## Current compiled callable arguments

The following lists are the Python invocation contracts of the compiled CuTe
callables. `Aux` means `AuxData(tensors=[maximum], scalars=None)`. All listed
`None` values are fixed for this bounded square-MHA specialization. CUDA streams
are supplied through the TVM FFI environment stream; the compiler receives a
fake environment stream and the runtime invocation omits it.

Forward, 18 runtime positional arguments:

```text
F(q_bshd, k_bshd, v_bshd, out_bshd, logsum, scale,
  None, None, None, None, None, None, None, None, None, Aux, None, None)
```

There are nine `None` placeholders between `scale` and `Aux`, followed by two
tile-scheduler placeholders. The first nine are cumulative query/key lengths,
used query/key lengths, page table, left/right windows, sink, and block sparsity.
The direct source signature is `_forward.FlashAttentionForwardSm80.__call__`.

Preparation, 14 runtime positional arguments:

```text
P(out_bshd, dout_bshd, delta, logsum, lognorm, dq_accum,
  None, None, None, None, None, 1.0, None, None)
```

The five placeholders after `dq_accum` represent cumulative query lengths,
used query lengths, derivative of LSE, alternate row maximum, alternate
probability scale. The scalar `1.0` occupies
the scale slot; the final two placeholders are scheduler metadata and O-low.
The direct source signature is the pinned
`FlashAttentionBackwardPreprocess.__call__`; `_preprocess.py` overrides its GPU
kernel. The current Python call has 14 explicit positional values. Its optional
compile-time stream is supplied during compilation.

Joint main, 22 positional arguments:

```text
B(q_bshd, k_bshd, v_bshd, dout_bshd, lognorm, delta, dq_accum,
  dk_bshd, dv_bshd, scale,
  None, None, None, None, None, None, None, None, None,
  Aux, None, None)
```

The nine placeholders before `Aux` are cumulative query/key lengths, used
query/key lengths, left/right windows, and dQ/dK/dV semaphores. The final two
are block-sparse tensors and scheduler metadata. The direct signature is the
pinned `FlashAttentionBackwardSm80.__call__`; `_backward.py` overrides its
main kernel and tile computation.

dQ conversion, seven runtime positional arguments:

```text
C(dq_accum, dq_bshd, scale, None, None, None, None)
```

The placeholders are cumulative lengths, used lengths, sink tensors, and
scheduler metadata. Compilation must use actual architecture, main thread
count 256, `backward_m=64`, `AtomLayoutMdQ=2`, and `swap_ab=False`. A four-warp
conversion reads a different scratch layout and permutes dQ.

These invocation lists include Python-specialized arguments. They do not assert
that an exported packed FFI function retains every `None` slot. The exported
schema must be inspected before calling a raw packed function from C++.

## Flat host entry

A native bridge can compile a small CuTe host object that accepts flat tensor
arguments and invokes the static program objects above. This keeps optional
metadata inside the compiled entry, using the existing `native.py` pattern:
register `function.__tvm_ffi_object__()`, retain its `tvm::ffi::Function`, form
non-owning DLTensor views, and call it under the current-device/current-stream
scope. Allocate buffers in C++ with the shapes above. Constructing the BSHD
layout inside the host program avoids repeated Python views and DLPack setup.

Cache keys must distinguish forward/backward, device, dtype, causal mode, tile
schedule, thread/warp layouts, stride/alignment classes, relevant shape
boundaries, and eligibility. Positive versus nonpositive scale must select the
correct program family: the shifted forward is positive-only, while signed and
zero scales retain native CuTe. Changing scale within the positive family is
a runtime scalar update. Validate changed operands, saved actual O/statistics,
tails, cache reuse, graph replay, and exported symbols after the host change.
