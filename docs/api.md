# Attention API

```python
from kohakufa_cute import attention, scaled_dot_product_attention, automatic_scaled_dot_product_attention
```

## Strict CuTe calls

```python
attention(q, k, v, *, mask=None, causal=False, scale=None, center_keys=False)
scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0,
    is_causal=False, scale=None, enable_gqa=False,
)
```

Inputs use `[batch, heads, sequence, head_dim]` (BHSD):

- Q/K/V share a CUDA device and FP16 or BF16 dtype.
- Tensors are four-dimensional and nonempty. Batch and head dimension match;
  K and V have equal shapes. Head dimension is 1–512.
- Query heads must be divisible by KV heads. `attention` enables GQA implicitly;
  SDPA wrappers require `enable_gqa=True` when head counts differ.
- Supported devices have capability SM80/86/89/90/100/103/120/121.
- Supported strided views use ordinary CuTe paths; tuned schedules often require
  contiguous, aligned, equal-shape tensors.

Boolean masks broadcast to the attention scores; `True` means visible. An
all-hidden query row returns zero. `attention` accepts `mask` and `causal=True`
together. For SDPA wrappers, fold causal visibility into an explicit mask.
The causal predicate uses token positions; rectangular causal alignment can
differ from standalone FlashAttention.

The default scale is `head_dim**-0.5`. Signed and zero scales are supported;
some tuned schedules select only positive normal FP32 scales. Optional key
centering is described in [precision](precision.md#optional-key-centering).
Dropout, additive masks, FP32 inputs, and packed variable-length inputs are
unsupported by CuTe. Strict APIs propagate validation, build, and kernel errors.

## Automatic selection

`automatic_scaled_dot_product_attention` takes the SDPA arguments above. It uses
CuTe for supported calls with available CUDA dependencies and native PyTorch
SDPA otherwise, including CPU/MPS. It never patches PyTorch globally.
Errors from an already selected CuTe kernel propagate.

Selection does not predict the fastest backend. Optional tuned routes have
narrower [device and workload bounds](README.md#gpu-guides).

## Compilation, graphs, and gradients

The standard APIs support eager and compiled autograd:

```python
compiled = torch.compile(automatic_scaled_dot_product_attention, fullgraph=True)
out = compiled(q, k, v, is_causal=True)
```

Compile and warm before capturing a CUDA graph. Replay must retain tensor
owners; changing data in retained input storage is supported. Optional eager
schedule selectors can fall back to ordinary CuTe for compiled calls.
Enable the desired [determinism policy](determinism.md) before forward and capture.
Atomic schedules do not promise bitwise-identical dQ between calls.

## Experimental neighborhood attention

`neighborhood_attention(q, k, v, *, spatial_shape, window_size, scale=None)`
adds an explicit noncausal 2D path for matching FP16/BF16
`[batch, heads, height*width, 64]` inputs, with stride and dilation one.
It generates spatial tiles without a dense mask and uses shifted boundary
windows. See the [neighborhood guide](neighborhood/README.md) for the complete
input contract, asymmetric backward visibility, usage and current evidence.
Qualification covers H100, B200, B300, RTX 4090, RTX 5090 and L40S; the guide
documents the corrected dQ+dK method now enabled by default on every device.
The [H100 schedule](neighborhood/reports/tuning.md#default-dispatch) is bounded
by device and workload and preserves the default precision contract.
