# Schedule controls

`fa4_sm80._attention.Schedule` is an immutable configuration for shared warp-MMA
methods. A configuration is an explicit control until a device selector qualifies
it. Kernel resource capacity or fewer instructions alone do not establish a win.

| Fields | Meaning |
| --- | --- |
| `forward_m`, `forward_n` | Forward query/key tile |
| `forward_stages`, `query_in_registers` | Forward staging and Q placement |
| `backward_m`, `backward_n` | Backward query/key tile |
| `query_stages`, `dout_stages` | Q/dO pipeline depth |
| `threads` | Backward block size |
| `score_m_warps`, `kv_m_warps`, `dq_m_warps` | MMA ownership layouts |
| `v_in_regs` | Preserve V in registers before Q reuses shared storage |
| `causal_head_group` | Interleave causal key tiles across groups of heads |

With resident V, dP must consume the register operand. Reloading the aliased
shared V view after Q staging reads the wrong data. Shared storage may be reused
only after every reader has finished; preserve producer/consumer barriers, tail
predicates, zero filling, and epilogue ownership when changing schedules.

Causal head groups are 1, 2, 4, 8, or 16; one is the identity mapping. Grouping
changes CTA traversal, not products or arithmetic. Cache keys include the group
and layouts. Partial-head/batch mappings still require qualification.

The native [host adapter](native-host.md) must understand every activated field.
Do not silently drop a configuration key or substitute defaults. Public selectors
have separate exact-device and workload predicates; see [GPU guides](../README.md#gpu-guides).

Measure complete eager and graph paths on the intended device, matching operand
conversions and accumulation. Recheck the [precision contract](../precision.md),
all gradients, changed poisoned graphs, and sanitizers before selecting changes.
Compact negative-control evidence lives with [the experiments](../../benchmarks/experiments/README.md).
