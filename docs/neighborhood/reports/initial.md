# Experimental 2D neighborhood attention

Historical method labels below describe the recorded campaign. The combined
dQ+dK method with column-staged statistics is now the public neighborhood
default; see the [current guide](../README.md#default-precision).

This is the initial H100 implementation report. Its nine-method suite and generic schedule describe that frozen source. Start with the [current guide](../README.md) for the later bounded H100 schedule, corrected precision default and six-GPU coverage.

This explicit experimental path has scoped correctness qualification on a full
H100 PCIe, including independent references, changed-input graph replay, and all
four Compute Sanitizer tools. It preserves the default arithmetic contract and
does not participate in automatic dispatch. The initial qualification measured
no latency or HDiT model performance. Subsequent operator timings are documented
in the [performance comparison](performance.md). The
[qualification summary](../../../results/neighborhood/qualification-h100.json) records
source hashes, acceptance criteria, results, and unresolved baseline failures.

```python
from kohakufa_cute import neighborhood_attention

out = neighborhood_attention(
    q, k, v, spatial_shape=(height, width), window_size=(7, 7), scale=1.0,
)
```

Q/K/V have equal, nonempty `[batch, heads, height*width, 64]` shapes, on one
supported CUDA device, in FP16 or BF16. Tokens use row-major spatial order.
The output has Q's shape and dtype. Ordinary strided views are accepted; the
copy path selects vector width from the actual strides and pointer alignment.
Logical negative views are materialized before passing storage to the kernel;
autograd maps the gradients back to their bases. Backward also resolves logical
views returned by saved-tensor hooks. First-order autograd is supported.
`window_size` accepts an integer square size or a pair, each positive and no
larger than the corresponding image extent. Signed and zero finite FP32 scales
are supported; the default is `1/sqrt(64)`. GQA, masks, dropout, causal windows,
dilation, stride changes, relative positional bias, and other head dimensions
are outside this initial API. Unsupported calls raise errors.

## Semantics and arithmetic

For query `(y, x)`, define:

```text
top  = clamp(y - window_h//2, 0, height - window_h)
left = clamp(x - window_w//2, 0, width - window_w)
keys = [top, top+window_h) x [left, left+window_w)
```

Every query attends exactly `window_h * window_w` keys. Windows shift inward at
boundaries without padding, wrapping, or truncation. Even windows use the anchor
at `window//2`, matching the current
[NATTEN mask definition](https://github.com/SHI-Labs/NATTEN/blob/39375ea1d2f3b1bb284315fa5f79f6ecbebb6382/csrc/include/natten/cuda/fna/na_utils.cuh).

The [existing precision contract](../../precision.md) applies unchanged: separate
signed unscaled maxima and log2 normalization, FP32 QK/dP and gradient
accumulation, one input-dtype P conversion before dS, one input-dtype dS
conversion, and attention scaling after dQ/dK accumulation. Delta uses the
actual rounded output. Stronger compensated-policy experiments are excluded.

Spatial traversal overrides the generic kernels' iteration and visibility
hooks. Forward and query backward enumerate aligned flattened tiles intersecting
the union of the owners' spatial windows. Key/value backward enumerates the
inverse visibility relation. There are no atomic gradient updates: the existing
query owner writes dQ and the key/value owner writes dK/dV. Backward launches
the existing output/dO delta kernel before those two owners on the same stream.

For a key interval `[a, b]` in an axis of extent `n` and window `w`, the inverse
query interval is:

```text
lower = 0 if a < w else a - (w - 1 - w//2)
upper = n if b >= n - w else b + w//2 + 1   # exclusive
```

For instance, with extent 5 and window 3, query 0 attends key 2, while query 2
does not attend key 0. Key 2 receives contributions from all five queries.
Applying the forward window around each key would lose contributions.

The scalar bounds and enumeration functions run unchanged on CPU and under
CuTe JIT. Enumeration deduplicates tiles spanning adjacent spatial rows and
predicates extra alignment slots before loads. Per-element visibility still
checks the true window and sequence tails. An owner spanning multiple spatial
rows conservatively uses the full horizontal extent; this is a correctness-first
schedule, with no claim of optimal spatial tiling. All devices use generic warp
MMA, without architecture-specific presets. No dense mask, mask packing, or
device neighborhood table is allocated. Saved output/statistics and gradient
workspace grow linearly with the input sequence length.

## Independent qualification

`benchmarks/neighborhood/reference.py` enumerates all legal window placements
and selects the nearest anchor. It shares neither clamp nor inverse formulas
with production. Its dense mask is restricted to small diagnostic cases and
feeds both `fp64_reference` and `rounded_reference`; backward uses actual O.
CPU local-window autograd independently checks the mathematical equations.

The CPU suite exhausts every image/window shape with extents 1 through 5,
including even windows, forward/inverse traversal, owner tails, and overlapping
aligned tiles. It checks edge coverage and duplicate elimination, including an
exact single-owner inverse check. API tests cover explicit routing, rejected
geometry, saved output, fake tensors, and full-graph AOTAutograd tracing.

GPU tests cover FP16/BF16, batches and heads, boundary rows, full-image windows,
partial tiles, noncontiguous and dimension-strided inputs, signed/zero scales,
sharp common-offset logits, expected probability underflow, exact asymmetric
dV contribution counts, and graph replay with changed inputs and poisoned outputs.
The nine GPU test methods also exercise wide images with gaps between selected
tiles (`17 x 131`, window `7 x 5`), aligned and offset dO pointers, logical
negative Q/K/V/dO views, saved-tensor hooks, and cached compiled calls with
different spatial geometry at the same sequence length.
Random cases retain separate per-head output/dQ/dK/dV norm-error limits of 1%
for BF16 and 0.5% for FP16. Directed rounding fixtures use the same limits
against the declared rounded computation; their mathematical diagnostics do not
redefine that computation. Zero-reference components must remain zero.

Run from the repository root with Python 3.13:

```sh
CUDA_VISIBLE_DEVICES='' python3 -m unittest tests.test_neighborhood -v
python3 -m unittest tests.test_neighborhood.NeighborhoodGPUTests -v
```

The qualification runner writes per-component, per-head errors against both
references, records source hashes and versions, and retains failed numerical
operands. It fails on CPU instead of recording skipped GPU work as success.
Install the repository's CUDA extra and a compatible NATTEN separately. The
qualified Linux stack used Python 3.13.18, PyTorch 2.13.0+cu130, CUTLASS DSL
4.8.1, TVM FFI 0.1.15.post0, CUDA bindings 13.0.3, and NATTEN
0.21.7+torch2130cu130. PyTorch 2.13 was retained for NATTEN's newest documented
prebuilt pairing, allowing comparison in the same process; a newer 2.14.1 wheel
was available in the cu130 index. Local CPU checks used PyTorch 2.15.0.
The H100 had 114 SMs, an unpartitioned 80 GB device, a 350 W power limit, and
driver 580.159.03. Qualification does not extend to other runtime stacks or GPUs.

Choose an off-repository artifact directory in `NEIGHBORHOOD_RESULTS`. Before
using an ephemeral GPU, archive the full source snapshot and launch commands
outside that machine following the repository's campaign instructions.

```sh
python3 -m benchmarks.neighborhood.qualify --source-revision "$(git rev-parse HEAD)" \
    --output "$NEIGHBORHOOD_RESULTS/qualification.json" \
    --compare-natten --natten-backend cutlass-fna --reference-agreement --profile
```

Select a NATTEN backend actually supported by the target device. The optional
comparison calls the documented
[`natten.na2d`](https://natten.org/operations/#natten.na2d), with identical rounded
inputs, window, scale and output layout. Each implementation is assessed
independently. NATTEN does not define the oracle or the candidate's arithmetic
policy; comparator failures remain in the report. Profiling uses separate calls
and records actual launched CUDA kernel names. The runner reports correctness
only, with no latency samples or HDiT application claims.

The H100 matrix passed all 22 candidate cases against the required CPU and
device-reference gates. These comprise 18 random cases across three geometries,
two dtypes and positive/negative/zero scales, plus four directed common-offset
cases. The complete implementations use identical rounded operands; their
internal arithmetic policies are not assumed equal. NATTEN's explicit
`cutlass-fna` comparator passed six cases and failed sixteen under these gates.
Negative and zero scales are compatibility probes, so those failures do not
establish a general accuracy advantage. This does not compare every NATTEN
backend; traces identify SM80-template FNA kernels executing on H100, with
full-window cases able to select FMHA.

For the positive-scale common-offset fixture (`5 x 9`, window `3 x 3`, scale
`0.125`), worst-head dQ relative norm errors were:

| dtype | candidate / rounded | NATTEN / rounded | candidate / FP64 | NATTEN / FP64 |
| --- | ---: | ---: | ---: | ---: |
| FP16 | 0.0280% | 1.6494% | 1.6092% | 1.7979% |
| BF16 | 0.1744% | 7.2507% | 12.1567% | 12.6729% |

This demonstrates closer agreement with the declared rounded computation on
these fixtures. The larger mathematical errors show the operand-rounding floor;
the directed gate does not require universal FP64 agreement. The fixture sets
Q's first two channels to `(64, 1)`, K's to `(64, index % 7)`, all remaining Q/K
channels to zero, and generates V/dO with seed 792. The runner retains separate
output/dQ/dK/dV diagnostics for every head and preserves comparator failures.

The final nine-method suite passed uninstrumented and under memcheck, racecheck,
initcheck and synccheck, each with exit status zero and zero reported errors or
hazards. An earlier CUDA-bindings 13.4.4 run reported 41 driver API lookup errors
under memcheck despite passing the numerical tests. Aligning bindings to 13.0.3
removed those reports; final runs used the normal checks without suppressions.
The initial compile failure and an intermediate graph-test stream-lifetime
failure are retained with the successful final receipts.

Ordinary attention regression checks ran 72 test methods (26 skipped) and had
six failing subcases across three methods. The same six subcases failed on the
untouched base revision in the identical H100 runtime: large-logit key/value
gradients, short-query offset logits, and D256 cancellation. Their existing
rounded-reference thresholds were preserved, with all six failing operands
retained. Their causes remain unresolved; the scoped neighborhood qualification
does not establish that the entire repository's GPU suite passes.

Off-host archives retain every source snapshot, exact launch scripts, stage
receipts, failing operands, per-case Chrome traces, loaded-library hashes, and
compiler PTX/cubins. Representative traces show four candidate CuTe kernels
(forward, delta, dQ, dK/dV), plus two autograd statistic-gradient fills.
Candidate cubins have SASS/resource exports. The actual installed NATTEN binary
is retained; selected symbols have packaged SM90 SASS, but no selected packaged
PTX was obtained and the disassembly is not a CUPTI binding to a loaded module.
There are no timing, occupancy, utilization, or best-backend claims.

After CPU/device reference agreement outside instrumentation, run each sanitizer
separately and retain its actual exit status:

```sh
compute-sanitizer --tool memcheck --error-exitcode 1 python3 -m unittest tests.test_neighborhood.NeighborhoodGPUTests -v
compute-sanitizer --tool racecheck --error-exitcode 1 python3 -m unittest tests.test_neighborhood.NeighborhoodGPUTests -v
compute-sanitizer --tool initcheck --error-exitcode 1 python3 -m unittest tests.test_neighborhood.NeighborhoodGPUTests -v
compute-sanitizer --tool synccheck --error-exitcode 1 python3 -m unittest tests.test_neighborhood.NeighborhoodGPUTests -v
```

References in the GPU unittest run on CPU copies so sanitizer instrumentation
covers implementation kernels without instrumenting expensive reference GEMMs.
Re-run relevant ordinary attention GPU checks because the traversal hooks live
in the shared kernel. Before optimizing or enabling automatic dispatch, retain
successful compilation, reference agreement, correctness and sanitizer receipts,
kernel traces, and device/toolchain metadata. Evaluate HDiT only after capturing
its actual post-normalization and post-rotary tensors; its transforms, projections,
and output layers remain required parts of any complete-model comparison.
The integration scope was inspected in HDiT's
[neighborhood block](https://github.com/crowsonkb/k-diffusion/blob/4601bf085320592473f681a62808ed873d17fad5/k_diffusion/models/image_transformer_v2.py#L363).
