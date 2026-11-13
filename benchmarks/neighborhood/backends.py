"""Complete-call BHSD adapters for neighborhood attention comparisons.

Geometry preparation is explicit and reusable. Required tensor-layout copies
remain in the measured call. External backends retain their own arithmetic;
they are complete implementations, not arithmetic-policy substitutions.
"""

from dataclasses import dataclass, field
from time import perf_counter

import torch


def neighborhood_predicate(spatial, window):
    """FlexAttention's scalar tensor predicate, including shifted boundaries."""
    height, width = spatial
    window_h, window_w = window
    length = height * width

    def visible(batch, head, query, key):
        top = torch.clamp(query // width - window_h // 2, 0, height - window_h)
        left = torch.clamp(query % width - window_w // 2, 0, width - window_w)
        return (
            (query < length)
            & (key < length)
            & (key // width >= top)
            & (key // width < top + window_h)
            & (key % width >= left)
            & (key % width < left + window_w)
        )

    return visible


def spatial_block_lists(spatial, window, block_size=128):
    """Generate sparse forward block lists without a token-by-token mask.

    Visit each query and each spatial row in its window, adding the aligned
    key blocks covering that row interval. The official BlockMask factory
    generates the transpose for backward. Indices have the factory's required
    padded block-grid shape, O(ceil(sequence/block_size)**2), not sequence**2.
    """
    height, width = spatial
    window_h, window_w = window
    length = height * width
    blocks = (length + block_size - 1) // block_size
    counts, indices = [], []
    for begin in range(0, length, block_size):
        selected = set()
        for query in range(begin, min(begin + block_size, length)):
            top = min(max(query // width - window_h // 2, 0), height - window_h)
            left = min(max(query % width - window_w // 2, 0), width - window_w)
            for y in range(top, top + window_h):
                first = (y * width + left) // block_size
                last = (y * width + left + window_w - 1) // block_size
                selected.update(range(first, last + 1))
        ordered = sorted(selected)
        counts.append(len(ordered))
        indices.append(ordered + [0] * (blocks - len(ordered)))
    return counts, indices


def _geometry(spatial, window):
    spatial = tuple(spatial)
    window = (window, window) if isinstance(window, int) else tuple(window)
    if len(spatial) != 2 or len(window) != 2:
        raise ValueError("Expected two-dimensional spatial_shape and window_size.")
    if any(type(n) is not int or n <= 0 for n in spatial + window):
        raise ValueError("Spatial dimensions and window dimensions must be positive integers.")
    if any(w > n for w, n in zip(window, spatial)):
        raise ValueError("Window must fit within the image.")
    return spatial, window


@dataclass
class PreparedBackend:
    name: str
    spatial: tuple
    window: tuple
    scale: float
    device: torch.device
    block_size: int = 128
    compile_mode: str = "max-autotune-no-cudagraphs"
    metadata: dict = field(default_factory=dict)
    _mask: object = field(default=None, repr=False)
    _kernel: object = field(default=None, repr=False)
    _natten_backend: str | None = field(default=None, repr=False)

    def prepare(self):
        """Rebuild geometry preparation; cached calls read the updated state."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            begin.record()
        started = perf_counter()
        if self.name == "flex":
            from torch.nn.attention.flex_attention import BlockMask

            counts, indices = spatial_block_lists(self.spatial, self.window, self.block_size)
            count_tensor = torch.tensor(counts, device=self.device, dtype=torch.int32)[None, None]
            index_tensor = torch.tensor(indices, device=self.device, dtype=torch.int32)[None, None]
            length = self.spatial[0] * self.spatial[1]
            self._mask = BlockMask.from_kv_blocks(
                count_tensor,
                index_tensor,
                BLOCK_SIZE=self.block_size,
                mask_mod=neighborhood_predicate(self.spatial, self.window),
                seq_lengths=(length, length),
            )
            self.metadata["preparation_storage_bytes"] = sum(
                tensor.numel() * tensor.element_size()
                for name in ("kv_num_blocks", "kv_indices", "q_num_blocks", "q_indices")
                if isinstance(tensor := getattr(self._mask, name), torch.Tensor)
            )
            self.metadata["partial_kv_blocks"] = sum(counts)
        else:
            self.metadata["preparation_storage_bytes"] = 0
        if self.device.type == "cuda":
            end.record()
            end.synchronize()
            self.metadata["last_preparation_device_ms"] = begin.elapsed_time(end)
        self.metadata["last_preparation_host_ms"] = (perf_counter() - started) * 1000
        self.metadata.setdefault("first_preparation_host_ms", self.metadata["last_preparation_host_ms"])
        if "last_preparation_device_ms" in self.metadata:
            self.metadata.setdefault("first_preparation_device_ms", self.metadata["last_preparation_device_ms"])

    def call(self, q, k, v):
        if self.name == "kohakufa":
            from kohakufa_cute import neighborhood_attention

            return neighborhood_attention(
                q, k, v, spatial_shape=self.spatial, window_size=self.window, scale=self.scale
            ).contiguous()
        if self.name == "flex":
            return self._kernel(
                q, k, v, block_mask=self._mask, scale=self.scale, kernel_options={"BACKEND": "AUTO"}
            ).contiguous()
        from natten import na2d

        # NATTEN's public input contract is contiguous B,H_image,W_image,heads,D.
        # These copies and the output restoration belong to every BHSD call.
        inputs = [
            tensor.reshape(tensor.shape[0], tensor.shape[1], *self.spatial, tensor.shape[-1])
            .permute(0, 2, 3, 1, 4)
            .contiguous()
            for tensor in (q, k, v)
        ]
        if self._natten_backend is None:
            from natten.backends import choose_backend, choose_fmha_backend

            if self.window == self.spatial:
                selected = choose_fmha_backend(
                    *(tensor.flatten(1, 2) for tensor in inputs),
                    is_causal=False,
                    is_varlen=False,
                    torch_compile=False,
                )
            else:
                selected = choose_backend(*inputs, torch_compile=False)
            if selected.startswith("flex"):
                raise RuntimeError("NATTEN selected its Flex backend; an independent libnatten comparator is required.")
            self._natten_backend = selected
            self.metadata["selected_backend"] = self._natten_backend
        options = (
            {"attention_kwargs": {"backend": self._natten_backend}}
            if self.window == self.spatial
            else {"backend": self._natten_backend}
        )
        out = na2d(*inputs, kernel_size=self.window, stride=1, dilation=1, is_causal=False, scale=self.scale, **options)
        return out.permute(0, 3, 1, 2, 4).reshape(q.shape).contiguous()


def make_backend(name, spatial, window, scale=0.125, device="cuda", *, block_size=128, compile_mode=None):
    """Prepare a complete BHSD implementation; compilation warms on real inputs."""
    if name not in {"kohakufa", "flex", "natten"}:
        raise ValueError("Unknown neighborhood benchmark backend.")
    spatial, window = _geometry(spatial, window)
    if type(block_size) is not int or block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("Block size must be a positive power of two.")
    prepared = PreparedBackend(name, spatial, window, float(scale), torch.device(device), block_size)
    if compile_mode is not None:
        prepared.compile_mode = compile_mode
    prepared.metadata = {
        "name": name,
        "external_layout": "contiguous BHSD inputs and output",
        "spatial": list(spatial),
        "window": list(window),
        "scale": prepared.scale,
        "precision_policy": (
            "docs/precision.md" if name == "kohakufa" else "backend-native; complete implementation comparison"
        ),
        "geometry_preparation_reused": True,
        "selected_backend": "generic warp MMA" if name == "kohakufa" else "pending real-input dispatch",
    }
    if name == "flex":
        from torch.nn.attention.flex_attention import flex_attention

        prepared._kernel = torch.compile(flex_attention, fullgraph=True, dynamic=False, mode=prepared.compile_mode)
        prepared.metadata.update(
            selected_backend="FlexAttention AUTO; actual kernels retained by profiler",
            compile_mode=prepared.compile_mode,
            block_size=block_size,
            block_mask_builder="direct spatial block lists + BlockMask.from_kv_blocks; no token-level dense mask",
            block_mask_full_blocks="all selected blocks use the exact mask_mod",
        )
    elif name == "natten":
        prepared.metadata.update(
            backend_selection="libnatten's device/tensor capability priority; frozen after first actual call",
            layout_copies_included=["BHSD to contiguous BHWD for Q/K/V", "BHWD to contiguous BHSD for output"],
        )
    prepared.prepare()
    return prepared
