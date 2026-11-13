"""Compact volume ownership for the shared neighborhood forward arithmetic.

Only traversal and global-memory addressing differ from the generic owner.
The original order and storage of output/statistics are preserved for backward.
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync

from .neighborhood_geometry import compact_volume_bounds, compact_volume_row, compact_volume_tile
from .neighborhood_kernels import NeighborhoodAttention

_row = cute.jit(compact_volume_row, preprocess=False)
_bounds = cute.jit(compact_volume_bounds, preprocess=False)
_tile = cute.jit(compact_volume_tile, preprocess=False)


class VolumeForward(NeighborhoodAttention):
    def __init__(self, spatial, window, vector, tile=(2, 4, 8), key_tile=None):
        super().__init__("forward", spatial, window, vector, columns=64)
        self.tile_d, self.tile_h, self.tile_w = tile
        self.tile_size = self.tile_d * self.tile_h * self.tile_w
        assert self.tile_size in (16, 32, 64, 128)
        self.columns = self.rows = self.grid_rows = self.tile_size
        self.warps = self.score_warps = self.tile_size // 16
        assert self.height % self.tile_h == 0 and self.image_width % self.tile_w == 0
        self.blocks_h = self.height // self.tile_h
        self.blocks_w = self.image_width // self.tile_w
        self.blocks_d = (self.depth + self.tile_d - 1) // self.tile_d
        self.key_tile_d, self.key_tile_h, self.key_tile_w = tile if key_tile is None else key_tile
        self.columns = self.key_tile_d * self.key_tile_h * self.key_tile_w
        assert self.columns in (16, 32, 64, 128)
        assert self.columns != self.rows or key_tile is None or tuple(key_tile) == tuple(tile)
        assert self.height % self.key_tile_h == 0 and self.image_width % self.key_tile_w == 0
        self.key_blocks_w = self.image_width // self.key_tile_w
        self.key_blocks_h = self.height // self.key_tile_h
        self.key_blocks_d = (self.depth + self.key_tile_d - 1) // self.key_tile_d

    @cute.jit
    def owner_length(self, length):
        return self.blocks_d * self.blocks_h * self.blocks_w * self.tile_size

    @cute.jit
    def output_row(self, row):
        return _row(row, self.height, self.image_width, self.tile_d, self.tile_h, self.tile_w)

    @cute.jit
    def coordinates(self, row, key: cutlass.Constexpr = False):
        td, th, tw = (
            (self.key_tile_d, self.key_tile_h, self.key_tile_w)
            if cutlass.const_expr(key)
            else (self.tile_d, self.tile_h, self.tile_w)
        )
        bw, bh = self.image_width // tw, self.height // th
        size = td * th * tw
        block, inner = row // size, row % size
        x = block % bw * tw + inner % tw
        y = block // bw % bh * th + inner // tw % th
        z = block // (bw * bh) * td + inner // (tw * th)
        return z, y, x

    @cute.jit
    def visible(self, mask, batch, head, row, col, sq, sk, full):
        qz, qy, qx = self.coordinates(row)
        kz, ky, kx = self.coordinates(col, True)
        z = cutlass.min(cutlass.max(qz - self.window_d // 2, 0), self.depth - self.window_d)
        y = cutlass.min(cutlass.max(qy - self.window_h // 2, 0), self.height - self.window_h)
        x = cutlass.min(cutlass.max(qx - self.window_w // 2, 0), self.image_width - self.window_w)
        return (
            (True if self.depth % self.tile_d == 0 else qz < self.depth)
            and (True if self.depth % self.key_tile_d == 0 else kz < self.depth)
            and kz >= z
            and kz < z + self.window_d
            and ky >= y
            and ky < y + self.window_h
            and kx >= x
            and kx < x + self.window_w
        )

    @cute.jit
    def traversal(self, start, length):
        bounds = _bounds(
            start,
            self.depth,
            self.height,
            self.image_width,
            self.window_d,
            self.window_h,
            self.window_w,
            self.tile_d,
            self.tile_h,
            self.tile_w,
            key_tile_d=self.key_tile_d,
            key_tile_h=self.key_tile_h,
            key_tile_w=self.key_tile_w,
            minimum=cutlass.min,
            maximum=cutlass.max,
        )
        z0, z1, y0, y1, x0, x1 = bounds
        return 0, (z1 - z0) * (y1 - y0) * (x1 - x0), bounds

    @cute.jit
    def traversal_block(self, iteration, bounds):
        z0, z1, y0, y1, x0, x1 = bounds
        return _tile(iteration, z0, z1, y0, y1, x0, x1, self.key_blocks_h, self.key_blocks_w), True

    @cute.jit
    def mapped(self, source, batch, head, key: cutlass.Constexpr = False):
        td, th, tw = (
            (self.key_tile_d, self.key_tile_h, self.key_tile_w)
            if cutlass.const_expr(key)
            else (self.tile_d, self.tile_h, self.tile_w)
        )
        bw, bh, bd = self.image_width // tw, self.height // th, (self.depth + td - 1) // td
        stride = source.stride[2]
        return cute.make_tensor(
            source[batch, head, None, None].iterator,
            cute.make_layout(
                ((tw, th, td, bw, bh, bd), 64),
                stride=(
                    (
                        stride,
                        stride * self.image_width,
                        stride * self.height * self.image_width,
                        stride * tw,
                        stride * th * self.image_width,
                        stride * td * self.height * self.image_width,
                    ),
                    source.stride[3],
                ),
            ),
        )

    @cute.jit
    def load(self, source, shared, batch, head, start, tid):
        rows = cute.size(shared, mode=[0])
        key = rows != self.rows
        atom = cute.make_copy_atom(cpasync.CopyG2SOp(), source.element_type, num_bits_per_copy=16 * self.vector)
        columns = 32 // self.vector
        copy = cute.make_tiled_copy_tv(
            atom,
            cute.make_layout((32 * self.warps // columns, columns), stride=(columns, 1)),
            cute.make_layout((1, self.vector)),
        )
        thread = copy.get_slice(tid)
        tile = cute.local_tile(self.mapped(source, batch, head, key), (rows, 64), (start // rows, 0))
        coords = thread.partition_S(cute.make_identity_tensor((rows, 64)))
        src, dst = thread.partition_S(tile), thread.partition_D(shared)
        for r in cutlass.range_constexpr(cute.size(dst.shape[1])):
            for d in cutlass.range_constexpr(cute.size(dst.shape[2])):
                row, dim = coords[0, r, d]
                if cutlass.const_expr(self.depth % (self.key_tile_d if key else self.tile_d) == 0):
                    cute.copy(copy, src[None, r, d], dst[None, r, d])
                else:
                    if self.coordinates(start + row, key)[0] < self.depth:
                        cute.copy(copy, src[None, r, d], dst[None, r, d])
                    else:
                        dst[None, r, d].fill(0)

    @cute.jit
    def store_result(self, result, destination, shared, batch, head, start, offset, thread, tid, scale):
        coords = thread.partition_C(cute.make_identity_tensor((self.tile_size, 64)))
        for i in cutlass.range_constexpr(cute.size(coords)):
            row, dim = coords[i]
            shared[row, dim] = (result[i] * scale).to(destination.element_type)
        cute.arch.sync_threads()
        atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), destination.element_type, num_bits_per_copy=16 * self.vector
        )
        columns = 32 // self.vector
        copy = cute.make_tiled_copy_tv(
            atom,
            cute.make_layout((32 * self.warps // columns, columns), stride=(columns, 1)),
            cute.make_layout((1, self.vector)),
        )
        copier = copy.get_slice(tid)
        tile = cute.local_tile(self.mapped(destination, batch, head), (self.tile_size, 64), (start // self.tile_size, 0))
        coords = copier.partition_D(cute.make_identity_tensor((self.tile_size, 64)))
        src, dst = copier.partition_S(shared), copier.partition_D(tile)
        for r in cutlass.range_constexpr(cute.size(dst.shape[1])):
            for d in cutlass.range_constexpr(cute.size(dst.shape[2])):
                row, dim = coords[0, r, d]
                if cutlass.const_expr(self.depth % self.tile_d == 0):
                    cute.copy(copy, src[None, r, d], dst[None, r, d])
                elif self.output_row(start + row) < destination.shape[2]:
                    cute.copy(copy, src[None, r, d], dst[None, r, d])
        cute.arch.sync_threads()
