"""Independent 3D traversal, LTX layout, and corrected-gradient qualification."""

import unittest
from itertools import product
from math import prod
from types import SimpleNamespace
from unittest.mock import patch

import torch

from benchmarks.neighborhood.reference import neighborhoods
from kohakufa_cute import api, neighborhood_attention, neighborhood_attention_3d
from kohakufa_cute.neighborhood_geometry import volume_bounds, volume_tile
from tests import test_neighborhood as existing


class VolumeGeometryTests(unittest.TestCase):
    def check_geometry(self, spatial, window, rows, columns):
        edges = neighborhoods(spatial, window)
        inverse = [[] for _ in edges]
        for query, keys in enumerate(edges):
            for key in keys:
                inverse[key].append(query)
        depth, height, width = spatial
        for backward, adjacency in ((False, edges), (True, inverse)):
            for start in range(0, prod(spatial), rows):
                z0, z1, y0, y1, x0, x1 = volume_bounds(start, rows, *spatial, *window, backward)
                slots = (x1 - x0 + 2 * columns - 2) // columns
                tiles = [
                    block
                    for i in range((z1 - z0) * (y1 - y0) * slots)
                    for block, active in (volume_tile(i, z0, y0, y1, x0, x1, height, width, columns),)
                    if active
                ]
                box = {(z * height + y) * width + x for z in range(z0, z1) for y in range(y0, y1) for x in range(x0, x1)}
                self.assertEqual(set(tiles), {index // columns for index in box})
                self.assertEqual(len(tiles), len(set(tiles)), "Repeated tiles double probabilities and gradients")
                needed = {key for keys in adjacency[start : start + rows] for key in keys}
                self.assertTrue({key // columns for key in needed} <= set(tiles))
                if rows == 1:
                    self.assertEqual(needed, box)

    def test_exhaustive_small_windows_and_inverse(self):
        for spatial in product(range(1, 4), repeat=3):
            for window in product(*(range(1, extent + 1) for extent in spatial)):
                for rows, columns in ((1, 2), (4, 8), (64, 32)):
                    self.check_geometry(spatial, window, rows, columns)

    def test_unaligned_planes_and_even_windows(self):
        for spatial, window in (((5, 7, 9), (3, 4, 6)), ((3, 5, 33), (2, 3, 4)), ((5, 1, 65), (3, 1, 3))):
            for columns in (16, 32, 64):
                self.check_geometry(spatial, window, 64, columns)

    def test_temporal_boundary_is_asymmetric(self):
        edges = neighborhoods((5, 1, 1), (3, 1, 1))
        self.assertIn(2, edges[0])
        self.assertNotIn(0, edges[2])
        self.assertEqual(volume_bounds(2, 1, 5, 1, 1, 3, 1, 1, True), (0, 5, 0, 1, 0, 1))

    def test_large_volume_skips_distant_tiles(self):
        z0, z1, y0, y1, x0, x1 = volume_bounds((30 * 128 + 50) * 256 + 64, 64, 64, 128, 256, 3, 7, 7, False)
        self.assertLess((z1 - z0) * (y1 - y0) * ((x1 - x0 + 62) // 32), 120)


class VolumeAPITests(unittest.TestCase):
    def test_flattened_dispatch_and_invalid_geometry(self):
        from kohakufa_cute import neighborhood_ops

        q = SimpleNamespace(ndim=4, shape=(1, 2, 105, 64), device=torch.device("cuda", 0), dtype=torch.bfloat16)
        q.resolve_conj = q.resolve_neg = lambda: q
        with (
            patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
            patch.object(neighborhood_ops, "forward", return_value=(q, None, None)) as forward,
        ):
            self.assertIs(neighborhood_attention(q, q, q, spatial_shape=(3, 5, 7), window_size=(2, 3, 4)), q)
            forward.assert_called_once_with(q, q, q, [3, 5, 7], [2, 3, 4], 0.125)
            for spatial, window in (((3, 5, 8), 3), ((3, 5, 7), (3, 3)), ((3, 5, 7), 8), ((True, 5, 21), 1)):
                with self.assertRaises(ValueError):
                    neighborhood_attention(q, q, q, spatial_shape=spatial, window_size=window)

    def test_ltx_adapter_clips_windows_and_preserves_scale(self):
        q = torch.empty(1, 1, 5, 7, 2, 64)
        with patch.object(api, "neighborhood_attention", side_effect=lambda q, k, v, **kwargs: q) as call:
            out = neighborhood_attention_3d(q, q, q, (11, 11, 11), None, 1.0)
            self.assertEqual(out.shape, q.shape)
            self.assertEqual(call.call_args.kwargs, dict(spatial_shape=(1, 5, 7), window_size=(1, 5, 7), scale=1.0))
            self.assertEqual(call.call_args.args[0].shape, (1, 2, 35, 64))
            self.assertEqual(call.call_args.args[0].untyped_storage().data_ptr(), q.untyped_storage().data_ptr())
            for causal in (True, (True, False, False), (False,), (0, 0, 0)):
                with self.assertRaises(ValueError):
                    neighborhood_attention_3d(q, q, q, 3, causal)
            for window in (True, 0, (1, 2), (1, 2, -1)):
                with self.assertRaises(ValueError):
                    neighborhood_attention_3d(q, q, q, window)

    def test_fake_autograd_and_aot(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        from kohakufa_cute import neighborhood_ops

        with FakeTensorMode():
            q = torch.empty(1, 2, 105, 64, dtype=torch.bfloat16, requires_grad=True)

            def call(q, k, v):
                return neighborhood_ops.forward(q, k, v, [3, 5, 7], [2, 3, 4], 1.0)[0]

            compiled = torch.compile(call, backend="aot_eager", fullgraph=True)
            for function in (call, compiled):
                out = function(q, q, q)
                self.assertEqual(torch.autograd.grad(out.sum(), q)[0].shape, q.shape)


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA and the CuTe runtime")
class VolumeGPUTests(unittest.TestCase):
    qualify = existing.NeighborhoodGPUTests.qualify

    def setUp(self):
        torch.manual_seed(319)

    def test_boundaries_planes_tails_signed_scale_and_views(self):
        for dtype in (torch.float16, torch.bfloat16):
            for spatial, window in (((1, 1, 1), (1, 1, 1)), ((3, 5, 7), (2, 3, 4)), ((5, 3, 9), (3, 2, 5))):
                inputs = [torch.randn(1, 2, prod(spatial), 128, device="cuda", dtype=dtype)[..., ::2] for _ in range(3)]
                for scale in (1.0, -0.125, 0.0):
                    self.qualify(inputs, torch.randn_like(inputs[0]), spatial, window, scale)

    def test_ltx_layout_and_clipped_temporal_window(self):
        spatial = (2, 5, 7)
        for dtype in (torch.float16, torch.bfloat16):
            inputs = [torch.randn(1, *spatial, 2, 64, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)]
            dout = torch.randn_like(inputs[0])
            out = neighborhood_attention_3d(*inputs, (3, 7, 7), None, 1.0)
            actual = out, *torch.autograd.grad(out, inputs, dout)
            flattened = [t.flatten(1, 3).transpose(1, 2) for t in inputs]
            expected = self.qualify(flattened, dout.flatten(1, 3).transpose(1, 2), spatial, (2, 5, 7), 1.0)
            for got, want in zip(actual, expected):
                torch.testing.assert_close(got.flatten(1, 3).transpose(1, 2), want, atol=0, rtol=0)

    def test_sharp_common_offset(self):
        for dtype in (torch.float16, torch.bfloat16):
            for scale in (1.0, -1.0):
                inputs = [torch.zeros(1, 2, 105, 64, device="cuda", dtype=dtype) for _ in range(3)]
                q, k, v = inputs
                q[..., 0], q[..., 1], k[..., 0] = 64, 1, 64
                k[..., 1] = torch.arange(105, device="cuda") % 7
                v.copy_(torch.randn_like(v))
                self.qualify(inputs, torch.randn_like(q), (3, 5, 7), (2, 3, 4), scale, mathematical=False)

    def test_graph_replay_reads_changed_inputs_and_overwrites_outputs(self):
        inputs = [torch.randn(1, 2, 105, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
        dout = torch.randn_like(inputs[0])

        def call():
            out = neighborhood_attention(*inputs, spatial_shape=(3, 5, 7), window_size=(2, 3, 4), scale=1.0)
            return out, *torch.autograd.grad(out, inputs, dout)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                call()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = call()
        for _ in range(2):
            with torch.no_grad():
                for tensor in (*inputs, dout):
                    tensor.copy_(torch.randn_like(tensor))
                for tensor in actual:
                    tensor.fill_(torch.nan)
            graph.replay()
            torch.cuda.synchronize()
            expected = self.qualify(inputs, dout, (3, 5, 7), (2, 3, 4), 1.0)
            for got, want in zip(actual, expected):
                torch.testing.assert_close(got, want, atol=0, rtol=0)

    def test_sanitizer_temporal_seams(self):
        # Multiple partial planes in one owner, and a partial final CTA.
        for dtype in (torch.float16, torch.bfloat16):
            inputs = [torch.randn(1, 2, 105, 128, device="cuda", dtype=dtype)[..., ::2] for _ in range(3)]
            self.qualify(inputs, torch.randn_like(inputs[0]), (3, 5, 7), (2, 3, 4), 1.0)

    def test_compilation_separates_2d_and_3d_geometry(self):
        compiled = torch.compile(neighborhood_attention, fullgraph=True)
        inputs = [torch.randn(1, 2, 105, 64, device="cuda", dtype=torch.float16, requires_grad=True) for _ in range(3)]
        dout = torch.randn_like(inputs[0])
        for spatial, window in (((3, 5, 7), (2, 3, 4)), ((7, 3, 5), (4, 2, 3)), ((15, 7), (3, 4)), ((3, 5, 7), (2, 3, 4))):
            out = compiled(*inputs, spatial_shape=spatial, window_size=window)
            actual = out, *torch.autograd.grad(out, inputs, dout)
            expected = self.qualify(inputs, dout, spatial, window)
            for got, want in zip(actual, expected):
                torch.testing.assert_close(got, want, atol=0, rtol=0)
