"""Compact owners: independent edge coverage, original-order stores and replay."""

import unittest
from math import prod
from unittest.mock import patch

import torch

from benchmarks.neighborhood.reference import neighborhoods
from kohakufa_cute.neighborhood_geometry import compact_volume_bounds, compact_volume_row, compact_volume_tile
from tests import test_neighborhood as existing


class CompactGeometryTests(unittest.TestCase):
    def test_unique_owners_and_exact_tile_coverage(self):
        for tile in ((1, 4, 4), (2, 2, 4), (1, 4, 8), (2, 4, 8), (4, 4, 4), (1, 8, 8)):
            for spatial in ((1, 8, 8), (3, 8, 16), (5, 16, 8)):
                depth, height, width = spatial
                size = prod(tile)
                length = ((depth + tile[0] - 1) // tile[0]) * tile[0] * height * width
                rows = [compact_volume_row(row, height, width, *tile) for row in range(length)]
                self.assertEqual(sorted(row for row in rows if row < prod(spatial)), list(range(prod(spatial))))
                for window in ((1, 1, 1), tuple(min(3, n) for n in spatial), tuple(min(4, n) for n in spatial), spatial):
                    edges = neighborhoods(spatial, window)
                    for start in range(0, length, size):
                        bounds = compact_volume_bounds(start, *spatial, *window, *tile)
                        z0, z1, y0, y1, x0, x1 = bounds
                        blocks = [
                            compact_volume_tile(i, *bounds, height // tile[1], width // tile[2])
                            for i in range((z1 - z0) * (y1 - y0) * (x1 - x0))
                        ]
                        self.assertEqual(len(blocks), len(set(blocks)))
                        actual = {rows[b * size + i] for b in blocks for i in range(size)}
                        required = {key for row in rows[start : start + size] if row < len(edges) for key in edges[row]}
                        self.assertTrue(required <= actual)


class _CompactVolumeChecks:
    qualify = existing.NeighborhoodGPUTests.qualify
    configuration = ((1, 8, 8), True)

    def setUp(self):
        from kohakufa_cute.neighborhood_kernels import set_experiment_volume

        self.previous = set_experiment_volume(self.configuration)
        torch.manual_seed(1729)

    def tearDown(self):
        from kohakufa_cute.neighborhood_kernels import set_experiment_volume

        set_experiment_volume(self.previous[0], residual=self.previous[1])

    def test_compact_default_outputs_gradients_and_signed_scales(self):
        for dtype in (torch.float16, torch.bfloat16):
            for spatial, window in (((3, 8, 16), (2, 4, 5)), ((5, 16, 16), (3, 7, 7))):
                inputs = [torch.randn(1, prod(spatial), 2, 64, dtype=dtype, device="cuda").transpose(1, 2) for _ in range(3)]
                for scale in (1.0, -0.125, 0.0):
                    self.qualify(inputs, torch.randn_like(inputs[0]), spatial, window, scale)

    def test_sanitizer_compact(self):
        for dtype in (torch.float16, torch.bfloat16):
            spatial = (3, 8, 16)
            inputs = [torch.randn(1, prod(spatial), 2, 64, dtype=dtype, device="cuda").transpose(1, 2) for _ in range(3)]
            self.qualify(inputs, torch.randn_like(inputs[0]), spatial, (2, 4, 5), 1.0)

    def test_graph_changed_data_and_poisoned_outputs(self):
        from kohakufa_cute import neighborhood_attention

        inputs = [
            torch.randn(1, 384, 2, 64, device="cuda", dtype=torch.bfloat16).transpose(1, 2).requires_grad_()
            for _ in range(3)
        ]
        dout = torch.randn_like(inputs[0])

        def call():
            out = neighborhood_attention(*inputs, spatial_shape=(3, 8, 16), window_size=(2, 4, 5), scale=1.0)
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
        with torch.no_grad():
            for t in (*inputs, dout):
                t.copy_(torch.randn_like(t))
            for t in actual:
                t.fill_(torch.nan)
        graph.replay()
        torch.cuda.synchronize()
        expected = self.qualify(inputs, dout, (3, 8, 16), (2, 4, 5), 1.0)
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, atol=0, rtol=0)

    def test_forward_error_reaches_output_rounding_floor(self):
        from benchmarks.neighborhood.reference import neighborhood_mask, two_term_forward_reference
        from kohakufa_cute import neighborhood_attention
        from kohakufa_cute.neighborhood_kernels import set_experiment_volume

        mask = neighborhood_mask((3, 8, 16), (2, 4, 5))
        for dtype in (torch.float16, torch.bfloat16):
            inputs = [torch.randn(1, 2, 384, 64, device="cuda", dtype=dtype) for _ in range(3)]
            cpu = [t.cpu().double() for t in inputs]
            scores = (cpu[0] @ cpu[1].transpose(-1, -2)) * 0.125
            expected = scores.masked_fill(~mask, float("-inf")).softmax(-1) @ cpu[2]
            rounded = two_term_forward_reference([t.cpu() for t in inputs], (3, 8, 16), (2, 4, 5), 0.125)
            corrected = neighborhood_attention(*inputs, spatial_shape=(3, 8, 16), window_size=(2, 4, 5)).cpu().double()
            set_experiment_volume(False, residual=False)
            previous = neighborhood_attention(*inputs, spatial_shape=(3, 8, 16), window_size=(2, 4, 5)).cpu().double()
            set_experiment_volume(self.configuration)
            for head in range(2):
                floor = torch.linalg.vector_norm(expected[0, head].to(dtype).double() - expected[0, head])
                error = torch.linalg.vector_norm(corrected[0, head] - expected[0, head])
                old_error = torch.linalg.vector_norm(previous[0, head] - expected[0, head])
                self.assertLess(float(error), 1.05 * float(floor))
                self.assertLess(float(error), float(old_error))
                rounded_error = torch.linalg.vector_norm(corrected[0, head] - rounded[0, head])
                self.assertLess(float(rounded_error), 1.05 * float(floor))


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0), "Requires Hopper and the CuTe runtime"
)
class CompactVolumeGPUTests(_CompactVolumeChecks, unittest.TestCase):
    pass


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() in ((8, 0), (8, 6), (8, 9)),
    "Requires an Ampere or Ada GPU and the CuTe runtime",
)
class WarpCompactVolumeGPUTests(_CompactVolumeChecks, unittest.TestCase):
    configuration = ((1, 4, 8), False)

    def test_sanitizer_compact(self):
        spatial, window = (3, 8, 16), (2, 4, 5)
        for dtype in (torch.float16, torch.bfloat16):
            inputs = [
                torch.randn(2, prod(spatial), 3, 96, device="cuda", dtype=dtype)[..., 8:72].transpose(1, 2) for _ in range(3)
            ]
            self.qualify(inputs, torch.randn_like(inputs[0]), spatial, window, 1.0)

    def test_public_dispatch_and_changed_graph_inputs(self):
        from kohakufa_cute import neighborhood_attention
        from kohakufa_cute.neighborhood_kernels import set_experiment_volume
        from kohakufa_cute.neighborhood_schedule import default_volume_schedule
        from kohakufa_cute.neighborhood_volume_forward import VolumeForward

        spatial, window = (13, 64, 96), (11, 11, 11)
        properties = torch.cuda.get_device_properties(0)
        selected = default_volume_schedule(
            device_name=properties.name,
            capability=torch.cuda.get_device_capability(),
            sm_count=properties.multi_processor_count,
            shape=(1, 2, prod(spatial), 64),
            spatial=spatial,
            window=window,
            dtype="bfloat16",
            vector=8,
        )
        if selected != self.configuration:
            self.skipTest("This device has no qualified compact warp default")
        inputs = [
            torch.randn(1, prod(spatial), 2, 64, device="cuda", dtype=torch.bfloat16).transpose(1, 2).requires_grad_()
            for _ in range(3)
        ]
        dout = torch.randn_like(inputs[0])

        def call():
            out = neighborhood_attention(*inputs, spatial_shape=spatial, window_size=window)
            return out, *torch.autograd.grad(out, inputs, dout)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        created = []
        initialize = VolumeForward.__init__

        def record_owner(program, *args, **kwargs):
            initialize(program, *args, **kwargs)
            created.append((program.tile_d, program.tile_h, program.tile_w))

        set_experiment_volume()
        with torch.cuda.stream(stream):
            with patch.object(VolumeForward, "__init__", record_owner):
                actual = call()
            self.assertEqual(created, [self.configuration[0]])
            set_experiment_volume(self.configuration)
            explicit = call()
            set_experiment_volume()
            for _ in range(2):
                call()
        stream.synchronize()
        for got, want in zip(actual, explicit):
            torch.testing.assert_close(got, want, atol=0, rtol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = call()
        with torch.no_grad():
            for tensor in (*inputs, dout):
                tensor.copy_(torch.randn_like(tensor))
            for tensor in actual:
                tensor.fill_(torch.nan)
        graph.replay()
        torch.cuda.synchronize()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            expected = call()
        stream.synchronize()
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, atol=0, rtol=0)

    def test_large_window_outputs_and_individual_gradients(self):
        spatial, window = (11, 12, 16), (11, 11, 11)
        for dtype in (torch.float16, torch.bfloat16):
            inputs = [torch.randn(1, prod(spatial), 2, 64, device="cuda", dtype=dtype).transpose(1, 2) for _ in range(3)]
            self.qualify(inputs, torch.randn_like(inputs[0]), spatial, window, 0.125)

    def test_common_offset_and_sharp_logits(self):
        spatial, window = (3, 8, 16), (2, 4, 5)
        for dtype in (torch.float16, torch.bfloat16):
            inputs = [torch.zeros(2, 2, prod(spatial), 64, device="cuda", dtype=dtype) for _ in range(3)]
            q, k, v = inputs
            q[..., 0], q[..., 1], k[..., 0] = 64, 1, 64
            k[..., 1] = torch.arange(prod(spatial), device="cuda") % 7
            v.copy_(torch.randn_like(v))
            for scale in (1.0, -1.0):
                self.qualify(inputs, torch.randn_like(q), spatial, window, scale, mathematical=False)
            q.copy_(torch.randn_like(q) * 8)
            k.copy_(torch.randn_like(k) * 8)
            self.qualify(inputs, torch.randn_like(q), spatial, window, 0.125, mathematical=False)
