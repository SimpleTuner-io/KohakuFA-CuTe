"""Direct union traversal: shifted edges, strided copies and saved statistics."""

import itertools
import unittest
from math import prod
from unittest.mock import patch

import torch

from benchmarks.neighborhood.reference import neighborhoods
from kohakufa_cute.neighborhood_geometry import compact_volume_row, volume_union_coordinates, volume_union_start
from kohakufa_cute.neighborhood_schedule import VolumeUnionSchedule
from tests import test_neighborhood_volume_forward as compact


class UnionGeometryTests(unittest.TestCase):
    def test_independently_enumerated_shifted_windows(self):
        for spatial, window in (((3, 8, 16), (2, 4, 5)), ((5, 16, 16), (3, 7, 7))):
            edges = neighborhoods(spatial, window)
            for tile in ((1, 4, 4), (1, 2, 8), (1, 4, 8), (2, 2, 4)):
                union = tuple(w + t - 1 for w, t in zip(window, tile))
                length = ((spatial[0] + tile[0] - 1) // tile[0]) * tile[0] * prod(spatial[1:])
                for start in range(0, length, prod(tile)):
                    first = compact_volume_row(start, *spatial[1:], *tile)
                    z, rem = divmod(first, prod(spatial[1:]))
                    y, x = divmod(rem, spatial[2])
                    origin = tuple(volume_union_start(q, n, w, t) for q, n, w, t in zip((z, y, x), spatial, window, tile))
                    coords = [volume_union_coordinates(i, *origin, *union[1:]) for i in range(prod(union))]
                    self.assertEqual(len(coords), len(set(coords)))
                    self.assertTrue(all(all(0 <= q < n for q, n in zip(c, spatial)) for c in coords))
                    actual = {(z * spatial[1] + y) * spatial[2] + x for z, y, x in coords}
                    for row in range(start, start + prod(tile)):
                        query = compact_volume_row(row, *spatial[1:], *tile)
                        if query < prod(spatial):
                            self.assertTrue(set(edges[query]) <= actual)


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA and the CuTe runtime")
class UnionVolumeGPUTests(compact._CompactVolumeChecks, unittest.TestCase):
    configuration = VolumeUnionSchedule((1, 4, 4), 32)
    test_common_offset_and_sharp_logits = compact.WarpCompactVolumeGPUTests.test_common_offset_and_sharp_logits
    test_sanitizer_compact = compact.WarpCompactVolumeGPUTests.test_sanitizer_compact

    def test_large_window_outputs_and_individual_gradients(self):
        from kohakufa_cute.neighborhood_kernels import set_experiment_volume

        spatial, window = (11, 16, 24), (11, 11, 11)
        for columns, dtype in itertools.product((32, 64), (torch.float16, torch.bfloat16)):
            set_experiment_volume(VolumeUnionSchedule((1, 4, 8), columns))
            inputs = [torch.randn(1, prod(spatial), 1, 64, device="cuda", dtype=dtype).transpose(1, 2) for _ in range(3)]
            self.qualify(inputs, torch.randn_like(inputs[0]), spatial, window, 0.125)

    def test_sanitizer_large_union(self):
        from kohakufa_cute.neighborhood_kernels import set_experiment_volume

        spatial, window = (11, 16, 24), (11, 11, 11)
        for dtype in (torch.float16, torch.bfloat16):
            set_experiment_volume(VolumeUnionSchedule((1, 4, 8), 32))
            inputs = [
                torch.randn(1, prod(spatial), 1, 80, device="cuda", dtype=dtype)[..., 8:72].transpose(1, 2) for _ in range(3)
            ]
            self.qualify(inputs, torch.randn_like(inputs[0]), spatial, window, 0.125)

    def test_partial_temporal_owner_and_key_tile(self):
        from kohakufa_cute.neighborhood_kernels import set_experiment_volume

        set_experiment_volume(VolumeUnionSchedule((2, 2, 4), 32))
        spatial, window = (3, 8, 16), (2, 4, 5)
        for dtype in (torch.float16, torch.bfloat16):
            inputs = [torch.randn(1, 1, prod(spatial), 64, device="cuda", dtype=dtype) for _ in range(3)]
            self.qualify(inputs, torch.randn_like(inputs[0]), spatial, window, -0.125)

    def test_partial_spatial_owners_and_saved_statistics(self):
        from kohakufa_cute.neighborhood_kernels import set_experiment_volume

        set_experiment_volume(VolumeUnionSchedule((2, 4, 8), 32))
        spatial, window = (3, 9, 17), (2, 4, 5)
        for dtype in (torch.float16, torch.bfloat16):
            inputs = [
                torch.randn(1, prod(spatial), 2, 80, device="cuda", dtype=dtype)[..., 8:72].transpose(1, 2) for _ in range(3)
            ]
            self.qualify(inputs, torch.randn_like(inputs[0]), spatial, window, -0.125)

    def test_sanitizer_spatial_union(self):
        from kohakufa_cute.neighborhood_kernels import set_experiment_volume

        set_experiment_volume(VolumeUnionSchedule((1, 4, 4), 32))
        spatial, window = (13, 17, 25), (11, 11, 11)
        for dtype in (torch.float16, torch.bfloat16):
            inputs = [
                torch.randn(1, prod(spatial), 1, 80, device="cuda", dtype=dtype)[..., 8:72].transpose(1, 2) for _ in range(3)
            ]
            self.qualify(inputs, torch.randn_like(inputs[0]), spatial, window, 0.125)

    def test_operator_schema_fake_and_compiled_autograd(self):
        from kohakufa_cute.neighborhood_kernels import set_experiment_volume

        set_experiment_volume()
        from kohakufa_cute.neighborhood_ops import forward

        inputs = [torch.randn(1, 1, 384, 64, device="cuda", dtype=torch.float16, requires_grad=True) for _ in range(3)]
        torch.library.opcheck(forward, (*inputs, [3, 8, 16], [2, 4, 5], 0.125))

    def test_compiled_symbolic_geometry_and_backward(self):
        from kohakufa_cute.neighborhood_kernels import set_experiment_volume

        set_experiment_volume()
        from kohakufa_cute.neighborhood_ops import forward

        def call(q, k, v):
            return forward(q, k, v, [q.shape[2] // 128, 8, 16], [2, 4, 5], 0.125)[0]

        compiled = torch.compile(call, backend="aot_eager", fullgraph=True, dynamic=True)
        for depth in (3, 5, 3):
            inputs = [
                torch.randn(1, 1, depth * 128, 64, device="cuda", dtype=torch.float16, requires_grad=True) for _ in range(3)
            ]
            dout = torch.randn_like(inputs[0])
            got, expected = compiled(*inputs), call(*inputs)
            actual = got, *torch.autograd.grad(got, inputs, dout)
            reference = expected, *torch.autograd.grad(expected, inputs, dout)
            for a, b in zip(actual, reference):
                torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_native_cache_layout_geometry_alignment_and_scale(self):
        from kohakufa_cute import neighborhood_attention
        from kohakufa_cute.native import extension
        from kohakufa_cute.neighborhood_kernels import set_experiment_configuration, set_experiment_volume
        from kohakufa_cute.neighborhood_schedule import BASELINE_COLUMNS

        extension().clear_cache()
        set_experiment_volume()
        for dtype, offset, spatial in itertools.product((torch.float16, torch.bfloat16), (1, 8), ((5, 16, 16), (5, 8, 32))):
            inputs = [
                torch.randn(1, prod(spatial), 2, 80, device="cuda", dtype=dtype)[..., offset : offset + 64]
                .transpose(1, 2)
                .requires_grad_()
                for _ in range(3)
            ]
            dout = torch.randn_like(inputs[0])
            for scale in (0.0, -0.125):

                def call():
                    out = neighborhood_attention(*inputs, spatial_shape=spatial, window_size=(3, 5, 5), scale=scale)
                    return out, *torch.autograd.grad(out, inputs, dout)

                set_experiment_configuration(None)
                actual = call()
                cached = extension().cache_size()
                # Identical scheduling through Python is an independent host path.
                set_experiment_configuration(BASELINE_COLUMNS)
                try:
                    expected = call()
                finally:
                    set_experiment_configuration(None)
                for got, want in zip(actual, expected):
                    torch.testing.assert_close(got, want, atol=0, rtol=0)
                call()
                self.assertEqual(extension().cache_size(), cached)

    def test_public_dispatch_and_changed_graph_inputs(self):
        from kohakufa_cute import neighborhood_attention
        from kohakufa_cute.neighborhood_kernels import _compile, set_experiment_volume
        from kohakufa_cute.neighborhood_schedule import default_volume_schedule
        from kohakufa_cute.neighborhood_volume_forward import UnionVolumeForward

        properties = torch.cuda.get_device_properties(0)
        for spatial, window in (
            ((5, 16, 16), (3, 7, 7)),
            ((13, 64, 96), (11, 11, 11)),
            ((13, 64, 65), (11, 11, 11)),
        ):
            selected = default_volume_schedule(
                device_name=properties.name,
                capability=torch.cuda.get_device_capability(),
                sm_count=properties.multi_processor_count,
                shape=(1, 1, prod(spatial), 64),
                spatial=spatial,
                window=window,
                dtype="bfloat16",
                vector=8,
            )
            if not isinstance(selected, VolumeUnionSchedule):
                self.skipTest("This device has no qualified union default")
            inputs = [
                torch.randn(1, prod(spatial), 1, 64, device="cuda", dtype=torch.bfloat16).transpose(1, 2).requires_grad_()
                for _ in range(3)
            ]
            dout = torch.randn_like(inputs[0])

            def call():
                out = neighborhood_attention(*inputs, spatial_shape=spatial, window_size=window)
                return out, *torch.autograd.grad(out, inputs, dout)

            created = []
            initialize = UnionVolumeForward.__init__

            def record_owner(program, *args, **kwargs):
                initialize(program, *args, **kwargs)
                created.append(VolumeUnionSchedule((program.tile_d, program.tile_h, program.tile_w), program.columns))

            from kohakufa_cute.native import extension

            extension().clear_cache()
            _compile.cache_clear()
            set_experiment_volume()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                with patch.object(UnionVolumeForward, "__init__", record_owner):
                    actual = call()
                self.assertEqual(created, [selected])
                set_experiment_volume(selected)
                explicit = call()
                set_experiment_volume()
                for _ in range(3):
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
