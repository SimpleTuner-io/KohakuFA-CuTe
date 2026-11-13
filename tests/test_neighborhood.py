"""Independent spatial, reference, API, and device qualification."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from benchmarks.neighborhood.key_precision_reference import key_precision_reference
from benchmarks.neighborhood.reference import neighborhood_mask, neighborhoods
from benchmarks.reference import fp64_reference, metrics, rounded_reference
from kohakufa_cute import api, neighborhood_attention
from kohakufa_cute.neighborhood_geometry import spatial_bounds, spatial_tile


class NeighborhoodGeometryTests(unittest.TestCase):
    def test_exhaustive_small_forward_and_inverse_tiles(self):
        for height in range(1, 6):
            for width in range(1, 6):
                for wh in range(1, height + 1):
                    for ww in range(1, width + 1):
                        edges = neighborhoods((height, width), (wh, ww))
                        inverse = tuple(tuple(q for q, keys in enumerate(edges) if key in keys) for key in range(len(edges)))
                        for rows in (1, 4, 64):
                            for columns in (8, 32):
                                for backward, adjacency in ((False, edges), (True, inverse)):
                                    for start in range(0, height * width, rows):
                                        bounds = spatial_bounds(start, rows, height, width, wh, ww, backward)
                                        y0, y1, x0, x1 = bounds
                                        slots = (x1 - x0 + 2 * columns - 2) // columns
                                        tiles = [
                                            block
                                            for i in range((y1 - y0) * slots)
                                            for block, selected in (spatial_tile(i, y0, x0, x1, width, columns),)
                                            if selected
                                        ]
                                        expected_tiles = {
                                            (y * width + x) // columns for y in range(y0, y1) for x in range(x0, x1)
                                        }
                                        self.assertEqual(set(tiles), expected_tiles)
                                        self.assertEqual(len(tiles), len(set(tiles)), "Repeated tiles double gradients")
                                        needed = {key for keys in adjacency[start : start + rows] for key in keys}
                                        self.assertTrue({key // columns for key in needed} <= set(tiles))
                                        if rows == 1:
                                            # For a single owner, inverse bounds must be exact.
                                            self.assertEqual(
                                                needed, {y * width + x for y in range(y0, y1) for x in range(x0, x1)}
                                            )

    def test_shifted_boundary_is_asymmetric(self):
        edges = neighborhoods((1, 5), (1, 3))
        self.assertEqual(edges, ((0, 1, 2), (0, 1, 2), (1, 2, 3), (2, 3, 4), (2, 3, 4)))
        self.assertIn(2, edges[0])
        self.assertNotIn(0, edges[2])
        self.assertEqual(spatial_bounds(2, 1, 1, 5, 1, 3, True), (0, 1, 0, 5))

    def test_large_width_skips_distant_spatial_tiles(self):
        for inverse in (False, True):
            bounds = spatial_bounds(128 * 1024 + 256, 64, 256, 1024, 7, 7, inverse)
            y0, y1, x0, x1 = bounds
            slots = (x1 - x0 + 62) // 32
            self.assertLess((y1 - y0) * slots, 40)
            self.assertGreater(256 * 1024 // 32, 8000)

    def test_unaligned_spatial_rows_and_partial_owner_tiles(self):
        for spatial, window in (((17, 131), (7, 5)), ((1, 65), (1, 3)), ((65, 1), (3, 1)), ((11, 33), (4, 6))):
            height, width = spatial
            edges = neighborhoods(spatial, window)
            inverse = [[] for _ in edges]
            for query, keys in enumerate(edges):
                for key in keys:
                    inverse[key].append(query)
            for backward, adjacency in ((False, edges), (True, inverse)):
                for start in range(0, height * width, 64):
                    y0, y1, x0, x1 = spatial_bounds(start, 64, height, width, *window, backward)
                    slots = (x1 - x0 + 62) // 32
                    tiles = [
                        block
                        for i in range((y1 - y0) * slots)
                        for block, selected in (spatial_tile(i, y0, x0, x1, width, 32),)
                        if selected
                    ]
                    expected = {(y * width + x) // 32 for y in range(y0, y1) for x in range(x0, x1)}
                    self.assertEqual(set(tiles), expected)
                    self.assertEqual(len(tiles), len(expected))
                    self.assertTrue({key // 32 for keys in adjacency[start : start + 64] for key in keys} <= expected)

    def test_enumerated_fp64_equations_match_local_autograd(self):
        torch.manual_seed(61)
        spatial, window = (3, 5), (2, 3)
        inputs = [torch.randn(1, 2, 15, 4, dtype=torch.float64, requires_grad=True) for _ in range(3)]
        q, k, v = inputs
        dout = torch.randn_like(q)
        for scale in (-0.3, 0, 0.3):
            pieces = []
            for row, keys in enumerate(neighborhoods(spatial, window)):
                scores = (q[:, :, row : row + 1] @ k[:, :, list(keys)].transpose(-1, -2)) * scale
                pieces.append(scores.softmax(-1) @ v[:, :, list(keys)])
            out = torch.cat(pieces, dim=2)
            gradients = torch.autograd.grad(out, inputs, dout)
            expected = fp64_reference(inputs, dout, neighborhood_mask(spatial, window), scale=scale)
            for got, want in zip((out, *gradients), expected):
                torch.testing.assert_close(got, want, atol=1e-12, rtol=1e-12)

    def test_qualification_runner_keeps_head_and_component_failures_separate(self):
        from benchmarks.neighborhood.qualify import assess

        torch.manual_seed(62)
        inputs = [torch.randn(2, 2, 15, 4, dtype=torch.float16) for _ in range(3)]
        dout = torch.randn_like(inputs[0])
        reference = rounded_reference(inputs, dout, neighborhood_mask((3, 5), (3, 3)))
        actual = [t.to(torch.float16) for t in reference]
        # Delta must use the rounded O just supplied to the candidate assessment.
        reference = rounded_reference(inputs, dout, neighborhood_mask((3, 5), (3, 3)), output=actual[0])
        actual[1:] = [t.to(torch.float16) for t in reference[1:]]
        records = assess(actual, inputs, dout, (3, 5), (3, 3), None, directed=True, policy="baseline")
        self.assertTrue(all(r["pass"] for r in records))
        actual[2][1, 0].zero_()
        records = assess(actual, inputs, dout, (3, 5), (3, 3), None, directed=True, policy="baseline")
        failures = [(r["component"], r["batch"], r["head"]) for r in records if not r["pass"]]
        self.assertEqual(failures, [("dk", 1, 0)])


class NeighborhoodAPITests(unittest.TestCase):
    def test_rejects_cpu_and_invalid_geometry(self):
        q = torch.zeros(1, 1, 15, 64, dtype=torch.bfloat16)
        for spatial, window in (((3, 5), 3), ((3, 5), 6), ((0, 5), 1), ((True, 5), 1), ((3, 5), (2, 0))):
            with self.subTest(spatial=spatial, window=window), self.assertRaises(ValueError):
                neighborhood_attention(q, q, q, spatial_shape=spatial, window_size=window)

    def test_explicit_dispatch_and_validation(self):
        from kohakufa_cute import neighborhood_ops

        q = SimpleNamespace(ndim=4, shape=(1, 2, 15, 64), device=torch.device("cuda", 0), dtype=torch.bfloat16)
        q.resolve_conj = q.resolve_neg = lambda: q
        with (
            patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)),
            patch.object(neighborhood_ops, "forward", return_value=(q, None, None)) as forward,
            patch.object(api, "_extension", side_effect=AssertionError("Dense dispatch")),
        ):
            self.assertIs(neighborhood_attention(q, q, q, spatial_shape=(3, 5), window_size=3), q)
            forward.assert_called_once_with(q, q, q, 3, 5, 3, 3, 0.125)
            for scale in (float("nan"), float("inf"), 1e50):
                with self.assertRaises(ValueError):
                    neighborhood_attention(q, q, q, spatial_shape=(3, 5), window_size=3, scale=scale)
            for spatial, window in (((3, 6), 3), ((3, 5), True)):
                with self.assertRaises(ValueError):
                    neighborhood_attention(q, q, q, spatial_shape=spatial, window_size=window)

    def test_backward_saves_actual_output_and_geometry(self):
        from kohakufa_cute import neighborhood_ops

        inputs = [torch.randn(1, 2, 15, 64, dtype=torch.float16) for _ in range(3)]
        out = torch.zeros_like(inputs[0])
        stats = torch.empty(1, 2, 15)
        ctx = SimpleNamespace(mark_non_differentiable=lambda *args: None, set_materialize_grads=Mock())
        ctx.save_for_backward = lambda *args: setattr(ctx, "saved_tensors", args)
        neighborhood_ops.setup_context(ctx, (*inputs, 3, 5, 2, 3, -0.125), (out, stats, stats))
        ctx.set_materialize_grads.assert_called_once_with(False)
        with patch.object(neighborhood_ops, "backward", return_value=tuple(inputs)) as backward:
            self.assertEqual(len(neighborhood_ops.autograd_backward(ctx, out, None, None)), 8)
        self.assertIs(backward.call_args.args[3], out)
        self.assertEqual(backward.call_args.args[-5:], (3, 5, 2, 3, -0.125))
        with patch.object(neighborhood_ops, "backward", side_effect=AssertionError("Unused output backward")):
            self.assertEqual(neighborhood_ops.autograd_backward(ctx, None, None, None), (None,) * 8)

    def test_fake_and_compiled_fake_shapes(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        from kohakufa_cute import neighborhood_ops

        with FakeTensorMode():
            q = torch.empty(1, 2, 15, 64, dtype=torch.bfloat16, requires_grad=True)

            def call(q, k, v):
                return neighborhood_ops.forward(q, k, v, 3, 5, 2, 3, 0.125)[0]

            out = call(q, q, q)
            self.assertEqual(out.shape, q.shape)
            self.assertEqual(out.dtype, q.dtype)
            self.assertEqual(torch.autograd.grad(out.sum(), q)[0].shape, q.shape)
            compiled = torch.compile(call, backend="aot_eager", fullgraph=True)
            out = compiled(q, q, q)
            self.assertEqual(torch.autograd.grad(out.sum(), q)[0].shape, q.shape)


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA and the CuTe runtime")
class NeighborhoodGPUTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(621)

    def qualify(self, inputs, dout, spatial, window, scale=None, *, mathematical=True):
        operands = [t.detach().requires_grad_() for t in inputs]
        out = neighborhood_attention(*operands, spatial_shape=spatial, window_size=window, scale=scale)
        actual = out, *torch.autograd.grad(out, operands, dout)
        mask = neighborhood_mask(spatial, window)
        # Keep references off the GPU, including during Compute Sanitizer runs.
        cpu = [t.detach().cpu() for t in inputs]
        rounded = key_precision_reference(
            cpu,
            dout.cpu(),
            spatial,
            window,
            0.125 if scale is None else scale,
            out.detach().cpu(),
            method="weighted_dq_reconstructed_dk",
        )
        references = [rounded]
        if mathematical:
            references.append(fp64_reference(cpu, dout.cpu(), mask, scale=scale))
        limit = 0.01 if out.dtype == torch.bfloat16 else 0.005
        for reference in references:
            for component, (got, want) in enumerate(zip(actual, reference)):
                for batch in range(got.shape[0]):
                    for head in range(got.shape[1]):
                        error = metrics(got[batch, head].cpu(), want[batch, head])
                        self.assertTrue(error["finite"], (component, batch, head, error))
                        if error["reference_norm"]:
                            self.assertLess(error["relative_error"], limit, (component, batch, head, error))
                        else:
                            self.assertEqual(got[batch, head].count_nonzero().item(), 0)
        return actual

    def test_boundaries_tails_signed_scale_and_strided_views(self):
        for dtype in (torch.float16, torch.bfloat16):
            for spatial, window in (((1, 1), (1, 1)), ((3, 5), (3, 3)), ((9, 17), (3, 4)), ((5, 7), (5, 7))):
                for scale in (None, 0.0, -0.125):
                    with self.subTest(dtype=dtype, spatial=spatial, window=window, scale=scale):
                        length = spatial[0] * spatial[1]
                        inputs = [
                            torch.randn(2, length, 2, 128, device="cuda", dtype=dtype)[:, :, :, ::2].transpose(1, 2)
                            for _ in range(3)
                        ]
                        if length == 1:
                            inputs[0].zero_()
                            inputs[1].zero_()
                        self.qualify(inputs, torch.randn_like(inputs[0]), spatial, window, scale)

    def test_inverse_boundary_counts_have_exact_rounded_gradients(self):
        for dtype in (torch.float16, torch.bfloat16):
            inputs = [torch.zeros(1, 1, 5, 64, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)]
            out = neighborhood_attention(*inputs, spatial_shape=(1, 5), window_size=(1, 3))
            dq, dk, dv = torch.autograd.grad(out, inputs, torch.ones_like(out))
            self.assertEqual(out.count_nonzero().item(), 0)
            self.assertEqual(dq.count_nonzero().item(), 0)
            self.assertEqual(dk.count_nonzero().item(), 0)
            probability = torch.tensor(1 / 3, device="cuda", dtype=dtype).float()
            expected = (torch.tensor([2, 3, 5, 3, 2], device="cuda") * probability).to(dtype)
            torch.testing.assert_close(dv, expected[None, None, :, None].expand_as(dv), atol=0, rtol=0)

    def test_spatial_gaps_with_vector_loads_and_offset_gradient(self):
        spatial, window = (17, 131), (7, 5)
        for dtype in (torch.float16, torch.bfloat16):
            inputs = [torch.randn(1, 2, 2227, 64, device="cuda", dtype=dtype) for _ in range(3)]
            gradient = torch.randn_like(inputs[0])
            for offset in (0, 1):
                with self.subTest(dtype=dtype, gradient_storage_offset=offset):
                    dout = torch.empty(inputs[0].numel() + offset, device="cuda", dtype=dtype)[offset:].view_as(inputs[0])
                    dout.copy_(gradient)
                    self.qualify(inputs, dout, spatial, window)

    def test_logical_negative_views_differentiate_to_strided_bases(self):
        from tests.test_native_view_flags import base_gradients, logical_inputs

        spatial, window = (5, 9), (3, 3)
        mask = neighborhood_mask(spatial, window)
        patterns = (
            (True, False, False, False),
            (False, True, False, False),
            (False, False, True, False),
            (False, False, False, True),
            (True, True, True, True),
        )
        for dtype in (torch.float16, torch.bfloat16):
            for negative in patterns:
                with self.subTest(dtype=dtype, negative=negative):
                    bases = [torch.randn(1, 2, 45, 128, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)]
                    inputs = logical_inputs(bases, negative[:3], strided=True)
                    gradient = torch.randn_like(inputs[0])
                    dout = torch.ops.aten._neg_view.default(gradient) if negative[3] else gradient
                    saved = []

                    def pack(tensor):
                        saved.append(tensor)
                        return -tensor

                    # Value-preserving hooks return flagged views of every
                    # saved operand, including the FP32 softmax statistics.
                    with torch.autograd.graph.saved_tensors_hooks(pack, torch.ops.aten._neg_view.default):
                        out = neighborhood_attention(*inputs, spatial_shape=spatial, window_size=window)
                    actual = out, *torch.autograd.grad(out, bases, dout)
                    self.assertTrue(all(not t.is_neg() and not t.is_conj() for t in saved[:3]))
                    cpu = [t.detach().cpu() for t in inputs]
                    base_cpu = [t.detach().cpu() for t in bases]
                    exact = fp64_reference(cpu, dout.cpu(), mask)
                    rounded = key_precision_reference(
                        cpu,
                        dout.cpu(),
                        spatial,
                        window,
                        0.125,
                        out.detach().cpu(),
                        method="weighted_dq_reconstructed_dk",
                    )
                    for reference in (exact, rounded):
                        expected = reference[0], *base_gradients(reference[1:], negative[:3], base_cpu, strided=True)
                        for got, want in zip(actual, expected):
                            for head in range(got.shape[1]):
                                error = metrics(got[:, head].cpu(), want[:, head])
                                self.assertTrue(error["finite"])
                                self.assertLess(error["relative_error"], 0.01 if dtype == torch.bfloat16 else 0.005)

    def test_negative_view_graph_reads_changed_base_values(self):
        from tests.test_native_view_flags import base_gradients, logical_inputs

        bases = [torch.randn(1, 2, 45, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
        gradient = torch.randn_like(bases[0][..., ::2])

        def call():
            inputs = logical_inputs(bases, (True, True, True), strided=True)
            dout = torch.ops.aten._neg_view.default(gradient)
            out = neighborhood_attention(*inputs, spatial_shape=(5, 9), window_size=3)
            return out, *torch.autograd.grad(out, bases, dout)

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
                for tensor in (*bases, gradient):
                    tensor.copy_(torch.randn_like(tensor))
                for tensor in actual:
                    tensor.fill_(torch.nan)
            graph.replay()
            torch.cuda.synchronize()
            inputs = logical_inputs(bases, (True, True, True), strided=True)
            dout = torch.ops.aten._neg_view.default(gradient)
            logical = self.qualify(inputs, dout, (5, 9), (3, 3))
            expected = logical[0], *base_gradients(logical[1:], (True, True, True), bases, strided=True)
            for got, want in zip(actual, expected):
                torch.testing.assert_close(got, want, atol=0, rtol=0)

    def test_probability_underflow_remains_part_of_the_contract(self):
        inputs = [torch.zeros(1, 1, 5, 64, device="cuda", dtype=torch.float16) for _ in range(3)]
        inputs[0][..., 0] = 1
        inputs[1][..., 0, 0] = -20
        dout = torch.full_like(inputs[0], 8192)
        actual = self.qualify(inputs, dout, (1, 5), (1, 3), 1.0, mathematical=False)
        exact = fp64_reference([t.cpu() for t in inputs], dout.cpu(), neighborhood_mask((1, 5), (1, 3)), scale=1)
        self.assertGreater(exact[3][0, 0, 0, 0].item(), 0)
        self.assertEqual(actual[3][0, 0, 0].count_nonzero().item(), 0)

    def test_compiled_calls_separate_geometry_with_equal_sequence_length(self):
        compiled = torch.compile(neighborhood_attention, fullgraph=True)
        inputs = [torch.randn(1, 2, 15, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
        dout = torch.randn_like(inputs[0])
        for spatial, window in (((3, 5), (3, 3)), ((5, 3), (2, 3)), ((3, 5), (3, 3))):
            out = compiled(*inputs, spatial_shape=spatial, window_size=window)
            actual = out, *torch.autograd.grad(out, inputs, dout)
            expected = self.qualify(inputs, dout, spatial, window)
            for got, want in zip(actual, expected):
                torch.testing.assert_close(got, want, atol=0, rtol=0)

    def test_sharp_common_offset_rounding(self):
        for dtype in (torch.float16, torch.bfloat16):
            for scale in (0.125, -0.125):
                inputs = [torch.zeros(1, 1, 45, 64, device="cuda", dtype=dtype) for _ in range(3)]
                q, k, v = inputs
                q[..., 0] = 64
                q[..., 1] = 1
                k[..., 0] = 64
                k[..., 1] = torch.arange(45, device="cuda") % 7
                v.copy_(torch.randn_like(v))
                self.qualify(inputs, torch.randn_like(q), (5, 9), (3, 3), scale, mathematical=False)

    def test_changed_graph_inputs_and_poisoned_outputs(self):
        inputs = [torch.randn(1, 2, 153, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
        dout = torch.randn_like(inputs[0])

        def call():
            out = neighborhood_attention(*inputs, spatial_shape=(9, 17), window_size=(3, 3))
            return out, *torch.autograd.grad(out, inputs, dout)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                call()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = call()
        for _ in range(2):
            with torch.no_grad():
                for t in (*inputs, dout):
                    t.copy_(torch.randn_like(t))
                for t in actual:
                    t.fill_(torch.nan)
            graph.replay()
            torch.cuda.synchronize()
            expected = self.qualify(inputs, dout, (9, 17), (3, 3))
            for got, want in zip(actual, expected):
                torch.testing.assert_close(got, want, atol=0, rtol=0)


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA and the CuTe runtime")
class ProductionNeighborhoodGPUTests(unittest.TestCase):
    def test_default_fastpath_changed_graph_and_poisoned_outputs(self):
        import os
        from pathlib import Path

        from benchmarks.neighborhood.benchmark import assess, sparse_references
        from kohakufa_cute import neighborhood_kernels

        properties = torch.cuda.get_device_properties(0)
        normalized_name = "".join(properties.name.upper().split())
        if (
            "H100" not in normalized_name
            or "80GB" not in normalized_name
            or "HBM3" not in normalized_name
            or "MIG" in normalized_name
            or "NVL" in normalized_name
            or torch.cuda.get_device_capability() != (9, 0)
            or properties.multi_processor_count != 132
        ):
            self.skipTest("Production fast-path qualification requires a full 132-SM H100 80GB HBM3.")
        # A forced experimental schedule would not validate production dispatch.
        self.assertIsNone(neighborhood_kernels._experiment_columns)
        self.assertIsNone(neighborhood_kernels._experiment_exact_contiguous)
        torch.manual_seed(1949)
        for dtype in (torch.float16, torch.bfloat16):
            for spatial, batch, heads in (((16, 64), 1, 8), ((33, 49), 2, 4)):
                with self.subTest(dtype=dtype, spatial=spatial):
                    shape = (batch, heads, spatial[0] * spatial[1], 64)
                    inputs = [torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)]
                    dout = torch.empty(inputs[0].numel() + 1, device="cuda", dtype=dtype)[1:].view(shape)
                    dout.copy_(torch.randn_like(dout))
                    self.assertEqual(dout.storage_offset(), 1)
                    columns, traversal = neighborhood_kernels.effective_schedule(inputs, spatial, (7, 7))
                    self.assertEqual(columns, (32, 32, 64))
                    self.assertEqual(traversal, (True, False, True) if spatial[1] == 64 else (False, False, False))
                    retained = {
                        "inputs": [tensor.detach().cpu().clone() for tensor in inputs],
                        "dout": dout.cpu().clone(),
                        "spatial": spatial,
                        "window": (7, 7),
                        "columns": columns,
                        "traversal": traversal,
                        "dout_storage_offset": dout.storage_offset(),
                        "phase": "initial",
                    }
                    actual = None

                    def call():
                        out = neighborhood_attention(*inputs, spatial_shape=spatial, window_size=(7, 7))
                        return out, *torch.autograd.grad(out, inputs, dout)

                    def qualify(values):
                        self.assertTrue(all(bool(torch.isfinite(tensor).all()) for tensor in values))
                        # The sparse equations enumerate their own neighborhoods and
                        # run on CPU, including under Compute Sanitizer. Actual O
                        # supplies delta for the independent rounded diagnostics.
                        references = sparse_references(inputs, dout, values[0], spatial, (7, 7), scale=0.125)
                        errors = assess(values, references, dtype)
                        retained["accuracy"] = errors
                        self.assertTrue(all(error["passed"] for error in errors), errors)

                    try:
                        # Autograd records each leaf's first-use stream. Keep its
                        # initial call, capture warmup, and fresh expectations on
                        # the same side stream before capturing these leaves.
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            actual = call()
                        stream.synchronize()
                        qualify(actual)
                        with torch.cuda.stream(stream):
                            for _ in range(3):
                                call()
                        torch.cuda.current_stream().wait_stream(stream)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=stream):
                            actual = call()
                        stream.synchronize()
                        for replay in range(2):
                            with torch.no_grad():
                                for tensor in (*inputs, dout):
                                    tensor.copy_(torch.randn_like(tensor))
                                for tensor in actual:
                                    tensor.fill_(torch.nan)
                            retained.update(
                                inputs=[tensor.detach().cpu().clone() for tensor in inputs],
                                dout=dout.cpu().clone(),
                                phase="changed-graph",
                                replay=replay,
                            )
                            graph.replay()
                            torch.cuda.synchronize()
                            stream.wait_stream(torch.cuda.current_stream())
                            with torch.cuda.stream(stream):
                                expected = call()
                            stream.synchronize()
                            for got, want in zip(actual, expected):
                                self.assertTrue(torch.equal(got, want), "Changed graph output differs from fresh eager.")
                            qualify(actual)
                    except BaseException:
                        destination = os.environ.get("NEIGHBORHOOD_RESULTS")
                        if destination:
                            directory = Path(destination).resolve()
                            source_root = Path(__file__).resolve().parents[1]
                            if not directory.is_relative_to(source_root):
                                directory.mkdir(parents=True, exist_ok=True)
                                try:
                                    retained["actual"] = (
                                        None if actual is None else [tensor.detach().cpu() for tensor in actual]
                                    )
                                except Exception as transfer_error:
                                    retained["actual_transfer_error"] = type(transfer_error).__name__
                                label = str(dtype).split(".")[-1]
                                torch.save(retained, directory / f"production-{spatial[0]}x{spatial[1]}-{label}-failure.pt")
                        raise


if __name__ == "__main__":
    unittest.main()
