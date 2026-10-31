"""Qualify selected RTX 3090 Ti length classes and numerical policies."""

import itertools
import math
import unittest
from unittest.mock import patch

import torch

from benchmarks.reference import fp64_reference, metrics
from kohakufa_cute import attention


def qualified_device():
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_name() == "NVIDIA GeForce RTX 3090 Ti"
        and torch.cuda.get_device_properties(0).multi_processor_count == 84
    )


@unittest.skipUnless(qualified_device(), "Requires a full RTX 3090 Ti")
class AmpereScheduleTests(unittest.TestCase):
    def setUp(self):
        from kohakufa_cute import kernels, native

        native.extension().clear_cache()
        kernels._compile_kernel.cache_clear()

    def compare(self, actual, q, k, v, dout, *, causal=False, scale=None, mask=None):
        if max(q.shape[2], k.shape[2]) >= 2048:
            parts = [
                fp64_reference(
                    (q[:, head : head + 2], k[:, head : head + 2], v[:, head : head + 2]),
                    dout[:, head : head + 2],
                    causal=causal,
                    scale=scale,
                    mask=mask,
                )
                for head in range(0, q.shape[1], 2)
            ]
            reference = tuple(torch.cat([part[i] for part in parts], dim=1) for i in range(4))
        else:
            reference = fp64_reference((q, k, v), dout, causal=causal, scale=scale, mask=mask)
        atol = rtol = 0.02 if q.dtype == torch.bfloat16 else 0.003
        for got, want in zip(actual, reference):
            torch.testing.assert_close(got.double(), want, atol=atol, rtol=rtol)
            error = metrics(got, want)
            self.assertTrue(error["finite"])
            if error["reference_norm"]:
                self.assertLess(error["relative_error"], 0.01 if q.dtype == torch.bfloat16 else 0.005)
            else:
                self.assertEqual(error["max_absolute_error"], 0)

    @staticmethod
    def run_attention(q, k, v, dout, *, causal=False, scale=None, function=attention):
        out = function(q, k, v, causal=causal, scale=scale)
        return out, *torch.autograd.grad(out, (q, k, v), dout)

    def test_d16_cache_boundaries_and_changed_graph_replay(self):
        from kohakufa_cute import kernels, native

        for dtype, causal in itertools.product((torch.float16, torch.bfloat16), (False, True)):
            native.extension().clear_cache()
            kernels._compile_kernel.cache_clear()
            torch.manual_seed(1297)
            with patch.object(native, "compile_kernel", wraps=native.compile_kernel) as compiler:
                for queries, keys in ((1023, 1025), (1024, 1024), (1025, 1031), (1088, 1152), (1025, 1031)):
                    with self.subTest(dtype=dtype, causal=causal, queries=queries, keys=keys):
                        q = torch.randn(1, 16, queries, 16, device="cuda", dtype=dtype, requires_grad=True)
                        k, v = [
                            torch.randn(1, 16, keys, 16, device="cuda", dtype=dtype, requires_grad=True) for _ in range(2)
                        ]
                        dout = torch.randn_like(q)
                        actual = self.run_attention(q, k, v, dout, causal=causal)
                        self.compare(actual, q, k, v, dout, causal=causal)
                self.assertEqual(compiler.call_count, 2)
            q, k, v = [tensor.detach().requires_grad_() for tensor in (q, k, v)]
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.run_attention(q, k, v, dout, causal=causal)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    actual = self.run_attention(q, k, v, dout, causal=causal)
            torch.cuda.current_stream().wait_stream(stream)
            with torch.no_grad():
                for tensor in (q, k, v, dout):
                    tensor.copy_(torch.randn_like(tensor) * 0.75)
                for tensor in actual:
                    tensor.fill_(float("nan"))
            graph.replay()
            self.compare(actual, q, k, v, dout, causal=causal)

    def test_short_both_axis_bounds_strides_and_compiled_calls(self):
        from kohakufa_cute import kernels, native

        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        for dtype, dimension, causal in itertools.product((torch.float16, torch.bfloat16), (128, 256, 512), (False, True)):
            torch.compiler.reset()
            compiled = torch.compile(attention, fullgraph=True, dynamic=True)
            native.extension().clear_cache()
            kernels._compile_kernel.cache_clear()
            for index, (queries, keys) in enumerate(((127, 113), (129, 113), (113, 129), (128, 128))):
                with self.subTest(dtype=dtype, dimension=dimension, causal=causal, queries=queries, keys=keys):
                    torch.manual_seed(1301)
                    q = torch.randn(1, 16, queries + 4, dimension, device="cuda", dtype=dtype)[:, :, :queries]
                    k, v = [
                        torch.randn(1, 16, keys + 4, dimension, device="cuda", dtype=dtype)[:, :, :keys] for _ in range(2)
                    ]
                    q, k, v = [tensor.requires_grad_() for tensor in (q, k, v)]
                    dout = torch.randn_like(q)
                    actual = self.run_attention(q, k, v, dout, causal=causal, function=compiled if index % 2 else attention)
                    self.compare(actual, q, k, v, dout, causal=causal)

    def test_signed_scales_negative_statistics_and_tiny_gradients(self):
        for dtype, dimension, causal, case in itertools.product(
            (torch.float16, torch.bfloat16), (16, 128, 256), (False, True), ("negative", "zero", "maxima", "sharp")
        ):
            with self.subTest(dtype=dtype, dimension=dimension, causal=causal, case=case):
                torch.manual_seed(1207)
                sequence = 1025 if dimension == 16 else 4097 if dimension == 256 else 128
                q, k, v, dout = [torch.randn(1, 16, sequence, dimension, device="cuda", dtype=dtype) for _ in range(4)]
                scale = -0.25 if case == "negative" else 0.0 if case == "zero" else dimension**-0.5
                if case == "maxima":
                    q.mul_(0.3)
                    k.mul_(0.3)
                    q[..., 0], k[..., 0] = 20, -20
                    from kohakufa_cute import native

                    _, maximum, logsum, _ = native.extension().forward(q, k, v, None, scale, causal)
                    self.assertTrue((maximum < 0).all())
                    self.assertTrue((maximum * scale + logsum * math.log(2) < 0).all())
                elif case == "sharp":
                    for tensor in (q, k, v, dout):
                        tensor.zero_()
                    q[..., 0], k[..., 0], k[:, :, 0, 0] = 10, 9 if dtype == torch.float16 else 5, 10
                    v[..., 0], v[:, :, 0, 0], dout[..., 0] = -1, 1, 1
                    scale = 1.0
                q, k, v = [tensor.requires_grad_() for tensor in (q, k, v)]
                actual = self.run_attention(q, k, v, dout, causal=causal, scale=scale)
                self.compare(actual, q, k, v, dout, causal=causal, scale=scale)
                if case == "sharp":
                    self.assertGreater(actual[1].double().norm().item(), 0)
                    self.assertGreater(actual[2].double().norm().item(), 0)

    def test_key_gradients_preserve_common_and_invisible_query_rows(self):
        for dtype, causal in itertools.product((torch.float16, torch.bfloat16), (False, True)):
            with self.subTest(dtype=dtype, causal=causal):
                torch.manual_seed(1277)
                q, k, v, dout = [torch.randn(1, 16, 128, 128, device="cuda", dtype=dtype) for _ in range(4)]
                q[..., 0], k[..., 0] = 20, -20
                # This query is invisible to most causal keys. Its magnitude
                # must not become a substitute for the real dS residual product.
                q[:, :, 0].zero_()
                q[:, :, 0, 0] = 32768
                q, k, v = [tensor.requires_grad_() for tensor in (q, k, v)]
                self.compare(self.run_attention(q, k, v, dout, causal=causal), q, k, v, dout, causal=causal)

    def test_d256_thresholds_tails_rectangles_and_poisoned_graphs(self):
        from benchmarks.compare_sm86 import graph_metadata
        from kohakufa_cute import kernels, native

        cases = (
            (1023, 1023),
            (1024, 1024),
            (1025, 1025),
            (1025, 1031),
            (4095, 4095),
            (4096, 4096),
            (4097, 4097),
            (4097, 4095),
            (1024, 1024),
        )
        for dtype, causal in itertools.product((torch.float16, torch.bfloat16), (False, True)):
            native.extension().clear_cache()
            kernels._compile_kernel.cache_clear()
            with patch.object(native, "compile_kernel", wraps=native.compile_kernel) as compiler:
                for queries, keys in cases:
                    with self.subTest(dtype=dtype, causal=causal, queries=queries, keys=keys):
                        torch.manual_seed(1319)
                        q = torch.randn(1, 16, queries + 4, 256, device="cuda", dtype=dtype)[:, :, 1 : queries + 1]
                        k, v = [
                            torch.randn(1, 16, keys + 4, 256, device="cuda", dtype=dtype)[:, :, 2 : keys + 2]
                            for _ in range(2)
                        ]
                        q, k, v = [tensor.requires_grad_(True) for tensor in (q, k, v)]
                        dout = torch.randn_like(q)
                        actual = self.run_attention(q, k, v, dout, causal=causal)
                        self.compare(actual, q, k, v, dout, causal=causal)
                        if queries in (1024, 1025, 4097):
                            q, k, v = [tensor.detach().requires_grad_() for tensor in (q, k, v)]
                            stream = torch.cuda.Stream()
                            stream.wait_stream(torch.cuda.current_stream())
                            with torch.cuda.stream(stream):
                                for _ in range(3):
                                    self.run_attention(q, k, v, dout, causal=causal)
                                graph = torch.cuda.CUDAGraph(keep_graph=True)
                                with torch.cuda.graph(graph, stream=stream):
                                    replayed = self.run_attention(q, k, v, dout, causal=causal)
                            torch.cuda.current_stream().wait_stream(stream)
                            names = [node["symbol"] for node in graph_metadata(graph)]
                            self.assertTrue(any("MediumKeyValue" in name for name in names))
                            self.assertEqual(
                                any("kernel_queries" in name for name in names), queries == keys and queries >= 4096
                            )
                            if queries == keys == 4097:
                                with torch.no_grad():
                                    for tensor in (q, k, v, dout):
                                        tensor.copy_(torch.randn_like(tensor) * 0.5)
                                    for tensor in replayed:
                                        tensor.fill_(float("nan"))
                                graph.replay()
                                self.compare(replayed, q, k, v, dout, causal=causal)
                self.assertEqual(compiler.call_count, 2)

    @staticmethod
    def capture_layouts(function, inputs, dout, causal):
        operands = tuple(tensor.detach().requires_grad_() for tensor in inputs)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())

        def operation():
            out = function(*operands, is_causal=causal)
            return out, *torch.autograd.grad(out, operands, dout)

        with torch.cuda.stream(stream):
            for _ in range(3):
                operation()
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, stream=stream):
                actual = operation()
        torch.cuda.current_stream().wait_stream(stream)
        graph.replay()
        torch.cuda.synchronize()
        return graph, (operands, dout, None, actual)

    def test_d64_long_bounds_cache_compiled_calls_and_poisoned_replay(self):
        from benchmarks.compare_sm86 import graph_metadata
        from kohakufa_cute import kernels, native, scaled_dot_product_attention

        cases = (
            (1023, 1023),
            (1024, 1024),
            (1025, 1025),
            (1025, 1031),
            (4095, 4095),
            (4096, 4096),
            (4097, 4097),
            (4096, 4095),
            (1024, 1024),
        )
        for dtype, causal in itertools.product((torch.float16, torch.bfloat16), (False, True)):
            native.extension().clear_cache()
            kernels._compile_kernel.cache_clear()
            with patch.object(native, "compile_kernel", wraps=native.compile_kernel) as compiler:
                for queries, keys in cases:
                    with self.subTest(dtype=dtype, causal=causal, queries=queries, keys=keys):
                        torch.manual_seed(1847)
                        inputs = tuple(
                            torch.randn(1, 16, length + 4, 64, device="cuda", dtype=dtype)[:, :, offset : offset + length]
                            for length, offset in ((queries, 1), (keys, 2), (keys, 2))
                        )
                        dout = torch.randn_like(inputs[0])
                        graph, storage = self.capture_layouts(scaled_dot_product_attention, inputs, dout.clone(), causal)
                        self.assertEqual(storage[0][0].stride(), inputs[0].stride())
                        expected_n2 = dtype == torch.bfloat16 and queries == keys and 1024 <= queries <= 4096
                        names = [node["symbol"] for node in graph_metadata(graph)]
                        self.assertEqual(any("D64Values" in name for name in names), expected_n2)
                        self.compare(storage[3], *inputs, dout, causal=causal)
                        if queries == keys and queries in (1024, 4097):
                            operands, graph_dout, _, outputs = storage
                            with torch.no_grad():
                                for tensor in (*operands, graph_dout):
                                    tensor.copy_(torch.randn_like(tensor) * 0.5)
                                for tensor in outputs:
                                    tensor.fill_(float("nan"))
                            graph.replay()
                            self.compare(outputs, *operands, graph_dout, causal=causal)
                # Native caches retain a separate D64 max-length >= 1024 class.
                self.assertEqual(compiler.call_count, 4)
            torch.compiler.reset()
            compiled = torch.compile(attention, fullgraph=True, dynamic=True)
            for length in (1024, 4097, 1025):
                inputs = tuple(torch.randn(1, 16, length, 64, device="cuda", dtype=dtype).requires_grad_() for _ in range(3))
                dout = torch.randn_like(inputs[0])
                actual = self.run_attention(*inputs, dout, causal=causal, function=compiled)
                self.compare(actual, *inputs, dout, causal=causal)
            torch.compiler.reset()

    def test_d64_masks_gqa_batch_and_vector_fallbacks(self):
        from benchmarks.compare_sm86 import graph_metadata

        for dtype, kind in itertools.product((torch.float16, torch.bfloat16), ("stride", "batch", "gqa", "padding", "full")):
            with self.subTest(dtype=dtype, kind=kind):
                torch.manual_seed(1859)
                batch, heads, kv_heads = (2, 4, 4) if kind == "batch" else (1, 16, 8 if kind == "gqa" else 16)
                stride = 2 if kind == "stride" else 1
                inputs = tuple(
                    torch.randn(batch, count, 1024, 64 * stride, device="cuda", dtype=dtype)[..., ::stride]
                    for count in (heads, kv_heads, kv_heads)
                )
                dout = torch.randn_like(inputs[0])
                mask = None
                if kind in ("padding", "full"):
                    mask = torch.ones(batch, 1, 1024 if kind == "full" else 1, 1024, device="cuda", dtype=torch.bool)
                    mask[..., -1] = False
                    if kind == "full":
                        mask[:, :, -1] = False

                def function(q, k, v, is_causal=False):
                    return attention(q, k, v, causal=is_causal, mask=mask)

                graph, storage = self.capture_layouts(function, inputs, dout.clone(), True)
                self.assertEqual(storage[0][0].stride(), inputs[0].stride())
                self.assertFalse(any("D64Values" in node["symbol"] for node in graph_metadata(graph)))
                self.compare(storage[3], *inputs, dout, causal=True, mask=mask)
                if mask is not None:
                    self.assertEqual(torch.count_nonzero(storage[3][2][:, :, -1]).item(), 0)
                    self.assertEqual(torch.count_nonzero(storage[3][3][:, :, -1]).item(), 0)
                    if kind == "full":
                        self.assertEqual(torch.count_nonzero(storage[3][0][:, :, -1]).item(), 0)
                        self.assertEqual(torch.count_nonzero(storage[3][1][:, :, -1]).item(), 0)

    def test_d64_selected_bf16_and_fp16_fallback_preserve_directed_coordinates(self):
        from benchmarks.compare_sm86 import graph_metadata
        from tests.test_key_precision import key_cancellation, key_coordinate
        from tests.test_value_precision import square_value_cancellation, value_residual_underflow

        for dtype, causal in itertools.product((torch.float16, torch.bfloat16), (False, True)):
            with self.subTest(dtype=dtype, causal=causal, recipe="key"):
                inputs, dout = key_cancellation(64, dtype, "cuda", square=True)
                if causal:
                    # Active rows see all 1031 non-negligible keys, preserving
                    # the same closed form under causal visibility.
                    inputs = (inputs[0].roll(1031, 2), inputs[1], inputs[2])
                    dout = dout.roll(1031, 2)

                def function(q, k, v, is_causal=False):
                    return attention(q, k, v, causal=is_causal)

                graph, storage = self.capture_layouts(function, inputs, dout.clone(), causal)
                self.assertEqual(
                    any("D64Values" in node["symbol"] for node in graph_metadata(graph)), dtype == torch.bfloat16
                )
                self.compare(storage[3], *inputs, dout, causal=causal)
                coordinate = storage[3][2][:, :, 0, 0].double()
                self.assertTrue((coordinate < 0).all())
                self.assertTrue((coordinate != 0).all())
                torch.testing.assert_close(coordinate, torch.full_like(coordinate, key_coordinate(64)), atol=0, rtol=0.02)
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype, recipe="value"):
                inputs, dout, scale, target = square_value_cancellation(64, dtype, "cuda", length=1024)

                def function(q, k, v, is_causal=False):
                    return attention(q, k, v, causal=is_causal, scale=scale)

                graph, storage = self.capture_layouts(function, inputs, dout.clone(), False)
                self.assertEqual(
                    any("D64Values" in node["symbol"] for node in graph_metadata(graph)), dtype == torch.bfloat16
                )
                self.compare(storage[3], *inputs, dout, scale=scale)
                coordinate = storage[3][3][:, :, 0, 0].double()
                self.assertTrue((coordinate < 0).all())
                torch.testing.assert_close(coordinate, torch.full_like(coordinate, target), atol=0, rtol=0.02)
        inputs, dout, scale, target = value_residual_underflow(64, "cuda", length=1024)

        def function(q, k, v, is_causal=False):
            return attention(q, k, v, causal=is_causal, scale=scale)

        graph, storage = self.capture_layouts(function, inputs, dout.clone(), False)
        self.assertFalse(any("D64Values" in node["symbol"] for node in graph_metadata(graph)))
        self.compare(storage[3], *inputs, dout, scale=scale)
        coordinate = storage[3][3][:, :, 0, 0].double()
        self.assertTrue((coordinate > 0).all())
        torch.testing.assert_close(coordinate, torch.full_like(coordinate, target), atol=0, rtol=0.005)
