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

    def compare(self, actual, q, k, v, dout, *, causal=False, scale=None):
        if max(q.shape[2], k.shape[2]) >= 2048:
            parts = [
                fp64_reference(
                    (q[:, head : head + 2], k[:, head : head + 2], v[:, head : head + 2]),
                    dout[:, head : head + 2],
                    causal=causal,
                    scale=scale,
                )
                for head in range(0, q.shape[1], 2)
            ]
            reference = tuple(torch.cat([part[i] for part in parts], dim=1) for i in range(4))
        else:
            reference = fp64_reference((q, k, v), dout, causal=causal, scale=scale)
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
