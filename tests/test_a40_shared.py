import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from kohakufa_cute import api, fa4_sm80
from kohakufa_cute.fa4_sm80 import a40


def operand(shape=(1, 16, 4096, 128), *, dtype=torch.bfloat16, pointer=4096, contiguous=True):
    return SimpleNamespace(
        shape=shape,
        ndim=4,
        device=torch.device("cuda", 0),
        dtype=dtype,
        data_ptr=lambda: pointer,
        is_contiguous=lambda: contiguous,
    )


class A40SharedTests(unittest.TestCase):
    def setUp(self):
        self.properties = SimpleNamespace(
            name="NVIDIA A40", major=8, minor=6, multi_processor_count=84, shared_memory_per_block_optin=101376
        )
        a40._is_device.cache_clear()
        self.addCleanup(a40._is_device.cache_clear)
        self.patches = [
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(torch.cuda, "get_device_capability", return_value=(8, 6)),
            patch.object(torch.cuda, "get_device_properties", return_value=self.properties),
            patch.object(fa4_sm80, "available", return_value=True),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def test_measured_families_and_fallback_boundaries(self):
        for dimension in (128,):
            for sequence in (1024, 4096):
                for causal in (False, True):
                    for dtype in (torch.float16, torch.bfloat16):
                        q = operand((1, 16, sequence, dimension), dtype=dtype)
                        self.assertTrue(a40.compatible(q, q, q, causal=causal))
        for sequence in (1024, 4096):
            q = operand((1, 16, sequence, 64))
            self.assertTrue(a40.compatible(q, q, q, causal=True))
            self.assertFalse(a40.compatible(q, q, q, causal=False))
        for shape in (
            (1, 16, 128, 128),
            (1, 16, 1025, 128),
            (1, 16, 2048, 128),
            (2, 16, 4096, 128),
            (1, 8, 4096, 128),
            (1, 16, 4096, 16),
            (1, 16, 4096, 64),
            (1, 16, 4096, 256),
            (1, 16, 4096, 512),
        ):
            q = operand(shape)
            self.assertFalse(a40.compatible(q, q, q), shape)
        q = operand()
        for scale in (0.0, -0.125, 1e-40, float("inf"), float("nan")):
            self.assertFalse(a40.compatible(q, q, q, scale=scale))
        for options in ({"mask": object()}, {"dropout": 0.1}, {"gqa": True}):
            self.assertFalse(a40.compatible(q, q, q, **options))
        for position in range(3):
            for bad in (operand(pointer=4098), operand(contiguous=False)):
                values = [q, q, q]
                values[position] = bad
                self.assertFalse(a40.compatible(*values))

    def test_partition_and_runtime_guards(self):
        q = operand()
        for name, value in (
            ("major", 9),
            ("minor", 9),
            ("name", "Other SM89 device"),
            ("multi_processor_count", 64),
            ("shared_memory_per_block_optin", 98304),
        ):
            with patch.object(self.properties, name, value):
                a40._is_device.cache_clear()
                self.assertFalse(a40.compatible(q, q, q))
        a40._is_device.cache_clear()
        with patch.object(fa4_sm80, "available", return_value=False):
            self.assertFalse(a40.compatible(q, q, q))
        with patch.object(torch, "are_deterministic_algorithms_enabled", return_value=True):
            self.assertFalse(a40.compatible(q, q, q))
        with patch.object(torch.compiler, "is_compiling", return_value=True):
            self.assertFalse(a40.compatible(q, q, q))

    def test_dense_storage_repair_keeps_causal_geometry(self):
        dense = a40.schedule(128, False)
        causal = a40.schedule(128, True)
        self.assertEqual((dense.backward_m, dense.backward_n, dense.v_in_regs), (32, 128, False))
        self.assertEqual((causal.backward_m, causal.backward_n, causal.v_in_regs), (64, 64, True))
        self.assertEqual((dense.query_stages, dense.dout_stages), (1, 1))

    def test_only_device_classification_is_cached_per_index(self):
        q = operand()
        with patch.object(torch.cuda, "get_device_properties", return_value=self.properties) as properties:
            self.assertTrue(a40.compatible(q, q, q))
            self.assertTrue(a40.compatible(q, q, q))
            self.assertEqual(properties.call_count, 1)
            q.device = torch.device("cuda", 1)
            self.assertFalse(a40.compatible(q, q, q))
            with patch.object(torch.cuda, "current_device", return_value=1):
                self.assertTrue(a40.compatible(q, q, q))
                self.assertEqual(properties.call_count, 2)
                with patch.object(torch, "are_deterministic_algorithms_enabled", return_value=True):
                    self.assertFalse(a40.compatible(q, q, q))

    def test_causal_work_order_presets_preserve_dense(self):
        short = a40.schedule(64, True, 1024)
        long = a40.schedule(64, True, 4096)
        self.assertEqual((short.causal_head_group, long.causal_head_group), (16, 4))
        self.assertEqual((short.query_stages, short.dout_stages, short.kv_m_warps, short.dq_m_warps), (2, 1, 4, 4))
        self.assertTrue(short.v_in_regs)
        self.assertEqual(a40.schedule(128, True, 1024).causal_head_group, 4)
        self.assertEqual(a40.schedule(128, False, 4096).causal_head_group, 1)

    def test_benchmark_control_retains_public_entry_and_group(self):
        from benchmarks.a40_shared_candidates import make_attention

        q = operand()
        with patch.object(a40, "native_runner", return_value=Mock(return_value="selected")) as runner:
            control = make_attention(head_group=1)
            self.assertEqual(control(q, q, q, is_causal=True), "selected")
            self.assertEqual(runner.call_args.args[0].causal_head_group, 1)
            a40.scaled_dot_product_attention(q, q, q, is_causal=True)
            self.assertEqual(runner.call_args.args[0].causal_head_group, 4)

    def test_selected_errors_propagate_and_fallback_keeps_arguments(self):
        q = operand()
        with patch.object(a40, "native_runner", side_effect=RuntimeError("selected kernel failed")):
            with self.assertRaisesRegex(RuntimeError, "selected kernel"):
                a40.scaled_dot_product_attention(q, q, q)
        mask = object()
        fallback = Mock(return_value="fallback")
        with patch.object(api, "scaled_dot_product_attention", fallback):
            self.assertEqual(
                a40.scaled_dot_product_attention(q, q, q, mask, 0.2, True, scale=-0.125, enable_gqa=True),
                "fallback",
            )
            fallback.assert_called_once_with(q, q, q, mask, 0.2, True, scale=-0.125, enable_gqa=True)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class A40SharedGpuTests(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_name() != "NVIDIA A40" or not fa4_sm80.available():
            self.skipTest("The qualified full A40 runtime is required")

    def check_call(self, inputs, dout, *, causal=False, scale=None, selected=False):
        from benchmarks.reference import fp64_reference, metrics

        operands = tuple(t.detach().requires_grad_(True) for t in inputs)
        if selected:
            with patch.object(api, "scaled_dot_product_attention", side_effect=AssertionError("unexpected fallback")):
                output = a40.scaled_dot_product_attention(*operands, is_causal=causal, scale=scale)
        else:
            output = a40.scaled_dot_product_attention(*operands, is_causal=causal, scale=scale)
        actual = output, *torch.autograd.grad(output, operands, dout)
        for head in range(inputs[0].shape[1]):
            reference = fp64_reference(
                tuple(t[:, head : head + 1] for t in inputs), dout[:, head : head + 1], causal=causal, scale=scale
            )
            for got, expected in zip(actual, reference):
                error = metrics(got[:, head : head + 1], expected)
                self.assertTrue(error["finite"])
                if error["reference_norm"]:
                    self.assertLess(error["relative_error"], 0.01 if got.dtype == torch.bfloat16 else 0.005)
                else:
                    self.assertEqual(error["max_absolute_error"], 0)

    def test_selected_scale_cache_reads_changed_inputs(self):
        for dtype in (torch.bfloat16, torch.float16):
            for dimension, causal in ((128, False), (128, True), (64, True)):
                for seed, scale in ((109, None), (113, 0.125), (127, 0.0625)):
                    torch.manual_seed(seed)
                    q, k, v, dout = [torch.randn(1, 16, 1024, dimension, dtype=dtype, device="cuda") for _ in range(4)]
                    self.check_call((q, k, v), dout, causal=causal, scale=scale, selected=True)

    def test_grouped_public_graphs_read_changed_inputs(self):
        from benchmarks.compare_ada import capture
        from benchmarks.validate_sm80_static import assess

        for dtype in (torch.bfloat16, torch.float16):
            for dimension, sequence in ((64, 1024), (64, 4096), (128, 1024), (128, 4096)):
                torch.manual_seed(151)
                q, k, v, dout = [torch.randn(1, 16, sequence, dimension, dtype=dtype, device="cuda") for _ in range(4)]
                self.assertTrue(a40.compatible(q, k, v, causal=True))
                graph, storage = capture(a40.scaled_dot_product_attention, (q, k, v), dout, True, "joint")
                operands, _, _, actual = storage
                with torch.no_grad():
                    for tensor in (*operands, dout):
                        tensor.copy_(torch.randn_like(tensor))
                    for tensor in actual:
                        tensor.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                assess(actual, operands, dout, causal=True, scale=None, directed=False, pointwise=False)

    def test_other_dimensions_keep_native_math(self):
        for dtype in (torch.bfloat16, torch.float16):
            for dimension in (16, 64, 256, 512):
                for causal in (False, True):
                    torch.manual_seed(131)
                    q, k, v, dout = [torch.randn(1, 2, 129, dimension, dtype=dtype, device="cuda") for _ in range(4)]
                    self.assertFalse(a40.compatible(q, k, v, causal=causal))
                    self.check_call((q, k, v), dout, causal=causal)


if __name__ == "__main__":
    unittest.main()
