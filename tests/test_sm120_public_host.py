import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from kohakufa_cute import fa4_sm80
from kohakufa_cute.fa4_sm80 import _sm120


def operand(shape=(1, 16, 4096, 128), *, dtype=torch.bfloat16, pointer=4096, contiguous=True, requires_grad=True):
    return SimpleNamespace(
        ndim=4,
        shape=shape,
        dtype=dtype,
        device=torch.device("cuda", 0),
        requires_grad=requires_grad,
        data_ptr=lambda: pointer,
        is_contiguous=lambda: contiguous,
    )


class Sm120PublicSelectionTests(unittest.TestCase):
    def setUp(self):
        _sm120.target_device.cache_clear()

    def tearDown(self):
        _sm120.target_device.cache_clear()

    def runtime(self):
        return (
            patch.object(torch.cuda, "get_device_capability", return_value=(12, 0)),
            patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(
                    multi_processor_count=170, name="NVIDIA GeForce RTX 5090", shared_memory_per_block_optin=101376
                ),
            ),
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(fa4_sm80, "available", return_value=True),
        )

    def test_qualified_family_and_unchanged_schedule_fields(self):
        capability, properties, current, available = self.runtime()
        with capability, properties, current, available:
            for length in (4096,):
                for dimension in (64, 128):
                    for dtype in (torch.bfloat16, torch.float16):
                        for causal in (False, True):
                            q = operand((1, 16, length, dimension), dtype=dtype)
                            self.assertTrue(fa4_sm80.compatible(q, q, q, None, 0, causal, None, False))
            for dimension in (64, 128):
                schedule = _sm120.schedule(dimension)
                self.assertEqual((schedule.backward_m, schedule.backward_n), (64, 64))
                self.assertEqual((schedule.query_stages, schedule.dout_stages, schedule.threads), (2, 1, 256))
                self.assertEqual((schedule.score_m_warps, schedule.kv_m_warps, schedule.dq_m_warps), (4, 2, 2))
                self.assertFalse(schedule.v_in_regs)
                self.assertEqual(schedule.causal_head_group, 1)
                causal_schedule = _sm120.schedule(dimension, True)
                self.assertEqual(causal_schedule.causal_head_group, 16)
                self.assertEqual(causal_schedule.query_stages, schedule.query_stages)
                self.assertEqual(causal_schedule.backward_n, schedule.backward_n)

    def test_unqualified_shapes_layouts_and_modes_fall_through(self):
        capability, properties, current, available = self.runtime()
        with capability, properties, current, available:
            q = operand()
            for shape in ((1, 16, 128, 128), (1, 16, 1024, 128), (1, 16, 2048, 128), (1, 24, 4096, 128), (2, 16, 4096, 128)):
                value = operand(shape)
                self.assertFalse(fa4_sm80.compatible(value, value, value, None, 0, False, None, False))
            for value in (
                operand(contiguous=False),
                operand(pointer=4098),
                operand(requires_grad=False),
                operand(dtype=torch.float32),
            ):
                self.assertFalse(fa4_sm80.compatible(value, value, value, None, 0, False, None, False))
            for scale in (0, -0.125, 1e-50, float("nan"), float("inf")):
                self.assertFalse(fa4_sm80.compatible(q, q, q, None, 0, False, scale, False))
            self.assertFalse(fa4_sm80.compatible(q, q, q, object(), 0, False, None, False))
            self.assertFalse(fa4_sm80.compatible(q, q, q, None, 0.1, False, None, False))
            with torch.no_grad():
                self.assertFalse(fa4_sm80.compatible(q, q, q, None, 0, False, None, False))
            with patch.object(torch.compiler, "is_compiling", return_value=True):
                self.assertFalse(fa4_sm80.compatible(q, q, q, None, 0, False, None, False))
            with patch.object(fa4_sm80, "available", return_value=False):
                self.assertFalse(fa4_sm80.compatible(q, q, q, None, 0, False, None, False))

    def test_public_entry_reuses_the_canonical_host_helper(self):
        from kohakufa_cute.fa4_sm80 import _host, _l40s

        q = operand()
        runner = Mock(return_value=object())
        entry = fa4_sm80.attention
        with (
            patch.object(_l40s, "is_device", return_value=False),
            patch.object(_sm120, "target_device", return_value=True),
            patch.object(_host, "native_runner", return_value=runner) as factory,
        ):
            for _ in range(2):
                fa4_sm80.attention(q, q, q, causal=True)
                self.assertIs(fa4_sm80.attention, entry)
            self.assertEqual(factory.call_args.args[0], _sm120.schedule(128, True))
            self.assertEqual(runner.call_count, 2)


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.cuda.get_device_capability() == (12, 0)
    and torch.cuda.get_device_properties(0).multi_processor_count == 170
    and fa4_sm80.available(),
    "Requires the qualified full RTX 5090 runtime",
)
class Sm120PublicKernelTests(unittest.TestCase):
    def check_case(self, inputs, dout, *, causal=False, scale=None, directed=False, pointwise=False, function=None):
        from benchmarks.validate_sm80_static import assess, statistics
        from kohakufa_cute import api

        if function is None:
            from kohakufa_cute.fa4_sm80._host import native_runner

            def function(q, k, v, *, is_causal=False, scale=None):
                if q.shape[2] == 4096:
                    return api.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale)
                return native_runner(_sm120.schedule(q.shape[-1]))(q, k, v, causal=is_causal, scale=scale)

        operands = [t.detach().requires_grad_(True) for t in inputs]
        self.assertEqual(fa4_sm80.compatible(*operands, None, 0, causal, scale, False), inputs[0].shape[2] == 4096)
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
            out = function(*operands, is_causal=causal, scale=scale)
        self.assertEqual(len(saved), 6)
        statistics(operands, saved[-2], saved[-1], causal, inputs[0].shape[-1] ** -0.5 if scale is None else scale)
        actual = out, *torch.autograd.grad(out, operands, dout)
        assess(actual, operands, dout, causal=causal, scale=scale, directed=directed, pointwise=pointwise)

    def test_final_public_bounds(self):
        from benchmarks.compare_ada import capture
        from benchmarks.compare_l40s import graph_kernels
        from kohakufa_cute import api

        for length in (1024, 4096):
            inputs = [torch.randn(1, 16, length, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
            dout = torch.randn_like(inputs[0])
            operands = [t.detach().requires_grad_(True) for t in inputs]
            self.assertEqual(fa4_sm80.compatible(*operands, None, 0, True, None, False), length == 4096)
            graph, _ = capture(api.scaled_dot_product_attention, inputs, dout, True, "joint", keep_graph=True)
            symbols = [row["symbol"] for row in graph_kernels(graph)]
            self.assertEqual(any("fa4_sm80_backward" in symbol for symbol in symbols), length == 4096)

    def test_selected_matrix_and_public_frontdoors(self):
        from kohakufa_cute import api

        for dtype in (torch.bfloat16, torch.float16):
            for length in (4096,):
                for dimension in (64, 128):
                    for causal in (False, True):
                        torch.manual_seed(947)
                        q, k, v, dout = [torch.randn(1, 16, length, dimension, device="cuda", dtype=dtype) for _ in range(4)]
                        self.check_case((q, k, v), dout, causal=causal)
        inputs = [torch.randn(1, 16, 4096, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        dout = torch.randn_like(inputs[0])
        for function in (api.attention, api.automatic_scaled_dot_product_attention):
            if function is api.attention:

                def entry(q, k, v, *, is_causal=False, scale=None):
                    return function(q, k, v, causal=is_causal, scale=scale)

            else:
                entry = function
            self.check_case(inputs, dout, causal=True, function=entry)

    def test_selected_directed_rounding_and_negative_maxima(self):
        from tests.test_key_precision import key_cancellation
        from tests.test_value_precision import square_value_cancellation, value_residual_underflow

        for dimension in (64, 128):
            for dtype in (torch.bfloat16, torch.float16):
                inputs, dout = key_cancellation(dimension, dtype, "cuda", heads=16, square=True)
                self.check_case(inputs, dout, directed=True, pointwise=True)
                inputs, dout, scale, _ = square_value_cancellation(dimension, dtype, "cuda", heads=16, length=1024)
                self.check_case(inputs, dout, scale=scale, directed=True, pointwise=True)
            inputs, dout, scale, _ = value_residual_underflow(dimension, "cuda", heads=16, length=1024)
            self.check_case(inputs, dout, scale=scale, directed=True, pointwise=True)
        for dtype in (torch.bfloat16, torch.float16):
            torch.manual_seed(953)
            q, k, v, dout = [torch.randn(1, 16, 1024, 64, device="cuda", dtype=dtype) for _ in range(4)]
            q.mul_(0.3)
            k.mul_(0.3)
            q[..., 0], k[..., 0] = 20, -20
            self.check_case((q, k, v), dout, directed=True)

    def test_changed_graph_and_actual_selected_components(self):
        from benchmarks.compare_ada import capture
        from benchmarks.compare_l40s import graph_kernels
        from benchmarks.validate_sm80_static import assess
        from kohakufa_cute import api

        for dtype, dimension, causal in ((torch.bfloat16, 64, True), (torch.float16, 128, True)):
            inputs = [torch.randn(1, 16, 4096, dimension, device="cuda", dtype=dtype) for _ in range(3)]
            dout = torch.randn_like(inputs[0])
            graph, storage = capture(api.scaled_dot_product_attention, inputs, dout, causal, "joint", keep_graph=True)
            components = graph_kernels(graph)
            self.assertEqual(len(components), 4)
            self.assertTrue(any("fa4_sm80" in row["symbol"] and "BackwardSm80" in row["symbol"] for row in components))
            operands, _, _, actual = storage
            for seed in (967, 971):
                torch.manual_seed(seed)
                with torch.no_grad():
                    for tensor in (*operands, dout):
                        tensor.copy_(torch.randn_like(tensor))
                    for tensor in actual:
                        tensor.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                assess(actual, operands, dout, causal=causal, scale=None, directed=False, pointwise=False)


if __name__ == "__main__":
    unittest.main()
