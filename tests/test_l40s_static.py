import sys
import unittest
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from kohakufa_cute import api, fa4_sm80
from tests.test_fa4_sm80 import cuda_operand


def training_operand(*args, **kwargs):
    operand = cuda_operand(*args, **kwargs)
    operand.requires_grad = True
    return operand


class L40sSelectionTests(unittest.TestCase):
    def setUp(self):
        from kohakufa_cute.fa4_sm80 import _l40s

        _l40s.is_device.cache_clear()
        self.addCleanup(_l40s.is_device.cache_clear)
        self.properties = SimpleNamespace(
            name="NVIDIA L40S", multi_processor_count=142, shared_memory_per_block_optin=101376
        )
        for control in (
            patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)),
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(torch.cuda, "get_device_properties", return_value=self.properties),
            patch.object(fa4_sm80, "available", return_value=True),
        ):
            control.start()
            self.addCleanup(control.stop)

    def test_bounded_full_device_calls_and_fallbacks(self):
        for dtype in (torch.bfloat16, torch.float16):
            for sequence in (1024, 4096):
                for dimension in (64, 128):
                    q = training_operand((1, 16, sequence, dimension), dtype)
                    for causal in (False, True):
                        self.assertEqual(
                            fa4_sm80.compatible(q, q, q, None, 0, causal, None, False),
                            not (causal and dimension == 64 and sequence != 4096),
                        )
        for shape in (
            (1, 16, 128, 64),
            (1, 16, 1023, 128),
            (1, 16, 1152, 128),
            (1, 16, 1280, 128),
            (1, 16, 2048, 128),
            (1, 16, 4608, 128),
            (1, 16, 4097, 64),
            (1, 8, 1024, 64),
            (2, 16, 1024, 64),
            (1, 16, 1024, 256),
        ):
            q = training_operand(shape)
            self.assertFalse(fa4_sm80.compatible(q, q, q, None, 0, False, None, False))
        q = training_operand()
        for scale in (0, -0.1, 1e-50, float("inf"), float("nan"), 3e38):
            self.assertFalse(fa4_sm80.compatible(q, q, q, None, 0, False, scale, False))
        self.assertFalse(fa4_sm80.compatible(q, q, q, object(), 0, False, None, False))
        self.assertFalse(fa4_sm80.compatible(q, q, q, None, 0.1, False, None, False))
        for position in range(3):
            operands = [q, q, q]
            operands[position] = training_operand(pointer=4098)
            self.assertFalse(fa4_sm80.compatible(*operands, None, 0, False, None, False))
        with patch.object(torch.compiler, "is_compiling", return_value=True):
            self.assertFalse(fa4_sm80.compatible(q, q, q, None, 0, False, None, False))
        for field, value in (
            ("name", "NVIDIA GeForce RTX 4090"),
            ("multi_processor_count", 141),
            ("shared_memory_per_block_optin", 65536),
        ):
            from kohakufa_cute.fa4_sm80 import _l40s

            _l40s.is_device.cache_clear()
            with patch.object(self.properties, field, value):
                self.assertFalse(fa4_sm80.compatible(q, q, q, None, 0, False, None, False))

    def test_device_classification_and_schedules_are_cached(self):
        from kohakufa_cute.fa4_sm80 import _l40s

        q = training_operand()
        with patch.object(torch.cuda, "get_device_properties", return_value=self.properties) as properties:
            for _ in range(2):
                self.assertTrue(fa4_sm80.compatible(q, q, q, None, 0, False, None, False))
            properties.assert_called_once()
        self.assertIs(_l40s.schedule(128, 4096, False), _l40s.schedule(128, 4096, False))
        _l40s.is_device.cache_clear()
        with patch.object(self.properties, "name", "NVIDIA GeForce RTX 4090"):
            with patch("kohakufa_cute.fa4_sm80._attention.attention", return_value=object()) as preceding:
                fa4_sm80.attention(q, q, q)
            preceding.assert_called_once()

    def test_a100_does_not_enter_l40s_tensor_validation(self):
        from kohakufa_cute.fa4_sm80 import _l40s

        q = training_operand()
        with (
            patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)),
            patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(multi_processor_count=108)),
            patch.object(_l40s, "compatible", side_effect=AssertionError("Unexpected L40S validation")),
        ):
            self.assertTrue(fa4_sm80.compatible(q, q, q, None, 0, False, None, False))

    def test_inference_and_no_grad_keep_the_preceding_dispatch(self):
        q = training_operand()
        with torch.no_grad():
            self.assertFalse(fa4_sm80.compatible(q, q, q, None, 0, False, None, False))
        q.requires_grad = False
        self.assertFalse(fa4_sm80.compatible(q, q, q, None, 0, False, None, False))

    def test_public_entry_uses_cached_runner_and_preserves_configuration(self):
        from kohakufa_cute.fa4_sm80 import _host, _l40s

        _host.native_runner.cache_clear()
        self.addCleanup(_host.native_runner.cache_clear)
        runner = Mock(return_value=object())
        extension = Mock()
        extension.AttentionRunner.return_value = runner
        q = training_operand()
        native = SimpleNamespace(extension=lambda: extension)
        with patch.dict(sys.modules, {"kohakufa_cute.native": native}):
            for function in (api.attention, api.scaled_dot_product_attention, api.automatic_scaled_dot_product_attention):
                function(q, q, q)
        selected = _l40s.schedule(128, 4096, False)
        extension.AttentionRunner.assert_called_once_with("warp", asdict(selected))
        self.assertEqual(len(asdict(selected)), 13)
        self.assertEqual(runner.call_count, 3)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_name() == "NVIDIA L40S" and fa4_sm80.available(),
    "Requires the qualified full-device L40S runtime",
)
class L40sPublicKernelTests(unittest.TestCase):
    def test_native_public_statistics_and_gradients(self):
        from benchmarks.validate_sm80_static import assess, statistics

        for dtype in (torch.bfloat16, torch.float16):
            for dimension in (64, 128):
                for sequence in (1024, 4096):
                    for causal in (False, True):
                        if causal and dimension == 64 and sequence != 4096:
                            continue
                        torch.manual_seed(10681)
                        inputs = [
                            torch.randn(1, 16, sequence, dimension, device="cuda", dtype=dtype, requires_grad=True)
                            for _ in range(3)
                        ]
                        dout = torch.randn_like(inputs[0])
                        self.assertTrue(fa4_sm80.compatible(*inputs, None, 0, causal, None, False))
                        saved = []
                        with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
                            with patch.object(api, "_extension", side_effect=AssertionError("Unexpected native fallback")):
                                out = api.scaled_dot_product_attention(*inputs, is_causal=causal)
                        self.assertEqual(len(saved), 6)
                        statistics(inputs, saved[-2], saved[-1], causal, dimension**-0.5)
                        actual = out, *torch.autograd.grad(out, inputs, dout)
                        assess(actual, inputs, dout, causal=causal, scale=None, directed=False, pointwise=False)

    def test_all_three_public_entries_execute_the_native_schedule(self):
        from benchmarks.validate_sm80_static import assess
        from kohakufa_cute.fa4_sm80 import _l40s

        inputs = [torch.randn(1, 16, 1024, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
        dout = torch.randn_like(inputs[0])
        with patch.object(_l40s, "attention", wraps=_l40s.attention) as selected:
            for function in (api.attention, api.scaled_dot_product_attention, api.automatic_scaled_dot_product_attention):
                output = function(*inputs)
                actual = output, *torch.autograd.grad(output, inputs, dout)
                assess(actual, inputs, dout, causal=False, scale=None, directed=False, pointwise=False)
            self.assertEqual(selected.call_count, 3)

    def test_inference_and_short_training_retain_native_fallback(self):
        from kohakufa_cute import native

        for sequence, dimension, causal, inference in (
            (128, 128, False, False),
            (1024, 64, True, False),
            (1024, 128, False, True),
        ):
            inputs = [
                torch.randn(1, 16, sequence, dimension, device="cuda", dtype=torch.bfloat16, requires_grad=not inference)
                for _ in range(3)
            ]
            self.assertFalse(fa4_sm80.compatible(*inputs, None, 0, causal, None, False))
            module = native.extension()
            operations = (
                (lambda: api.attention(*inputs, causal=causal), lambda: module.attention(*inputs, None, None, causal)),
                (
                    lambda: api.scaled_dot_product_attention(*inputs, is_causal=causal),
                    lambda: module.sdpa(*inputs, None, 0.0, causal, None, False),
                ),
                (
                    lambda: api.automatic_scaled_dot_product_attention(*inputs, is_causal=causal),
                    lambda: module.automatic_sdpa(*inputs, None, 0.0, causal, None, False),
                ),
            )
            for public, preceding in operations:
                with patch(
                    "kohakufa_cute.fa4_sm80._l40s.attention", side_effect=AssertionError("Unexpected L40S selection")
                ):
                    output = public()
                torch.testing.assert_close(output, preceding(), atol=0, rtol=0)

    def test_changed_poisoned_graph_and_unaligned_output_gradient(self):
        from benchmarks.compare_ada import capture
        from benchmarks.validate_sm80_static import assess

        for dtype in (torch.bfloat16, torch.float16):
            for dimension, causal in ((64, True), (128, False)):
                inputs = [torch.randn(1, 16, 4096, dimension, device="cuda", dtype=dtype) for _ in range(3)]
                dout = torch.empty(inputs[0].numel() + 1, device="cuda", dtype=dtype)[1:].view_as(inputs[0])
                dout.normal_()
                graph, storage = capture(api.scaled_dot_product_attention, inputs, dout, causal, "joint")
                operands, _, _, actual = storage
                for seed in (10687, 10691):
                    torch.manual_seed(seed)
                    with torch.no_grad():
                        for tensor in (*operands, dout):
                            tensor.copy_(torch.randn_like(tensor))
                        for tensor in actual:
                            tensor.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize()
                    assess(actual, operands, dout, causal=causal, scale=None, directed=False, pointwise=False)
