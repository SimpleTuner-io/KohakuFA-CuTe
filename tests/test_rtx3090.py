"""Unsupported opt-in calls preserve the ordinary attention boundary."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from kohakufa_cute import rtx3090_attention
from kohakufa_cute.rtx3090 import _device_matches


class Rtx3090RoutingTests(unittest.TestCase):
    def test_selected_runner_preserves_the_public_call_arguments(self):
        for sequence, causal, group in ((1024, True, 16), (1025, True, 4), (4096, True, 4), (2048, False, 1)):
            q = torch.empty(1, 16, sequence, 64, dtype=torch.float16)
            operation = Mock(return_value=object())
            with (
                self.subTest(sequence=sequence, causal=causal),
                patch("kohakufa_cute.rtx3090._eligible", return_value=True),
                patch("kohakufa_cute.rtx3090._runner", return_value=operation) as runner,
            ):
                self.assertIs(rtx3090_attention(q, q, q, is_causal=causal, scale=0.125), operation.return_value)
                runner.assert_called_once_with(64, causal, group)
                operation.assert_called_once_with(q, q, q, scale=0.125, causal=causal)

    def test_short_owned_runner_receives_the_public_arguments(self):
        q = torch.empty(1, 16, 128, 16, dtype=torch.float16)
        operation = Mock(return_value=object())
        with (
            patch("kohakufa_cute.rtx3090._eligible", return_value=True),
            patch("kohakufa_cute.rtx3090._short_runner", return_value=operation),
        ):
            self.assertIs(rtx3090_attention(q, q, q, is_causal=True), operation.return_value)
            operation.assert_called_once_with(q, q, q, scale=None, causal=True)

    def test_only_the_qualified_full_ordinary_device_matches(self):
        for name, sms, expected in (
            ("NVIDIA GeForce RTX 3090", 82, True),
            ("NVIDIA GeForce RTX 3090 Ti", 84, False),
            ("NVIDIA A40", 84, False),
            ("NVIDIA GeForce RTX 3090", 41, False),
        ):
            with (
                self.subTest(name=name, sms=sms),
                patch(
                    "torch.cuda.get_device_properties",
                    return_value=SimpleNamespace(major=8, minor=6, multi_processor_count=sms, name=name),
                ),
            ):
                _device_matches.cache_clear()
                self.assertEqual(_device_matches(0), expected)
        _device_matches.cache_clear()

    def test_cpu_call_delegates_all_sdpa_arguments(self):
        q = torch.zeros(1, 16, 5, 64)
        mask = torch.ones(5, 5, dtype=torch.bool)
        sentinel = object()
        with patch("kohakufa_cute.api.scaled_dot_product_attention", return_value=sentinel) as ordinary:
            self.assertIs(rtx3090_attention(q, q, q, mask, 0.0, True, -0.25, True), sentinel)
        ordinary.assert_called_once_with(q, q, q, mask, 0.0, True, -0.25, True)

    def test_compiled_call_does_not_query_hardware_or_optional_runtime(self):
        q = torch.zeros(1, 16, 1024, 128)
        sentinel = object()
        with (
            patch("torch.compiler.is_compiling", return_value=True),
            patch("kohakufa_cute.rtx3090._device_matches", side_effect=AssertionError),
            patch("kohakufa_cute.api.scaled_dot_product_attention", return_value=sentinel),
        ):
            self.assertIs(rtx3090_attention(q, q, q), sentinel)


if __name__ == "__main__":
    unittest.main()
