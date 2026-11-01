"""Check H200 optional-package routing and native fallback boundaries."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from kohakufa_cute import fa4_hopper


class TensorDescription:
    ndim = 4
    shape = (1, 16, 4096, 128)
    dtype = torch.bfloat16
    device = torch.device("cuda", 0)

    def is_contiguous(self):
        return True


class HopperEligibilityTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for target, value in (
            ("torch.compiler.is_compiling", False),
            ("torch.are_deterministic_algorithms_enabled", False),
            ("torch.cuda.current_device", 0),
            ("torch.cuda.get_device_capability", (9, 0)),
            (
                "torch.cuda.get_device_properties",
                SimpleNamespace(name="NVIDIA H200 NVL", multi_processor_count=132, shared_memory_per_block_optin=232448),
            ),
        ):
            self.stack.enter_context(patch(target, return_value=value))
        self.stack.enter_context(patch.object(fa4_hopper, "available", return_value=True))
        self.inputs = [TensorDescription() for _ in range(3)]

    def compatible(self, **kwargs):
        parameters = dict(mask=None, dropout=0.0, causal=False, scale=None, gqa=False)
        parameters.update(kwargs)
        return fa4_hopper.compatible(*self.inputs, **parameters)

    def test_experimental_full_h200_calls_and_both_visibility_modes(self):
        self.assertTrue(self.compatible())
        self.assertTrue(self.compatible(causal=True, scale=0.25))

    def test_masks_dropout_signed_subnormal_and_nonfinite_scales_keep_fallback(self):
        for parameters in (
            dict(mask=object()),
            dict(dropout=0.1),
            dict(scale=0),
            dict(scale=-0.25),
            dict(scale=1e-100),
            dict(scale=float("nan")),
            dict(scale=float("inf")),
        ):
            with self.subTest(parameters=parameters):
                self.assertFalse(self.compatible(**parameters))

    def test_short_tails_other_heads_and_dimensions_keep_fallback(self):
        for shape in ((1, 16, 128, 128), (1, 16, 4095, 128), (1, 8, 4096, 128), (1, 16, 4096, 256)):
            with self.subTest(shape=shape):
                for tensor in self.inputs:
                    tensor.shape = shape
                self.assertFalse(self.compatible())

    def test_other_device_partitions_and_strides_keep_fallback(self):
        with patch(
            "torch.cuda.get_device_properties",
            return_value=SimpleNamespace(name="NVIDIA H200", multi_processor_count=66, shared_memory_per_block_optin=232448),
        ):
            self.assertFalse(self.compatible())
        with patch.object(self.inputs[0], "is_contiguous", return_value=False):
            self.assertFalse(self.compatible())
        with patch("torch.are_deterministic_algorithms_enabled", return_value=True):
            self.assertFalse(self.compatible())
        with patch("torch.compiler.is_compiling", return_value=True):
            self.assertFalse(self.compatible())


if __name__ == "__main__":
    unittest.main()
