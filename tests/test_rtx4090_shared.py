import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from kohakufa_cute import api, fa4_sm80
from kohakufa_cute.fa4_sm80 import rtx4090


def operand(shape=(1, 16, 4096, 128), *, dtype=torch.bfloat16, pointer=4096, contiguous=True):
    return SimpleNamespace(
        shape=shape,
        ndim=4,
        device=torch.device("cuda", 0),
        dtype=dtype,
        data_ptr=lambda: pointer,
        is_contiguous=lambda: contiguous,
    )


class Rtx4090SharedTests(unittest.TestCase):
    def setUp(self):
        self.properties = SimpleNamespace(
            name="NVIDIA GeForce RTX 4090", multi_processor_count=128, shared_memory_per_block_optin=101376
        )
        self.patches = [
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)),
            patch.object(torch.cuda, "get_device_properties", return_value=self.properties),
            patch.object(fa4_sm80, "available", return_value=True),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def test_measured_families_and_fallback_boundaries(self):
        for dimension in (64, 128):
            for sequence in (1024, 4096):
                for causal in (False, True):
                    for dtype in (torch.float16, torch.bfloat16):
                        q = operand((1, 16, sequence, dimension), dtype=dtype)
                        self.assertTrue(rtx4090.compatible(q, q, q, causal=causal))
        for shape in (
            (1, 16, 128, 128),
            (1, 16, 1025, 128),
            (1, 16, 2048, 128),
            (2, 16, 4096, 128),
            (1, 8, 4096, 128),
            (1, 16, 4096, 16),
            (1, 16, 4096, 256),
            (1, 16, 4096, 512),
        ):
            q = operand(shape)
            self.assertFalse(rtx4090.compatible(q, q, q), shape)
        q = operand()
        for scale in (0.0, -0.125, 1e-40, float("inf"), float("nan")):
            self.assertFalse(rtx4090.compatible(q, q, q, scale=scale))
        for options in ({"mask": object()}, {"dropout": 0.1}, {"gqa": True}):
            self.assertFalse(rtx4090.compatible(q, q, q, **options))
        for position in range(3):
            for bad in (operand(pointer=4098), operand(contiguous=False)):
                values = [q, q, q]
                values[position] = bad
                self.assertFalse(rtx4090.compatible(*values))

    def test_partition_and_runtime_guards(self):
        q = operand()
        for name, value in (
            ("name", "Other SM89 device"),
            ("multi_processor_count", 64),
            ("shared_memory_per_block_optin", 98304),
        ):
            with patch.object(self.properties, name, value):
                self.assertFalse(rtx4090.compatible(q, q, q))
        with patch.object(fa4_sm80, "available", return_value=False):
            self.assertFalse(rtx4090.compatible(q, q, q))
        with patch.object(torch, "are_deterministic_algorithms_enabled", return_value=True):
            self.assertFalse(rtx4090.compatible(q, q, q))
        with patch.object(torch.compiler, "is_compiling", return_value=True):
            self.assertFalse(rtx4090.compatible(q, q, q))

    def test_dense_storage_repair_keeps_causal_geometry(self):
        dense = rtx4090.schedule(128, False)
        causal = rtx4090.schedule(128, True)
        self.assertEqual((dense.backward_m, dense.backward_n, dense.v_in_regs), (32, 128, False))
        self.assertEqual((causal.backward_m, causal.backward_n, causal.v_in_regs), (64, 64, True))
        self.assertEqual((dense.query_stages, dense.dout_stages), (1, 1))
        self.assertEqual((dense.causal_head_group, causal.causal_head_group), (1, 16))
        self.assertEqual(rtx4090.schedule(64, True).forward_m, 64)
        self.assertEqual(rtx4090.schedule(64, False).forward_m, 128)

    def test_selected_errors_propagate_and_fallback_keeps_arguments(self):
        q = operand()
        with patch.object(rtx4090, "native_runner", side_effect=RuntimeError("selected kernel failed")):
            with self.assertRaisesRegex(RuntimeError, "selected kernel"):
                rtx4090.scaled_dot_product_attention(q, q, q)
        mask = object()
        fallback = Mock(return_value="fallback")
        with patch.object(api, "scaled_dot_product_attention", fallback):
            self.assertEqual(
                rtx4090.scaled_dot_product_attention(q, q, q, mask, 0.2, True, scale=-0.125, enable_gqa=True),
                "fallback",
            )
            fallback.assert_called_once_with(q, q, q, mask, 0.2, True, scale=-0.125, enable_gqa=True)


if __name__ == "__main__":
    unittest.main()
