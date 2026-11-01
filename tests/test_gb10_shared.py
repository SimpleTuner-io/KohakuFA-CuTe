import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from kohakufa_cute import api, fa4_sm80
from kohakufa_cute.fa4_sm80 import gb10


def operand(shape=(1, 16, 4096, 128), *, dtype=torch.bfloat16, pointer=4096, contiguous=True):
    return SimpleNamespace(
        shape=shape,
        ndim=4,
        device=torch.device("cuda", 0),
        dtype=dtype,
        data_ptr=lambda: pointer,
        is_contiguous=lambda: contiguous,
    )


class Gb10SharedTests(unittest.TestCase):
    def setUp(self):
        self.properties = SimpleNamespace(name="NVIDIA GB10", multi_processor_count=48, shared_memory_per_block_optin=101376)
        self.patches = [
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(torch.cuda, "get_device_capability", return_value=(12, 1)),
            patch.object(torch.cuda, "get_device_properties", return_value=self.properties),
            patch.object(fa4_sm80, "available", return_value=True),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def test_measured_families_and_fallback_boundaries(self):
        for heads in (4, 16):
            for sequence in (512, 1024, 4096):
                for causal in (False, True):
                    for dtype in (torch.float16, torch.bfloat16):
                        q = operand((1, heads, sequence, 128), dtype=dtype)
                        self.assertEqual(gb10.compatible(q, q, q, causal=causal), not (causal and sequence == 1024))
        for shape in (
            (1, 16, 128, 128),
            (1, 16, 513, 128),
            (1, 16, 2048, 128),
            (2, 16, 4096, 128),
            (1, 8, 4096, 128),
            (1, 16, 4096, 16),
            (1, 16, 4096, 64),
            (1, 16, 4096, 256),
            (1, 16, 4096, 512),
        ):
            q = operand(shape)
            self.assertFalse(gb10.compatible(q, q, q), shape)
        q = operand()
        for scale in (0.0, -0.125, 1e-40, float("inf"), float("nan")):
            self.assertFalse(gb10.compatible(q, q, q, scale=scale))
        for options in ({"mask": object()}, {"dropout": 0.1}, {"gqa": True}):
            self.assertFalse(gb10.compatible(q, q, q, **options))
        for position in range(3):
            for bad in (operand(pointer=4098), operand(contiguous=False)):
                values = [q, q, q]
                values[position] = bad
                self.assertFalse(gb10.compatible(*values))

    def test_partition_and_runtime_guards(self):
        q = operand()
        for name, value in (
            ("name", "Other SM121 device"),
            ("multi_processor_count", 64),
            ("shared_memory_per_block_optin", 98304),
        ):
            with patch.object(self.properties, name, value):
                self.assertFalse(gb10.compatible(q, q, q))
        with patch.object(fa4_sm80, "available", return_value=False):
            self.assertFalse(gb10.compatible(q, q, q))
        with patch.object(torch, "are_deterministic_algorithms_enabled", return_value=True):
            self.assertFalse(gb10.compatible(q, q, q))
        with patch.object(torch.compiler, "is_compiling", return_value=True):
            self.assertFalse(gb10.compatible(q, q, q))

    def test_short_dense_and_causal_keep_the_qualified_query_tile(self):
        dense = gb10.schedule(4096, False, dtype=torch.float16)
        for sequence, causal in ((512, False), (512, True), (4096, True)):
            selected = gb10.schedule(sequence, causal)
            self.assertEqual(
                (selected.backward_m, selected.backward_n, selected.query_stages, selected.v_in_regs), (64, 64, 2, True)
            )
        self.assertEqual((dense.backward_m, dense.backward_n, dense.query_stages, dense.v_in_regs), (32, 128, 1, False))

    def test_paired_controls_keep_bf16_long_and_h16_medium_on_m64(self):
        for heads in (4, 16):
            self.assertEqual(gb10.schedule(4096, False, heads, torch.bfloat16).backward_m, 64)
            self.assertEqual(gb10.schedule(4096, False, heads, torch.float16).backward_m, 32)
        for dtype in (torch.bfloat16, torch.float16):
            self.assertEqual(gb10.schedule(1024, False, 4, dtype).backward_m, 32)
            self.assertEqual(gb10.schedule(1024, False, 16, dtype).backward_m, 64)

    def test_selected_errors_propagate_and_fallback_keeps_arguments(self):
        q = operand()
        with patch.object(gb10, "native_runner", side_effect=RuntimeError("selected kernel failed")):
            with self.assertRaisesRegex(RuntimeError, "selected kernel"):
                gb10.scaled_dot_product_attention(q, q, q)
        mask = object()
        fallback = Mock(return_value="fallback")
        with patch.object(api, "scaled_dot_product_attention", fallback):
            self.assertEqual(
                gb10.scaled_dot_product_attention(q, q, q, mask, 0.2, True, scale=-0.125, enable_gqa=True),
                "fallback",
            )
            fallback.assert_called_once_with(q, q, q, mask, 0.2, True, scale=-0.125, enable_gqa=True)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 1) and fa4_sm80.available(),
    "Requires the qualified GB10 optional runtime",
)
class Gb10FallbackGpuTests(unittest.TestCase):
    def test_tails_masks_gqa_strides_and_signed_scales_use_native_arithmetic(self):
        from benchmarks.reference import fp64_reference, metrics, rounded_reference

        for dtype in (torch.bfloat16, torch.float16):
            for case in ("tail", "padding", "hidden", "strided", "gqa", "zero", "negative"):
                with self.subTest(dtype=dtype, case=case):
                    torch.manual_seed(2031)
                    length = 512 if case in ("padding", "hidden", "zero", "negative") else 129
                    q, k, v = [torch.randn(1, 4, length, 128, device="cuda", dtype=dtype) for _ in range(3)]
                    if case == "strided":
                        q = torch.randn(1, 4, length, 256, device="cuda", dtype=dtype)[..., ::2]
                    if case == "gqa":
                        k, v = [torch.randn(1, 2, 71, 128, device="cuda", dtype=dtype) for _ in range(2)]
                    mask = None
                    if case in ("padding", "hidden"):
                        mask = torch.ones(length, length, device="cuda", dtype=torch.bool)
                        mask[:, -7:] = False
                        mask[-3:] = False
                        if case == "hidden":
                            mask.zero_()
                    scale = {"zero": 0.0, "negative": -0.125}.get(case)
                    operands = tuple(t.detach().requires_grad_(True) for t in (q, k, v))
                    dout = torch.randn_like(q)
                    self.assertFalse(gb10.compatible(*operands, mask=mask, scale=scale, gqa=case == "gqa"))
                    with patch.object(gb10, "native_runner", side_effect=AssertionError("Unexpected selected runner")):
                        out = gb10.scaled_dot_product_attention(
                            *operands, attn_mask=mask, scale=scale, enable_gqa=case == "gqa"
                        )
                    actual = out, *torch.autograd.grad(out, operands, dout)
                    exact = fp64_reference(operands, dout, mask=mask, scale=scale)
                    rounded = rounded_reference(operands, dout, mask=mask, scale=scale, output=out)
                    for got, mathematical, declared in zip(actual, exact, rounded):
                        for head in range(got.shape[1]):
                            error = metrics(got[:, head : head + 1], mathematical[:, head : head + 1])
                            contract = metrics(got[:, head : head + 1], declared[:, head : head + 1])
                            self.assertTrue(error["finite"], (case, error, contract))
                            if error["reference_norm"]:
                                self.assertLess(
                                    error["relative_error"],
                                    0.01 if dtype == torch.bfloat16 else 0.005,
                                    (case, error, contract),
                                )
                            else:
                                self.assertEqual(error["max_absolute_error"], 0, (case, error, contract))


if __name__ == "__main__":
    unittest.main()
