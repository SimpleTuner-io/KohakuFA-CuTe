"""Automatic training eligibility, live metadata and actual API routing."""

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from kohakufa_cute import fa4_sm80
from kohakufa_cute.fa4_sm80 import rtx4090


class AutomaticPolicyTests(unittest.TestCase):
    def tearDown(self):
        rtx4090._automatic_metadata_predicate.cache_clear()

    def test_only_function_handle_is_cached(self):
        rtx4090._automatic_metadata_predicate.cache_clear()
        function = Mock()
        loader = Mock(return_value=SimpleNamespace(rtx4090_automatic_metadata_compatible=function))
        with patch.dict(sys.modules, {"kohakufa_cute.native": SimpleNamespace(extension=loader)}):
            self.assertIs(rtx4090._automatic_metadata_predicate(), function)
            self.assertIs(rtx4090._automatic_metadata_predicate(), function)
        loader.assert_called_once()

    def test_unsupported_policy_rejects_without_loading(self):
        for options in ((None, 0.1, None), (object(), 0, None), (None, 0, -0.1), (None, 0, 0), (None, 0, float("nan"))):
            with patch.object(rtx4090, "_automatic_metadata_predicate", side_effect=AssertionError("loaded")):
                self.assertFalse(
                    rtx4090.automatic_compatible(None, None, None, options[0], options[1], False, options[2], False)
                )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class AutomaticGpuTests(unittest.TestCase):
    def setUp(self):
        if not rtx4090._short_device(torch.cuda.current_device()) or not fa4_sm80.available():
            self.skipTest("Qualified full RTX 4090 runtime required")

    def tearDown(self):
        result = self._outcome.result
        if hasattr(self, "fixture") and any(test is self for test, _ in result.failures + result.errors):
            folder = Path(os.environ.get("KOHAKUFA_FAILURE_DIR", ".publication/failing-operands"))
            folder.mkdir(parents=True, exist_ok=True)
            torch.save(self.fixture, folder / (self._testMethodName + ".pt"))

    def test_live_metadata_and_policy_boundaries(self):
        from kohakufa_cute.native import extension

        predicate = extension().rtx4090_automatic_metadata_compatible
        self.assertFalse(predicate(None, None, None, False))
        cpu = torch.empty(1, 16, 128, 16, requires_grad=True)
        self.assertFalse(predicate(cpu, cpu, cpu, False))
        for dtype in (torch.bfloat16, torch.float16):
            for sequence in (128, 129, 512, 1024, 1025, 4096):
                for dimension in (16, 64, 128, 256, 512):
                    q = torch.empty(1, 16, sequence, dimension, device="cuda", dtype=dtype, requires_grad=True)
                    expected = (sequence, dimension) == (128, 16) or (sequence == 4096 and dimension in (64, 128))
                    for causal in (False, True):
                        self.assertEqual(predicate(q, q, q, causal), expected)
                        self.assertEqual(fa4_sm80.compatible(q, q, q, None, 0, causal, None, True), expected)
            q = torch.empty(1, 16, 128, 16, device="cuda", dtype=dtype, requires_grad=True)
            for flags in range(8):
                values = [q.detach().clone().requires_grad_(bool(flags & (1 << i))) for i in range(3)]
                self.assertEqual(predicate(*values, False), bool(flags))
            with torch.no_grad():
                self.assertFalse(predicate(q, q, q, False))
                self.assertFalse(fa4_sm80.compatible(q, q, q, None, 0, False, None, False))
            for bad in (q.float(), q[:, :8], q.transpose(1, 2), q.detach().cpu()):
                self.assertFalse(predicate(bad, q, q, False))
            padded = torch.empty(q.numel() + 16, device="cuda", dtype=dtype)
            mutable = padded[: q.numel()].view_as(q).requires_grad_(True)
            for offset in (0, 1, 0):
                with torch.no_grad():
                    mutable.set_(padded.untyped_storage(), offset, q.shape, q.stride())
                self.assertEqual(predicate(mutable, q, q, False), offset == 0)
            negative = torch._neg_view(q)
            self.assertFalse(predicate(negative, q, q, False))
            for guard, target in ((torch.compiler, "is_compiling"), (torch, "are_deterministic_algorithms_enabled")):
                with patch.object(guard, target, return_value=True):
                    self.assertFalse(rtx4090.automatic_compatible(q, q, q, None, 0, False, None, False))
            with patch.object(fa4_sm80, "available", return_value=False):
                self.assertFalse(rtx4090.automatic_compatible(q, q, q, None, 0, False, None, False))

    def test_public_training_parameters_subclasses(self):
        from benchmarks.experiments.ampere.validate_sm80_static import assess
        from kohakufa_cute import api

        class PlainTensor(torch.Tensor):
            pass

        for dtype in (torch.bfloat16, torch.float16):
            for sequence, dimension in ((128, 16), (1024, 64), (1024, 128), (4096, 64), (4096, 128)):
                for causal in (False, True):
                    torch.manual_seed(941019)
                    inputs = [
                        torch.nn.Parameter(torch.randn(1, 16, sequence, dimension, device="cuda", dtype=dtype))
                        for _ in range(3)
                    ]
                    dout = torch.randn_like(inputs[0])
                    self.fixture = {"inputs": inputs, "dout": dout, "causal": causal, "seed": 941019}
                    for operation in (api.scaled_dot_product_attention, api.automatic_scaled_dot_product_attention):
                        output = operation(*inputs, is_causal=causal, enable_gqa=True)
                        assess(
                            (output, *torch.autograd.grad(output, inputs, dout)),
                            inputs,
                            dout,
                            causal=causal,
                            scale=None,
                            directed=False,
                            pointwise=False,
                        )
                    output = api.attention(*inputs, causal=causal)
                    assess(
                        (output, *torch.autograd.grad(output, inputs, dout)),
                        inputs,
                        dout,
                        causal=causal,
                        scale=None,
                        directed=False,
                        pointwise=False,
                    )
                    for values in ((inputs[0].as_subclass(PlainTensor), *inputs[1:]), tuple(inputs)):
                        self.assertTrue(fa4_sm80.compatible(*values, None, 0, causal, None, False))

    def test_fallback_namespace_matches_parent(self):
        from benchmarks.experiments.ada.rtx4090_automatic_controls import parent_api
        from kohakufa_cute import api

        before = parent_api()
        for sequence, dimension, scale in (
            (129, 64, None),
            (128, 128, None),
            (128, 16, -0.125),
            (128, 16, 0.0),
            (128, 256, None),
            (128, 512, None),
        ):
            q, k, v = [
                torch.randn(1, 16, sequence, dimension, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                for _ in range(3)
            ]
            for causal in (False, True):
                torch.testing.assert_close(
                    api.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale),
                    before.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale),
                    atol=0,
                    rtol=0,
                )
                with torch.no_grad():
                    torch.testing.assert_close(
                        api.scaled_dot_product_attention(q, k, v, is_causal=causal),
                        before.scaled_dot_product_attention(q, k, v, is_causal=causal),
                        atol=0,
                        rtol=0,
                    )

    def test_actual_grad_subsets_follow_live_flags(self):
        from benchmarks.reference import fp64_reference, metrics
        from kohakufa_cute import api

        for dtype in (torch.bfloat16, torch.float16):
            for sequence, dimension in ((128, 16), (4096, 64)):
                for causal in (False, True):
                    inputs = [
                        torch.randn(1, 16, sequence, dimension, device="cuda", dtype=dtype, requires_grad=True)
                        for _ in range(3)
                    ]
                    dout = torch.randn_like(inputs[0])
                    output = api.scaled_dot_product_attention(*inputs, is_causal=causal)
                    full = torch.autograd.grad(output, inputs, dout)
                    mathematical = [
                        fp64_reference(tuple(t[:, head : head + 1] for t in inputs), dout[:, head : head + 1], causal=causal)
                        for head in range(16)
                    ]
                    for flags in range(1, 8):
                        values = [t.detach().requires_grad_(bool(flags & (1 << i))) for i, t in enumerate(inputs)]
                        self.fixture = {"inputs": values, "dout": dout, "causal": causal, "flags": flags}
                        actual = api.scaled_dot_product_attention(*values, is_causal=causal)
                        selected = [t for t in values if t.requires_grad]
                        gradients = torch.autograd.grad(actual, selected, dout)
                        indices = [i for i in range(3) if flags & (1 << i)]
                        torch.testing.assert_close(actual, output, atol=0, rtol=0)
                        # Existing key-owned dQ uses FP32 atomic accumulation.
                        # Check each requested gradient against the independent
                        # mathematical reference, not a new bitwise-order gate.
                        for got, index in zip(gradients, indices):
                            for head in range(16):
                                error = metrics(got[:, head : head + 1], mathematical[head][index + 1])
                                self.assertTrue(error["finite"])
                                self.assertLess(error["relative_error"], 0.01 if dtype == torch.bfloat16 else 0.005)

    def test_masks_rectangular_gqa_and_compiler_fallbacks(self):
        from benchmarks.experiments.ada.rtx4090_automatic_controls import parent_api
        from kohakufa_cute import api

        before = parent_api()
        for queries, keys, heads, kvheads, masked in (
            (129, 129, 16, 16, True),
            (128, 129, 16, 16, False),
            (129, 129, 16, 8, False),
        ):
            inputs = [
                torch.randn(
                    1,
                    heads if i == 0 else kvheads,
                    queries if i == 0 else keys,
                    64,
                    device="cuda",
                    dtype=torch.bfloat16,
                    requires_grad=True,
                )
                for i in range(3)
            ]
            mask = (torch.arange(keys, device="cuda") % 3 != 0).reshape(1, 1, 1, keys) if masked else None
            dout = torch.randn_like(inputs[0])
            self.fixture = {"inputs": inputs, "dout": dout, "mask": mask}
            outputs = [
                function(*inputs, attn_mask=mask, enable_gqa=True)
                for function in (api.scaled_dot_product_attention, before.scaled_dot_product_attention)
            ]
            torch.testing.assert_close(*outputs, atol=0, rtol=0)
            gradients = [torch.autograd.grad(output, inputs, dout) for output in outputs]
            for left, right in zip(*gradients):
                torch.testing.assert_close(left, right, atol=0, rtol=0)
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        q = torch.randn(1, 16, 128, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        operation = torch.compile(api.scaled_dot_product_attention, backend="eager", fullgraph=True)
        with patch.object(rtx4090, "_automatic_metadata_predicate", side_effect=AssertionError("compiled metadata route")):
            actual = operation(q, q, q)
        expected = before.scaled_dot_product_attention(q, q, q)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
