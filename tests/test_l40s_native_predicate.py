"""Actual-object and live-storage parity for the L40S native metadata check."""

import sys
import unittest
from functools import lru_cache
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from benchmarks.l40s_public_metadata_controls import python_compatible
from kohakufa_cute import fa4_sm80
from kohakufa_cute.fa4_sm80 import _l40s


class L40sPredicateHandleTests(unittest.TestCase):
    def test_function_handle_only_is_cached(self):
        function = Mock()
        module = SimpleNamespace(l40s_metadata_compatible=function)
        loader = Mock(return_value=module)
        _l40s._metadata_predicate.cache_clear()
        self.addCleanup(_l40s._metadata_predicate.cache_clear)
        with patch.dict(sys.modules, {"kohakufa_cute.native": SimpleNamespace(extension=loader)}):
            self.assertIs(_l40s._metadata_predicate(), function)
            self.assertIs(_l40s._metadata_predicate(), function)
        loader.assert_called_once()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class L40sNativePredicateTests(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_name() != "NVIDIA L40S" or not fa4_sm80.available():
            self.skipTest("The qualified full L40S runtime is required")

    def test_actual_live_metadata_matches_preceding_python_policy(self):
        checked = 0

        def check(q, k, v, **options):
            nonlocal checked
            args = (
                q,
                k,
                v,
                options.get("mask"),
                options.get("dropout", 0),
                options.get("causal", False),
                options.get("scale"),
                options.get("gqa", False),
            )
            self.assertEqual(_l40s.compatible(*args), python_compatible(*args))
            checked += 1

        for dtype in (torch.bfloat16, torch.float16):
            for dimension in (16, 64, 128, 256, 512):
                for sequence in (128, 1024, 1025, 1152, 1280, 2048, 4096, 4608):
                    q = torch.empty(1, 16, sequence, dimension, device="cuda", dtype=dtype, requires_grad=True)
                    for causal in (False, True):
                        for gqa in (False, True):
                            check(q, q, q, causal=causal, gqa=gqa)
            q = torch.empty(1, 16, 1024, 128, device="cuda", dtype=dtype, requires_grad=True)
            for options in (
                {"mask": torch.ones(1, device="cuda", dtype=torch.bool)},
                {"dropout": 0.1},
                {"scale": 0.0},
                {"scale": -0.125},
                {"scale": 1e-40},
                {"scale": float("inf")},
                {"scale": float("nan")},
                {"scale": 2e38},
                {"scale": torch.finfo(torch.float32).tiny},
                {"scale": 3e38},
                {"scale": 0.125},
                {"gqa": True},
            ):
                check(q, q, q, **options)
            padded = torch.empty(q.numel() + 16, device="cuda", dtype=dtype)
            misaligned = padded[1 : 1 + q.numel()].view_as(q).requires_grad_(True)
            noncontiguous = torch.empty(1, 16, 1024, 129, device="cuda", dtype=dtype, requires_grad=True)[..., :128]
            for position in range(3):
                for bad in (
                    misaligned,
                    noncontiguous,
                    q.detach().cpu().requires_grad_(True),
                    q.float(),
                    q[:, :8],
                    q[:, :, :512],
                ):
                    values = [q, q, q]
                    values[position] = bad
                    check(*values, causal=True)
            mutable = padded[: q.numel()].view_as(q).requires_grad_(True)
            for offset in (0, 1, 0):
                with torch.no_grad():
                    mutable.set_(padded.untyped_storage(), offset, q.shape, q.stride())
                check(mutable, q, q, causal=True)
            values = [q.detach().clone() for _ in range(3)]
            check(*values, causal=True)
            for position in range(3):
                values[position].requires_grad_(True)
                check(*values, causal=True)
                values[position].requires_grad_(False)
            with torch.no_grad():
                check(q, q, q, causal=True)
            with patch.object(torch, "are_deterministic_algorithms_enabled", return_value=True):
                check(q, q, q, causal=True)
            with patch.object(torch.compiler, "is_compiling", return_value=True):
                check(q, q, q, causal=True)
            with patch.object(fa4_sm80, "available", return_value=False):
                check(q, q, q, causal=True)
            if torch.cuda.device_count() > 1:
                with torch.cuda.device(1):
                    check(q, q, q, causal=True)
        self.assertGreater(checked, 400)

    def test_parameters_subclasses_and_parametrized_modules(self):
        class PlainTensor(torch.Tensor):
            pass

        from torch.nn.utils import parametrize

        for dtype in (torch.bfloat16, torch.float16):
            q = torch.randn(1, 16, 1024, 128, device="cuda", dtype=dtype, requires_grad=True)
            parameter = torch.nn.Parameter(q.detach().clone())
            subclass = q.as_subclass(PlainTensor)
            holder = torch.nn.Module()
            holder.register_parameter("weight", parameter)
            parametrize.register_parametrization(holder, "weight", torch.nn.Identity())
            for wrapped in (parameter, subclass, holder.weight):
                self.assertTrue(_l40s.compatible(wrapped, q, q, None, 0, True, None, True))
                self.assertTrue(python_compatible(wrapped, q, q, None, 0, True, None, True))
            parameter.requires_grad_(False)
            with torch.no_grad():
                self.assertFalse(_l40s.compatible(holder.weight, q, q, None, 0, True, None, False))

    def test_native_rejects_cpu_and_non_tensors_without_cuda_metadata_reads(self):
        from kohakufa_cute.native import extension

        predicate = extension().l40s_metadata_compatible
        q = torch.empty(1, 16, 1024, 128, requires_grad=True)
        self.assertFalse(predicate(q, q, q, True))
        self.assertFalse(predicate(None, None, None, True))
        self.assertFalse(predicate(object(), object(), object(), True))

    def test_unsupported_cold_calls_do_not_initialize_the_native_helper(self):
        @lru_cache(maxsize=1)
        def unavailable():
            raise RuntimeError("Cold native loading must happen only for a selected call")

        q = torch.empty(1, 16, 1024, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        with patch.object(_l40s, "_metadata_predicate", unavailable):
            for bad in (q.float(), q[:, :8], q[:, :, :128], q.detach(), q.detach().cpu().requires_grad_(True)):
                self.assertFalse(_l40s.compatible(bad, bad, bad, None, 0, True, None, False))
            with torch.no_grad():
                self.assertFalse(_l40s.compatible(q, q, q, None, 0, True, None, False))
            self.assertEqual(unavailable.cache_info().misses, 0)
            with self.assertRaisesRegex(RuntimeError, "Cold native loading"):
                _l40s.compatible(q, q, q, None, 0, True, None, True)

    def test_missing_required_helper_propagates(self):
        q = torch.empty(1, 16, 1024, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        with patch.object(_l40s, "_metadata_predicate", side_effect=RuntimeError("required guard failed")):
            from kohakufa_cute import api

            for function, options in (
                (api.attention, {"causal": True}),
                (api.scaled_dot_product_attention, {"is_causal": True}),
                (api.automatic_scaled_dot_product_attention, {"is_causal": True}),
            ):
                with self.assertRaisesRegex(RuntimeError, "required guard failed"):
                    function(q, q, q, **options)

    def test_actual_public_namespaces_cold_and_warm_and_parameter_gradients(self):
        from benchmarks.l40s_public_metadata_controls import parent_api
        from benchmarks.validate_sm80_static import assess
        from kohakufa_cute import api, native
        from kohakufa_cute.fa4_sm80 import _host

        parent = parent_api()
        for dtype in (torch.bfloat16, torch.float16):
            torch.manual_seed(19331)
            inputs = [torch.nn.Parameter(torch.randn(1, 16, 1024, 128, device="cuda", dtype=dtype)) for _ in range(3)]
            dout = torch.randn_like(inputs[0])
            _l40s._metadata_predicate.cache_clear()
            _host.native_runner.cache_clear()
            native.extension().clear_cache()
            for repetition in range(2):
                for function, options in (
                    (api.attention, {"causal": True}),
                    (api.scaled_dot_product_attention, {"is_causal": True, "enable_gqa": True}),
                    (api.automatic_scaled_dot_product_attention, {"is_causal": True, "enable_gqa": True}),
                    (fa4_sm80.attention, {"causal": True}),
                    (parent.scaled_dot_product_attention, {"is_causal": True, "enable_gqa": True}),
                ):
                    with self.subTest(
                        dtype=dtype, repetition=repetition, function=function.__module__ + "." + function.__name__
                    ):
                        output = function(*inputs, **options)
                        actual = output, *torch.autograd.grad(output, inputs, dout)
                        assess(actual, inputs, dout, causal=True, scale=None, directed=False, pointwise=False)

    def test_public_parent_fallbacks_preserve_mask_scale_shape_and_grad_mode(self):
        from benchmarks.l40s_public_metadata_controls import parent_api
        from benchmarks.validate_sm80_static import assess
        from kohakufa_cute import api

        parent = parent_api()
        for causal, scale, mask_kind, sequence, heads in (
            (True, -0.125, None, 1024, 16),
            (False, 0.0, None, 1024, 16),
            (False, None, "mask", 129, 16),
            (True, None, None, 129, 16),
            (False, None, None, 1024, 8),
        ):
            torch.manual_seed(19333)
            inputs = [
                torch.randn(1, heads, sequence, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                for _ in range(3)
            ]
            mask = (
                None
                if mask_kind is None
                else (torch.arange(sequence, device="cuda") % 3 != 0).reshape(1, 1, 1, sequence)
            )
            dout = torch.randn_like(inputs[0])
            for current, before in (
                (api.attention, parent.attention),
                (api.scaled_dot_product_attention, parent.scaled_dot_product_attention),
                (api.automatic_scaled_dot_product_attention, parent.automatic_scaled_dot_product_attention),
            ):
                kwargs = (
                    {"mask": mask, "causal": causal, "scale": scale}
                    if current is api.attention
                    else {"attn_mask": mask, "is_causal": causal, "scale": scale}
                )
                first = current(*inputs, **kwargs)
                second = before(*inputs, **kwargs)
                torch.testing.assert_close(first, second, atol=0.02, rtol=0.02)
                grads = torch.autograd.grad(first, inputs, dout)
                expected = torch.autograd.grad(second, inputs, dout)
                for got, want in zip(grads, expected):
                    torch.testing.assert_close(got, want, atol=0.02, rtol=0.02)
            if mask is None:
                assess(
                    (first, *grads),
                    inputs,
                    dout,
                    causal=causal,
                    scale=scale,
                    directed=scale is not None,
                    pointwise=False,
                )
        q = torch.randn(1, 16, 1024, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for inference in (torch.no_grad(), torch.inference_mode()):
            with inference:
                torch.testing.assert_close(
                    api.scaled_dot_product_attention(q, q, q, is_causal=True),
                    parent.scaled_dot_product_attention(q, q, q, is_causal=True),
                    atol=0.02,
                    rtol=0.02,
                )

    def test_compiled_call_keeps_the_existing_fallback(self):
        from benchmarks.validate_sm80_static import assess
        from kohakufa_cute import api

        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        inputs = [
            torch.randn(1, 16, 129, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)
        ]
        dout = torch.randn_like(inputs[0])
        operation = torch.compile(api.scaled_dot_product_attention, backend="eager", fullgraph=True)
        with patch.object(
            _l40s, "_metadata_predicate", side_effect=AssertionError("Compiled call entered the eager metadata helper")
        ):
            output = operation(*inputs, is_causal=True)
            actual = output, *torch.autograd.grad(output, inputs, dout)
        assess(actual, inputs, dout, causal=True, scale=None, directed=False, pointwise=False)


if __name__ == "__main__":
    unittest.main()
