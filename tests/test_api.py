import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from kohakufa_cute import api


class DispatchTests(unittest.TestCase):
    @unittest.skipUnless(torch.backends.mps.is_available(), "Requires MPS")
    def test_mps_native_dispatch(self):
        q = torch.randn(1, 2, 5, 16, device="mps", requires_grad=True)
        k, v = [torch.randn(1, 2, 7, 16, device="mps", requires_grad=True) for _ in range(2)]
        mask = torch.randn(5, 7, device="mps")
        with patch.object(api, "_extension", side_effect=AssertionError("CUDA build on MPS")):
            actual = api.automatic_scaled_dot_product_attention(q, k, v, attn_mask=mask)
            expected = api._native_sdpa(q, k, v, attn_mask=mask)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            dout = torch.randn_like(actual)
            for got, want in zip(
                torch.autograd.grad(actual, (q, k, v), dout), torch.autograd.grad(expected, (q, k, v), dout)
            ):
                torch.testing.assert_close(got, want, atol=0, rtol=0)

    def test_import_and_cpu_dispatch_do_not_load_cuda_dependencies(self):
        code = """
import sys
import torch
from kohakufa_cute import automatic_scaled_dot_product_attention
q = torch.randn(1, 2, 3, 16, requires_grad=True)
out = automatic_scaled_dot_product_attention(q, q, q)
out.sum().backward()
assert torch.isfinite(q.grad).all()
assert not any(name.startswith(('cutlass', 'tvm_ffi', 'kohakufa_cute.native', 'simpletuner')) for name in sys.modules)
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cpu_native_outputs_and_gradients_match(self):
        q = torch.randn(2, 4, 5, 16, requires_grad=True)
        k, v = [torch.randn(2, 2, 7, 16, requires_grad=True) for _ in range(2)]
        masks = (None, torch.rand(5, 7) > 0.25, torch.randn(5, 7))
        with patch.object(api, "_extension", side_effect=AssertionError("CUDA build on CPU")):
            for mask in masks:
                with self.subTest(mask=mask):
                    kwargs = dict(attn_mask=mask, enable_gqa=True, scale=-0.2)
                    out = api.automatic_scaled_dot_product_attention(q, k, v, **kwargs)
                    reference = api._native_sdpa(q, k, v, **kwargs)
                    torch.testing.assert_close(out, reference, atol=0, rtol=0)
                    dout = torch.randn_like(out)
                    for got, want in zip(
                        torch.autograd.grad(out, (q, k, v), dout), torch.autograd.grad(reference, (q, k, v), dout)
                    ):
                        torch.testing.assert_close(got, want, atol=0, rtol=0)

    def test_cpu_fullgraph_compile_stays_native(self):
        q = torch.randn(1, 2, 3, 16, requires_grad=True)
        compiled = torch.compile(api.automatic_scaled_dot_product_attention, backend="eager", fullgraph=True)
        out = compiled(q, q, q, scale=0.3)
        expected = api._native_sdpa(q, q, q, scale=0.3)
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
        torch.testing.assert_close(torch.autograd.grad(out.sum(), q)[0], torch.autograd.grad(expected.sum(), q)[0])

    def test_strict_cpu_calls_fail_before_importing_cuda(self):
        q = torch.randn(1, 2, 3, 16)
        for fn in (api.attention, api.scaled_dot_product_attention):
            with self.subTest(fn=fn.__name__), self.assertRaisesRegex(ValueError, "CUDA"):
                fn(q, q, q)

    def test_missing_dependency_selects_native(self):
        from importlib.metadata import PackageNotFoundError

        with patch.object(api, "version", side_effect=PackageNotFoundError):
            self.assertFalse(api._runtime_available())

    def test_architecture_and_feature_selection(self):
        def tensor(heads=2, dtype=torch.bfloat16, shape=None):
            return SimpleNamespace(ndim=4, device=torch.device("cuda", 0), dtype=dtype, shape=shape or (1, heads, 5, 64))

        q, k, v = tensor(), tensor(), tensor()
        for capability, expected in [((8, 9), True), ((9, 0), True), ((10, 0), False), ((12, 0), False)]:
            with (
                self.subTest(capability=capability),
                patch.object(torch.cuda, "get_device_capability", return_value=capability),
            ):
                self.assertEqual(api._compatible(q, k, v, None, 0, False, False), expected)
        with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
            self.assertFalse(api._compatible(q, k, v, None, 0.1, False, False))
            self.assertFalse(api._compatible(q, k, v, tensor(dtype=torch.float32), 0, False, False))
            self.assertTrue(api._compatible(tensor(4), k, v, None, 0, False, True))
            self.assertFalse(api._compatible(tensor(4), k, v, None, 0, False, False))

    def test_selected_kernel_errors_are_not_suppressed(self):
        q = SimpleNamespace(device=torch.device("cuda", 0))
        builder = Mock(side_effect=RuntimeError("build failure"))
        builder.cache_info.return_value = SimpleNamespace(currsize=0)
        with (
            patch.object(api, "_extension", builder),
            patch.object(api, "_compatible", return_value=True),
            patch.object(api, "_runtime_available", return_value=True),
            patch.object(api, "scaled_dot_product_attention", side_effect=RuntimeError("kernel failure")),
        ):
            with self.assertRaisesRegex(RuntimeError, "kernel failure"):
                api.automatic_scaled_dot_product_attention(q, q, q)


if __name__ == "__main__":
    unittest.main()
