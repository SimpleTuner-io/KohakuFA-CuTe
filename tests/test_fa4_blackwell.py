"""Bound the optional SM103 schedule before loading its CUDA dependency."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from kohakufa_cute import fa4_blackwell


def tensor_metadata(*, shape=(1, 16, 4096, 128), dtype=torch.bfloat16, device="cuda", last_stride=1):
    return SimpleNamespace(
        shape=shape, ndim=len(shape), dtype=dtype, device=torch.device(device), stride=lambda axis: last_stride
    )


class BlackwellDispatchTests(unittest.TestCase):
    def setUp(self):
        qualification = patch.object(fa4_blackwell, "_AUTOMATIC_DISPATCH_QUALIFIED", True)
        qualification.start()
        self.addCleanup(qualification.stop)

    def test_unqualified_schedule_stays_disabled_in_public_dispatch(self):
        from kohakufa_cute import api

        q = tensor_metadata()
        with (
            patch.object(fa4_blackwell, "_AUTOMATIC_DISPATCH_QUALIFIED", False),
            patch.object(fa4_blackwell, "available", return_value=True) as available,
            patch.object(fa4_blackwell, "run_attention") as run,
            patch.object(torch.cuda, "get_device_capability") as geometry,
        ):
            self.assertFalse(self.compatible(q))
            self.assertIsNone(api._blackwell_attention(q, q, q, None, 0, False, None, False))
            available.assert_not_called()
            geometry.assert_not_called()
            run.assert_not_called()

    def compatible(self, q=None, *, mask=None, dropout=0, scale=None):
        q = tensor_metadata() if q is None else q
        return fa4_blackwell.compatible(q, q, q, mask, dropout, False, scale, False)

    def test_qualified_square_calls_select_only_the_optional_runtime(self):
        with (
            patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)),
            patch.object(torch, "are_deterministic_algorithms_enabled", return_value=False),
            patch.object(fa4_blackwell, "available", return_value=True) as available,
        ):
            self.assertTrue(self.compatible())
            self.assertTrue(self.compatible(scale=0.125))
            self.assertEqual(available.call_count, 2)

    def test_unsupported_metadata_does_not_load_the_optional_backend(self):
        cases = [
            tensor_metadata(shape=(1, 16, 1024, 128)),
            tensor_metadata(shape=(1, 16, 16384, 128)),
            tensor_metadata(shape=(1, 16, 4096, 512)),
            tensor_metadata(shape=(2, 16, 4096, 128)),
            tensor_metadata(dtype=torch.float32),
            tensor_metadata(device="cpu"),
            tensor_metadata(last_stride=2),
        ]
        with (
            patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)),
            patch.object(torch, "are_deterministic_algorithms_enabled", return_value=False),
            patch.object(fa4_blackwell, "available") as available,
        ):
            for q in cases:
                with self.subTest(shape=q.shape, dtype=q.dtype, device=q.device):
                    self.assertFalse(self.compatible(q))
            for scale in (0, -1, 3e38, float("inf"), float("nan")):
                self.assertFalse(self.compatible(scale=scale))
            self.assertFalse(self.compatible(mask=object()))
            self.assertFalse(self.compatible(dropout=0.1))
            available.assert_not_called()

    def test_other_architectures_and_deterministic_execution_retain_cute(self):
        with (
            patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)),
            patch.object(torch, "are_deterministic_algorithms_enabled", return_value=False),
            patch.object(fa4_blackwell, "available") as available,
        ):
            self.assertFalse(self.compatible())
            available.assert_not_called()
        with (
            patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)),
            patch.object(torch, "are_deterministic_algorithms_enabled", return_value=True),
            patch.object(fa4_blackwell, "available") as available,
        ):
            self.assertFalse(self.compatible())
            available.assert_not_called()

    def test_runtime_entry_survives_implementation_module_import(self):
        from importlib import import_module

        implementation = import_module("kohakufa_cute.fa4_blackwell.attention")
        with patch.object(implementation, "attention", return_value="selected") as run:
            for _ in range(2):
                self.assertEqual(fa4_blackwell.run_attention(None, None, None), "selected")
            self.assertEqual(run.call_count, 2)
        self.assertTrue(callable(fa4_blackwell.run_attention))

    def test_missing_optional_runtime_retains_cute(self):
        with (
            patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)),
            patch.object(torch, "are_deterministic_algorithms_enabled", return_value=False),
            patch.object(fa4_blackwell, "available", return_value=False),
        ):
            self.assertFalse(self.compatible())


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 3), "Requires a full SM103 CUDA target"
)
class BlackwellPublicKernelTests(unittest.TestCase):
    @patch.object(fa4_blackwell, "_AUTOMATIC_DISPATCH_QUALIFIED", True)
    def test_experimental_partial_tiles_and_changed_input_graphs_preserve_the_declared_contract(self):
        if not fa4_blackwell.available():
            self.skipTest("Requires the qualified optional FA4 runtime")
        from benchmarks.compare_ada import capture
        from benchmarks.reference import fp64_reference, metrics
        from kohakufa_cute import scaled_dot_product_attention

        torch.manual_seed(2239)
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                inputs = tuple(torch.randn(1, 3, 2049, 64, device="cuda", dtype=dtype) for _ in range(3))
                dout = torch.randn_like(inputs[0])
                operands = [t.detach().requires_grad_() for t in inputs]
                with patch.object(fa4_blackwell, "run_attention", wraps=fa4_blackwell.run_attention) as selected:
                    out = scaled_dot_product_attention(*operands, is_causal=True)
                    actual = out, *torch.autograd.grad(out, operands, dout)
                    self.assertGreater(selected.call_count, 0)
                expected = fp64_reference(inputs, dout, causal=True)
                limit = 0.01 if dtype == torch.bfloat16 else 0.005
                for got, want in zip(actual, expected):
                    for head in range(3):
                        error = metrics(got[:, head : head + 1], want[:, head : head + 1])
                        self.assertTrue(error["finite"])
                        if error["reference_norm"]:
                            self.assertLess(error["relative_error"], limit)
                        else:
                            self.assertEqual(error["max_absolute_error"], 0)
                capture(scaled_dot_product_attention, inputs, dout, True, "joint")


if __name__ == "__main__":
    unittest.main()
