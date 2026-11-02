"""Bounded L40S short-owner routing and actual public-call qualification."""

import unittest
from unittest.mock import Mock, patch

import torch

from kohakufa_cute import api, fa4_sm80
from kohakufa_cute.fa4_sm80 import _l40s
from tests.test_l40s_static import training_operand


class L40sShortSelectionTests(unittest.TestCase):
    def setUp(self):
        for control in (
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(_l40s, "is_device", return_value=True),
            patch.object(fa4_sm80, "available", return_value=True),
        ):
            control.start()
            self.addCleanup(control.stop)

    def test_short_metadata_and_training_boundaries(self):
        q = training_operand((1, 16, 128, 16))
        self.assertTrue(_l40s._cold_compatible(q, q, q, None, 0, False, None, True))
        for shape in ((1, 16, 127, 16), (1, 16, 129, 16), (1, 8, 128, 16), (2, 16, 128, 16)):
            bad = training_operand(shape)
            self.assertFalse(_l40s._cold_compatible(bad, bad, bad, None, 0, False, None, False))
        for scale in (0, -0.25, 1e-40, float("inf"), float("nan")):
            self.assertFalse(_l40s._cold_compatible(q, q, q, None, 0, False, scale, False))
        q.requires_grad = False
        self.assertFalse(_l40s._cold_compatible(q, q, q, None, 0, False, None, False))
        k = training_operand((1, 16, 128, 16))
        self.assertTrue(_l40s._cold_compatible(q, k, q, None, 0, False, None, False))
        with torch.no_grad():
            self.assertFalse(_l40s._cold_compatible(q, k, q, None, 0, False, None, False))

    def test_only_short_dimension_selects_short_runner(self):
        q = training_operand((1, 16, 128, 16))
        runner = Mock(return_value="owned")
        with patch.object(_l40s, "_short_runner", return_value=runner):
            self.assertEqual(_l40s.attention(q, q, q, causal=True), "owned")
        runner.assert_called_once_with(q, q, q, causal=True, scale=None)
        with patch.object(_l40s, "_short_runner", side_effect=AssertionError("wrong short route")):
            with patch("kohakufa_cute.fa4_sm80._host.native_runner", return_value=Mock()) as native:
                long = training_operand((1, 16, 4096, 128))
                _l40s.attention(long, long, long)
            native.assert_called_once_with(_l40s.schedule(128, 4096, False))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class L40sShortPublicTests(unittest.TestCase):
    def setUp(self):
        if not _l40s.is_device(torch.device("cuda", torch.cuda.current_device())) or not fa4_sm80.available():
            self.skipTest("Qualified full L40S runtime required")
        torch.set_num_threads(4)

    def test_four_entries_and_each_live_gradient_flag(self):
        from benchmarks.validate_sm80_static import assess

        torch.manual_seed(296653)
        for dtype in (torch.bfloat16, torch.float16):
            for causal in (False, True):
                for selected in range(3):
                    inputs = [torch.randn(1, 16, 128, 16, device="cuda", dtype=dtype) for _ in range(3)]
                    inputs[selected] = torch.nn.Parameter(inputs[selected])
                    dout = torch.randn_like(inputs[0])
                    for function, options in (
                        (api.attention, {"causal": causal}),
                        (api.scaled_dot_product_attention, {"is_causal": causal}),
                        (api.automatic_scaled_dot_product_attention, {"is_causal": causal}),
                        (fa4_sm80.attention, {"causal": causal}),
                    ):
                        with patch.object(_l40s, "_short_runner", wraps=_l40s._short_runner) as factory:
                            out = function(*inputs, **options)
                        factory.assert_called_once()
                        got = torch.autograd.grad(out, inputs[selected], dout)[0]
                        all_inputs = [t.detach().requires_grad_(True) for t in inputs]
                        complete = api.scaled_dot_product_attention(*all_inputs, is_causal=causal)
                        gradients = torch.autograd.grad(complete, all_inputs, dout)
                        torch.testing.assert_close(got, gradients[selected], rtol=0, atol=0)
                        assess(
                            tuple(t.cpu() for t in (complete, *gradients)),
                            tuple(t.cpu() for t in all_inputs),
                            dout.cpu(),
                            causal=causal,
                            scale=None,
                            directed=False,
                            pointwise=False,
                        )

    def test_inference_policy_and_long_routes_match_parent(self):
        from benchmarks.l40s_short_controls import parent_api

        parent = parent_api()
        for shape, gradients in (
            ((1, 16, 128, 16), False),
            ((1, 16, 128, 64), True),
            ((1, 16, 1024, 128), True),
            ((1, 16, 4096, 64), True),
        ):
            inputs = [
                torch.randn(*shape, device="cuda", dtype=torch.bfloat16, requires_grad=gradients) for _ in range(3)
            ]
            for causal in (False, True):
                with patch.object(_l40s, "_short_runner", side_effect=AssertionError("short selected")):
                    out = api.scaled_dot_product_attention(*inputs, is_causal=causal)
                torch.testing.assert_close(
                    out, parent.scaled_dot_product_attention(*inputs, is_causal=causal), rtol=0, atol=0
                )
        inputs = [
            torch.randn(1, 16, 128, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)
        ]
        for options in (
            {"scale": 0.0},
            {"scale": -0.25},
            {"attn_mask": torch.ones(128, 128, device="cuda", dtype=torch.bool)},
        ):
            with patch.object(_l40s, "_short_runner", side_effect=AssertionError("short selected")):
                out = api.scaled_dot_product_attention(*inputs, **options)
            torch.testing.assert_close(out, parent.scaled_dot_product_attention(*inputs, **options), rtol=0, atol=0)
        with torch.no_grad(), patch.object(_l40s, "_short_runner", side_effect=AssertionError("short selected")):
            out = api.scaled_dot_product_attention(*inputs)
            torch.testing.assert_close(out, parent.scaled_dot_product_attention(*inputs), rtol=0, atol=0)
        q = inputs[0]
        native = __import__("kohakufa_cute.native", fromlist=["extension"]).extension()
        self.assertFalse(native.l40s_metadata_compatible(q, q, q, False))
        self.assertTrue(native.l40s_metadata_compatible(q, q, q, False, True))


if __name__ == "__main__":
    unittest.main()
