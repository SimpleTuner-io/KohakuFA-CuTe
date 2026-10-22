"""CPU checks for isolated Clef attention replacement and cleanup."""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as functional

from benchmarks.clef_forward import attention_scope, fp64_forward_chunks, parser, run


class AttentionCall(torch.nn.Module):
    def forward(self, q):
        return functional.scaled_dot_product_attention(q, q, q)


class TwoScopes(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = AttentionCall()
        self.head = AttentionCall()

    def forward(self, q):
        return self.self_attn(q), self.head(q)


class ClefForwardBenchmarkTests(unittest.TestCase):
    def test_other_model_identifiers_require_explicit_revision(self):
        args = parser().parse_args(
            ["--model", "weights", "--model-id", "Cloudflare/clef", "--output", "../unused-clef-capture"]
        )
        with self.assertRaisesRegex(ValueError, "immutable --revision"):
            run(args)

    def test_chunked_oracle_preserves_causal_offsets_gqa_and_hidden_rows(self):
        from benchmarks.reference import fp64_reference

        torch.manual_seed(818)
        q = torch.randn(1, 4, 7, 8, dtype=torch.bfloat16)
        k, v = [torch.randn(1, 2, 11, 8, dtype=q.dtype) for _ in range(2)]
        mask = (torch.arange(11) < 5).expand(1, 1, 7, 11).clone()
        mask[..., 0, :] = False
        for visible, causal, scale in ((None, False, 0.25), (mask, True, -0.25), (mask, False, 0.0)):
            expected = fp64_reference((q, k, v), torch.zeros_like(q), visible, causal, scale)[0]
            for chunk in (1, 3, 8):
                with self.subTest(causal=causal, scale=scale, chunk=chunk):
                    actual = fp64_forward_chunks(q, k, v, visible, causal=causal, scale=scale, chunk=chunk)
                    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)

    def test_only_requested_scope_uses_replacement(self):
        q = torch.randn(1, 2, 3, 8)
        model = TwoScopes()
        native = functional.scaled_dot_product_attention
        expected = native(q, q, q)
        for backend, changed in (("cute-backbone", 0), ("cute-head", 1)):
            with self.subTest(backend=backend):
                with patch("kohakufa_cute.scaled_dot_product_attention", return_value=expected + 1) as replacement:
                    with attention_scope(model, backend):
                        outputs = model(q)
                self.assertEqual(replacement.call_count, 1)
                torch.testing.assert_close(outputs[changed], expected + 1)
                torch.testing.assert_close(outputs[1 - changed], expected)
                self.assertIs(functional.scaled_dot_product_attention, native)

    def test_exception_restores_sdpa_and_removes_hooks(self):
        model = TwoScopes()
        q = torch.randn(1, 2, 3, 8)
        native = functional.scaled_dot_product_attention
        with patch("kohakufa_cute.scaled_dot_product_attention", side_effect=RuntimeError("probe")):
            with self.assertRaisesRegex(RuntimeError, "probe"):
                with attention_scope(model, "cute-head"):
                    model(q)
        self.assertIs(functional.scaled_dot_product_attention, native)
        for module in model.modules():
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)


if __name__ == "__main__":
    unittest.main()
