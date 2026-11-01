"""Qualified dispatch must not depend on warming an unrelated loader cache."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from kohakufa_cute import api


class SelectedValidationTests(unittest.TestCase):
    def test_qualified_call_avoids_generic_validation_with_cold_loader(self):
        inputs = object(), object(), object()
        expected = object()
        loader = Mock()
        loader.cache_info.return_value = SimpleNamespace(currsize=0)
        with (
            patch.object(api.torch.compiler, "is_compiling", return_value=False),
            patch.object(api.fa4_sm80, "compatible", return_value=True),
            patch.object(api.fa4_sm80, "attention", return_value=expected) as selected,
            patch.object(api, "_extension", loader),
            patch.object(api, "_validate", side_effect=AssertionError("Duplicate validation")),
        ):
            self.assertIs(api.scaled_dot_product_attention(*inputs), expected)
        selected.assert_called_once_with(*inputs, causal=False, scale=None)
        loader.assert_not_called()
        loader.cache_info.assert_not_called()

    def test_selected_failure_propagates_without_loading_fallback(self):
        with (
            patch.object(api.torch.compiler, "is_compiling", return_value=False),
            patch.object(api.fa4_sm80, "compatible", return_value=True),
            patch.object(api.fa4_sm80, "attention", side_effect=RuntimeError("Selected backend failed")),
            patch.object(api, "_extension", side_effect=AssertionError("Unexpected fallback")),
            patch.object(api, "_validate", side_effect=AssertionError("Duplicate validation")),
        ):
            with self.assertRaisesRegex(RuntimeError, "Selected backend failed"):
                api.scaled_dot_product_attention(object(), object(), object())

    def test_unselected_cold_call_still_validates_and_uses_native(self):
        inputs = object(), object(), object()
        expected = object()
        extension = SimpleNamespace(sdpa=Mock(return_value=expected))
        loader = Mock(return_value=extension)
        loader.cache_info.return_value = SimpleNamespace(currsize=0)
        with (
            patch.object(api.torch.compiler, "is_compiling", return_value=False),
            patch.object(api.fa4_sm80, "compatible", return_value=False),
            patch.object(api, "_extension", loader),
            patch.object(api, "_validate") as validate,
        ):
            self.assertIs(api.scaled_dot_product_attention(*inputs), expected)
        validate.assert_called_once_with(*inputs, None, 0.0, False, False)
        extension.sdpa.assert_called_once_with(*inputs, None, 0.0, False, None, False)

    def test_compiled_call_keeps_validation_and_custom_op(self):
        inputs = object(), object(), object()
        expected = object()
        with (
            patch.object(api.torch.compiler, "is_compiling", return_value=True),
            patch.object(api.fa4_sm80, "compatible", side_effect=AssertionError("Compiler negotiation")),
            patch.object(api, "_validate") as validate,
            patch.object(api._ops, "attention", return_value=expected) as selected,
        ):
            self.assertIs(api.scaled_dot_product_attention(*inputs), expected)
        validate.assert_called_once_with(*inputs, None, 0.0, False, False)
        selected.assert_called_once_with(*inputs, mask=None, causal=False, scale=None)


if __name__ == "__main__":
    unittest.main()
