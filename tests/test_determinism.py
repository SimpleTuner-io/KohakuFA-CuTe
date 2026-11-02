"""Live deterministic-policy checks at atomic backward execution."""

import unittest
import warnings
from contextlib import contextmanager

import torch

from kohakufa_cute import api, fa4_sm80
from kohakufa_cute.determinism import check_atomic_backward


@contextmanager
def deterministic(enabled, *, warn_only=False):
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn)


class DeterminismPolicyTests(unittest.TestCase):
    def test_policy_changes_are_live_and_warning_only_continues(self):
        with deterministic(False), warnings.catch_warnings(record=True) as caught:
            check_atomic_backward()
        self.assertFalse(caught)
        with deterministic(True), self.assertRaisesRegex(RuntimeError, "atomic_backward"):
            check_atomic_backward()
        with deterministic(True, warn_only=True), self.assertWarnsRegex(UserWarning, "atomic_backward"):
            check_atomic_backward()
        with deterministic(False):
            check_atomic_backward()

    def test_python_warp_guard_precedes_optional_runtime_and_tensor_access(self):
        from kohakufa_cute.fa4_sm80._attention import _backward

        with deterministic(True), self.assertRaisesRegex(RuntimeError, "atomic_backward"):
            _backward(*([None] * 11))

    def test_python_hopper_guard_precedes_padding_and_optional_runtime(self):
        from kohakufa_cute.fa4_hopper._attention import _Attention

        with deterministic(True), self.assertRaisesRegex(RuntimeError, "atomic_backward"):
            _Attention.backward(None, None)


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.cuda.get_device_capability() in ((8, 0), (8, 6), (8, 9), (12, 0), (12, 1))
    and fa4_sm80.available(),
    "Requires a qualified CUDA warp runtime",
)
class NativeDeterminismTests(unittest.TestCase):
    def operands(self, dtype, length=257, dimension=64, heads=2):
        torch.manual_seed(7193)
        return [torch.randn(1, heads, length, dimension, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)]

    def runner(self):
        from kohakufa_cute.fa4_sm80._attention import Schedule
        from kohakufa_cute.fa4_sm80._host import native_runner

        return native_runner(Schedule())

    def test_native_strict_policy_before_forward_and_after_warmed_forward(self):
        for dtype in (torch.float16, torch.bfloat16):
            inputs = self.operands(dtype)
            dout = torch.randn_like(inputs[0])
            runner = self.runner()
            with deterministic(False):
                warm = runner(*inputs, causal=True)
                torch.autograd.grad(warm, inputs, dout)
            for before_forward in (False, True):
                with self.subTest(dtype=dtype, before_forward=before_forward):
                    with deterministic(before_forward):
                        output = runner(*inputs, causal=True)
                    with deterministic(True), self.assertRaisesRegex(RuntimeError, "atomic_backward"):
                        torch.autograd.grad(output, inputs, dout)

    def test_native_warning_only_reports_and_preserves_math(self):
        from benchmarks.validate_sm80_static import assess

        for dtype in (torch.float16, torch.bfloat16):
            inputs = self.operands(dtype)
            dout = torch.randn_like(inputs[0])
            with deterministic(True, warn_only=True):
                output = self.runner()(*inputs, causal=True)
                with self.assertWarnsRegex(UserWarning, "atomic_backward"):
                    gradients = torch.autograd.grad(output, inputs, dout)
            assess((output, *gradients), inputs, dout, causal=True, scale=None, directed=False, pointwise=False)

    def test_unconfigured_sm80_backward_uses_the_same_policy(self):
        from kohakufa_cute.native import extension

        if torch.cuda.get_device_capability() != (8, 0):
            self.skipTest("The unconfigured SM80 entry requires SM80")
        inputs = self.operands(torch.bfloat16, length=129)
        with deterministic(False):
            output = extension().sm80_attention(*inputs, 0.125, True)
        with deterministic(True), self.assertRaisesRegex(RuntimeError, "atomic_backward"):
            torch.autograd.grad(output, inputs, torch.randn_like(output))

    def test_public_strict_fallback_repeats_bitwise(self):
        from benchmarks.validate_sm80_static import assess

        for dtype in (torch.float16, torch.bfloat16):
            inputs = self.operands(dtype, length=129)
            dout = torch.randn_like(inputs[0])

            def run():
                output = api.scaled_dot_product_attention(*inputs, is_causal=True)
                return output, *torch.autograd.grad(output, inputs, dout)

            with deterministic(True):
                expected = tuple(t.detach().clone() for t in run())
                for _ in range(3):
                    for actual, reference in zip(run(), expected):
                        self.assertTrue(torch.equal(actual.contiguous().view(torch.uint8), reference.view(torch.uint8)))
            assess(expected, inputs, dout, causal=True, scale=None, directed=False, pointwise=False)

    def test_python_warp_backward_enforces_strict_and_warning_only(self):
        from benchmarks.validate_sm80_static import assess
        from kohakufa_cute.fa4_sm80._attention import attention

        for dtype in (torch.float16, torch.bfloat16):
            inputs = self.operands(dtype, length=129)
            dout = torch.randn_like(inputs[0])
            with deterministic(False):
                output = attention(*inputs, causal=True)
            with deterministic(True), self.assertRaisesRegex(RuntimeError, "atomic_backward"):
                torch.autograd.grad(output, inputs, dout)
            with deterministic(True, warn_only=True):
                output = attention(*inputs, causal=True)
                with self.assertWarnsRegex(UserWarning, "atomic_backward"):
                    gradients = torch.autograd.grad(output, inputs, dout)
            assess((output, *gradients), inputs, dout, causal=True, scale=None, directed=False, pointwise=False)

    def test_native_warning_only_graph_reads_changed_inputs(self):
        from benchmarks.compare_ada import capture
        from benchmarks.validate_sm80_static import assess

        inputs = self.operands(torch.bfloat16, length=129)
        dout = torch.randn_like(inputs[0])
        runner = self.runner()

        def operation(q, k, v, *, is_causal=False):
            return runner(q, k, v, causal=is_causal)

        with deterministic(True, warn_only=True), self.assertWarnsRegex(UserWarning, "atomic_backward"):
            graph, storage = capture(operation, [t.detach() for t in inputs], dout, True, "joint")
        operands, graph_dout, _, actual = storage
        with torch.no_grad():
            for tensor in (*operands, graph_dout):
                tensor.copy_(torch.randn_like(tensor))
            for tensor in actual:
                tensor.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assess(actual, operands, graph_dout, causal=True, scale=None, directed=False, pointwise=False)

    def test_owned_short_backward_is_allowed(self):
        from benchmarks.validate_sm80_static import assess
        from kohakufa_cute.native import extension

        prop = torch.cuda.get_device_properties(0)
        if prop.name != "NVIDIA GeForce RTX 4090" or prop.multi_processor_count != 128:
            self.skipTest("Qualified full RTX 4090 short owner required")
        runner = extension().AttentionRunner("short-optimized", {})
        for dtype in (torch.float16, torch.bfloat16):
            inputs = self.operands(dtype, length=128, dimension=16, heads=16)
            dout = torch.randn_like(inputs[0])
            with deterministic(True):
                output = runner(*inputs, causal=True)
                gradients = torch.autograd.grad(output, inputs, dout)
            assess((output, *gradients), inputs, dout, causal=True, scale=None, directed=False, pointwise=False)


if __name__ == "__main__":
    unittest.main()
