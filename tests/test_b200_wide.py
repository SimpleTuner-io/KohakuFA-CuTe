"""B200 wide-owner selection, arithmetic, and dynamic-cache boundaries."""

import unittest

import torch


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.cuda.get_device_capability() == (10, 0)
    and "B200" in torch.cuda.get_device_name()
    and "MIG" not in torch.cuda.get_device_name(),
    "Requires a full B200",
)
class B200WideTests(unittest.TestCase):
    def test_directed_rounding_uses_actual_output(self):
        from benchmarks.reference import metrics, rounded_reference
        from kohakufa_cute import attention, native
        from tests.test_key_precision import key_cancellation
        from tests.test_value_precision import square_value_cancellation

        native.extension().clear_cache()
        self.addCleanup(native.extension().clear_cache)
        key_inputs, key_dout = key_cancellation(512, torch.bfloat16, "cuda", heads=8, square=True)
        key_inputs = tuple(t[:, :, :1024].contiguous() for t in key_inputs)
        key_dout = key_dout[:, :, :1024].contiguous()
        value_inputs, value_dout, scale, _ = square_value_cancellation(512, torch.bfloat16, "cuda", heads=8, length=1024)
        for inputs, dout, scale in ((key_inputs, key_dout, None), (value_inputs, value_dout, scale)):
            with self.subTest(scale=scale):
                operands = [t.detach().requires_grad_() for t in inputs]
                out = attention(*operands, scale=scale)
                actual = out, *torch.autograd.grad(out, operands, dout)
                reference = rounded_reference(inputs, dout, scale=scale, output=out)
                for got, want in zip(actual, reference):
                    for head in range(8):
                        error = metrics(got[:, head : head + 1], want[:, head : head + 1])
                        self.assertTrue(error["finite"])
                        if error["reference_norm"]:
                            self.assertLess(error["relative_error"], 0.01)
                        else:
                            self.assertEqual(error["max_absolute_error"], 0)

    def test_runtime_length_boundaries_and_changed_graphs(self):
        from unittest.mock import patch

        from benchmarks.compare_ada import capture
        from benchmarks.compare_l40s import graph_kernels
        from benchmarks.reference import fp64_reference, metrics, rounded_reference
        from kohakufa_cute import kernels, native, scaled_dot_product_attention
        from kohakufa_cute.blackwell_register_aux import AdaptiveBackward

        native.extension().clear_cache()
        kernels._compile_kernel.cache_clear()
        self.addCleanup(native.extension().clear_cache)
        self.addCleanup(kernels._compile_kernel.cache_clear)
        compiler = kernels.cute.compile

        def checked_compile(program, *args, **kwargs):
            if isinstance(program, AdaptiveBackward):
                self.assertEqual(program.maximum_length, 1024)
                self.assertEqual(program.minimum_length, 1024 if args[0].shape[1] == 8 else 512)
                for owner in (program.wide.q, program.wide.kv):
                    self.assertTrue(owner.baseline_precision)
                    self.assertFalse(owner.hardware_reductions)
                    self.assertTrue(owner.role_statistics)
            return compiler(program, *args, **kwargs)

        torch.manual_seed(9391)
        with patch.object(kernels.cute, "compile", checked_compile):
            for length, keys, causal, scale, heads in (
                (511, 511, False, None, 16),
                (512, 512, False, None, 16),
                (513, 513, True, None, 16),
                (1024, 1024, False, None, 16),
                (1025, 1025, False, None, 16),
                (512, 513, False, None, 16),
                (512, 512, False, -0.125, 16),
                (513, 513, True, 0.0, 16),
                (1023, 1023, False, None, 8),
                (1024, 1024, False, None, 8),
                (1025, 1025, False, None, 8),
            ):
                with self.subTest(length=length, keys=keys, causal=causal, scale=scale):
                    inputs = [
                        torch.randn(1, heads, n, 512, device="cuda", dtype=torch.bfloat16) for n in (length, keys, keys)
                    ]
                    dout = torch.randn_like(inputs[0])

                    def attention(q, k, v, *, is_causal=False):
                        return scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale)

                    operands = [t.detach().requires_grad_() for t in inputs]
                    out = attention(*operands, is_causal=causal)
                    actual = out, *torch.autograd.grad(out, operands, dout)
                    for reference in (
                        fp64_reference(inputs, dout, causal=causal, scale=scale),
                        rounded_reference(inputs, dout, causal=causal, scale=scale, output=out),
                    ):
                        for got, want in zip(actual, reference):
                            for head in range(heads):
                                error = metrics(got[:, head : head + 1], want[:, head : head + 1])
                                self.assertTrue(error["finite"])
                                if error["reference_norm"]:
                                    self.assertLess(error["relative_error"], 0.01)
                                else:
                                    self.assertEqual(error["max_absolute_error"], 0)
                    graph, retained = capture(attention, inputs, dout, causal, "joint", keep_graph=True)
                    selected = [k for k in graph_kernels(graph) if "RegisterAux" in k["symbol"]]
                    minimum = 1024 if heads == 8 else 512
                    self.assertEqual(len(selected), 2 if minimum <= length <= 1024 and length == keys else 0)


if __name__ == "__main__":
    unittest.main()
