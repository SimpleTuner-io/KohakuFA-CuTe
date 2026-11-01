import unittest
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from kohakufa_cute import api, fa4_sm80


def cuda_operand(shape=(1, 16, 4096, 128), dtype=torch.bfloat16, contiguous=True, pointer=4096):
    return SimpleNamespace(
        shape=shape,
        ndim=4,
        dtype=dtype,
        device=torch.device("cuda", 0),
        is_contiguous=lambda: contiguous,
        data_ptr=lambda: pointer,
    )


class Sm80SelectionTests(unittest.TestCase):
    def test_architecture_shape_scale_and_runtime_boundary(self):
        q = cuda_operand()
        with (
            patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)),
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(multi_processor_count=108)),
            patch.object(fa4_sm80, "available", return_value=True),
        ):
            for causal in (False, True):
                for dtype in (torch.bfloat16, torch.float16):
                    q = cuda_operand(dtype=dtype)
                    self.assertTrue(fa4_sm80.compatible(q, q, q, None, 0, causal, None, False))
            medium = cuda_operand((1, 8, 2048, 128))
            self.assertTrue(fa4_sm80.compatible(medium, medium, medium, None, 0, False, None, False))
            self.assertFalse(fa4_sm80.compatible(medium, medium, medium, None, 0, True, None, False))
            for scale in (-0.1, 0, 1e-50, torch.finfo(torch.float32).tiny / 2, 3e38, float("nan"), float("inf")):
                self.assertFalse(fa4_sm80.compatible(q, q, q, None, 0, False, scale, False))
            self.assertTrue(fa4_sm80.compatible(q, q, q, None, 0, False, torch.finfo(torch.float32).tiny, False))
            for shape in (
                (1, 16, 2047, 128),
                (1, 16, 4097, 128),
                (2, 16, 4096, 128),
                (1, 17, 4096, 128),
                (1, 7, 4096, 128),
                (1, 16, 4096, 256),
            ):
                operand = cuda_operand(shape)
                self.assertFalse(fa4_sm80.compatible(operand, operand, operand, None, 0, False, None, False))
            self.assertFalse(fa4_sm80.compatible(q, cuda_operand((1, 2, 257, 128)), q, None, 0, False, None, True))
            self.assertFalse(fa4_sm80.compatible(q, q, q, object(), 0, False, None, False))
            self.assertFalse(fa4_sm80.compatible(q, q, q, None, 0.1, False, None, False))
            operand = cuda_operand(contiguous=False)
            self.assertFalse(fa4_sm80.compatible(operand, operand, operand, None, 0, False, None, False))
            for position in range(3):
                operands = [q, q, q]
                operands[position] = cuda_operand(pointer=4098)
                self.assertFalse(fa4_sm80.compatible(*operands, None, 0, False, None, False))
            with patch.object(torch.compiler, "is_compiling", return_value=True):
                self.assertFalse(fa4_sm80.compatible(q, q, q, None, 0, False, None, False))
            with patch.object(torch, "are_deterministic_algorithms_enabled", return_value=True):
                self.assertFalse(fa4_sm80.compatible(q, q, q, None, 0, False, None, False))
            with patch.object(torch.cuda, "get_device_capability", return_value=(8, 6)):
                self.assertFalse(fa4_sm80.compatible(q, q, q, None, 0, False, None, False))
            with patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(multi_processor_count=14)):
                self.assertFalse(fa4_sm80.compatible(q, q, q, None, 0, False, None, False))
            with patch.object(torch.cuda, "current_device", return_value=1):
                self.assertFalse(fa4_sm80.compatible(q, q, q, None, 0, False, None, False))

    def test_a100_policy_keeps_explicit_schedule_defaults_stable(self):
        from kohakufa_cute.fa4_sm80._attention import Schedule
        from kohakufa_cute.fa4_sm80._schedules import a100_schedule

        baseline = Schedule()
        for sequence in (128, 1024, 2047):
            self.assertEqual(a100_schedule(64, sequence, False), baseline)
        for sequence in (2048, 4096):
            for causal in (False, True):
                selected = a100_schedule(64, sequence, causal)
                self.assertEqual((selected.backward_m, selected.query_stages, selected.dout_stages), (128, 1, 1))
                self.assertEqual((selected.score_m_warps, selected.kv_m_warps, selected.dq_m_warps), (4, 4, 4))
                self.assertFalse(selected.v_in_regs)
        self.assertEqual(a100_schedule(128, 4096, False), baseline)
        self.assertEqual(a100_schedule(128, 4096, True).forward_m, 64)
        self.assertEqual((Schedule().backward_m, Schedule().dout_stages, Schedule().v_in_regs), (64, 2, False))

    def test_missing_optional_dependency_retains_native_schedule(self):
        fa4_sm80.available.cache_clear()
        try:
            with patch.object(fa4_sm80, "version", side_effect=PackageNotFoundError):
                self.assertFalse(fa4_sm80.available())
        finally:
            fa4_sm80.available.cache_clear()

    def test_selected_errors_propagate_even_with_warm_native_cache(self):
        q = cuda_operand()
        extension = Mock()
        extension.cache_info.return_value = SimpleNamespace(currsize=1)
        with (
            patch.object(api, "_extension", extension),
            patch.object(fa4_sm80, "compatible", return_value=True),
            patch.object(fa4_sm80, "attention", side_effect=RuntimeError("selected schedule failed")),
        ):
            for function in (api.attention, api.scaled_dot_product_attention, api.automatic_scaled_dot_product_attention):
                with self.subTest(function=function.__name__), self.assertRaisesRegex(RuntimeError, "selected schedule"):
                    function(q, q, q)
        extension.assert_not_called()

    def test_repeated_calls_preserve_package_entry_function(self):
        from kohakufa_cute.fa4_sm80 import _host

        entry = fa4_sm80.attention
        q = cuda_operand()
        runner = Mock(return_value=object())
        with patch.object(_host, "native_runner", return_value=runner) as selected:
            for _ in range(2):
                fa4_sm80.attention(q, q, q)
                self.assertIs(fa4_sm80.attention, entry)
            self.assertEqual(selected.call_count, 2)
            self.assertEqual(runner.call_count, 2)

    def test_native_runner_cache_retains_every_selected_schedule_field(self):
        from dataclasses import asdict, replace

        from kohakufa_cute.fa4_sm80._host import native_runner
        from kohakufa_cute.fa4_sm80._schedules import a100_schedule

        schedule = a100_schedule(64, 4096, False)
        different = replace(schedule, query_stages=2)
        module = Mock()
        native = SimpleNamespace(extension=Mock(return_value=module))
        native_runner.cache_clear()
        try:
            with patch.dict("sys.modules", {"kohakufa_cute.native": native}):
                first = native_runner(schedule)
                self.assertIs(first, native_runner(schedule))
                native_runner(different)
            self.assertEqual(module.AttentionRunner.call_count, 2)
            self.assertEqual(module.AttentionRunner.call_args_list[0].args, ("warp", asdict(schedule)))
            self.assertEqual(module.AttentionRunner.call_args_list[1].args, ("warp", asdict(different)))
        finally:
            native_runner.cache_clear()


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 0) and fa4_sm80.available(),
    "Requires the qualified optional A100 schedule",
)
class Sm80KernelTests(unittest.TestCase):
    def public_output(self, inputs, *, causal=False, scale=None, function=None):
        from benchmarks.validate_sm80_static import statistics

        function = api.scaled_dot_product_attention if function is None else function
        selected = fa4_sm80.compatible(*inputs, None, 0, causal, scale, False)
        saved = []
        with (
            patch.object(fa4_sm80, "attention", wraps=fa4_sm80.attention) as call,
            torch.autograd.graph.saved_tensors_hooks(lambda tensor: saved.append(tensor) or tensor, lambda tensor: tensor),
        ):
            if function is api.attention:
                output = function(*inputs, causal=causal, scale=scale)
            else:
                output = function(*inputs, is_causal=causal, scale=scale)
        self.assertEqual(call.call_count, int(selected))
        if selected:
            self.assertEqual(len(saved), 6)
            statistics(inputs, saved[-2], saved[-1], causal, inputs[0].shape[-1] ** -0.5 if scale is None else scale)
        return output

    def test_tiny_positive_scales_retain_native_zero_aware_fallback(self):
        from benchmarks.reference import metrics

        for dtype in (torch.bfloat16, torch.float16):
            torch.manual_seed(887)
            inputs = [torch.randn(1, 8, 2048, 64, device="cuda", dtype=dtype) for _ in range(3)]
            dout = torch.randn_like(inputs[0])
            for scale in (1e-50, torch.finfo(torch.float32).tiny / 2):
                for causal in (False, True):
                    with self.subTest(dtype=dtype, scale=scale, causal=causal):
                        operands = [tensor.detach().requires_grad_(True) for tensor in inputs]
                        self.assertFalse(fa4_sm80.compatible(*operands, None, 0, causal, scale, False))
                        out = self.public_output(operands, causal=causal, scale=scale)
                        actual = out, *torch.autograd.grad(out, operands, dout)
                        operands = [tensor.detach().requires_grad_(True) for tensor in inputs]
                        with patch.object(fa4_sm80, "compatible", return_value=False):
                            control = api.scaled_dot_product_attention(*operands, is_causal=causal, scale=scale)
                            expected = control, *torch.autograd.grad(control, operands, dout)
                        for got, want in zip(actual, expected):
                            error = metrics(got, want.double())
                            self.assertTrue(error["finite"])
                            if error["reference_norm"]:
                                self.assertLess(error["relative_error"], 0.01 if dtype == torch.bfloat16 else 0.005)
                            else:
                                self.assertEqual(error["max_absolute_error"], 0)

    def test_unaligned_inputs_use_native_and_output_gradient_is_materialized(self):
        from benchmarks.reference import fp64_reference, metrics

        for dtype in (torch.bfloat16, torch.float16):
            torch.manual_seed(839)
            shape = (1, 8, 2048, 64)
            inputs = [torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)]
            unaligned = torch.empty(inputs[0].numel() + 1, device="cuda", dtype=dtype)[1:].view(shape)
            unaligned.copy_(inputs[0].detach())
            unaligned.requires_grad_(True)
            self.assertTrue(unaligned.is_contiguous())
            self.assertNotEqual(unaligned.data_ptr() % 16, 0)
            for position in range(3):
                operands = list(inputs)
                operands[position] = unaligned
                self.assertFalse(fa4_sm80.compatible(*operands, None, 0, False, None, False))
                out = self.public_output(operands)
                torch.autograd.grad(out, operands, torch.ones_like(out))
            out = self.public_output(inputs)
            dout = torch.empty(out.numel() + 1, device="cuda", dtype=dtype)[1:].view(shape)
            dout.normal_()
            self.assertTrue(dout.is_contiguous())
            self.assertNotEqual(dout.data_ptr() % 16, 0)
            actual = out, *torch.autograd.grad(out, inputs, dout)
            for head in range(shape[1]):
                reference = fp64_reference(
                    [tensor[:, head : head + 1].detach() for tensor in inputs], dout[:, head : head + 1], causal=False
                )
                for got, expected in zip(actual, reference):
                    error = metrics(got[:, head : head + 1], expected)
                    self.assertTrue(error["finite"])
                    self.assertLess(error["relative_error"], 0.01 if dtype == torch.bfloat16 else 0.005)

    def test_public_selected_boundaries_and_native_exception(self):
        from benchmarks.reference import fp64_reference, metrics

        for dtype in (torch.bfloat16, torch.float16):
            for sequence in (2048, 2049, 4095):
                for heads, dimension in ((8, 64), (8, 128), (16, 64), (16, 128)):
                    for causal in (False, True):
                        with self.subTest(dtype=dtype, sequence=sequence, heads=heads, dimension=dimension, causal=causal):
                            torch.manual_seed(827)
                            inputs = [
                                torch.randn(1, heads, sequence, dimension, device="cuda", dtype=dtype, requires_grad=True)
                                for _ in range(3)
                            ]
                            dout = torch.randn_like(inputs[0])
                            out = self.public_output(inputs, causal=causal)
                            actual = out, *torch.autograd.grad(out, inputs, dout)
                            for head in range(heads):
                                part = [tensor[:, head : head + 1].detach() for tensor in inputs]
                                reference = fp64_reference(part, dout[:, head : head + 1], causal=causal)
                                for got, expected in zip(actual, reference):
                                    error = metrics(got[:, head : head + 1], expected)
                                    self.assertTrue(error["finite"])
                                    self.assertLess(error["relative_error"], 0.01 if dtype == torch.bfloat16 else 0.005)

    def test_public_repeated_calls_graph_symbols_and_compiled_native_fallback(self):
        from benchmarks.compare_ada import capture
        from benchmarks.compare_l40s import graph_kernels

        torch.manual_seed(817)
        inputs = [torch.randn(1, 16, 4096, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        dout = torch.randn_like(inputs[0])
        for function in (api.attention, api.scaled_dot_product_attention, api.automatic_scaled_dot_product_attention):
            for _ in range(2):
                out = self.public_output([tensor.detach().requires_grad_(True) for tensor in inputs], function=function)
                self.assertTrue(callable(fa4_sm80.attention))
        graph, owners = capture(api.scaled_dot_product_attention, inputs, dout, False, "joint", keep_graph=True)
        symbols = [record["symbol"] for record in graph_kernels(graph)]
        self.assertEqual(len(symbols), 4)
        self.assertTrue(any("fa4_sm80" in symbol and "Forward" in symbol for symbol in symbols))
        self.assertTrue(any("fa4_sm80" in symbol and "BackwardSm80" in symbol for symbol in symbols))
        compiled = torch.compile(api.scaled_dot_product_attention, backend="eager", fullgraph=True)
        graph, owners = capture(compiled, inputs, dout, False, "joint", keep_graph=True)
        symbols = [record["symbol"] for record in graph_kernels(graph)]
        self.assertTrue(all("fa4_sm80" not in symbol for symbol in symbols))
        self.assertTrue(any("kohakufa_cute" in symbol for symbol in symbols))

    def test_cached_public_scales_statistics_and_changed_graph(self):
        from benchmarks.compare_ada import capture
        from benchmarks.validate_sm80_static import assess

        for dtype, heads, dimension, causal in (
            (torch.bfloat16, 8, 64, False),
            (torch.float16, 16, 128, True),
        ):
            with self.subTest(dtype=dtype, dimension=dimension, causal=causal):
                torch.manual_seed(947)
                inputs = [torch.randn(1, heads, 2048, dimension, device="cuda", dtype=dtype) for _ in range(3)]
                dout = torch.randn_like(inputs[0])
                for scale in (0.17, 0.41, 0.17):
                    operands = [tensor.detach().requires_grad_(True) for tensor in inputs]
                    out = self.public_output(operands, causal=causal, scale=scale)
                    actual = out, *torch.autograd.grad(out, operands, dout)
                    assess(actual, operands, dout, causal=causal, scale=scale, directed=False, pointwise=False)

                def function(q, k, v, *, is_causal=False):
                    return api.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=0.17)

                graph, storage = capture(function, [tensor.detach() for tensor in inputs], dout, causal, "joint")
                operands, _, _, actual = storage
                for _ in range(2):
                    with torch.no_grad():
                        for tensor in (*operands, dout):
                            tensor.copy_(torch.randn_like(tensor))
                        for tensor in actual:
                            tensor.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize()
                    assess(actual, operands, dout, causal=causal, scale=0.17, directed=False, pointwise=False)

    def test_direct_tails_heads_and_expanded_output_gradient(self):
        from benchmarks.reference import fp64_reference, metrics
        from kohakufa_cute.fa4_sm80._attention import attention

        for dtype in (torch.bfloat16, torch.float16):
            for heads, dimension, causal in ((1, 64, False), (3, 128, True)):
                with self.subTest(dtype=dtype, heads=heads, dimension=dimension, causal=causal):
                    torch.manual_seed(809)
                    inputs = [
                        torch.randn(1, heads, 257, dimension, device="cuda", dtype=dtype, requires_grad=True)
                        for _ in range(3)
                    ]
                    out = attention(*inputs, causal=causal)
                    self.assertTrue(hasattr(out.grad_fn, "schedule"))
                    actual = out, *torch.autograd.grad(out.sum(), inputs)
                    for head in range(heads):
                        part = [tensor[:, head : head + 1].detach() for tensor in inputs]
                        reference = fp64_reference(part, torch.ones_like(part[0]), causal=causal)
                        for got, expected in zip(actual, reference):
                            error = metrics(got[:, head : head + 1], expected)
                            self.assertTrue(error["finite"])
                            self.assertLess(error["relative_error"], 0.01 if dtype == torch.bfloat16 else 0.005)


if __name__ == "__main__":
    unittest.main()
