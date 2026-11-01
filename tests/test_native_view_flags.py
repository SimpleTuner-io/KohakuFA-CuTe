"""Logical view values must survive the native DLPack boundary and autograd."""

import os
import unittest
from pathlib import Path

import torch

from benchmarks.reference import fp64_reference, metrics, rounded_reference


def logical_inputs(bases, negative, strided=False):
    inputs = [base[..., ::2] if strided else base for base in bases]
    return [torch.ops.aten._neg_view.default(value) if flag else value for value, flag in zip(inputs, negative)]


def base_gradients(logical, negative, bases, strided=False):
    result = []
    for value, flag, base in zip(logical, negative, bases):
        value = -value if flag else value
        if strided:
            full = torch.zeros_like(base, dtype=value.dtype)
            full[..., ::2] = value
            value = full
        result.append(value)
    return result


class ViewFlagFixtureTests(unittest.TestCase):
    def test_reference_maps_negative_strided_views_to_original_base(self):
        bases = [torch.randn(1, 2, 5, 16, requires_grad=True) for _ in range(3)]
        inputs = logical_inputs(bases, (True, False, True), True)
        torch.autograd.backward(sum(value.sum() for value in inputs))
        expected = base_gradients([torch.ones_like(value) for value in inputs], (True, False, True), bases, True)
        for base, value in zip(bases, expected):
            torch.testing.assert_close(base.grad, value)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class NativeViewFlagTests(unittest.TestCase):
    def retain_inputs(self, label, bases, dout, negative, strided=False):
        folder = os.environ.get("KOHAKUFA_FAILURE_DIR")
        if folder:
            destination = Path(folder)
            destination.mkdir(parents=True, exist_ok=True)
            self._case_number = getattr(self, "_case_number", 0) + 1
            torch.save(
                {
                    "bases": [tensor.detach().cpu() for tensor in bases],
                    "dout": dout.detach().cpu(),
                    "negative": negative,
                    "strided": strided,
                },
                destination / f"{self._case_number:03}-{label}.pt",
            )

    def check_case(self, operation, dtype, negative, *, strided=False, sequence=129):
        torch.manual_seed(42013)
        width = 128 if strided else 64
        bases = [torch.randn(1, 2, sequence, width, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)]
        inputs = logical_inputs(bases, negative[:3], strided)
        dout_base = torch.randn_like(inputs[0])
        dout = torch.ops.aten._neg_view.default(dout_base) if negative[3] else dout_base
        self.retain_inputs("view-case", bases, dout, negative, strided)
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda tensor: saved.append(tensor) or tensor, lambda tensor: tensor):
            output = operation(*inputs, is_causal=True)
        actual = output, *torch.autograd.grad(output, bases, dout)
        self.assertGreaterEqual(len(saved), 6)
        for tensor in saved[:3]:
            self.assertFalse(tensor.is_neg())
            self.assertFalse(tensor.is_conj())
        exact = fp64_reference(inputs, dout, causal=True)
        rounded = rounded_reference(inputs, dout, causal=True, output=output)
        exact = (exact[0], *base_gradients(exact[1:], negative[:3], bases, strided))
        rounded = (rounded[0], *base_gradients(rounded[1:], negative[:3], bases, strided))
        for component, value, mathematical, contract in zip(
            ("out", "dq_base", "dk_base", "dv_base"), actual, exact, rounded
        ):
            for head in range(value.shape[1]):
                error = metrics(value[:, head : head + 1], mathematical[:, head : head + 1])
                self.assertTrue(error["finite"], (component, error))
                self.assertLess(error["relative_error"], 0.01 if dtype == torch.bfloat16 else 0.005, (component, error))
                diagnostic = metrics(value[:, head : head + 1], contract[:, head : head + 1])
                self.assertTrue(diagnostic["finite"])

    def test_native_view_values_bases_cache_and_graphs(self):
        from kohakufa_cute import api, fa4_sm80
        from kohakufa_cute.native import extension

        module = extension()
        operations = [("generic", api.scaled_dot_product_attention)]
        if fa4_sm80.available() and torch.cuda.get_device_capability() in ((8, 0), (8, 6), (8, 9), (12, 0), (12, 1)):
            runner = module.AttentionRunner(
                "warp",
                {
                    "forward_m": 64,
                    "backward_n": 64,
                    "query_stages": 1,
                    "dout_stages": 1,
                    "score_m_warps": 4,
                    "v_in_regs": True,
                },
            )
            operations.append(("configured", lambda q, k, v, is_causal=False: runner(q, k, v, causal=is_causal)))
        patterns = (
            (True, False, False, False),
            (False, True, False, False),
            (False, False, True, False),
            (False, False, False, True),
            (True, True, True, True),
        )
        print("native view regression initial program cache:", module.cache_size(), flush=True)
        for name, operation in operations:
            for dtype in (torch.bfloat16, torch.float16):
                for pattern in patterns:
                    for repeat in range(2):
                        with self.subTest(path=name, dtype=dtype, negative=pattern, repeat=repeat):
                            self.check_case(operation, dtype, pattern)
                if name == "generic":
                    with self.subTest(path=name, dtype=dtype, layout="strided-negative"):
                        self.check_case(operation, dtype, (True, True, True, True), strided=True)
        self.check_raw_forward_backward(module)
        self.check_changed_graph(api.scaled_dot_product_attention)
        if len(operations) == 2:
            self.check_changed_graph(operations[-1][1])

    def check_raw_forward_backward(self, module):
        torch.manual_seed(42019)
        bases = [torch.randn(1, 2, 129, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        inputs = logical_inputs(bases, (True, True, True))
        dout = torch.ops.aten._neg_view.default(torch.randn_like(inputs[0]))
        self.retain_inputs("raw-native", bases, dout, (True, True, True, True))
        out, maximum, logsum, blocks = module.forward(*inputs, None, 0.125, True)
        actual = out, *module.backward(*inputs, None, maximum, logsum, dout, blocks, out, 0.125, True)
        expected = fp64_reference(inputs, dout, causal=True, scale=0.125)
        for got, want in zip(actual, expected):
            self.assertLess(metrics(got, want)["relative_error"], 0.01)

        def call():
            result, maximum, logsum, blocks = module.forward(*inputs, None, 0.125, True)
            return result, *module.backward(*inputs, None, maximum, logsum, dout, blocks, result, 0.125, True)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                call()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=stream):
            actual = call()
        with torch.no_grad():
            for tensor in (*bases, dout):
                tensor.copy_(torch.randn_like(tensor))
            for tensor in actual:
                tensor.fill_(float("nan"))
        self.retain_inputs("raw-native-changed-graph", bases, dout, (True, True, True, True))
        graph.replay()
        torch.cuda.synchronize()
        expected = fp64_reference(inputs, dout, causal=True, scale=0.125)
        for got, want in zip(actual, expected):
            self.assertLess(metrics(got, want)["relative_error"], 0.01)

    def check_changed_graph(self, operation):
        from benchmarks.compare_ada import capture

        torch.manual_seed(42023)
        bases = [torch.randn(1, 2, 129, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
        dout_base = torch.randn_like(bases[0])
        dout = torch.ops.aten._neg_view.default(dout_base)

        def flagged(q, k, v, *, is_causal=False):
            return operation(*logical_inputs((q, k, v), (True, True, True)), is_causal=is_causal)

        graph, storage = capture(flagged, [tensor.detach() for tensor in bases], dout, True, "joint", keep_graph=True)
        operands, dout, _, actual = storage
        for _ in range(2):
            with torch.no_grad():
                for tensor in (*operands, dout):
                    tensor.copy_(torch.randn_like(tensor))
                for tensor in actual:
                    tensor.fill_(float("nan"))
            self.retain_inputs("changed-graph", operands, dout, (True, True, True, True))
            graph.replay()
            torch.cuda.synchronize()
            inputs = logical_inputs(operands, (True, True, True))
            expected = fp64_reference(inputs, dout, causal=True)
            expected = (expected[0], *base_gradients(expected[1:], (True, True, True), operands))
            for got, want in zip(actual, expected):
                self.assertLess(metrics(got, want)["relative_error"], 0.01)


if __name__ == "__main__":
    unittest.main()
