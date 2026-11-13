"""Independent residual-centered dK equations and focused GPU qualification."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from benchmarks.neighborhood_key_precision_reference import (
    METHODS,
    NEW_METHODS,
    key_precision_reference,
    key_terms,
    sparse_key_precision_reference,
)
from benchmarks.neighborhood_precision_reference import mathematical_reference, precision_reference, reference_gate
from benchmarks.neighborhood_reference import neighborhoods
from benchmarks.qualify_neighborhood import execute


class NeighborhoodKeyPrecisionReferenceTests(unittest.TestCase):
    def test_prior_controls_keep_their_original_equations(self):
        for dtype in (torch.float16, torch.bfloat16):
            torch.manual_seed(190)
            inputs = [torch.randn(1, 2, 15, 64, dtype=dtype) for _ in range(3)]
            dout = torch.randn_like(inputs[0])
            out = mathematical_reference(inputs, dout, (3, 5), (3, 3), -0.125)[0].to(dtype)
            for method, options in (
                ("baseline", {}),
                ("fp32_p_weighted_dq", dict(fp32_p=True, correction="normalized")),
            ):
                got = key_precision_reference(inputs, dout, (3, 5), (3, 3), -0.125, out, method=method)
                want = precision_reference(inputs, dout, (3, 5), (3, 3), -0.125, out, **options)
                for a, b in zip(got, want):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_normalized_residual_correction_cancels_a_dyadic_output_bias(self):
        # Every legal 2x2 window has exact dyadic probabilities. Integer inputs
        # make the separate residual and correction reductions exact here.
        for dtype in (torch.float16, torch.bfloat16):
            q = torch.ones(1, 1, 6, 4, dtype=dtype)
            k = torch.zeros_like(q)
            v = torch.arange(24, dtype=dtype).reshape_as(q)
            inputs = [q, k, v]
            dout = torch.ones_like(q)
            out = mathematical_reference(inputs, dout, (2, 3), (2, 2), 0.5)[0].to(dtype)
            for method in NEW_METHODS:
                first = key_precision_reference(inputs, dout, (2, 3), (2, 2), 0.5, out, method=method)
                shifted = key_precision_reference(inputs, dout, (2, 3), (2, 2), 0.5, out + 1, method=method)
                torch.testing.assert_close(first[2], shifted[2], atol=0, rtol=0)

    def test_key_product_uses_independently_enumerated_inverse_edges(self):
        q = torch.arange(24, dtype=torch.float16).reshape(1, 1, 6, 4) / 8
        k = torch.zeros_like(q)
        v = torch.arange(24, dtype=torch.float16).reshape_as(q) / 4
        dout = torch.ones_like(q)
        out = mathematical_reference([q, k, v], dout, (2, 3), (2, 2), -0.5)[0].half()
        terms = key_terms([q, k, v], dout, (2, 3), (2, 2), -0.5, out)
        expected = torch.zeros_like(q, dtype=torch.float64)
        for query, keys in enumerate(neighborhoods((2, 3), (2, 2))):
            for key in keys:
                expected[:, :, key] += terms["ds"][:, :, query, key, None].double() * q[:, :, query].double() * -0.5
        actual = key_precision_reference([q, k, v], dout, (2, 3), (2, 2), -0.5, out, method="weighted_dq_reconstructed_dk")
        torch.testing.assert_close(actual[2], expected, atol=0, rtol=0)

    def test_sparse_reference_matches_dense_dyadic_equations(self):
        q = torch.arange(24, dtype=torch.float16).reshape(1, 1, 6, 4) / 8
        k = torch.zeros_like(q)
        v = torch.arange(24, dtype=torch.float16).reshape_as(q) / 4
        dout = torch.ones_like(q)
        out = mathematical_reference([q, k, v], dout, (2, 3), (2, 2), -0.5)[0].half()
        positions = [0, 2, 5]
        for method in METHODS:
            dense = key_precision_reference([q, k, v], dout, (2, 3), (2, 2), -0.5, out, method=method)
            sparse = sparse_key_precision_reference(
                [q, k, v], dout, (2, 3), (2, 2), -0.5, out, method=method, positions=positions, chunk_size=2
            )
            self.assertEqual(sparse["positions"], positions)
            for a, b in zip(sparse["policy"], dense):
                torch.testing.assert_close(a, b[:, :, positions], atol=0, rtol=0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA qualification requires a rented GPU")
class NeighborhoodKeyPrecisionGPUTests(unittest.TestCase):
    def test_cpu_cuda_reference_agreement(self):
        q = torch.ones(1, 1, 6, 64, dtype=torch.float16)
        k = torch.zeros_like(q)
        v = torch.arange(6, dtype=q.dtype)[None, None, :, None].expand_as(q).contiguous()
        dout = torch.ones_like(q)
        out = mathematical_reference([q, k, v], dout, (2, 3), (2, 2), -0.125)[0].half()
        for method in NEW_METHODS:
            cpu = key_precision_reference([q, k, v], dout, (2, 3), (2, 2), -0.125, out, method=method)
            cuda = key_precision_reference(
                [x.cuda() for x in (q, k, v)], dout.cuda(), (2, 3), (2, 2), -0.125, out.cuda(), method=method
            )
            for a, b in zip(cpu, cuda):
                torch.testing.assert_close(a, b.cpu(), atol=0, rtol=0)

    def test_formula_and_changed_poisoned_graph(self):
        from kohakufa_cute import neighborhood_attention

        from benchmarks.neighborhood_precision import key_statistics, variant

        spatial, window = (9, 17), (3, 4)
        for dtype in (torch.float16, torch.bfloat16):
            for method in METHODS:
                statistics = "column_staged" if method == "weighted_dq_reconstructed_dk" else "direct"
                with self.subTest(dtype=dtype, method=method), variant(method), key_statistics(statistics):
                    torch.manual_seed(191)
                    inputs = [torch.randn(1, 2, 153, 64, dtype=dtype, device="cuda").requires_grad_() for _ in range(3)]
                    dout = torch.randn(*inputs[0].shape[:-1], 129, dtype=dtype, device="cuda")[..., 1::2]
                    self.assertEqual(dout.shape, inputs[0].shape)
                    self.assertEqual(dout.stride(-1), 2)
                    self.assertEqual(dout.storage_offset(), 1)
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())

                    def call():
                        return execute(
                            lambda q, k, v: neighborhood_attention(
                                q, k, v, spatial_shape=spatial, window_size=window, scale=-0.125
                            ),
                            inputs,
                            dout,
                        )

                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            actual = call()
                    stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        actual = call()
                    for turn in range(2):
                        with torch.no_grad(), torch.cuda.stream(stream):
                            for index, operand in enumerate((*inputs, dout)):
                                operand.mul_(0.5).add_(0.015625 * (index + turn + 1))
                            for tensor in actual:
                                tensor.fill_(float("nan"))
                            graph.replay()
                        stream.synchronize()
                        with torch.cuda.stream(stream):
                            got = tuple(x.detach().clone() for x in actual)
                            fresh = call()
                        stream.synchronize()
                        for a, b in zip(got, fresh):
                            torch.testing.assert_close(a, b, atol=0, rtol=0)
                        cpu, gradient = [x.detach().cpu() for x in inputs], dout.cpu()
                        expected = key_precision_reference(
                            cpu, gradient, spatial, window, -0.125, got[0].cpu(), method=method
                        )
                        self.assertTrue(reference_gate(got, expected, dtype)["passed"])
                        exact = mathematical_reference(cpu, gradient, spatial, window, -0.125)
                        self.assertTrue(reference_gate(got, exact, dtype)["passed"])

    def test_resolved_schedule_changed_poisoned_graph(self):
        from kohakufa_cute import neighborhood_attention, neighborhood_ops
        from kohakufa_cute.neighborhood_kernels import effective_schedule

        from benchmarks.benchmark_neighborhood import sample_positions, sparse_references
        from benchmarks.neighborhood_precision import key_statistics, variant

        properties = torch.cuda.get_device_properties(0)
        normalized_name = "".join(properties.name.upper().split())
        full_h100 = (
            torch.cuda.get_device_capability() == (9, 0)
            and properties.multi_processor_count == 132
            and all(token in normalized_name for token in ("H100", "80GB", "HBM3"))
            and all(token not in normalized_name for token in ("NVL", "MIG"))
        )
        # This fixture selects the bounded H100 preset only on that qualified
        # device. Every other architecture must exercise its real C32 path;
        # skipping here would leave its memory ownership unqualified.
        expected_columns = (32, 32, 64) if full_h100 else (32, 32, 32)
        expected_traversal = (True, False, True) if full_h100 else (False, False, False)
        spatial, window = (16, 64), (7, 7)
        positions = sample_positions(spatial, count=6)
        original_arguments = neighborhood_ops._arguments
        for dtype in (torch.float16, torch.bfloat16):
            for method in NEW_METHODS:
                ownership = []

                def checked_arguments(*arguments, **kwargs):
                    tensors = original_arguments(*arguments, **kwargs)
                    dout = kwargs.get("dout", arguments[6] if len(arguments) > 6 else None)
                    if dout is not None:
                        # Observe buffers from the actual API call rather than
                        # constructing an independent scratch-allocation probe.
                        ownership.append(tensors[8].data_ptr() != tensors[13].data_ptr())
                    return tensors

                statistics = "column_staged" if method == "weighted_dq_reconstructed_dk" else "direct"
                with (
                    self.subTest(dtype=dtype, method=method),
                    variant(method),
                    key_statistics(statistics),
                    patch.object(neighborhood_ops, "_arguments", checked_arguments),
                ):
                    torch.manual_seed(192)
                    inputs = [torch.randn(1, 8, 1024, 64, device="cuda", dtype=dtype).requires_grad_() for _ in range(3)]
                    dout = torch.randn(1, 8, 1024, 129, device="cuda", dtype=dtype)[..., 1::2]
                    self.assertEqual(dout.stride(-1), 2)
                    self.assertEqual(dout.storage_offset(), 1)
                    columns, traversal = effective_schedule(tuple(inputs), spatial, window)
                    self.assertEqual(columns, expected_columns)
                    self.assertEqual(traversal, expected_traversal)
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())

                    def call():
                        return execute(
                            lambda q, k, v: neighborhood_attention(
                                q, k, v, spatial_shape=spatial, window_size=window, scale=-0.125
                            ),
                            inputs,
                            dout,
                        )

                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            call()
                    stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        actual = call()
                    for turn in range(3):
                        with torch.no_grad(), torch.cuda.stream(stream):
                            if turn:
                                for index, operand in enumerate((*inputs, dout)):
                                    operand.mul_(0.5).add_(0.015625 * (index + turn + 1))
                            for tensor in actual:
                                tensor.fill_(float("nan"))
                            graph.replay()
                        stream.synchronize()
                        with torch.cuda.stream(stream):
                            got = tuple(value.detach().clone() for value in actual)
                            fresh = call()
                        stream.synchronize()
                        for a, b in zip(got, fresh):
                            torch.testing.assert_close(a, b, atol=0, rtol=0)
                        # All reference arithmetic is bounded CPU work, including
                        # every query contributing to a sampled inverse key owner.
                        references = sparse_references(
                            inputs, dout, got[0], spatial, window, scale=-0.125, positions=positions, chunk_size=32
                        )
                        policy = sparse_key_precision_reference(
                            inputs,
                            dout,
                            spatial,
                            window,
                            -0.125,
                            got[0],
                            method=method,
                            positions=positions,
                            chunk_size=32,
                        )
                        selected = tuple(value.cpu()[:, :, positions] for value in got)
                        self.assertTrue(reference_gate(selected, policy["policy"], dtype)["passed"])
                        self.assertTrue(reference_gate(selected, references["fp64"], dtype)["passed"])
                    self.assertTrue(ownership, "No actual backward scratch buffers were observed")
                    self.assertTrue(all(ownership), "The reconstruction and actual-O delta buffers alias")

    def test_frozen_sharp_math_fp64(self):
        from kohakufa_cute import neighborhood_attention

        from benchmarks.neighborhood_precision import variant
        from benchmarks.qualify_neighborhood_accuracy import tensor_hash
        from benchmarks.test_neighborhood_key_precision import (
            fixtures,
            historical_hashes,
            required_math_components,
            selected_gate,
        )

        retained = os.environ.get("NEIGHBORHOOD_FROZEN_OPERANDS")
        frozen = Path(retained) if retained else None
        hashes = historical_hashes(Path("docs/experimental/neighborhood-key-precision.json"))
        checked = 0
        for index, fixture in enumerate(fixtures(frozen)):
            label, inputs, dout, spatial, window, scale, stress, _ = fixture
            if label not in ("sharp-signed", "zero-dp-sharp", "constant-dp-sharp"):
                continue
            cpu, gradient = [x.detach().cpu() for x in inputs], dout.cpu()
            self.assertEqual(tensor_hash([*cpu, gradient]), hashes[index])
            exact = mathematical_reference(cpu, gradient, spatial, window, scale)
            for method in NEW_METHODS:
                with self.subTest(fixture=label, dtype=inputs[0].dtype, scale=scale, method=method), variant(method):
                    actual = execute(
                        lambda q, k, v: neighborhood_attention(
                            q, k, v, spatial_shape=spatial, window_size=window, scale=scale
                        ),
                        inputs,
                        dout,
                    )
                    torch.cuda.synchronize()
                    expected = key_precision_reference(
                        cpu, gradient, spatial, window, scale, actual[0].detach().cpu(), method=method
                    )
                    self.assertTrue(reference_gate(actual, expected, inputs[0].dtype)["passed"])
                    gate = selected_gate(actual, exact, inputs[0].dtype, required_math_components(method, stress, label))
                    self.assertTrue(gate["passed"], gate)
            checked += 1
        self.assertEqual(checked, 12)


if __name__ == "__main__":
    unittest.main()
