"""Independent experimental equations and focused device qualification.

Run CPU and GPU methods on the rented host. Sanitizers should select the GPU
formula/graph method; its numerical references run on CPU outside GPU tooling.
"""

import unittest

import torch

from benchmarks.neighborhood_precision_reference import mathematical_reference, precision_reference, reference_gate
from benchmarks.neighborhood_reference import neighborhood_mask
from benchmarks.qualify_neighborhood import execute
from benchmarks.reference import rounded_reference


class NeighborhoodPrecisionReferenceTests(unittest.TestCase):
    def test_baseline_is_the_existing_independent_rounded_equation(self):
        for dtype in (torch.float16, torch.bfloat16):
            torch.manual_seed(149)
            inputs = [torch.randn(1, 2, 15, 64, dtype=dtype) for _ in range(3)]
            dout = torch.randn_like(inputs[0])
            out = mathematical_reference(inputs, dout, (3, 5), (3, 3), -0.125)[0].to(dtype)
            got = precision_reference(inputs, dout, (3, 5), (3, 3), -0.125, out)
            want = rounded_reference(inputs, dout, neighborhood_mask((3, 5), (3, 3)), scale=-0.125, output=out)
            for a, b in zip(got, want):
                torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)

    def test_normalized_weighted_equation_removes_exact_common_key(self):
        for dtype in (torch.float16, torch.bfloat16):
            torch.manual_seed(150)
            inputs = [torch.zeros(1, 1, 15, 64, dtype=dtype) for _ in range(3)]
            q, k, v = inputs
            q[..., 0], k[..., 0] = 1024, 1024
            q[..., 1] = 1
            k[..., 1] = torch.arange(15) % 7
            v.copy_(torch.randn_like(v))
            dout = torch.randn_like(q)
            out = mathematical_reference(inputs, dout, (3, 5), (3, 3), 0.125)[0].to(dtype)
            uncorrected = precision_reference(inputs, dout, (3, 5), (3, 3), 0.125, out)
            corrected = precision_reference(inputs, dout, (3, 5), (3, 3), 0.125, out, correction="normalized")
            self.assertGreater(float(uncorrected[1][..., 0].abs().max()), 0)
            torch.testing.assert_close(corrected[1][..., 0], torch.zeros_like(corrected[1][..., 0]), atol=1e-12, rtol=0)
            for a, b in zip(corrected[2:], uncorrected[2:]):
                torch.testing.assert_close(a, b, atol=0, rtol=0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA qualification requires a rented GPU")
class NeighborhoodPrecisionGPUTests(unittest.TestCase):
    def test_cpu_and_cuda_reference_agree_on_exact_small_operands(self):
        # Dyadic logits/dP make this a cross-device equation check, independent
        # of a different FP32 reduction order crossing a dS rounding boundary.
        q = torch.zeros(1, 1, 15, 64, dtype=torch.float16)
        k, v = q.clone(), q.clone()
        q[..., 0] = 1
        k[..., 0] = 2
        v[..., 0] = 0.5
        dout = torch.ones_like(q)
        output = torch.zeros_like(q)
        output[..., 0] = 0.5
        for fp32_p in (False, True):
            for correction in ("none", "weighted", "normalized"):
                kwargs = dict(fp32_p=fp32_p, correction=correction)
                cpu = precision_reference([q, k, v], dout, (3, 5), (3, 3), 0.125, output, **kwargs)
                gpu = precision_reference(
                    [x.cuda() for x in (q, k, v)], dout.cuda(), (3, 5), (3, 3), 0.125, output.cuda(), **kwargs
                )
                for a, b in zip(cpu, gpu):
                    torch.testing.assert_close(a, b.cpu(), atol=1e-12, rtol=1e-12)

    def test_all_variants_formula_and_changed_poisoned_graph(self):
        from benchmarks.neighborhood_precision import VARIANTS, variant
        from kohakufa_cute import neighborhood_attention

        options = {
            "baseline": {},
            "fp32_p": dict(fp32_p=True),
            "weighted_dq": dict(correction="normalized"),
            "fp32_p_weighted_dq": dict(fp32_p=True, correction="normalized"),
            "legacy_weighted": dict(fp32_p=True, correction="weighted"),
        }
        spatial, window = (9, 17), (3, 4)
        for dtype in (torch.float16, torch.bfloat16):
            for name in VARIANTS:
                with self.subTest(dtype=dtype, variant=name), variant(name):
                    torch.manual_seed(151)
                    inputs = [torch.randn(1, 2, 153, 64, dtype=dtype, device="cuda").requires_grad_() for _ in range(3)]
                    dout = torch.randn_like(inputs[0])
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

                    # First autograd call, capture and eager probes all share
                    # this stream; AccumulateGrad retains its first stream.
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            actual = call()
                    stream.synchronize()
                    cpu = [x.detach().cpu() for x in inputs]
                    expected = precision_reference(
                        cpu, dout.cpu(), spatial, window, -0.125, actual[0].detach().cpu(), **options[name]
                    )
                    self.assertTrue(reference_gate(actual, expected, dtype)["passed"])
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
                        expected = precision_reference(
                            [x.detach().cpu() for x in inputs],
                            dout.cpu(),
                            spatial,
                            window,
                            -0.125,
                            got[0].cpu(),
                            **options[name],
                        )
                        self.assertTrue(reference_gate(got, expected, dtype)["passed"])


if __name__ == "__main__":
    unittest.main()
