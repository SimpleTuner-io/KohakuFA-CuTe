"""Directed contracts from open NVIDIA/cudnn-frontend attention issues.

Dense-view/cache tests adapt issues 1567/1570/624; they do not claim a THD
or execute-time cuDNN override API. Issues 1428/1435/1071/1083 supply the
remaining mathematical shapes and contracts. Issue 1573 supplies grid limits.
"""

import math
import os
import unittest

import torch

from benchmarks.reference import fp64_reference, metrics


def run_attention(inputs, dout, *, mask=None, causal=False, scale=None):
    from kohakufa_cute import attention

    variables = [value.detach().requires_grad_() for value in inputs]
    if os.environ.get("KOH_CUTE_TRACKER_BACKEND", "cute") == "cute":
        out = attention(*variables, mask=mask, causal=causal, scale=scale)
        return out, *torch.autograd.grad(out, variables, dout)
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if mask is not None and causal:
        visible = (
            torch.arange(inputs[1].shape[2], device=mask.device)[None]
            <= torch.arange(inputs[0].shape[2], device=mask.device)[:, None]
        )
        mask, causal = mask & visible, False
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        out = torch.nn.functional.scaled_dot_product_attention(
            *variables,
            attn_mask=mask,
            is_causal=causal,
            scale=scale,
            enable_gqa=inputs[0].shape[1] != inputs[1].shape[1],
        )
        return out, *torch.autograd.grad(out, variables, dout)


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.cuda.get_device_capability() in {(8, 6), (8, 9), (9, 0), (10, 0), (10, 3), (12, 0), (12, 1)},
    "Requires supported CuTe CUDA kernels",
)
class TrackerRegressionTests(unittest.TestCase):
    run_attention = staticmethod(run_attention)

    def assert_reference(self, values, reference):
        for name, actual, expected in zip(("out", "dq", "dk", "dv"), values, reference):
            for region in (slice(None), slice(0, 1), slice(-1, None)):
                got, want = actual[..., region, :], expected[..., region, :]
                result = metrics(got, want)
                self.assertTrue(result["finite"], (name, region, result))
                if result["reference_norm"]:
                    self.assertIsNotNone(result["relative_error"])
                    self.assertLess(
                        result["relative_error"], 0.03 if actual.dtype == torch.bfloat16 else 0.01, (name, region, result)
                    )
                else:
                    self.assertEqual(got.count_nonzero().item(), 0)

    def test_live_kv_views_ignore_capacity_tail(self):
        torch.manual_seed(1567)
        for dtype in (torch.float16, torch.bfloat16):
            q = torch.randn(1, 4, 96, 64, device="cuda", dtype=dtype)
            storage = [torch.randn(1, 4, 256, 64, device="cuda", dtype=dtype) for _ in range(2)]
            clean = [value.clone() for value in storage]
            dout = torch.randn_like(q)
            # Exercise the same cache in both length orders and then restore.
            for length in (256, 96, 192, 200, 255, 64, 256):
                for poison in (1234.0, float("nan")):
                    with self.subTest(dtype=dtype, length=length, poison=str(poison)):
                        for value, original in zip(storage, clean):
                            value.copy_(original)
                            value[..., length:, :] = poison
                        inputs = (q, *(value[..., :length, :] for value in storage))
                        reference = fp64_reference(inputs, dout)
                        actual = self.run_attention(inputs, dout)
                        self.assert_reference(actual, reference)
                        for value in storage:
                            tail = value[..., length:, :]
                            if math.isnan(poison):
                                self.assertTrue(torch.isnan(tail).all())
                            else:
                                self.assertTrue((tail == poison).all())

    def test_cached_strides_and_batch_changes(self):
        torch.manual_seed(1570)
        for dtype in (torch.float16, torch.bfloat16):
            for batch, length, layout in (
                (2, 256, "bhsd"),
                (2, 256, "bshd"),
                (2, 128, "bhsd"),
                (1, 256, "bhsd"),
                (2, 256, "bshd"),
                (2, 256, "bhsd"),
            ):
                with self.subTest(dtype=dtype, batch=batch, length=length, layout=layout):
                    inputs = [torch.randn(batch, 4, length, 64, device="cuda", dtype=dtype) for _ in range(3)]
                    if layout == "bshd":
                        inputs = [value.transpose(1, 2).contiguous().transpose(1, 2) for value in inputs]
                    dout = torch.randn_like(inputs[0])
                    self.assert_reference(self.run_attention(inputs, dout), fp64_reference(inputs, dout))

    def test_nonpositive_scales_with_masks(self):
        torch.manual_seed(1428)
        for dtype in (torch.float16, torch.bfloat16):
            inputs = [torch.randn(1, 2, 129, 128, device="cuda", dtype=dtype) for _ in range(3)]
            dout = torch.randn_like(inputs[0])
            mask = torch.arange(129, device="cuda")[None, :] < 97
            mask = mask.expand(129, -1).clone()
            mask[0] = False
            for scale in (0.0, -0.1):
                for causal in (False, True):
                    for masked in (False, True):
                        with self.subTest(dtype=dtype, scale=scale, causal=causal, masked=masked):
                            selected = mask if masked else None
                            actual = self.run_attention(inputs, dout, mask=selected, causal=causal, scale=scale)
                            reference = fp64_reference(inputs, dout, selected, causal, scale)
                            self.assert_reference(actual, reference)
                            if scale == 0:
                                self.assertEqual(actual[1].count_nonzero().item(), 0)
                                self.assertEqual(actual[2].count_nonzero().item(), 0)

    def test_single_head_single_query(self):
        torch.manual_seed(1071)
        for dtype in (torch.float16, torch.bfloat16):
            inputs = [torch.randn(1, 1, length, 64, device="cuda", dtype=dtype) for length in (1, 128, 128)]
            dout = torch.randn_like(inputs[0])
            self.assert_reference(self.run_attention(inputs, dout), fp64_reference(inputs, dout))

    def test_single_query_gqa_statistics(self):
        from kohakufa_cute import native

        if os.environ.get("KOH_CUTE_TRACKER_BACKEND", "cute") != "cute":
            self.skipTest("CuTe saved-statistics contract; no ragged cuDNN API equivalence")
        torch.manual_seed(1083)
        for heads, kvheads in ((8, 2), (32, 8)):
            inputs = [torch.randn(1, heads, 1, 128, device="cuda", dtype=torch.bfloat16)]
            inputs += [torch.randn(1, kvheads, 300, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
            dout = torch.randn_like(inputs[0])
            expected = fp64_reference(inputs, dout)
            self.assert_reference(self.run_attention(inputs, dout), expected)
            scale = 128**-0.5
            native.extension().forward(*inputs, None, scale, False)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out, maximum, logsum, _ = native.extension().forward(*inputs, None, scale, False)
            for _ in range(3):
                maximum.fill_(float("nan"))
                logsum.fill_(float("nan"))
                out.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                self.assertTrue(torch.isfinite(maximum).all())
                self.assertTrue(torch.isfinite(logsum).all())
                keys = inputs[1].double().repeat_interleave(heads // kvheads, dim=1)
                scores = inputs[0].double() @ keys.transpose(-1, -2)
                recovered = maximum.double() * scale + logsum.double() * math.log(2)
                torch.testing.assert_close(recovered, (scores * scale).logsumexp(-1), atol=1e-3, rtol=1e-5)
                torch.testing.assert_close(out.double(), expected[0], atol=0.004, rtol=0.03)

    def test_zero_stride_output_gradient(self):
        torch.manual_seed(100)
        inputs = [torch.randn(1, 2, 129, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        expanded = torch.tensor(0.25, device="cuda", dtype=torch.bfloat16).expand_as(inputs[0])
        self.assertEqual(expanded.stride(), (0, 0, 0, 0))
        reference = fp64_reference(inputs, expanded)
        self.assert_reference(self.run_attention(inputs, expanded), reference)
        self.assert_reference(self.run_attention(inputs, expanded.contiguous()), reference)


if __name__ == "__main__":
    unittest.main()
