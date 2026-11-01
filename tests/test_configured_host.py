import gc
import unittest

import torch

from kohakufa_cute import fa4_sm80


def configuration(dimension, *, query_stages=1):
    return dict(
        forward_m=64,
        backward_m=64,
        backward_n=128 if dimension == 64 else 64,
        query_stages=query_stages,
        dout_stages=1,
        score_m_warps=2 if dimension == 64 else 4,
        kv_m_warps=4 if dimension == 64 else 2,
        dq_m_warps=4 if dimension == 64 else 2,
        v_in_regs=True,
    )


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.cuda.get_device_capability() in ((8, 0), (8, 6), (8, 9), (12, 0), (12, 1))
    and fa4_sm80.available(),
    "Requires a supported warp device and the qualified optional runtime",
)
class ConfiguredHostTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9703)

    def test_configured_frames_statistics_and_cache_isolation(self):
        from benchmarks.validate_sm80_static import assess, statistics
        from kohakufa_cute.native import extension

        module = extension()
        for dtype in (torch.bfloat16, torch.float16):
            for dimension in (64, 128):
                for sequence in (129, 257):
                    for causal in (False, True):
                        with self.subTest(dtype=dtype, dimension=dimension, sequence=sequence, causal=causal):
                            inputs = [
                                torch.randn(1, 2, sequence, dimension, device="cuda", dtype=dtype, requires_grad=True)
                                for _ in range(3)
                            ]
                            dout = torch.randn_like(inputs[0])
                            for stages in (1, 2):
                                runner = module.AttentionRunner("warp", configuration(dimension, query_stages=stages))
                                saved = []
                                with torch.autograd.graph.saved_tensors_hooks(
                                    lambda tensor: saved.append(tensor) or tensor, lambda t: t
                                ):
                                    out = runner(*inputs, causal=causal)
                                self.assertEqual(len(saved), 6)
                                statistics(inputs, saved[-2], saved[-1], causal, dimension**-0.5)
                                actual = out, *torch.autograd.grad(out, inputs, dout)
                                assess(actual, inputs, dout, causal=causal, scale=None, directed=False, pointwise=False)

    def test_changed_graph_with_unaligned_output_gradient(self):
        from benchmarks.compare_ada import capture
        from benchmarks.validate_sm80_static import assess
        from kohakufa_cute.native import extension

        for dtype in (torch.bfloat16, torch.float16):
            for dimension, stages in ((64, 2), (128, 1)):
                runner = extension().AttentionRunner("warp", configuration(dimension, query_stages=stages))

                def function(q, k, v, *, is_causal=False):
                    return runner(q, k, v, causal=is_causal)

                inputs = [torch.randn(1, 2, 129, dimension, device="cuda", dtype=dtype) for _ in range(3)]
                dout = torch.empty(inputs[0].numel() + 1, device="cuda", dtype=dtype)[1:].view_as(inputs[0])
                dout.copy_(torch.randn_like(dout))
                graph, storage = capture(function, inputs, dout, True, "joint")
                operands, _, _, actual = storage
                for _ in range(2):
                    with torch.no_grad():
                        for tensor in (*operands, dout):
                            tensor.copy_(torch.randn_like(tensor))
                        for tensor in actual:
                            tensor.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize()
                    assess(actual, operands, dout, causal=True, scale=None, directed=False, pointwise=False)

    def test_runner_lifetime_and_rejected_contracts(self):
        from benchmarks.validate_sm80_static import assess
        from kohakufa_cute.native import extension

        inputs = [torch.randn(1, 2, 129, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
        dout = torch.randn_like(inputs[0])
        module = extension()
        runner = module.AttentionRunner("warp", configuration(64))
        out = runner(*inputs, causal=True)
        del runner
        gc.collect()
        actual = out, *torch.autograd.grad(out, inputs, dout)
        assess(actual, inputs, dout, causal=True, scale=None, directed=False, pointwise=False)
        for config in (
            {"misspelled_field": 1},
            {"backward_m": 0},
            {"query_stages": 1, "dout_stages": 2},
            {"forward_stages": 2, "query_in_registers": True},
        ):
            with self.assertRaises(ValueError):
                module.AttentionRunner("warp", config)
        runner = module.AttentionRunner("warp", configuration(64))
        for scale in (0, -0.1, 1e-50, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                runner(*inputs, scale=scale)
        for scale in (0.17, 0.41, 0.17):
            out = runner(*inputs, scale=scale, causal=True)
            actual = out, *torch.autograd.grad(out, inputs, dout)
            assess(actual, inputs, dout, causal=True, scale=scale, directed=False, pointwise=False)

    def test_wide_query_tile_tail_scratch(self):
        from benchmarks.validate_sm80_static import assess, statistics
        from kohakufa_cute.native import extension

        options = configuration(64)
        options.update(backward_m=128, backward_n=64, score_m_warps=4, kv_m_warps=4, dq_m_warps=4, v_in_regs=False)
        runner = extension().AttentionRunner("warp", options)
        for dtype in (torch.bfloat16, torch.float16):
            for sequence in (129, 257):
                for causal in (False, True):
                    with self.subTest(dtype=dtype, sequence=sequence, causal=causal):
                        inputs = [
                            torch.randn(1, 2, sequence, 64, device="cuda", dtype=dtype, requires_grad=True)
                            for _ in range(3)
                        ]
                        dout = torch.randn_like(inputs[0])
                        saved = []
                        with torch.autograd.graph.saved_tensors_hooks(
                            lambda tensor: saved.append(tensor) or tensor, lambda t: t
                        ):
                            out = runner(*inputs, causal=causal)
                        statistics(inputs, saved[-2], saved[-1], causal, 64**-0.5)
                        actual = out, *torch.autograd.grad(out, inputs, dout)
                        assess(actual, inputs, dout, causal=causal, scale=None, directed=False, pointwise=False)


if __name__ == "__main__":
    unittest.main()
