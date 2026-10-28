"""Large batch/head grids retain exact singleton-key forward/backward results."""

import unittest

import torch


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.cuda.get_device_capability() in {(8, 6), (8, 9), (9, 0), (10, 0), (10, 3), (12, 0), (12, 1)},
    "Requires supported CuTe CUDA kernels",
)
class LaunchLimitTests(unittest.TestCase):
    def run_attention(self, inputs, dout, *, mask=None):
        from tests.test_tracker_regressions import run_attention

        return run_attention(inputs, dout, mask=mask)

    def test_grid_limits(self):
        # One visible key gives an exact oracle without a huge FP64 score matrix.
        shapes = (
            (1, 65534, 1, 64),
            (1, 65535, 1, 64),
            (1, 65536, 1, 64),
            (256, 256, 1, 64),
            (65536, 1, 1, 64),
            (1, 65536, 129, 8),
            (256, 256, 1, 257),
        )
        for batch, heads, queries, dimension in shapes:
            with self.subTest(batch=batch, heads=heads, queries=queries, dimension=dimension):
                inputs = [
                    torch.randn(batch, heads, length, dimension, device="cuda", dtype=torch.float16)
                    for length in (queries, 1, 1)
                ]
                dout = torch.ones_like(inputs[0])
                out, dq, dk, dv = self.run_attention(inputs, dout)
                torch.testing.assert_close(out, inputs[2].expand_as(out), atol=0, rtol=0)
                self.assertEqual(dq.count_nonzero().item(), 0)
                self.assertEqual(dk.count_nonzero().item(), 0)
                torch.testing.assert_close(dv, dout.sum(2, keepdim=True), atol=0, rtol=0)

        # Query and KV launches can lie on opposite sides of the grid limit.
        q = torch.randn(256, 256, 1, 64, device="cuda", dtype=torch.float16)
        k, v = [torch.randn(256, 1, 1, 64, device="cuda", dtype=torch.float16) for _ in range(2)]
        out, dq, dk, dv = self.run_attention((q, k, v), torch.ones_like(q))
        torch.testing.assert_close(out, v.expand_as(q), atol=0, rtol=0)
        self.assertEqual(dq.count_nonzero().item(), 0)
        self.assertEqual(dk.count_nonzero().item(), 0)
        torch.testing.assert_close(dv, torch.full_like(v, 256), atol=0, rtol=0)

    def test_masked_grid_limits(self):
        batch, heads = 256, 256
        inputs = [torch.randn(batch, heads, 1, 64, device="cuda", dtype=torch.float16) for _ in range(3)]
        mask = (torch.arange(batch * heads, device="cuda").reshape(batch, heads, 1, 1) % 2) == 0
        dout = torch.ones_like(inputs[0])
        out, dq, dk, dv = self.run_attention(inputs, dout, mask=mask)
        torch.testing.assert_close(out, inputs[2].masked_fill(~mask, 0), atol=0, rtol=0)
        self.assertEqual(dq.count_nonzero().item(), 0)
        self.assertEqual(dk.count_nonzero().item(), 0)
        torch.testing.assert_close(dv, dout.masked_fill(~mask, 0), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
