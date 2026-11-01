"""Check literal unscaled statistics from the explicit Hopper pipeline."""

import itertools
import math
import unittest

import torch

from tests.hopper_runtime import full_h200_runtime


@unittest.skipUnless(
    full_h200_runtime(),
    "Requires full H200 and the pinned optional runtime",
)
class HopperShiftedStatisticsTests(unittest.TestCase):
    def test_negative_common_offset_retains_unscaled_maximum_and_lognorm(self):
        from kohakufa_cute.fa4_hopper import attention

        function = lambda *args, is_causal=False, **kwargs: attention(*args, causal=is_causal, **kwargs)
        for dtype, dimension, causal in itertools.product((torch.bfloat16, torch.float16), (64, 128), (False, True)):
            with self.subTest(dtype=dtype, dimension=dimension, causal=causal):
                length = 129
                q = torch.zeros(1, 2, length, dimension, device="cuda", dtype=dtype)
                k = torch.zeros_like(q)
                q[..., 0], k[..., 0] = 512, -512
                q[..., 1] = 1
                k[..., 1] = (torch.arange(length, device="cuda") % 7 - 3).to(dtype)
                v = torch.randn_like(q)
                inputs = [t.requires_grad_() for t in (q, k, v)]
                saved = []

                def retain(tensor):
                    saved.append(tensor)
                    return tensor

                with torch.autograd.graph.saved_tensors_hooks(retain, lambda tensor: tensor):
                    out = function(*inputs, is_causal=causal, scale=0.25)
                self.assertEqual(len(saved), 6)
                maximum, lognorm = saved[-2:]
                scores = q.double() @ k.double().transpose(-2, -1)
                if causal:
                    visible = torch.arange(length, device="cuda")
                    scores = scores.masked_fill(visible[None, :] > visible[:, None], -torch.inf)
                expected_maximum = scores.amax(-1)
                expected_lognorm = torch.logsumexp((scores - expected_maximum[..., None]) * 0.25, -1) / math.log(2)
                # These sparse integer products and maxima are exactly
                # representable in FP32; this is a literal-statistic fixture,
                # not a general requirement on tensor-core reduction order.
                self.assertTrue(torch.equal(maximum.double(), expected_maximum))
                torch.testing.assert_close(lognorm.double(), expected_lognorm, atol=2e-5, rtol=2e-6)
                self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
