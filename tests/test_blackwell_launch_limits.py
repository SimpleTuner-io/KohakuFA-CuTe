"""Force specialized launch mappings across the batch/head grid boundary."""

import unittest
from unittest.mock import patch

import torch


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() in {(10, 0), (10, 3)},
    "Requires datacentre Blackwell",
)
class BlackwellLaunchLimitTests(unittest.TestCase):
    def check_singleton_key(self, dimension, configuration, *, wide_aux=False):
        from kohakufa_cute import attention, blackwell_register_aux, kernels, native

        torch.manual_seed(1573)
        q, k, v = [torch.randn(256, 256, 1, dimension, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        dout = torch.ones_like(q)
        native.extension().clear_cache()
        kernels._compile_kernel.cache_clear()
        try:
            with patch.object(kernels, "kernel_configuration", configuration):
                if wide_aux:
                    # Exercise the wide worker without allocating S=512 at every head.
                    def worker(causal, fallback, base_vector):
                        return blackwell_register_aux.JointBackward(causal=causal, base_vector=base_vector)

                    with patch.object(blackwell_register_aux, "AdaptiveBackward", worker):
                        inputs = [x.detach().requires_grad_() for x in (q, k, v)]
                        out = attention(*inputs)
                        gradients = torch.autograd.grad(out, inputs, dout)
                else:
                    inputs = [x.detach().requires_grad_() for x in (q, k, v)]
                    out = attention(*inputs)
                    gradients = torch.autograd.grad(out, inputs, dout)
            torch.testing.assert_close(out, v, atol=0, rtol=0)
            self.assertEqual(gradients[0].count_nonzero().item(), 0)
            self.assertEqual(gradients[1].count_nonzero().item(), 0)
            torch.testing.assert_close(gradients[2], dout, atol=0, rtol=0)
        finally:
            native.extension().clear_cache()
            kernels._compile_kernel.cache_clear()

    def test_native_grid(self):
        for dimension in (64, 512):

            def configuration(device, mode, descriptors):
                selected = (32, min(dimension, 256), 4, False, mode != "forward", True, False, False, 64)
                return (*selected, True) if mode != "forward" and dimension == 512 else selected

            with self.subTest(dimension=dimension):
                self.check_singleton_key(dimension, configuration)

    def test_pipeline_grid(self):
        from kohakufa_cute import kernels

        for dimension in (64, 512):

            def configuration(device, mode, descriptors):
                return kernels._SM100_FUSED_CONFIGURATIONS[(dimension, mode)]

            with self.subTest(dimension=dimension):
                self.check_singleton_key(dimension, configuration)

    def test_register_aux_grid(self):
        if torch.cuda.get_device_capability() != (10, 3):
            self.skipTest("The adaptive wide worker is selected only on SM103")
        from kohakufa_cute import kernels

        def configuration(device, mode, descriptors):
            return kernels._SM103_FUSED_CONFIGURATIONS[(512, mode)]

        self.check_singleton_key(512, configuration, wide_aux=True)

    def test_split_forward_grid(self):
        from kohakufa_cute import kernels

        # Explicit scratch selects the split forward even though normal dispatch
        # reserves it for low-occupancy Ada/Hopper shapes.
        q = torch.zeros(1, 65536, 3, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.zeros(1, 65536, 1, 64, device=q.device, dtype=q.dtype)
        v = torch.randn_like(k)
        out = torch.full_like(q, float("nan"))
        maximum = torch.full(q.shape[:-1], float("nan"), device=q.device, dtype=torch.float32)
        logsum = maximum.clone()
        partial = torch.empty(*q.shape[:-1], 2, 64, device=q.device, dtype=torch.float32)
        stats = torch.empty(*q.shape[:-1], 2, 2, device=q.device, dtype=torch.float32)
        mask = torch.empty(1, device=q.device, dtype=torch.int32)
        blocks = torch.empty(1, 1, 1, 1, device=q.device, dtype=torch.int32)
        tensors = (q, k, v, mask, out, maximum, logsum, partial, stats, out, out, out, blocks, maximum)
        torch.cuda.synchronize()
        function = kernels.compile_kernel(q.device.index, "forward", False, False, tuple(map(kernels.descriptor, tensors)))
        function(*tensors, 0.125)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, v.expand_as(q), atol=0, rtol=0)
        torch.testing.assert_close(maximum, torch.zeros_like(maximum), atol=0, rtol=0)
        torch.testing.assert_close(logsum, torch.zeros_like(logsum), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
