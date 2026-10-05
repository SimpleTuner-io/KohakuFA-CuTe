import copy
import unittest

import torch
from torch.utils.checkpoint import checkpoint

from kohakufa_cute import automatic_scaled_dot_product_attention, scaled_dot_product_attention


class AdapterLinear(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.base = torch.nn.Linear(width, width).requires_grad_(False)
        self.a = torch.nn.Linear(width, 4, bias=False)
        self.b = torch.nn.Linear(4, width, bias=False)
        torch.nn.init.zeros_(self.b.weight)

    def forward(self, x):
        return self.base(x) + self.b(self.a(x))


class TrainingBlock(torch.nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.attention = attention
        self.q, self.k, self.v, self.proj = [AdapterLinear(128) for _ in range(4)]

    def forward(self, x):
        q, k, v = [layer(x).reshape(x.shape[0], x.shape[1], 2, 64).transpose(1, 2) for layer in (self.q, self.k, self.v)]
        out = self.attention(q, k, v)
        return x + self.proj(out.transpose(1, 2).reshape(x.shape))


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() in {(8, 9), (9, 0)},
    "Requires centered Ada/Hopper kernels",
)
class TrainingTests(unittest.TestCase):
    def test_fresh_compiled_automatic_call_selects_cute(self):
        import subprocess
        import sys
        import textwrap

        code = textwrap.dedent("""
            import torch
            from kohakufa_cute import api
            from benchmarks.reference import fp64_reference
            def unexpected_native(*args, **kwargs):
                raise AssertionError('Supported CUDA call selected native SDPA')
            api._native_sdpa = unexpected_native
            q = torch.zeros(1, 1, 1, 64, device='cuda', dtype=torch.bfloat16)
            k = torch.zeros(1, 1, 2, 64, device='cuda', dtype=torch.bfloat16)
            v, dout = torch.zeros_like(k), torch.zeros_like(q)
            q[..., 0] = 10
            k[..., 0] = torch.tensor([10, 5], device='cuda', dtype=q.dtype)
            v[..., 0] = torch.tensor([1, -1], device='cuda', dtype=q.dtype)
            dout[..., 0] = 1
            reference = fp64_reference((q, k, v), dout, scale=1.0)
            inputs = [tensor.requires_grad_() for tensor in (q, k, v)]
            fn = torch.compile(api.automatic_scaled_dot_product_attention, fullgraph=True)
            out = fn(*inputs, scale=1.0)
            gradients = torch.autograd.grad(out, inputs, dout)
            assert gradients[0][..., 0].item() > 0
            for actual, expected in zip((out, *gradients), reference):
                torch.testing.assert_close(actual.double(), expected, atol=0, rtol=0.02)
        """)
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_combined_causal_mask_in_cold_warm_and_compiled_attention(self):
        import subprocess
        import sys
        import textwrap

        code = textwrap.dedent("""
            import torch
            from kohakufa_cute import attention
            from benchmarks.reference import fp64_reference
            torch.manual_seed(105)
            q, k, v = [torch.randn(1, 2, 17, 64, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
            mask = torch.rand(17, 17, device='cuda') > 0.25
            dout = torch.randn_like(q)
            expected = fp64_reference((q, k, v), dout, mask, causal=True)
            for fn in (attention, attention, torch.compile(attention, fullgraph=True)):
                inputs = [tensor.detach().requires_grad_() for tensor in (q, k, v)]
                out = fn(*inputs, mask=mask, causal=True)
                gradients = torch.autograd.grad(out, inputs, dout)
                for actual, reference in zip((out, *gradients), expected):
                    torch.testing.assert_close(actual.double(), reference, atol=0.02, rtol=0.02)
        """)
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_checkpointed_lora_optimizer_steps_match_native(self):
        torch.manual_seed(99)
        initial = TrainingBlock(torch.nn.functional.scaled_dot_product_attention).cuda()
        x = torch.randn(1, 33, 128, device="cuda")
        target = torch.randn_like(x)
        histories, changes = [], []
        for fn, compiled in [
            (torch.nn.functional.scaled_dot_product_attention, False),
            (scaled_dot_product_attention, False),
            (automatic_scaled_dot_product_attention, True),
        ]:
            model = copy.deepcopy(initial)
            model.attention = fn
            parameters = [p for p in model.parameters() if p.requires_grad]
            original = [p.detach().clone() for p in parameters]
            current = torch.compile(model, fullgraph=True) if compiled else model
            optimizer = torch.optim.AdamW(parameters, lr=0.001)
            history = []
            for _ in range(5):
                optimizer.zero_grad(set_to_none=True)
                inputs = x.detach().requires_grad_()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = checkpoint(current, inputs, use_reentrant=False)
                    loss = (output.float() - target).square().mean()
                loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters))
                optimizer.step()
                history.append(loss.detach())
            histories.append(torch.stack(history))
            changes.append(torch.cat([(p.detach() - start).flatten() for p, start in zip(parameters, original)]))
        for history, change in zip(histories[1:], changes[1:]):
            torch.testing.assert_close(history, histories[0], atol=0.001, rtol=0.01)
            self.assertLess((change - changes[0]).norm().item(), 0.05 * changes[0].norm().item())
            self.assertGreater(change.norm().item(), 0)
            self.assertLess(history[-1].item(), history[0].item())

    def test_automatic_fullgraph_mixes_kernel_and_native_calls(self):
        compiled = torch.compile(automatic_scaled_dot_product_attention, fullgraph=True)
        compiled_native = torch.compile(torch.nn.functional.scaled_dot_product_attention, fullgraph=True)
        q = torch.randn(1, 2, 17, 64, device="cuda", dtype=torch.bfloat16)
        for tensor, kwargs, reference in [
            (q, {}, scaled_dot_product_attention),
            (q.float(), {}, compiled_native),
            (q, {"attn_mask": torch.randn(17, 17, device="cuda", dtype=q.dtype)}, compiled_native),
        ]:
            inputs = [tensor.detach().requires_grad_() for _ in range(3)]
            actual = compiled(*inputs, **kwargs)
            expected = reference(*inputs, **kwargs)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            dout = torch.randn_like(actual)
            for got, want in zip(torch.autograd.grad(actual, inputs, dout), torch.autograd.grad(expected, inputs, dout)):
                torch.testing.assert_close(got, want, atol=0, rtol=0)
