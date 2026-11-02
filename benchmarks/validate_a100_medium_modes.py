"""Check actual runtime gradient-mode boundaries for medium causal selection."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    from benchmarks.a100_medium_controls import make_parent
    from kohakufa_cute import api, fa4_sm80

    parent = make_parent()
    torch.manual_seed(839731)
    inputs = [torch.randn(1, 8, 2048, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    dout = torch.randn_like(inputs[0])
    report = {"rows": []}

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    for label, context in (("no_grad", torch.no_grad()), ("inference_mode", torch.inference_mode())):
        with context:
            assert not fa4_sm80.compatible(*inputs, None, 0, True, None, False)
            with patch.object(fa4_sm80, "attention", wraps=fa4_sm80.attention) as chosen:
                actual = api.scaled_dot_product_attention(*inputs, is_causal=True)
            expected = parent(*inputs, is_causal=True)
            assert chosen.call_count == 0
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0.03)
        report["rows"].append({"mode": label, "selected": False, "same_previous_output": True})
        save()
    for position in range(3):
        part = [t.detach().requires_grad_(i != position) for i, t in enumerate(inputs)]
        assert not fa4_sm80.compatible(*part, None, 0, True, None, False)
        with patch.object(fa4_sm80, "attention", wraps=fa4_sm80.attention) as chosen:
            out = api.scaled_dot_product_attention(*part, is_causal=True)
        expected_out = parent(*part, is_causal=True)
        leaves = [t for t in part if t.requires_grad]
        actual = (out, *torch.autograd.grad(out, leaves, dout))
        expected = (expected_out, *torch.autograd.grad(expected_out, leaves, dout))
        assert chosen.call_count == 0
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=0.03)
        report["rows"].append(
            {
                "mode": "partial_grad",
                "disabled_input": position,
                "selected": False,
                "same_previous_output_and_gradients": True,
            }
        )
        save()
    strided = torch.empty(1, 8, 4096, 128, device="cuda", dtype=inputs[0].dtype)[:, :, ::2].requires_grad_(True)
    assert not fa4_sm80.compatible(strided, inputs[1], inputs[2], None, 0, True, None, False)
    unaligned = (
        torch.empty(inputs[0].numel() + 1, device="cuda", dtype=inputs[0].dtype)[1:].view_as(inputs[0]).requires_grad_(True)
    )
    assert unaligned.data_ptr() % 16 and not fa4_sm80.compatible(unaligned, inputs[1], inputs[2], None, 0, True, None, False)
    report["rows"].append({"mode": "storage_views", "strided_selected": False, "unaligned_selected": False})
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(inputs[0], torch.ones_like(inputs[0]))
        assert not fa4_sm80.compatible(dual, inputs[1], inputs[2], None, 0, True, None, False)
    report["rows"].append({"mode": "forward_ad", "selected": False})
    save()

    graphs = []

    def backend(gm, examples):
        graphs.append([str(n.target) for n in gm.graph.nodes if n.op == "call_function"])
        return gm.forward

    current_compiled = torch.compile(api.scaled_dot_product_attention, backend=backend, fullgraph=True)
    parent_compiled = torch.compile(parent, backend=backend, fullgraph=True)
    errors = []
    values = []
    for function in (current_compiled, parent_compiled):
        try:
            out = function(*inputs, is_causal=True)
            values.append((out, *torch.autograd.grad(out, inputs, dout)))
            errors.append(None)
        except Exception as error:
            errors.append(type(error).__name__)
    assert errors[0] == errors[1], errors
    if not errors[0]:
        for a, b in zip(*values):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=0.03)
    report["rows"].append(
        {
            "mode": "compiled_fullgraph",
            "graphs": graphs,
            "previous_and_current_error": errors,
            "same_previous_result": not any(errors),
        }
    )
    report["status"] = "pass"
    save()


if __name__ == "__main__":
    main()
