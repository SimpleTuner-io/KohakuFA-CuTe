"""Final narrowed public selection; kernel math/tools are qualified separately."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    from benchmarks.compare_ada import capture
    from benchmarks.compare_l40s import graph_kernels
    from kohakufa_cute import api
    from kohakufa_cute.fa4_sm80 import _host
    from kohakufa_cute.native import extension

    report = {"rows": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for heads, sequence, dtype, causal in (
        (16, 1024, torch.bfloat16, True),
        (8, 4096, torch.float16, False),
        (16, 4096, torch.bfloat16, True),
        (8, 1024, torch.bfloat16, True),
        (16, 128, torch.float16, False),
    ):
        selected = sequence == 4096 or (heads == 16 and sequence == 1024)
        torch.manual_seed(4471)
        inputs = [torch.randn(1, heads, sequence, 256, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)]
        dout = torch.randn_like(inputs[0])
        runtime = extension()
        assert not runtime.a40_metadata_compatible(*inputs, causal)
        assert not runtime.gb10_inference_metadata_compatible(*inputs, causal)
        with patch.object(_host, "native_runner", wraps=_host.native_runner) as called:
            output = api.scaled_dot_product_attention(*inputs, is_causal=causal)
        assert called.called == selected
        actual = output, *torch.autograd.grad(output, inputs, dout)
        assert all(torch.isfinite(t).all() for t in actual)
        graph, storage = capture(
            api.scaled_dot_product_attention, [t.detach() for t in inputs], dout, causal, "joint", keep_graph=True
        )
        leaves, _, _, actual = storage
        with torch.no_grad():
            for t in (*leaves, dout):
                t.copy_(torch.randn_like(t))
            for t in actual:
                t.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert all(torch.isfinite(t).all() for t in actual)
        kernels = graph_kernels(graph)
        if selected:
            assert len(kernels) == 4 and all(k["BINARY_VERSION"] == 80 for k in kernels)
        report["rows"].append(
            {
                "heads": heads,
                "sequence": sequence,
                "dtype": str(dtype),
                "causal": causal,
                "selected": selected,
                "kernels": kernels,
            }
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = "pass"
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
