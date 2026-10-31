"""Compare directed cuDNN failure cases with rounded and mathematical references.

Run with ``python -m benchmarks.cudnn_regressions --output RESULTS.json``.
cuDNN and other native backends are forced individually; unsupported cases
are recorded separately from numerical failures and successful executions.
"""

import argparse
import dataclasses
import importlib.metadata
import json
import sys
from pathlib import Path

import torch

from benchmarks.adversarial import cases, logits_statistics, make_case
from benchmarks.reference import fp64_reference, metrics, rounded_reference


def execute(backend, case, inputs, dout, mask):
    variables = [value.detach().clone().requires_grad_() for value in inputs]
    if backend == "kohakufa-cute":
        from kohakufa_cute import attention

        output = attention(*variables, mask=mask, causal=case.causal, scale=case.scale)
        return output, *torch.autograd.grad(output, variables, dout)
    if backend.startswith("frontend"):
        return frontend(case, inputs, dout, mask, padding=backend.endswith("padding"))
    from torch.nn.attention import SDPBackend, sdpa_kernel

    selected = {"cudnn": SDPBackend.CUDNN_ATTENTION, "flash": SDPBackend.FLASH_ATTENTION, "math": SDPBackend.MATH}
    with sdpa_kernel(selected[backend]):
        output = torch.nn.functional.scaled_dot_product_attention(
            *variables,
            attn_mask=mask,
            is_causal=case.causal,
            scale=case.scale,
            enable_gqa=case.heads != case.kvheads,
        )
        return output, *torch.autograd.grad(output, variables, dout)


def frontend(case, inputs, dout, mask, *, padding=False):
    import cudnn

    if mask is not None:
        raise NotImplementedError("Direct frontend comparison uses unmasked fixtures only")
    q, k, v = inputs
    dtype = cudnn.data_type.BFLOAT16 if q.dtype == torch.bfloat16 else cudnn.data_type.HALF
    scale = case.dim**-0.5 if case.scale is None else case.scale

    def graph():
        return cudnn.pygraph(
            io_data_type=dtype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT
        )

    def options(g, bindings):
        if not padding:
            return {}
        values = [
            torch.full((1, 1, 1, 1), length, device=q.device, dtype=torch.int32) for length in (case.queries, case.keys)
        ]
        attrs = [g.tensor_like(value) for value in values]
        bindings.update(zip(attrs, values))
        return dict(use_padding_mask=True, seq_len_q=attrs[0], seq_len_kv=attrs[1])

    def mark(attr, tensor):
        attr.set_output(True).set_dim(tensor.shape).set_stride(tensor.stride())

    def run(g, bindings):
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A])
        g.check_support()
        g.build_plans()
        workspace = torch.empty(g.get_workspace_size(), device=q.device, dtype=torch.uint8)
        g.execute(bindings, workspace)
        torch.cuda.synchronize()

    forward = graph()
    attrs = [forward.tensor_like(value) for value in inputs]
    bindings = dict(zip(attrs, inputs))
    output_attr, stats_attr = forward.sdpa(
        q=attrs[0],
        k=attrs[1],
        v=attrs[2],
        generate_stats=True,
        attn_scale=scale,
        use_causal_mask=case.causal,
        **options(forward, bindings),
    )
    output = torch.empty_like(q)
    stats = torch.empty(1, case.heads, case.queries, 1, device=q.device, dtype=torch.float32)
    mark(output_attr, output)
    mark(stats_attr, stats)
    stats_attr.set_data_type(cudnn.data_type.FLOAT)
    bindings.update({output_attr: output, stats_attr: stats})
    run(forward, bindings)
    if not torch.isfinite(output).all() or not torch.isfinite(stats).all():
        raise FloatingPointError("Direct frontend saved output/statistics are nonfinite")
    backward = graph()
    saved = dict(q=q, k=k, v=v, o=output, dO=dout, stats=stats)
    attrs = {name: backward.tensor_like(value) for name, value in saved.items()}
    bindings = {attrs[name]: value for name, value in saved.items()}
    grads = backward.sdpa_backward(**attrs, attn_scale=scale, use_causal_mask=case.causal, **options(backward, bindings))
    gradients = [torch.empty_like(value) for value in inputs]
    for attr, value in zip(grads, gradients):
        mark(attr, value)
    bindings.update(zip(grads, gradients))
    run(backward, bindings)
    return output, *gradients


def assess(actual, expected, dtype):
    def acceptable(row, got):
        return row["finite"] and (
            row["relative_error"] < (0.05 if dtype == torch.bfloat16 else 0.015)
            if row["reference_norm"] > 0
            else int(got.count_nonzero()) == 0
        )

    result = {}
    passed = True
    for name, got, want in zip(("out", "dq", "dk", "dv"), actual, expected):
        row = metrics(got, want)
        row["nan_elements"] = int(torch.isnan(got).sum())
        row["inf_elements"] = int(torch.isinf(got).sum())
        # Norm-relative comparisons cannot hide tiny nonzero gradients under
        # a large absolute tolerance. Zero-reference cases require actual zero.
        row["passed"] = acceptable(row, got)
        row["edge_rows"] = {}
        for edge, region in (("first", slice(0, 1)), ("last", slice(-1, None))):
            value, reference = got[..., region, :], want[..., region, :]
            edge_result = metrics(value, reference)
            edge_result["passed"] = acceptable(edge_result, value)
            row["edge_rows"][edge] = edge_result
            row["passed"] &= edge_result["passed"]
        passed &= row["passed"]
        result[name] = row
    return {"status": "pass" if passed else "numerical_failure", "tensors": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("kohakufa-cute", "cudnn", "flash", "math", "frontend", "frontend-padding"),
        default=["kohakufa-cute", "cudnn", "flash"],
    )
    parser.add_argument("--case", action="append", help="Restrict to these named fixtures")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--reference", choices=("rounded", "fp64"), default="rounded")
    args = parser.parse_args()
    selected = [case for case in cases(extended=not args.quick) if not args.case or case.name in args.case]
    if not selected:
        parser.error("No selected fixtures")
    report = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "torch_cudnn": torch.backends.cudnn.version(),
        "dependencies": {
            name: importlib.metadata.version(name) for name in ("nvidia-cutlass-dsl", "apache-tvm-ffi", "triton")
        },
        "reference": args.reference,
        "reference_policy": "The selected reference is applied to every backend; single-rounded P/dS uses each actual O for delta. Independent mathematical FP64 diagnostics are retained.",
        "sources": [
            "https://github.com/NVIDIA/cudnn-frontend/issues/160",
            "https://github.com/NVIDIA/cudnn-frontend/issues/1072",
            "https://github.com/NVIDIA/cudnn-frontend/pull/628",
        ],
        "rows": [],
    }
    if any(name.startswith("frontend") for name in args.backends):
        import cudnn

        report["frontend"] = str(cudnn.__version__)
        report["frontend_backend"] = cudnn.backend_version()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for dtype in (torch.bfloat16, torch.float16):
        for case in selected:
            inputs, dout, mask = make_case(case, device="cuda", dtype=dtype)
            expected = fp64_reference(inputs, dout, mask, causal=case.causal, scale=case.scale)
            assert all(torch.isfinite(value).all() for value in (*inputs, dout, *expected))
            row = {
                "case": dataclasses.asdict(case),
                "dtype": str(dtype),
                "statistics": logits_statistics(inputs, mask, causal=case.causal, scale=case.scale),
                "backends": {},
            }
            for backend in args.backends:
                try:
                    actual = execute(backend, case, inputs, dout, mask)
                    mathematical = assess(actual, expected, dtype)
                    contract = rounded_reference(inputs, dout, mask, causal=case.causal, scale=case.scale, output=actual[0])
                    rounded = assess(actual, contract, dtype)
                    result = dict(mathematical if args.reference == "fp64" else rounded)
                    result["fp64_diagnostic"] = mathematical
                    result["rounded_contract"] = rounded
                    del actual
                except FloatingPointError as exc:
                    result = {"status": "numerical_failure", "exception": type(exc).__name__}
                except (RuntimeError, NotImplementedError) as exc:
                    # A selected CuTe failure is never classified as fallback.
                    result = {
                        "status": "execution_error" if backend == "kohakufa-cute" else "unsupported_or_error",
                        "exception": type(exc).__name__,
                    }
                row["backends"][backend] = result
            report["rows"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(str(dtype), case.name, {name: value["status"] for name, value in row["backends"].items()}, flush=True)
    failures = sum(row["backends"].get("kohakufa-cute", {}).get("status", "pass") != "pass" for row in report["rows"])
    report["cute_failures"] = failures
    for name, field in (("mathematical", "fp64_diagnostic"), ("contract", "rounded_contract")):
        report["cute_" + name + "_failures"] = sum(
            row["backends"].get("kohakufa-cute", {}).get(field, {}).get("status") == "numerical_failure"
            for row in report["rows"]
        )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
