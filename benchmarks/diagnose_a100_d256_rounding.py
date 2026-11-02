"""Preserved D256 operands: audit independent reference reduction policies."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from benchmarks.reference import fp64_reference, metrics, rounded_reference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    data = torch.load(args.input, map_location="cpu", weights_only=False)
    inputs, dout, actual = data["inputs"], data["dout"], data["actual"]
    causal, scale = data["case"]["causal"], data["case"]["scale"]
    scale = inputs[0].shape[-1] ** -0.5 if scale is None else scale
    gpu = tuple(t.cuda() for t in inputs)
    grad, observed = dout.cuda(), tuple(t.cuda() for t in actual)
    report = {"input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(), "case": data["case"], "heads": []}

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def intermediates(values, g, output):
        q, k, v = values
        old = torch.backends.cuda.matmul.fp32_precision
        try:
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            s = q.float() @ k.float().transpose(-1, -2)
            dp = g.float() @ v.float().transpose(-1, -2)
        finally:
            torch.backends.cuda.matmul.fp32_precision = old
        delta = (output.float() * g.float()).sum(-1, keepdim=True)
        return s, dp, delta

    def products(values, g, intermediate):
        q, k, v = (t.double() for t in values)
        s, dp, delta = intermediate
        shifted = s.double()
        if causal:
            visible = torch.ones(shifted.shape[-2:], device=s.device, dtype=torch.bool).tril()
            shifted = shifted.masked_fill(~visible, -torch.inf)
        maximum = shifted.amax(-1, keepdim=True)
        p = ((shifted - maximum) * scale).exp()
        p = (p / p.sum(-1, keepdim=True)).float().to(values[0].dtype).float()
        ds = (p * (dp - delta)).to(values[0].dtype)
        result = (
            (ds.double() @ k) * scale,
            (ds.double().transpose(-1, -2) @ q) * scale,
            p.double().transpose(-1, -2) @ g.double(),
        )
        return p, ds, result

    for head in range(inputs[0].shape[1]):
        c = tuple(t[:, head : head + 1] for t in inputs)
        g = tuple(t[:, head : head + 1] for t in gpu)
        cd, gd = dout[:, head : head + 1], grad[:, head : head + 1]
        co, go = actual[0][:, head : head + 1], observed[0][:, head : head + 1]
        cm, gm = intermediates(c, cd, co), intermediates(g, gd, go)
        cp, cs, cr = products(c, cd, cm)
        gp, gs, gr = products(g, gd, gm)
        cpu_fixed = products(c, cd, tuple(t.cpu() for t in gm))
        torch.save(
            {"cpu": cm, "gpu": tuple(t.cpu() for t in gm), "cpu_p": cp, "gpu_p": gp.cpu(), "cpu_ds": cs, "gpu_ds": gs.cpu()},
            args.output.with_name(args.output.stem + f".head-{head}-intermediates.pt"),
        )
        exact_scores = c[0].double() @ c[1].double().transpose(-1, -2)
        exact_dp = cd.double() @ c[2].double().transpose(-1, -2)
        row = {"head": head, "intermediate_cpu_gpu": {}, "reference_errors": {}, "common_gpu_intermediates_cpu_gpu": {}}
        for name, ca, ga in zip(("scores", "dp", "delta", "p", "ds"), (*cm, cp, cs), (*gm, gp, gs)):
            row["intermediate_cpu_gpu"][name] = {**metrics(ca, ga.cpu()), "different_elements": int((ca != ga.cpu()).sum())}
        row["intermediate_fp64_audit"] = {
            "cpu_scores": metrics(cm[0], exact_scores),
            "gpu_scores": metrics(gm[0].cpu(), exact_scores),
            "cpu_dp": metrics(cm[1], exact_dp),
            "gpu_dp": metrics(gm[1].cpu(), exact_dp),
        }
        for name, ref in (("cpu", cr), ("gpu", gr), ("common_gpu_intermediates_cpu", cpu_fixed[2])):
            row["reference_errors"][name] = {
                label: metrics(t[:, head : head + 1], r.cpu()) for label, t, r in zip(("dq", "dk", "dv"), actual[1:], ref)
            }
        for name, a, b in zip(("p", "ds", "dq", "dk", "dv"), (*cpu_fixed[:2], *cpu_fixed[2]), (gp, gs, *gr)):
            row["common_gpu_intermediates_cpu_gpu"][name] = {
                **metrics(a, b.cpu()),
                "different_elements": int((a != b.cpu()).sum()),
            }
        row["mathematical_errors"] = {
            label: metrics(t[:, head : head + 1], r)
            for label, t, r in zip(("out", "dq", "dk", "dv"), actual, fp64_reference(c, cd, causal=causal, scale=scale))
        }
        row["original_rounded_reference"] = {
            backend: {
                label: metrics(t[:, head : head + 1], r.cpu())
                for label, t, r in zip(
                    ("out", "dq", "dk", "dv"), actual, rounded_reference(v, d, causal=causal, scale=scale, output=o)
                )
            }
            for backend, v, d, o in (("cpu", c, cd, co), ("gpu", g, gd, go))
        }
        saved_max, saved_norm = data["saved"][-2][..., head : head + 1, :], data["saved"][-1][..., head : head + 1, :]
        for name, m in (("cpu", cm), ("gpu", gm)):
            scores = m[0].double()
            if causal:
                scores = scores.masked_fill(
                    ~torch.ones(scores.shape[-2:], device=scores.device, dtype=torch.bool).tril(), -torch.inf
                )
            maximum = scores.amax(-1)
            norm = torch.logsumexp((scores - maximum[..., None]) * scale, -1) / torch.log(
                torch.tensor(2.0, device=scores.device, dtype=torch.float64)
            )
            row.setdefault("saved_statistics_audit", {})[name] = {
                "maximum_absolute_error": float((saved_max - maximum.cpu()).abs().max()),
                "lognorm_absolute_error": float((saved_norm - norm.cpu()).abs().max()),
            }
        report["heads"].append(row)
        save()
    torch.save(
        {"cpu": cm, "gpu": tuple(t.cpu() for t in gm), "cpu_p": cp, "gpu_p": gp.cpu(), "cpu_ds": cs, "gpu_ds": gs.cpu()},
        args.output.with_suffix(".last-head-intermediates.pt"),
    )
    report["status"] = "complete_diagnostic_no_gate_change"
    save()


if __name__ == "__main__":
    main()
