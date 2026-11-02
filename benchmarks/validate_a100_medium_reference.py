"""Audit CPU/device references on identical captured medium-attention operands."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from benchmarks.reference import fp64_reference, metrics, rounded_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    properties = torch.cuda.get_device_properties(0)
    assert (properties.major, properties.minor, properties.multi_processor_count) == (8, 0, 108)
    torch.set_num_threads(4)
    report = {
        "scope": "Independent CPU/device equations on bit-identical inputs and actual rounded O. FP32 reduction orders may differ before P/dS conversion; agreement uses the existing component norm limits.",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.inputs.glob("torch.*-*.pt")):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        values, dout, actual = saved["inputs"], saved["dout"], saved["actual"]
        limit = 0.01 if values[0].dtype == torch.bfloat16 else 0.005
        for head in range(values[0].shape[1]):
            part = [tensor[:, head : head + 1] for tensor in values]
            grad = dout[:, head : head + 1]
            output = actual[0][:, head : head + 1]
            cpu = (
                fp64_reference(part, grad, causal=True, scale=None),
                rounded_reference(part, grad, causal=True, scale=None, output=output),
            )
            device_part, device_grad = [tensor.cuda() for tensor in part], grad.cuda()
            device = (
                fp64_reference(device_part, device_grad, causal=True, scale=None),
                rounded_reference(device_part, device_grad, causal=True, scale=None, output=output.cuda()),
            )
            for policy, left, right in zip(("mathematical_fp64", "rounded_actual_o"), cpu, device):
                for component, a, b in zip(("out", "dq", "dk", "dv"), left, right):
                    comparison = metrics(a, b.cpu())
                    report["rows"].append(
                        {"case": path.name, "head": head, "component": component, "policy": policy, "agreement": comparison}
                    )
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    assert comparison["finite"]
                    if comparison["reference_norm"]:
                        assert comparison["relative_error"] < limit, report["rows"][-1]
        del saved, values, dout, actual
    assert len(report["rows"]) == 4 * 8 * 2 * 4, len(report["rows"])
    report["status"] = "pass"
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
