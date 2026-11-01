"""Run unchanged attention gates on CPU copies while sanitizing GPU kernels."""

import argparse
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--agreement", action="store_true", help="Outside sanitizers, require both original CUDA and CPU gates."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    from benchmarks import validate_sm80_static as checks

    original_assess, original_statistics = checks.assess, checks.statistics
    report = {
        "target": args.target,
        "reference": "Unchanged FP64 and single-rounded equations on bit-preserving CPU copies, including actual O.",
        "scope": "GPU kernel execution and actual test operands unchanged; diagnostic GEMMs run on CPU.",
        "agreement": args.agreement,
        "assessments": [],
        "statistics": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def assess(actual, inputs, dout, **criteria):
        cuda = original_assess(actual, inputs, dout, **criteria) if args.agreement else None
        values = tuple(t.detach().cpu() for t in actual)
        operands = tuple(t.detach().cpu() for t in inputs)
        gradient = dout.detach().cpu()
        cpu = original_assess(values, operands, gradient, **criteria)
        record = {"shape": list(inputs[0].shape), "dtype": str(inputs[0].dtype), "criteria": criteria, "errors": cpu}
        if cuda is not None:
            differences = [
                abs((left[policy][metric] or 0) - (right[policy][metric] or 0))
                for left, right in zip(cpu, cuda)
                for policy in ("fp64", "rounded")
                for metric in ("relative_error", "max_absolute_error")
            ]
            record.update(both_original_gates_pass=True, maximum_error_metric_difference=max(differences))
        report["assessments"].append(record)
        save()
        return cpu

    def statistics(inputs, maximum, logsum, causal, scale):
        cuda = original_statistics(inputs, maximum, logsum, causal, scale) if args.agreement else None
        cpu = original_statistics(
            tuple(t.detach().cpu() for t in inputs), maximum.detach().cpu(), logsum.detach().cpu(), causal, scale
        )
        report["statistics"].append({"shape": list(inputs[0].shape), "cpu": cpu, "cuda": cuda})
        save()
        return cpu

    with patch.object(checks, "assess", assess), patch.object(checks, "statistics", statistics):
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName(args.target))
    report.update(
        status="pass" if result.wasSuccessful() and result.testsRun and not result.skipped else "error",
        tests_run=result.testsRun,
        skipped=len(result.skipped),
    )
    save()
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
