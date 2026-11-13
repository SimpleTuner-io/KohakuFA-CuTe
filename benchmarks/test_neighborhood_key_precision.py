"""Qualify residual-centered dK and measure its complete training-call cost.

Named controls retain their equations; the combined method is now the public
default. Historical fixtures reuse retained operands
when supplied and must match their previous hashes. Every mathematical output
and gradient remains visible, independently of prescribed-equation agreement.
"""

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import torch

from benchmarks import test_neighborhood_precision as previous
from benchmarks.benchmark_neighborhood import atomic_save, snapshot_gpu
from benchmarks.neighborhood_backends import make_backend
from benchmarks.neighborhood_key_precision_reference import METHODS, NEW_METHODS, key_precision_reference, key_terms
from benchmarks.neighborhood_precision import variant
from benchmarks.neighborhood_precision_reference import component_metrics, mathematical_reference, reference_gate
from benchmarks.qualify_neighborhood import execute
from benchmarks.qualify_neighborhood_accuracy import tensor_hash

COMPONENTS = ("out", "dq", "dk", "dv")


def context(name):
    return variant(name) if name in METHODS else contextlib.nullcontext()


def _generated_fixtures():
    """Describe the qualified 29 fixtures, including their original seeds."""
    yield from previous.fixtures()
    for dtype in (torch.float16, torch.bfloat16):
        for scale in (0.125, -0.125):
            torch.manual_seed(180)
            inputs = [torch.randn(1, 2, 45, 64, dtype=dtype, device="cuda") for _ in range(3)]
            inputs[0].mul_(16)
            inputs[1].mul_(16)
            inputs[2].fill_(1)
            inputs[2][..., 32:] = -1
            yield "zero-dp-sharp", inputs, torch.ones_like(inputs[0]), (5, 9), (3, 3), scale, True, 180
            constant = [value.clone() for value in inputs]
            constant[2].fill_(1)
            yield "constant-dp-sharp", constant, torch.full_like(inputs[0], 1 / 64), (5, 9), (3, 3), scale, True, 180


def fixtures(frozen_operands=None):
    """Reuse every retained fixture, including all eight seed-180 controls.

    Device RNG and conversion differences must not substitute different inputs
    during cross-architecture qualification. Generators establish fixture
    identity; the frozen CPU payload replaces all input and dO values.
    """
    for index, fixture in enumerate(_generated_fixtures()):
        label, inputs, dout, spatial, window, scale, stress, seed = fixture
        if frozen_operands is not None:
            artifact = frozen_operands / f"report-fixture-{index}.pt"
            retained = torch.load(artifact, map_location="cpu", weights_only=True)
            if (
                tuple(retained["spatial"]) != tuple(spatial)
                or tuple(retained["window"]) != tuple(window)
                or retained["scale"] != scale
                or len(retained["inputs"]) != len(inputs)
                or any(a.shape != b.shape or a.dtype != b.dtype for a, b in zip(retained["inputs"], inputs))
                or retained["dout"].shape != dout.shape
                or retained["dout"].dtype != dout.dtype
            ):
                raise ValueError("Retained operand geometry or dtype does not match its fixture identity.")
            inputs = [value.cuda() for value in retained["inputs"]]
            dout = retained["dout"].cuda()
        yield label, inputs, dout, spatial, window, scale, stress, seed


def historical_hashes(path):
    report = json.loads(path.read_text())
    result = {}
    for row in report["directed"]:
        index, value = row["fixture_index"], row["operand_sha256"]
        if index in result and result[index] != value:
            raise ValueError("Historical fixture hashes disagree across implementations.")
        result[index] = value
    if set(result) != set(range(29)):
        raise ValueError("Cross-architecture qualification requires all 29 previously qualified fixture identities.")
    return result


def selected_gate(actual, expected, dtype, components=COMPONENTS):
    gate = reference_gate(actual, expected, dtype)
    gate["required_components"] = list(components)
    gate["passed"] = all(record["passed"] for record in gate["errors"] if record["component"] in components)
    return gate


def required_math_components(method, stress, fixture=None):
    if fixture in ("zero-dp-sharp", "constant-dp-sharp") or method == "reconstructed_dk" and stress:
        # This explicit control keeps the old unprojected query method. Its
        # known stress dQ discrepancy remains reported; the combined candidate
        # is the stronger all-gradient target and must pass all four gates.
        return ("out", "dk", "dv")
    return COMPONENTS


def unchanged_components(actual, baseline, old_combined, method):
    checks = [
        dict(
            component="out",
            against="baseline",
            passed=torch.equal(actual[0].view(torch.uint8), baseline[0].view(torch.uint8)),
        ),
        dict(
            component="dv",
            against="baseline",
            passed=torch.equal(actual[3].view(torch.uint8), baseline[3].view(torch.uint8)),
        ),
    ]
    if method == "weighted_dq_reconstructed_dk":
        checks.append(
            dict(
                component="dq",
                against="fp32_p_weighted_dq",
                passed=torch.equal(actual[1].view(torch.uint8), old_combined[1].view(torch.uint8)),
            )
        )
    return dict(passed=all(record["passed"] for record in checks), checks=checks)


def directed(args, report):
    failures = 0
    expected_hashes = historical_hashes(args.previous_report)
    for index, fixture in enumerate(fixtures(args.frozen_operands)):
        label, inputs, dout, spatial, window, scale, stress, seed = fixture
        if args.fixture and label not in args.fixture:
            continue
        cpu, gradient = [x.detach().cpu() for x in inputs], dout.cpu()
        operand_hash = tensor_hash([*cpu, gradient])
        if operand_hash != expected_hashes.get(index):
            raise ValueError("Historical input hash changed; do not substitute a different numerical fixture.")
        artifact = args.output.parent / f"{args.output.stem}-fixture-{index}.pt"
        torch.save(dict(inputs=cpu, dout=gradient, spatial=spatial, window=window, scale=scale, seed=seed), artifact)
        exact = mathematical_reference(cpu, gradient, spatial, window, scale)
        baseline, old_combined = None, None
        for name in list(dict.fromkeys(["baseline", "fp32_p_weighted_dq", *args.variants, *args.backends])):
            row = dict(
                stage="directed",
                fixture=label,
                fixture_index=index,
                implementation=name,
                dtype=str(inputs[0].dtype),
                spatial=spatial,
                window=window,
                scale=scale,
                directed=stress,
                seed=seed,
                operand_artifact=artifact.name,
                operand_sha256=operand_hash,
                historical_operand_verified=True,
                status="running",
            )
            report["rows"].append(row)
            atomic_save(args.output, report)
            try:
                backend = make_backend("kohakufa" if name in METHODS else name, spatial, window, scale, device="cuda")
                with context(name):
                    actual = execute(backend.call, inputs, dout)
                torch.cuda.synchronize()
                actual = tuple(x.detach().cpu().contiguous() for x in actual)
                actual_path = args.output.parent / f"{args.output.stem}-fixture-{index}-{name}.pt"
                torch.save(dict(actual=actual, operand_artifact=artifact.name), actual_path)
                row.update(
                    actual_artifact=actual_path.name,
                    backend_metadata=backend.metadata,
                    fp64=component_metrics(actual, exact),
                )
                if name == "baseline":
                    baseline = actual
                elif name == "fp32_p_weighted_dq":
                    old_combined = actual
                if name in METHODS:
                    expected = key_precision_reference(cpu, gradient, spatial, window, scale, actual[0], method=name)
                    row["formula"] = reference_gate(actual, expected, inputs[0].dtype)
                    row["math_gate"] = selected_gate(
                        actual, exact, inputs[0].dtype, required_math_components(name, stress, label)
                    )
                    required_math = name in NEW_METHODS or not stress
                    row["math_gate"]["required_for_qualification"] = required_math
                    row["paired_invariants"] = unchanged_components(actual, baseline, old_combined, name)
                    passed = row["formula"]["passed"] and row["paired_invariants"]["passed"]
                    passed &= row["math_gate"]["passed"] if required_math else True
                    row["status"] = "pass" if passed else "qualification-failure"
                    failures += not passed
                    if name in NEW_METHODS:
                        terms = key_terms(cpu, gradient, spatial, window, scale, actual[0])
                        row["correction_observations"] = dict(
                            max_abs_correction=float(terms["correction"].abs().max()),
                            min_probability_mass=float(terms["mass"].min()),
                            max_probability_mass=float(terms["mass"].max()),
                        )
                else:
                    row["status"] = "diagnostic-complete"
            except Exception as error:
                row.update(status="execution-error", error_type=type(error).__name__, error=str(error))
                failures += name in METHODS
            atomic_save(args.output, report)
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("directed", "timing", "all"), default="all")
    parser.add_argument("--variants", choices=METHODS, nargs="+", default=list(METHODS))
    parser.add_argument("--backends", choices=("flex", "natten"), nargs="*", default=[])
    parser.add_argument("--previous-report", type=Path, default=Path("docs/experimental/neighborhood-key-precision.json"))
    parser.add_argument("--frozen-operands", type=Path)
    parser.add_argument("--fixture", action="append")
    parser.add_argument("--case", action="append")
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--target-ms", type=float, default=100)
    parser.add_argument("--capture-batch", type=int, default=8)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.resolve().is_relative_to(root):
        parser.error("Keep raw operands and reports outside the repository.")
    if args.samples < 3 or args.target_ms <= 0 or args.capture_batch < 1:
        parser.error("Need three samples, positive target and capture batch.")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if revision != args.source_revision:
        parser.error("Source base revision does not match frozen snapshot.")
    if not torch.cuda.is_available():
        parser.error("CUDA is required; a skipped run is not qualification.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch._dynamo.config.recompile_limit = 128
    torch._dynamo.config.accumulated_recompile_limit = 512
    sources = (
        sorted((root / "src/kohakufa_cute").rglob("*.py"))
        + list((root / "benchmarks").glob("*neighborhood*.py"))
        + [root / "benchmarks/reference.py", root / "docs/precision.md"]
    )
    versions = {}
    for package in ("torch", "nvidia-cutlass-dsl", "apache-tvm-ffi", "natten", "triton", "cuda-bindings"):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            versions[package] = importlib.metadata.version(package)
    properties = torch.cuda.get_device_properties(0)
    report = dict(
        schema="neighborhood-key-precision-v1",
        status="running",
        source_revision=revision,
        previous_report_sha256=hashlib.sha256(args.previous_report.read_bytes()).hexdigest(),
        source_sha256={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        versions=versions,
        cuda=torch.version.cuda,
        device=dict(
            name=properties.name,
            capability=list(torch.cuda.get_device_capability()),
            sm_count=properties.multi_processor_count,
            memory_bytes=properties.total_memory,
        ),
        protocol=dict(
            fixed_inputs="All 29 fixture Q/K/V/dO hashes must match the previous H100 dQ+dK qualification report.",
            reference="Native mathematical FP64 on identical retained rounded operands; independent FP32 policy reference has its own reduction-order floor.",
            formula="t=sum_FP32(P32*(dP32-delta_actualO))/sum_FP32(P32); dK uses LP[P32*((dP32-delta_actualO)-t)] and post-product scale.",
            target="New combined method must pass all four mathematical component gates on ordinary and historical directed fixtures. New zero/constant-dP controls target O/dK/dV mathematical accuracy while preserving and reporting the previous dQ policy. Unprojected-query dK control reports dQ stress error without claiming to repair it.",
            fp16_limit=0.005,
            bf16_limit=0.01,
            timing="Warmed complete resident forward plus backward CUDA graph device-event samples; immutable geometry preparation separate, required transforms included.",
            samples=args.samples,
            target_ms=args.target_ms,
            capture_batch=args.capture_batch,
        ),
        gpu_observations=[snapshot_gpu()],
        rows=[],
    )
    atomic_save(args.output, report)
    failures = 0
    try:
        if args.stage in ("directed", "all"):
            failures += directed(args, report)
        if args.stage in ("timing", "all"):
            # Reuse the already-qualified timing harness without extending its
            # old policy matrix or changing the old GPU tests' variant list.
            with patch.object(previous, "VARIANTS", tuple(dict.fromkeys([*previous.VARIANTS, *METHODS]))):
                failures += previous.timing(args, report)
        report.update(status="pass" if failures == 0 else "qualification-failure", candidate_failures=failures)
    except BaseException as error:
        report.update(status="execution-error", error_type=type(error).__name__, error=str(error))
        atomic_save(args.output, report)
        raise
    atomic_save(args.output, report)
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
