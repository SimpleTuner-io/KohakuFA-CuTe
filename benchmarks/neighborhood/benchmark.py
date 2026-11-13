"""Complete-call neighborhood attention timings with sparse accuracy probes.

Raw reports, failed operands and traces must be written outside the repository.
This compares complete implementations in the same process, not equal internal
precision policies or a complete HDiT application.
"""

import argparse
import contextlib
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from benchmarks.neighborhood.reference import neighborhoods
from benchmarks.reference import metrics


@dataclass(frozen=True)
class Workload:
    name: str
    batch: int
    heads: int
    spatial: tuple[int, int]
    window: tuple[int, int]
    dtype: str
    scale: float = 0.125

    @property
    def shape(self):
        return self.batch, self.heads, math.prod(self.spatial), 64


def workloads(suite="full"):
    geometries = [
        ("32sq-w7", 1, 8, (32, 32), (7, 7)),
        ("64sq-w7", 1, 8, (64, 64), (7, 7)),
        ("64sq-w13", 1, 8, (64, 64), (13, 13)),
        ("128sq-w7", 1, 8, (128, 128), (7, 7)),
        ("tail-33x49-w7", 2, 4, (33, 49), (7, 7)),
    ]
    if suite == "quick":
        geometries = [geometries[0], geometries[-1]]
    elif suite != "full":
        raise ValueError("Unknown workload suite.")
    return [
        Workload(name + "-" + dtype, b, h, spatial, window, dtype)
        for dtype in ("fp16", "bf16")
        for name, b, h, spatial, window in geometries
    ]


def sample_positions(spatial, count=12):
    height, width = spatial
    total = height * width
    preferred = [0, width - 1, (height - 1) * width, total - 1, total // 2, width, total - width - 1]
    preferred += [round(i * (total - 1) / max(1, count - 1)) for i in range(count)]
    return sorted(set(preferred))


@torch.no_grad()
def sparse_references(inputs, dout, output, spatial, window, *, scale=0.125, positions=None, chunk_size=64):
    """Independent local-window references at selected query and key owners.

    Legal windows come from independently enumerated placements. Inverse edges
    are found by membership in those forward lists, not production's inverse
    formula. Every query contributing to a selected key is included. Operands
    and actual rounded O live on CPU; no N-by-N mask is constructed.
    """
    q, k, v = [x.detach().cpu() for x in inputs]
    dout, output = dout.detach().cpu(), output.detach().cpu()
    edges = neighborhoods(spatial, window)
    selected = sample_positions(spatial) if positions is None else sorted(set(positions))
    if not selected or any(i < 0 or i >= len(edges) for i in selected):
        raise ValueError("Reference positions must be nonempty valid token indices.")
    query_slot = {query: slot for slot, query in enumerate(selected)}
    key_slot = torch.full((len(edges),), -1, dtype=torch.long)
    key_slot[selected] = torch.arange(len(selected))
    selected_set = set(selected)
    needed = [query for query, keys in enumerate(edges) if query in selected_set or selected_set.intersection(keys)]
    shape = (q.shape[0], q.shape[1], len(selected), q.shape[-1])
    result = {policy: [torch.zeros(shape, dtype=torch.float64) for _ in range(4)] for policy in ("fp64", "rounded")}
    return _sparse_equations(q, k, v, dout, output, edges, selected, needed, query_slot, key_slot, result, scale, chunk_size)


@torch.no_grad()
def _sparse_equations(q, k, v, dout, output, edges, selected, needed, query_slot, key_slot, result, scale, chunk_size):
    dtype = q.dtype
    for batch in range(q.shape[0]):
        for head in range(q.shape[1]):
            for first in range(0, len(needed), chunk_size):
                queries = needed[first : first + chunk_size]
                indices = torch.tensor([edges[index] for index in queries])
                qd = q[batch, head, queries].double()
                kd, vd = k[batch, head, indices].double(), v[batch, head, indices].double()
                grad = dout[batch, head, queries].double()
                exact_scores = torch.bmm(qd[:, None], kd.transpose(1, 2)).squeeze(1)
                rounded_scores = torch.bmm(qd.float()[:, None], kd.float().transpose(1, 2)).squeeze(1).double()
                exact_dp = torch.bmm(grad[:, None], vd.transpose(1, 2)).squeeze(1)
                rounded_dp = torch.bmm(grad.float()[:, None], vd.float().transpose(1, 2)).squeeze(1)
                delta = (output[batch, head, queries].float() * dout[batch, head, queries].float()).sum(-1, keepdim=True)
                slots = key_slot[indices]
                selected_edges = slots >= 0
                for policy, scores in (("fp64", exact_scores), ("rounded", rounded_scores)):
                    signed = scores if scale >= 0 else -scores
                    weights = torch.exp((signed - signed.amax(-1, keepdim=True)) * abs(scale))
                    probability = weights / weights.sum(-1, keepdim=True)
                    out = torch.bmm(probability[:, None], vd).squeeze(1)
                    if policy == "rounded":
                        probability = probability.float().to(dtype).float()
                        ds = (probability * (rounded_dp - delta)).to(dtype).double()
                        probability = probability.double()
                    else:
                        centered = exact_dp - exact_dp[:, :1]
                        ds = probability * (centered - (probability * centered).sum(-1, keepdim=True))
                    dq = torch.bmm(ds[:, None], kd).squeeze(1) * scale
                    target = result[policy]
                    for local, query in enumerate(queries):
                        if query in query_slot:
                            slot = query_slot[query]
                            target[0][batch, head, slot] = out[local]
                            target[1][batch, head, slot] = dq[local]
                    query_indices, neighbor_indices = selected_edges.nonzero(as_tuple=True)
                    destination = slots[query_indices, neighbor_indices]
                    target[2][batch, head].index_add_(
                        0, destination, ds[query_indices, neighbor_indices, None] * qd[query_indices] * scale
                    )
                    target[3][batch, head].index_add_(
                        0, destination, probability[query_indices, neighbor_indices, None] * grad[query_indices]
                    )
    return {"positions": selected, **{policy: tuple(values) for policy, values in result.items()}}


def assess(actual, references, dtype):
    limit = 0.005 if dtype == torch.float16 else 0.01
    positions = references["positions"]
    rows = []
    for component, got, exact, rounded in zip(("out", "dq", "dk", "dv"), actual, references["fp64"], references["rounded"]):
        got = got.detach().cpu()[:, :, positions]
        for batch in range(got.shape[0]):
            for head in range(got.shape[1]):
                mathematical = metrics(got[batch, head], exact[batch, head])
                contract = metrics(got[batch, head], rounded[batch, head])
                passed = mathematical["finite"] and (
                    mathematical["relative_error"] < limit
                    if mathematical["reference_norm"]
                    else mathematical["max_absolute_error"] == 0
                )
                rows.append(
                    dict(component=component, batch=batch, head=head, passed=passed, fp64=mathematical, rounded=contract)
                )
    return rows


def summarize(values):
    if not values:
        raise ValueError("Cannot summarize no samples.")
    return dict(
        median=statistics.median(values),
        minimum=min(values),
        maximum=max(values),
        mean=statistics.mean(values),
        stdev=statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def paired_order(names, repeat):
    names = list(names)
    if not names:
        return []
    offset = repeat % len(names)
    order = names[offset:] + names[:offset]
    return order if repeat % 2 == 0 else list(reversed(order))


def snapshot_gpu():
    fields = "name,uuid,driver_version,power.limit,power.draw,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,memory.used,memory.total"
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False
    )
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    return dict(
        utc=datetime.datetime.now(datetime.UTC).isoformat(),
        query_fields=fields.split(","),
        rows=result.stdout.strip().splitlines(),
        exit_status=result.returncode,
        compute_processes=processes.stdout.strip().splitlines(),
        process_query_exit_status=processes.returncode,
    )


def execute(function, inputs, dout, training):
    if training:
        out = function(*inputs)
        return out, *(gradient.contiguous() for gradient in torch.autograd.grad(out, inputs, dout))
    with torch.no_grad():
        return (function(*inputs),)


class Execution:
    def __init__(self, backend, inputs, dout, phase, mode, stream, capture_batch):
        self.backend, self.inputs, self.dout = backend, inputs, dout
        self.training = phase == "forward_backward"
        self.mode, self.stream = mode, stream
        self.batch = capture_batch if mode == "graph" else 1
        self.output = None
        self.graph = None
        with torch.cuda.stream(stream):
            for _ in range(5):
                self.eager()
            stream.synchronize()
            if mode == "graph":
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph, stream=stream):
                    for _ in range(self.batch):
                        self.eager()
                self.graph.replay()
        stream.synchronize()

    def eager(self):
        self.output = execute(self.backend.call, self.inputs, self.dout, self.training)
        return self.output

    def call(self):
        if self.graph is None:
            self.eager()
        else:
            self.graph.replay()

    def sample(self, repeats):
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        with torch.cuda.stream(self.stream):
            self.stream.synchronize()
            wall_start = time.perf_counter()
            start.record()
            for _ in range(repeats):
                self.call()
            end.record()
            end.synchronize()
            wall = time.perf_counter() - wall_start
        calls = repeats * self.batch
        return dict(
            device_ms=start.elapsed_time(end) / calls,
            wall_ms=wall * 1000 / calls,
            graph_replays=repeats if self.graph else None,
            complete_calls=calls,
        )

    def calibrate(self, target_ms):
        probe = self.sample(3)
        per_replay = max(probe["device_ms"] * self.batch, 0.001)
        return min(10000, max(1, math.ceil(target_ms / per_replay)))

    def changed_inputs(self):
        if self.graph is None:
            return None
        with torch.cuda.stream(self.stream):
            saved = [x.detach().clone() for x in (*self.inputs, self.dout)]
            before = [x.detach().clone() for x in self.output]
        try:
            with torch.no_grad(), torch.cuda.stream(self.stream):
                for index, tensor in enumerate((*self.inputs, self.dout)):
                    tensor.mul_(0.5).add_(0.03125 * (index + 1))
                self.graph.replay()
            self.stream.synchronize()
            with torch.cuda.stream(self.stream):
                got = tuple(x.detach().clone() for x in self.output)
                expected = execute(self.backend.call, self.inputs, self.dout, self.training)
            self.stream.synchronize()
            errors = []
            for component, actual, want in zip(("out", "dq", "dk", "dv"), got, expected):
                actual, want = actual.cpu(), want.cpu()
                for batch in range(actual.shape[0]):
                    for head in range(actual.shape[1]):
                        errors.append(
                            dict(
                                component=component,
                                batch=batch,
                                head=head,
                                **metrics(actual[batch, head], want[batch, head]),
                            )
                        )
            limit = 0.005 if self.inputs[0].dtype == torch.float16 else 0.01
            valid = all(
                error["finite"]
                and (error["relative_error"] < limit if error["reference_norm"] else error["max_absolute_error"] == 0)
                for error in errors
            )
            changed = bool(torch.any(got[0] != before[0]))
            if not (valid and changed):
                self.failed_probe = dict(
                    inputs=[tensor.detach().cpu() for tensor in self.inputs],
                    dout=self.dout.cpu(),
                    actual=[tensor.cpu() for tensor in got],
                    expected=[tensor.cpu() for tensor in expected],
                )
            return dict(passed=valid and changed, changed_output=changed, errors=errors)
        finally:
            with torch.no_grad(), torch.cuda.stream(self.stream):
                for tensor, original in zip((*self.inputs, self.dout), saved):
                    tensor.copy_(original)
                self.graph.replay()
            self.stream.synchronize()


def profile_execution(execution, path):
    execution.stream.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as capture:
        with torch.cuda.stream(execution.stream):
            execution.call()
        execution.stream.synchronize()
    capture.export_chrome_trace(str(path))
    kernels = [event.name for event in capture.events() if event.device_type == torch.autograd.DeviceType.CUDA]
    return dict(trace_file=path.name, launched_cuda_kernels=kernels, complete_calls=execution.batch)


def atomic_save(path, report):
    serializable = json.loads(json.dumps(report), parse_constant=lambda value: None)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(serializable, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def error_record(error):
    return dict(type=type(error).__name__, message=str(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=["quick", "full"], default="full")
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--target-ms", type=float, default=100.0)
    parser.add_argument("--capture-batch", type=int, default=8)
    parser.add_argument(
        "--backends", nargs="+", choices=["kohakufa", "flex", "natten"], default=["kohakufa", "flex", "natten"]
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--case", action="append", help="Run only named cases, preserving immutable resume boundaries.")
    args = parser.parse_args()
    if args.samples < 3 or args.target_ms <= 0 or args.capture_batch < 1:
        parser.error("Need at least three samples, positive target time and capture batch.")
    root = Path(__file__).resolve().parents[2]
    if args.output.resolve().is_relative_to(root):
        parser.error("Write raw artifacts outside the repository.")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if revision != args.source_revision:
        parser.error("Source base revision does not match.")
    if not torch.cuda.is_available():
        parser.error("CUDA is required; skipped work is not a performance result.")
    from benchmarks.neighborhood.backends import make_backend

    selected = [case for case in workloads(args.suite) if args.case is None or case.name in args.case]
    if not selected or args.case is not None and set(args.case) - {case.name for case in selected}:
        parser.error("No workloads selected, or an unknown case was requested.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch._dynamo.config.recompile_limit = 128
    torch._dynamo.config.accumulated_recompile_limit = 512
    properties = torch.cuda.get_device_properties(0)
    sources = sorted((root / "src/kohakufa_cute").rglob("*.py")) + [
        Path(__file__),
        root / "benchmarks/neighborhood/backends.py",
        root / "benchmarks/neighborhood/reference.py",
        root / "benchmarks/reference.py",
        root / "docs/precision.md",
    ]
    packages = ["torch", "nvidia-cutlass-dsl", "apache-tvm-ffi", "natten", "triton", "cuda-bindings"]
    versions = {}
    for package in packages:
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            versions[package] = importlib.metadata.version(package)
    report = dict(
        status="running",
        source_revision=revision,
        source_sha256={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        versions=versions,
        python=os.sys.version,
        device=dict(
            name=properties.name,
            capability=list(torch.cuda.get_device_capability()),
            sm_count=properties.multi_processor_count,
            memory_bytes=properties.total_memory,
        ),
        protocol=dict(
            scope="Synthetic D64 2D stride-one dilation-one neighborhood attention; complete implementation comparison, no HDiT application claim.",
            canonical_layout="contiguous BHSD Q/K/V/output/dQ/dK/dV",
            preparation="Geometry and BlockMask preparation measured separately; resident timings reuse immutable geometry. Mandatory tensor layout conversions remain inside every call.",
            phases=["forward", "forward_backward"],
            modes=["eager", "graph"],
            samples=args.samples,
            dynamo_recompile_limit=128,
            dynamo_accumulated_recompile_limit=512,
            target_ms=args.target_ms,
            capture_batch=args.capture_batch,
            sample_order="rotating and alternating backend order each paired round",
            accuracy="Independent sparse per-head FP64 probes for output/dQ/dK/dV, including every contributing query for sampled key gradients; rounded-reference diagnostics use actual O.",
            fp16_limit=0.005,
            bf16_limit=0.01,
            graph_validation="Changed Q/K/V/dO data compared against fresh same-backend eager execution, restored before timings",
            profiling="Separate from uninstrumented timings; per-case eager and graph traces",
            runtime_stack="Same process and packages across backends; internal precision policies differ.",
        ),
        gpu_observations=[snapshot_gpu()],
        rows=[],
    )
    atomic_save(args.output, report)
    any_failure = False
    for case_index, case in enumerate(selected):
        seed = 809 + workloads().index(case)
        torch.manual_seed(seed)
        dtype = torch.float16 if case.dtype == "fp16" else torch.bfloat16
        inputs = [torch.randn(case.shape, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)]
        dout = torch.randn_like(inputs[0])
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        active = {}
        case_rows = {}
        for name in args.backends:
            actual = None
            row = dict(
                workload=asdict(case),
                seed=seed,
                workload_sha256=hashlib.sha256(json.dumps(asdict(case), sort_keys=True).encode()).hexdigest(),
                shape=list(case.shape),
                backend=name,
                status="preparing",
                timings={},
                accuracy=None,
                graph_checks={},
                profiles={},
            )
            report["rows"].append(row)
            case_rows[name] = row
            atomic_save(args.output, report)
            try:
                torch.cuda.synchronize()
                started = time.perf_counter()
                backend = make_backend(name, spatial=case.spatial, window=case.window, scale=case.scale, device="cuda")
                torch.cuda.synchronize()
                row["construction_wall_ms"] = (time.perf_counter() - started) * 1000
                first_call = time.perf_counter()
                with torch.cuda.stream(stream):
                    actual = execute(backend.call, inputs, dout, True)
                stream.synchronize()
                row["first_forward_backward_wall_ms"] = (time.perf_counter() - first_call) * 1000
                preparation_samples = []
                for _ in range(3):
                    backend.prepare()
                    preparation_samples.append(
                        dict(
                            host_ms=backend.metadata["last_preparation_host_ms"],
                            device_interval_ms=backend.metadata.get("last_preparation_device_ms"),
                        )
                    )
                torch.cuda.synchronize()
                row["preparation"] = dict(
                    samples=preparation_samples,
                    host_ms=summarize([sample["host_ms"] for sample in preparation_samples]),
                    device_interval_ms=summarize([sample["device_interval_ms"] for sample in preparation_samples]),
                    boundary="Device event interval includes host builder enqueue gaps; host wall time includes complete immutable geometry preparation.",
                )
                if not all(bool(torch.isfinite(value).all()) for value in actual):
                    raise ArithmeticError("Nonfinite output or gradients.")
                references = sparse_references(inputs, dout, actual[0], case.spatial, case.window, scale=case.scale)
                errors = assess(actual, references, dtype)
                row["accuracy"] = dict(
                    passed=all(error["passed"] for error in errors), positions=references["positions"], errors=errors
                )
                row["backend_metadata"] = backend.metadata
                if not row["accuracy"]["passed"]:
                    raise ArithmeticError("Independent sampled FP64 accuracy gate failed.")
                active[name] = backend
                row["status"] = "ready"
            except Exception as error:
                row.update(status="qualification-error", error=error_record(error))
                filename = f"{case.name}-{name}-failure.pt"
                try:
                    torch.save(
                        dict(
                            inputs=[x.detach().cpu() for x in inputs],
                            dout=dout.cpu(),
                            actual=None if actual is None else [tensor.detach().cpu() for tensor in actual],
                            workload=asdict(case),
                        ),
                        args.output.parent / filename,
                    )
                    row["failed_operands"] = filename
                except Exception as retention_error:
                    row["operand_retention_error"] = error_record(retention_error)
                any_failure = True
            atomic_save(args.output, report)
        for phase in ("forward", "forward_backward"):
            for mode in ("eager", "graph"):
                executions = {}
                for name, backend in active.items():
                    row = case_rows[name]
                    key = phase + "/" + mode
                    timing = dict(status="preparing", samples=[])
                    row["timings"][key] = timing
                    atomic_save(args.output, report)
                    try:
                        started = time.perf_counter()
                        execution = Execution(backend, inputs, dout, phase, mode, stream, args.capture_batch)
                        timing["warm_and_capture_wall_ms"] = (time.perf_counter() - started) * 1000
                        if mode == "graph":
                            graph_check = execution.changed_inputs()
                            row["graph_checks"][phase] = graph_check
                            if not graph_check["passed"]:
                                failure_name = f"{case.name}-{name}-{phase}-graph-failure.pt"
                                torch.save(execution.failed_probe, args.output.parent / failure_name)
                                graph_check["failed_operands"] = failure_name
                                raise ArithmeticError("Graph changed-input validation failed.")
                        count = execution.calibrate(args.target_ms)
                        for _ in range(3):
                            execution.sample(count)
                        executions[name] = execution, count
                        timing.update(status="sampling", calls_per_graph=execution.batch, repeats_per_sample=count)
                    except Exception as error:
                        timing.update(status="execution-error", error=error_record(error))
                        any_failure = True
                    atomic_save(args.output, report)
                for repeat in range(args.samples):
                    for name in paired_order(executions, repeat) if executions else ():
                        execution, count = executions[name]
                        timing = case_rows[name]["timings"][phase + "/" + mode]
                        try:
                            timing["samples"].append(dict(round=repeat, **execution.sample(count)))
                        except Exception as error:
                            timing.update(status="execution-error", error=error_record(error))
                            any_failure = True
                        atomic_save(args.output, report)
                for name, (execution, count) in executions.items():
                    timing = case_rows[name]["timings"][phase + "/" + mode]
                    if timing["status"] != "execution-error":
                        timing.update(
                            status="pass",
                            device_ms=summarize([sample["device_ms"] for sample in timing["samples"]]),
                            wall_ms=summarize([sample["wall_ms"] for sample in timing["samples"]]),
                        )
                    if args.profile:
                        try:
                            path = args.output.parent / f"{case.name}-{name}-{phase}-{mode}.trace.json"
                            case_rows[name]["profiles"][phase + "/" + mode] = profile_execution(execution, path)
                        except Exception as error:
                            case_rows[name]["profiles"][phase + "/" + mode] = dict(
                                status="profile-error", error=error_record(error)
                            )
                            any_failure = True
                    atomic_save(args.output, report)
                del executions
                execution = None
                torch.cuda.synchronize()
        for name, row in case_rows.items():
            if row["status"] == "ready":
                row["status"] = (
                    "pass" if all(value["status"] == "pass" for value in row["timings"].values()) else "partial-failure"
                )
        report["gpu_observations"].append(snapshot_gpu())
        atomic_save(args.output, report)
        del active, inputs, dout
        backend = actual = references = None
        torch.cuda.empty_cache()
    report["status"] = "completed-with-failures" if any_failure else "pass"
    report["finished_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
    atomic_save(args.output, report)
    raise SystemExit(1 if any_failure else 0)


if __name__ == "__main__":
    main()
