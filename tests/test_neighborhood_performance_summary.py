"""Publication-safe performance comparisons preserve measurement boundaries."""

import copy
import hashlib
import json
import unittest

from benchmarks.neighborhood.summarize_performance import markdown, summarize_inputs


def report():
    workload = dict(name="32sq-w7-fp16", batch=1, heads=1, spatial=[32, 32], window=[7, 7], dtype="fp16", scale=0.125)
    digest = hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest()
    result = dict(
        status="pass",
        source_revision="a" * 40,
        source_sha256={"benchmarks/neighborhood/benchmark.py": "b" * 64},
        versions={"torch": "2.13.0+cu130"},
        device=dict(name="NVIDIA H100 PCIe", capability=[9, 0], sm_count=114, memory_bytes=80 * 1024**3),
        protocol=dict(
            canonical_layout="contiguous BHSD Q/K/V/output/dQ/dK/dV",
            samples=3,
            capture_batch=8,
            target_ms=100,
            fp16_limit=0.005,
            bf16_limit=0.01,
        ),
        gpu_observations=[dict(uuid="PRIVATE_DEVICE_UUID", process="/private/remote/process")],
        rows=[],
    )
    for backend, duration in (("kohakufa", 0.1), ("flex", 0.2), ("natten", 0.05)):
        accuracy = dict(
            passed=True,
            positions=[0, 1023],
            errors=[
                dict(
                    component=component,
                    batch=0,
                    head=0,
                    passed=True,
                    fp64=dict(relative_error=0.001),
                    rounded=dict(relative_error=0.0001),
                )
                for component in ("out", "dq", "dk", "dv")
            ],
        )
        row = dict(
            backend=backend,
            workload=workload,
            shape=[1, 1, 1024, 64],
            seed=809,
            workload_sha256=digest,
            status="pass",
            accuracy=accuracy,
            timings={},
            graph_checks={},
            profiles={},
            backend_metadata=dict(
                selected_backend="generic warp MMA" if backend == "kohakufa" else "cutlass-fna", preparation_storage_bytes=0
            ),
        )
        for phase in ("forward", "forward_backward"):
            row["graph_checks"][phase] = dict(passed=True, message="/private/secret/graph")
            for mode in ("eager", "graph"):
                key = phase + "/" + mode
                samples = [
                    dict(
                        round=index,
                        device_ms=duration * multiplier,
                        wall_ms=duration * multiplier * 1.1,
                        complete_calls=24 if mode == "graph" else 10,
                        graph_replays=3 if mode == "graph" else None,
                    )
                    for index, multiplier in enumerate((0.9, 1.0, 1.1))
                ]
                row["timings"][key] = dict(
                    status="pass", samples=samples, device_ms=dict(median=duration), wall_ms=dict(median=duration * 1.1)
                )
                row["profiles"][key] = dict(
                    launched_cuda_kernels=["NeighborhoodAttention_object_at_PRIVATE_POINTER"],
                    trace_file="/private/remote/trace.json",
                )
        result["rows"].append(row)
    return result


def compact(data):
    return summarize_inputs([("h100", json.dumps(data).encode())])


class NeighborhoodPerformanceSummaryTests(unittest.TestCase):
    def test_ratios_have_explicit_direction_and_keep_paired_variability(self):
        summary = compact(report())
        case = summary["runs"][0]["cases"][0]
        for key in ("forward/eager", "forward/graph", "forward_backward/eager", "forward_backward/graph"):
            ratios = case["comparisons"][key]
            self.assertEqual(ratios["flex"]["baseline_over_kohakufa"], 2)
            self.assertEqual(ratios["natten"]["baseline_over_kohakufa"], 0.5)
            self.assertEqual(ratios["flex"]["paired_round_ratios"]["samples"], [2, 2, 2])
        self.assertEqual(case["backends"]["kohakufa"]["timings"]["forward/eager"]["device_us"]["median"], 100)
        text = markdown(summary)
        self.assertIn("forward backward / graph", text)
        self.assertIn("2.000×", text)
        self.assertNotIn("geometric mean", text)

    def test_qualification_and_execution_failures_never_enter_ratios(self):
        for status in ("qualification-error", "execution-error"):
            data = report()
            data["rows"][1]["status"] = status
            summary = compact(data)
            case = summary["runs"][0]["cases"][0]
            self.assertFalse(case["comparisons"]["forward/eager"]["flex"]["eligible"])
            self.assertTrue(case["comparisons"]["forward/eager"]["natten"]["eligible"])

    def test_individual_passes_survive_partial_run_failure(self):
        data = report()
        data["status"] = "completed-with-failures"
        data["rows"][1]["status"] = "partial-failure"
        data["rows"][1]["timings"]["forward_backward/graph"]["status"] = "execution-error"
        case = compact(data)["runs"][0]["cases"][0]
        self.assertTrue(case["comparisons"]["forward/eager"]["flex"]["eligible"])
        self.assertFalse(case["comparisons"]["forward_backward/graph"]["flex"]["eligible"])

    def test_missing_rounds_trace_or_graph_probe_exclude_ratios(self):
        for failure in ("round", "trace", "graph", "accuracy", "normalization", "median"):
            data = report()
            row = data["rows"][1]
            key = "forward/graph"
            if failure == "round":
                row["timings"][key]["samples"].pop()
            elif failure == "trace":
                row["profiles"][key]["launched_cuda_kernels"] = []
            elif failure == "graph":
                row["graph_checks"]["forward"]["passed"] = False
            elif failure == "accuracy":
                row["accuracy"]["errors"][2]["passed"] = False
            elif failure == "normalization":
                row["timings"][key]["samples"][0]["complete_calls"] = 3
            elif failure == "median":
                row["timings"][key]["device_ms"]["median"] = 999
            case = compact(data)["runs"][0]["cases"][0]
            self.assertFalse(case["comparisons"][key]["flex"]["eligible"], failure)
            self.assertTrue(case["backends"]["flex"]["timings"][key]["exclusions"], failure)

    def test_distinct_inputs_are_never_paired(self):
        data = report()
        data["rows"][1]["seed"] = 810
        cases = compact(data)["runs"][0]["cases"]
        self.assertEqual(len(cases), 2)
        self.assertTrue(all(not case["comparisons"]["forward/eager"]["flex"]["eligible"] for case in cases))

    def test_different_source_reports_remain_separate(self):
        first, second = report(), report()
        second["source_sha256"]["benchmarks/neighborhood/benchmark.py"] = "c" * 64
        result = summarize_inputs([("h100-one", json.dumps(first).encode()), ("h100-two", json.dumps(second).encode())])
        self.assertEqual(len(result["configurations"]), 2)
        self.assertEqual(len(result["runs"]), 2)

    def test_private_paths_uuid_messages_and_symbols_are_removed(self):
        data = report()
        data["rows"][1]["error"] = dict(type="RuntimeError", message="/private/secret/error")
        data["rows"][1]["failed_operands"] = "/private/remote/32sq-w7-fp16-flex-failure.pt"
        output = json.dumps(compact(data))
        for secret in ("/private/", "PRIVATE_DEVICE_UUID", "PRIVATE_POINTER", "trace.json"):
            self.assertNotIn(secret, output)
        self.assertIn("32sq-w7-fp16-flex-failure.pt", output)

    def test_csv_hardware_metadata_whitelists_driver_and_power_only(self):
        data = report()
        data["gpu_observations"] = [
            dict(exit_status=1, query_fields=["name"], rows=["PRIVATE_FAILURE"]),
            dict(
                exit_status=0,
                query_fields=["name", "uuid", "driver_version", "power.limit", "power.draw"],
                rows=['"NVIDIA H100 PCIe",PRIVATE_UUID,580.159.03,350.00,PRIVATE_OTHER_FIELD'],
            ),
        ]
        device = compact(data)["runs"][0]["device"]
        self.assertEqual(device["driver_version"], "580.159.03")
        self.assertEqual(device["power_limit_watts"], 350)
        serialized = json.dumps(compact(data))
        self.assertNotIn("PRIVATE_UUID", serialized)
        self.assertNotIn("PRIVATE_OTHER_FIELD", serialized)
        data["gpu_observations"][1]["rows"] = ['"NVIDIA H100 PCIe",PRIVATE_UUID,N/A,[N/A],PRIVATE_OTHER_FIELD']
        device = compact(data)["runs"][0]["device"]
        self.assertIsNone(device["driver_version"])
        self.assertIsNone(device["power_limit_watts"])

    def test_ambiguous_or_mismatched_inputs_fail_closed(self):
        data = report()
        data["rows"].append(copy.deepcopy(data["rows"][0]))
        with self.assertRaises(ValueError):
            compact(data)
        data = report()
        data["rows"][0]["shape"] = [1, 1, 1023, 64]
        with self.assertRaises(ValueError):
            compact(data)
        data = report()
        data["source_sha256"] = {"/private/absolute/source.py": "a" * 64}
        with self.assertRaises(ValueError):
            compact(data)
        data = report()
        data["protocol"]["canonical_layout"] = "BHWD"
        with self.assertRaises(ValueError):
            compact(data)


if __name__ == "__main__":
    unittest.main()
