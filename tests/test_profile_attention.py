import unittest
from types import SimpleNamespace

import torch

from benchmarks.plot_profile import representative
from benchmarks.profile_attention import launch_report, profile_phase


class ProfileTests(unittest.TestCase):
    def test_incomplete_capture_does_not_invent_iteration_slots(self):
        event = SimpleNamespace(
            device_type=torch.autograd.DeviceType.CUDA,
            name="one kernel",
            device_time_total=3,
            time_range=SimpleNamespace(start=10),
        )
        report = launch_report([event], iterations=5)
        self.assertFalse(report["uniform_launch_pattern"])
        self.assertEqual(report["launch_slots"], [])
        with self.assertRaises(ValueError):
            representative(report, 5)

    def test_timeline_preserves_gaps_and_does_not_double_count_concurrent_kernels(self):
        phase = dict(
            uniform_launch_pattern=True,
            launches=[
                dict(name="a", start_us=100, duration_us=10),
                dict(name="b", start_us=105, duration_us=10),
                dict(name="c", start_us=120, duration_us=5),
            ],
        )
        result = representative(phase, 1)
        self.assertEqual(result["span_us"], 25)
        self.assertEqual(result["busy_union_us"], 20)
        self.assertEqual(result["idle_us"], 5)
        self.assertEqual(result["events"][1]["start_us"], 5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA activity attribution requires CUDA")
    def test_record_function_gpu_projection_is_not_counted_as_a_kernel(self):
        x = torch.ones(4096, device="cuda")
        x.square()
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as profile:
            for _ in range(3):
                with torch.profiler.record_function("profile_fixture"):
                    result = x.square()
            torch.cuda.synchronize()
        report = launch_report(profile.events(), 3)
        self.assertTrue(report["uniform_launch_pattern"])
        self.assertEqual(len(report["launches"]), 3)
        self.assertEqual(len(report["launch_slots"]), 1)
        self.assertNotEqual(report["launch_slots"][0]["name"], "profile_fixture")
        torch.testing.assert_close(result, x)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA graph profiling requires CUDA")
    def test_graph_profile_captures_then_replays_changed_inputs(self):
        x = torch.ones(4096, device="cuda", requires_grad=True)
        stream = torch.cuda.Stream()
        out, report, forward_graph = profile_phase(
            lambda: x.square(),
            backend="fixture",
            phase="forward",
            iterations=3,
            warmup=3,
            profiler="nvtx",
            trace_dir=None,
            execution="graph",
            stream=stream,
        )
        self.assertEqual(report["execution"], "graph")
        torch.testing.assert_close(out, torch.ones_like(x))
        grad, _, backward_graph = profile_phase(
            lambda: torch.autograd.grad(out, x, torch.ones_like(out), retain_graph=True),
            backend="fixture",
            phase="backward",
            iterations=3,
            warmup=3,
            profiler="nvtx",
            trace_dir=None,
            execution="graph",
            stream=stream,
        )
        with torch.no_grad():
            x.fill_(3)
        forward_graph.replay()
        backward_graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, torch.full_like(x, 9))
        torch.testing.assert_close(grad[0], torch.full_like(x, 6))


if __name__ == "__main__":
    unittest.main()
