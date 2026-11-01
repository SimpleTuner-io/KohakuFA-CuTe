import unittest
from types import SimpleNamespace

import torch

from benchmarks.paired_gb10_schedules import eager_call, graph_call


class Gb10BenchmarkCallTests(unittest.TestCase):
    def test_eager_and_graph_builders_preserve_visibility_operands_and_gradients(self):
        for causal in (False, True):
            for phase in ("backward", "joint"):
                for execution in ("eager", "graph"):
                    with self.subTest(causal=causal, phase=phase, execution=execution):
                        inputs = tuple(torch.randn(1, 2, 3, 4, requires_grad=True) for _ in range(3))
                        dout = torch.randn_like(inputs[0])
                        calls = []

                        def backend(q, k, v, *, is_causal):
                            calls.append((q, k, v, is_causal))
                            return q + 2 * k + 3 * v

                        if execution == "eager":
                            run = eager_call(backend, inputs, dout, causal=causal, phase=phase)
                        else:

                            def fake_capture(function, operands, gradient, visibility, selected_phase, *, keep_graph):
                                self.assertIs(function, backend)
                                self.assertIs(operands, inputs)
                                self.assertIs(gradient, dout)
                                self.assertEqual((visibility, selected_phase, keep_graph), (causal, phase, True))
                                return (
                                    SimpleNamespace(
                                        replay=eager_call(
                                            function, operands, gradient, causal=visibility, phase=selected_phase
                                        )
                                    ),
                                    None,
                                )

                            graph, _ = graph_call(
                                backend, inputs, dout, causal=causal, phase=phase, capture_function=fake_capture
                            )
                            run = graph.replay
                        for _ in range(2):
                            actual = run()
                            torch.testing.assert_close(actual[0], inputs[0] + 2 * inputs[1] + 3 * inputs[2])
                            for grad, factor in zip(actual[1:], (1, 2, 3)):
                                torch.testing.assert_close(grad, dout * factor)
                            self.assertTrue(all(t.grad is None for t in inputs))
                        self.assertTrue(calls)
                        for q, k, v, visibility in calls:
                            self.assertTrue(q is inputs[0] and k is inputs[1] and v is inputs[2])
                            self.assertEqual(visibility, causal)
                        self.assertEqual(len(calls), 1 if phase == "backward" else 2)


if __name__ == "__main__":
    unittest.main()
