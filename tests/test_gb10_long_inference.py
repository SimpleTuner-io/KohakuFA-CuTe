import sys
import unittest
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock, patch

from kohakufa_cute.fa4_sm80 import gb10
from kohakufa_cute.fa4_sm80._attention import Schedule
from kohakufa_cute.fa4_sm80._host import native_runner


class Gb10LongInferenceTests(unittest.TestCase):
    def tearDown(self):
        gb10._long_d256_runner.cache_clear()
        native_runner.cache_clear()

    def test_forward_thread_option_preserves_default_configuration(self):
        make = Mock()
        native = SimpleNamespace(extension=lambda: SimpleNamespace(AttentionRunner=make))
        selected = Schedule(forward_m=128, forward_n=32)
        with patch.dict(sys.modules, {"kohakufa_cute.native": native}):
            native_runner.cache_clear()
            native_runner(selected)
            make.assert_called_with("warp", asdict(selected))
            native_runner(selected, forward_threads=256)
            make.assert_called_with("warp", dict(asdict(selected), forward_threads=256))

    def test_public_selected_runner_is_cached_without_caching_operands(self):
        make = Mock()
        make.return_value.inference.return_value = "selected"
        with patch.object(gb10, "native_runner", make), patch.object(gb10, "inference_compatible", return_value=True):
            gb10._long_d256_runner.cache_clear()
            for sequence in (512, 1024, 4096):
                for causal in (False, True):
                    q = SimpleNamespace(shape=(1, 16, sequence, 256))
                    self.assertEqual(gb10.scaled_dot_product_attention(q, q, q, is_causal=causal), "selected")
                    changed = SimpleNamespace(shape=q.shape)
                    gb10.scaled_dot_product_attention(changed, changed, changed, is_causal=causal)
                    make.return_value.inference.assert_called_with(changed, changed, changed, scale=None, causal=causal)
            self.assertEqual(make.call_count, 6)
            for call in make.call_args_list:
                threads = call.kwargs["forward_threads"]
                self.assertIn(threads, (128, 256))
                self.assertGreaterEqual(call.args[0].forward_m, threads // 32 * 16)
        with self.assertRaises(ValueError):
            gb10._long_d256_runner(129, False)


if __name__ == "__main__":
    unittest.main()
