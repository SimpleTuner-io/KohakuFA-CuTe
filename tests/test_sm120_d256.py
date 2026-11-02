import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from kohakufa_cute import fa4_sm80
from kohakufa_cute.fa4_sm80 import _sm120
from tests import test_sm120_public_host as host_tests

operand = host_tests.operand


class Sm120D256SelectionTests(unittest.TestCase):
    def test_bounded_shape_and_preserved_short_fallback(self):
        capability, properties, current, available = host_tests.Sm120PublicSelectionTests().runtime()
        _sm120.target_device.cache_clear()
        with capability, properties, current, available:
            for dtype in (torch.bfloat16, torch.float16):
                for length in (1024, 4096):
                    q = operand((1, 16, length, 256), dtype=dtype)
                    for causal in (False, True):
                        self.assertTrue(fa4_sm80.compatible(q, q, q, None, 0, causal, None, False))
                for length in (128, 129, 257, 4095, 4097):
                    q = operand((1, 16, length, 256), dtype=dtype)
                    self.assertFalse(fa4_sm80.compatible(q, q, q, None, 0, True, None, False))
            q = operand((1, 16, 1024, 128), dtype=torch.float16)
            self.assertFalse(fa4_sm80.compatible(q, q, q, None, 0, True, None, False))
        _sm120.target_device.cache_clear()

    def test_geometry_and_native_forward_thread_override(self):
        extension = SimpleNamespace(AttentionRunner=Mock(return_value=object()))
        _sm120.d256_runner.cache_clear()
        with patch.dict(sys.modules, {"kohakufa_cute.native": SimpleNamespace(extension=lambda: extension)}):
            for sequence in (1024, 4096):
                for causal in (False, True):
                    _sm120.d256_runner(causal, sequence)
                    family, options = extension.AttentionRunner.call_args.args
                    self.assertEqual(family, "warp")
                    self.assertEqual((options["backward_m"], options["backward_n"]), (64, 32))
                    self.assertEqual((options["query_stages"], options["dout_stages"]), (1, 1))
                    self.assertEqual((options["score_m_warps"], options["kv_m_warps"], options["dq_m_warps"]), (4, 1, 2))
                    self.assertTrue(options["v_in_regs"])
                    self.assertEqual(options["causal_head_group"], 1)
                    expected = (128, 32, 256) if sequence == 1024 and not causal else (64, 64, 128)
                    self.assertEqual((options["forward_m"], options["forward_n"], options["forward_threads"]), expected)
        _sm120.d256_runner.cache_clear()


if __name__ == "__main__":
    unittest.main()
