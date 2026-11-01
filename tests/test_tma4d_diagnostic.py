"""The diagnostic must inspect the selected loaded function, not a nearby one."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from benchmarks.diagnostics.tma4d_store import (
    Interrupted,
    execute_process,
    function_body,
    inherited_lock_fds,
    validated_control,
)


class TestSelectedTmaFunction(unittest.TestCase):
    def test_excludes_other_producer_and_consumer(self):
        sass = """code for sm_120
        Function : tma4d_read
        /*0020*/ UTMASTG.4D [UR8], [UR4], desc[URZ] ;
        Function : tma4d_ordinary
        /*0040*/ STG.E.U16 [R2.64], R4 ;
        Function : tma4d_consume
        /*0060*/ LDG.E.128 R4, [R2.64] ;
        """
        body = function_body(sass, "tma4d_ordinary")
        self.assertIn("STG.E.U16", body)
        self.assertNotIn("UTMASTG", body)
        self.assertNotIn("LDG.E.128", body)

    def test_missing_function_is_not_empty_success(self):
        with self.assertRaises(RuntimeError):
            function_body("Function : unrelated\nUTMASTG.4D;\n", "tma4d_read")

    def test_multiple_loaded_functions_are_ambiguous(self):
        with self.assertRaises(RuntimeError):
            function_body("Function : tma4d_read\nA\nFunction : tma4d_read\nB\n", "tma4d_read")


class TestDiagnosticOwnership(unittest.TestCase):
    def test_present_lock_is_explicitly_inherited(self):
        with patch("benchmarks.diagnostics.tma4d_store.os.fstat", return_value=object()):
            process = Mock(pid=123)
            process.wait.return_value = 0
            with patch("benchmarks.diagnostics.tma4d_store.subprocess.Popen", return_value=process) as popen:
                self.assertEqual(execute_process(["probe"], Mock(), 45), {"returncode": 0})
            self.assertEqual(popen.call_args.kwargs["pass_fds"], (9,))
            self.assertTrue(popen.call_args.kwargs["start_new_session"])

    def test_absent_lock_does_not_pass_an_invalid_descriptor(self):
        with patch("benchmarks.diagnostics.tma4d_store.os.fstat", side_effect=OSError):
            self.assertEqual(inherited_lock_fds(), ())

    def test_term_cleans_target_group_even_if_leader_exits_first(self):
        import signal

        process = Mock(pid=123)
        process.wait.side_effect = [Interrupted(signal.SIGTERM), 0, 0]
        with (
            patch("benchmarks.diagnostics.tma4d_store.subprocess.Popen", return_value=process),
            patch("benchmarks.diagnostics.tma4d_store.os.killpg") as killpg,
        ):
            result = execute_process(["probe"], Mock(), 45)
        self.assertEqual(result, {"returncode": 143, "signal": signal.SIGTERM})
        self.assertEqual(killpg.call_args_list, [call(123, signal.SIGTERM), call(123, signal.SIGKILL)])


class TestSanitizerControlBinding(unittest.TestCase):
    def control(self, path):
        control = {
            "build": {"source_sha256": "source", "binary_sha256": "binary", "ptx_sha256": "ptx"},
            "mode": "read",
            "initialization": "uninitialized",
            "rows": 257,
            "capture": "enabled",
            "tool": "none",
            "returncode": 0,
            "code_binding": "passed",
            "physical_verification": "passed",
            "loaded_bindings": [{"loaded_cubin_sha256": "actual"}],
        }
        path.write_text(json.dumps(control))
        args = SimpleNamespace(
            capture="disabled", tool="initcheck", control_result=path, mode="read", initialization="uninitialized", rows=257
        )
        return control, args

    def test_accepts_same_executable_without_inventing_instrumented_binding(self):
        with tempfile.TemporaryDirectory() as folder:
            control, args = self.control(Path(folder) / "control.json")
            record = validated_control(args, control["build"])
        self.assertEqual(record["binary_sha256"], "binary")
        self.assertIn("Uninstrumented control only", record["scope"])

    def test_rejects_different_executable(self):
        with tempfile.TemporaryDirectory() as folder:
            control, args = self.control(Path(folder) / "control.json")
            with self.assertRaises(RuntimeError):
                validated_control(args, dict(control["build"], binary_sha256="changed"))

    def test_rejects_different_geometry_and_missing_direct_control(self):
        with tempfile.TemporaryDirectory() as folder:
            control, args = self.control(Path(folder) / "control.json")
            args.rows = 256
            with self.assertRaises(RuntimeError):
                validated_control(args, control["build"])
            args.rows = 257
            control["code_binding"] = "unavailable_under_sanitizer"
            args.control_result.write_text(json.dumps(control))
            with self.assertRaises(RuntimeError):
                validated_control(args, control["build"])


if __name__ == "__main__":
    unittest.main()
