"""Compiler retention preserves distinct code sharing an automatic filename."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from benchmarks.compiled_artifacts import artifact_bytes, retain_code


class CompiledArtifactTests(unittest.TestCase):
    def test_distinct_images_survive_reused_dump_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dump = root / "automatic.cubin"
            images = [b"\x7fELFfirst", b"\x7fELFsecond"]
            retained = []
            for image in images:
                dump.write_bytes(image)
                function = SimpleNamespace(__cubin__=str(dump), __ptx__=b".version 9.4\n.visible .entry kernel() {}")
                records = retain_code(function, root / "retained")
                retained.append(records[0])
                self.assertTrue(all(r["status"] == "retained" for r in records))
            self.assertNotEqual(retained[0]["sha256"], retained[1]["sha256"])
            self.assertEqual([(root / "retained" / r["file"]).read_bytes() for r in retained], images)

    def test_missing_paths_and_non_code_are_explicitly_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            records = retain_code(SimpleNamespace(__cubin__=Path(directory) / "missing", __ptx__=b"not PTX"), directory)
            self.assertTrue(all(r["status"] == "unavailable" for r in records))
            self.assertTrue(all("file" not in r for r in records))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_direct_code_and_path_bytes_preserve_exact_bytes(self):
        ptx = b".version 9.4\n.visible .entry kernel() {}"
        self.assertEqual(artifact_bytes(ptx.decode(), "ptx"), ptx)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kernel.cubin"
            path.write_bytes(b"\x7fELFimage")
            self.assertEqual(artifact_bytes(bytes(path), "cubin"), path.read_bytes())
        with self.assertRaises(ValueError):
            artifact_bytes(ptx, "cubin")


if __name__ == "__main__":
    unittest.main()
