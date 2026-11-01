"""Benchmark source metadata must remain truthful in frozen snapshots."""

import argparse
import hashlib
import io
import os
import shutil
import subprocess
import tempfile
import unittest
from contextlib import chdir, redirect_stderr
from pathlib import Path
from unittest.mock import patch

from benchmarks.compare_flashattention import parser, source_metadata


class FlashAttentionRevisionTests(unittest.TestCase):
    def make_sources(self, root):
        package = root / "src" / "kohakufa_cute"
        package.mkdir(parents=True)
        (package / "kernel.py").write_text("value = 1\n")
        (package / "native.cpp").write_text("// source fixture\n")
        (package / "ignored.txt").write_text("not implementation code\n")
        return package

    def expected_hashes(self, package):
        return {
            "src/kohakufa_cute/" + name: hashlib.sha256((package / name).read_bytes()).hexdigest()
            for name in ("kernel.py", "native.cpp")
        }

    def git_fixture(self, root, message, object_format=None):
        root.mkdir(parents=True, exist_ok=True)
        arguments = ["git", "-C", str(root), "init", "-q"]
        if object_format is not None:
            arguments.append("--object-format=" + object_format)
        subprocess.run(arguments, check=True, capture_output=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "-c",
                "user.name=Benchmark Fixture",
                "-c",
                "user.email=benchmark@example.invalid",
                "commit",
                "--allow-empty",
                "-qm",
                message,
            ],
            check=True,
            capture_output=True,
        )
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()

    def test_explicit_full_labels_preserve_source_hashes_without_git(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = self.make_sources(root)
            with patch("benchmarks.compare_flashattention.subprocess.run", side_effect=AssertionError("Unexpected Git")):
                for label in ("A" * 40, "B" * 64):
                    record = source_metadata(root, package, label)
                    self.assertEqual(record["base_revision"], label.lower())
                    self.assertEqual(record["base_revision_origin"], "explicit_installer_label")
                    self.assertEqual(record["source_sha256"], self.expected_hashes(package))
                    self.assertNotIn(directory, str(record))

    def test_missing_git_keeps_hashes_and_records_unavailable_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = self.make_sources(root)
            with patch("benchmarks.compare_flashattention.subprocess.run", side_effect=FileNotFoundError):
                record = source_metadata(root, package)
            self.assertIsNone(record["base_revision"])
            self.assertEqual(record["base_revision_origin"], "unavailable")
            self.assertEqual(record["source_sha256"], self.expected_hashes(package))

    @unittest.skipUnless(shutil.which("git"), "Requires Git")
    def test_non_git_snapshot_keeps_hashes_and_records_unavailable_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = self.make_sources(root)
            record = source_metadata(root, package)
            self.assertIsNone(record["base_revision"])
            self.assertEqual(record["base_revision_origin"], "unavailable")
            self.assertEqual(record["source_sha256"], self.expected_hashes(package))

    @unittest.skipUnless(shutil.which("git"), "Requires Git")
    def test_source_git_is_resolved_independently_of_caller_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, caller = root / "source", root / "caller"
            expected = self.git_fixture(source, "source fixture")
            other = self.git_fixture(caller, "caller fixture")
            package = self.make_sources(source)
            with chdir(caller):
                record = source_metadata(source, package)
            self.assertNotEqual(expected, other)
            self.assertEqual(record["base_revision"], expected)
            self.assertEqual(record["base_revision_origin"], "source_git")
            self.assertEqual(record["source_sha256"], self.expected_hashes(package))

    @unittest.skipUnless(shutil.which("git"), "Requires Git")
    def test_snapshot_nested_in_another_checkout_does_not_borrow_its_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.git_fixture(root, "unrelated fixture")
            snapshot = root / "frozen"
            package = self.make_sources(snapshot)
            record = source_metadata(snapshot, package)
            self.assertIsNone(record["base_revision"])
            self.assertEqual(record["base_revision_origin"], "unavailable")
            self.assertEqual(record["source_sha256"], self.expected_hashes(package))

    @unittest.skipUnless(shutil.which("git"), "Requires Git")
    def test_inherited_git_directory_cannot_replace_the_source_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, caller = root / "source", root / "caller"
            expected = self.git_fixture(source, "source fixture")
            self.git_fixture(caller, "caller fixture")
            package = self.make_sources(source)
            with patch.dict(os.environ, {"GIT_DIR": str(caller / ".git"), "GIT_WORK_TREE": str(caller)}):
                record = source_metadata(source, package)
            self.assertEqual(record["base_revision"], expected)
            self.assertEqual(record["base_revision_origin"], "source_git")

    def test_cli_accepts_full_sha1_and_sha256_labels(self):
        for label in ("a" * 40, "b" * 64):
            args = parser().parse_args(["--backends", "cute", "--output", "result.json", "--source-revision", label])
            self.assertEqual(args.source_revision, label)

    def test_cli_rejects_malformed_labels_without_echoing_input(self):
        with tempfile.TemporaryDirectory() as directory:
            values = ("a" * 39, "a" * 41, "b" * 63, "b" * 65, "g" * 40, str(Path(directory) / "private"))
            for value in values:
                with self.subTest(length=len(value)):
                    output = io.StringIO()
                    with redirect_stderr(output), self.assertRaises(SystemExit) as failure:
                        parser().parse_args(["--backends", "cute", "--output", "result.json", "--source-revision", value])
                    self.assertEqual(failure.exception.code, 2)
                    self.assertNotIn(value, output.getvalue())

    def test_programmatic_explicit_label_is_validated_too(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = self.make_sources(root)
            with self.assertRaises(argparse.ArgumentTypeError):
                source_metadata(root, package, "not-a-revision")


if __name__ == "__main__":
    unittest.main()
