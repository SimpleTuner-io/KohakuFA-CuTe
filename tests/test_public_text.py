"""Exercise the client-side guard without publishing any content."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

GUARD = Path(__file__).resolve().parents[1] / "tools" / "check_public_text.py"
BLOCKED = "Blocked: local machine identity was found in public text.\n"


class PublicTextTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.home = self.root / "private-config"
        self.home.mkdir()
        self.environment = dict(os.environ)
        for key in tuple(self.environment):
            if key.startswith("GIT_"):
                self.environment.pop(key)
        self.environment.update(HOME=str(self.home), GIT_CONFIG_NOSYSTEM="1")
        self.private_name = "Fixture" + " Private Identity"
        self.git("init", "--quiet")
        self.git("config", "--global", "user.name", self.private_name)
        self.git("config", "--global", "--add", "privacy.identity", self.private_name)
        self.git("config", "--global", "user.email", "fixture-private@example.invalid")
        self.git("config", "user.name", "Project Maintainers")
        self.git("config", "user.email", "maintainers@example.invalid")
        (self.repo / "tools").mkdir()
        shutil.copyfile(GUARD, self.repo / "tools" / GUARD.name)
        self.git("add", "tools")

    def git(self, *args, input=None):
        return subprocess.run(
            ["git", *args],
            cwd=self.repo,
            env=self.environment,
            input=input,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        ).stdout

    def guard(self, *args, input=None):
        return subprocess.run(
            [sys.executable, "tools/check_public_text.py", *args],
            cwd=self.repo,
            env=self.environment,
            input=input,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write(self, name, text, stage=False):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        if stage:
            self.git("add", name)

    def assert_blocked(self, result):
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(result.stderr.decode(), BLOCKED)

    def commit(self):
        self.git("commit", "--quiet", "-m", "Fixture change")

    def test_clean_staged_tree_metadata_and_exact_text(self):
        self.write("notes.txt", "Use tools/check_public_text.py before publishing.", stage=True)
        self.write("message.txt", "Improve the kernel pipeline")
        result = self.guard("--staged", "--metadata", "--message-file", "message.txt")
        self.assertEqual((result.returncode, result.stdout, result.stderr), (0, b"", b""))

    def test_full_index_includes_unchanged_file(self):
        self.write("old.txt", self.private_name, stage=True)
        self.commit()
        self.write("new.txt", "A safe change", stage=True)
        self.assert_blocked(self.guard("--staged"))

    def test_index_does_not_read_unstaged_content(self):
        self.write("notes.txt", "Safe staged content", stage=True)
        self.write("notes.txt", self.private_name)
        self.assertEqual(self.guard("--staged").returncode, 0)

    def test_staged_filename_is_scanned(self):
        self.write(self.private_name + ".txt", "Safe content", stage=True)
        self.assert_blocked(self.guard("--staged"))

    def test_private_commit_message_and_exact_publish_text(self):
        for option in ("--message-file", "--text-file", "--file"):
            with self.subTest(option=option):
                self.write("outgoing.txt", self.private_name)
                self.assert_blocked(self.guard(option, "outgoing.txt"))

    def test_commit_message_stdin_is_scanned(self):
        self.assert_blocked(self.guard("--message-stdin", input=self.private_name.encode()))

    def test_author_metadata_is_scanned(self):
        self.git("config", "user.name", self.private_name)
        self.assert_blocked(self.guard("--metadata"))

    def test_history_scans_deleted_blob_and_original_metadata(self):
        self.write("old.txt", self.private_name, stage=True)
        self.commit()
        self.git("rm", "--quiet", "old.txt")
        self.commit()
        self.assertEqual(self.guard("--staged").returncode, 0)
        self.assert_blocked(self.guard("--history"))

    def test_history_checks_old_commit_identity(self):
        self.git("config", "user.name", self.private_name)
        self.commit()
        self.git("config", "user.name", "Project Maintainers")
        self.assert_blocked(self.guard("--history"))

    def test_push_tip_outside_refs_is_scanned(self):
        self.commit()
        self.write("private.txt", self.private_name, stage=True)
        self.commit()
        tip = self.git("rev-parse", "HEAD").strip()
        self.git("reset", "--hard", "HEAD~1")
        payload = b"HEAD " + tip + b" refs/heads/review " + b"0" * 40 + b"\n"
        self.assert_blocked(self.guard("--history", "--push-refs", input=payload))

    def test_paths_identity_and_attribution_never_echo(self):
        values = [
            "/" + "Users" + "/fixture/report.txt",
            "/" + "home" + "/fixture/report.txt",
            "/" + "tmp" + "/capture.txt",
            "/" + "workspace" + "/capture.txt",
            "/" + "private" + "/var" + "/folders" + "/capture.txt",
            "C:" + "\\Users\\fixture\\capture.txt",
            "Co-authored-by: Fixture Person <person@example.invalid>",
            str(self.home),
            "fixture-host" + ".internal",
            "/" + "cache" + "/capture.txt",
        ]
        for index, value in enumerate(values):
            with self.subTest(kind=index):
                self.write("outgoing.txt", value)
                self.assert_blocked(self.guard("--text-file", "outgoing.txt"))

    def test_dynamic_environment_username(self):
        identity = "fixture" + "_private_account"
        self.environment["USERNAME"] = identity
        self.write("outgoing.txt", identity)
        self.assert_blocked(self.guard("--text-file", "outgoing.txt"))

    def test_compiler_flags_cannot_hide_absolute_paths(self):
        for flag in ("-I", "-L", "-isystem", "-iquote", "-idirafter", "-iprefix", "-isysroot", "--sysroot="):
            self.write("outgoing.txt", flag + "/" + "home" + "/fixture/include")
            self.assert_blocked(self.guard("--text-file", "outgoing.txt"))

    def test_prose_and_relative_slash_paths_are_allowed(self):
        for text in (
            "dynamic-shape/cache dispatch",
            "src/cache/module.py",
            "notes/home/configuration.md",
            "assets/workspace/diagram.svg",
            "data/var/cache/manifest.json",
        ):
            self.write("outgoing.txt", text)
            result = self.guard("--text-file", "outgoing.txt")
            self.assertEqual((result.returncode, result.stdout, result.stderr), (0, b"", b""))

    def test_local_account_substrings_are_private_including_urls(self):
        identity = "fixture" + "_private_account"
        self.environment["USERNAME"] = identity
        for template in ("prefix{}suffix", "https://example.invalid/prefix{}suffix"):
            self.write("outgoing.txt", template.format(identity.upper()))
            self.assert_blocked(self.guard("--text-file", "outgoing.txt"))

    def test_private_config_persists_multiple_identity_aliases(self):
        identities = ("legacy" + "_fixture_account", "former" + "_fixture_account")
        for identity in identities:
            self.git("config", "--global", "--add", "privacy.identity", identity)
        for identity in identities:
            self.write("outgoing.txt", "prefix" + identity + "suffix")
            self.assert_blocked(self.guard("--text-file", "outgoing.txt"))
        local_identity = "local" + "_fixture_account"
        self.git("config", "--add", "privacy.identity", local_identity)
        self.write("outgoing.txt", "prefix" + local_identity + "suffix")
        self.assert_blocked(self.guard("--text-file", "outgoing.txt"))

    def test_public_git_handle_and_automation_metadata_are_allowed(self):
        handle = "public" + "projecthandle"
        email = handle + "@example.invalid"
        self.git("config", "--global", "user.name", handle)
        self.git("config", "--global", "user.email", email)
        self.git("config", "user.name", handle)
        self.git("config", "user.email", email)
        self.environment.update(
            GIT_AUTHOR_NAME=handle,
            GIT_AUTHOR_EMAIL=email,
            GIT_COMMITTER_NAME=handle,
            GIT_COMMITTER_EMAIL=email,
        )
        self.write("outgoing.txt", "https://github.com/" + handle + "/project")
        result = self.guard("--metadata", "--text-file", "outgoing.txt")
        self.assertEqual((result.returncode, result.stdout, result.stderr), (0, b"", b""))
        self.git("config", "--add", "privacy.identity", handle)
        self.assert_blocked(self.guard("--metadata", "--text-file", "outgoing.txt"))

    def test_missing_or_absolute_input_fails_closed_without_details(self):
        self.assert_blocked(self.guard("--text-file", "missing.txt"))
        self.assert_blocked(self.guard("--text-file", str(self.home / "message.txt")))
        self.assert_blocked(self.guard("--unknown", self.private_name))

    def test_commit_hooks_block_before_creating_commit(self):
        shutil.copytree(GUARD.parents[1] / ".githooks", self.repo / ".githooks")
        self.git("config", "core.hooksPath", ".githooks")
        self.write("private.txt", self.private_name, stage=True)
        for private_content in (True, False):
            if not private_content:
                self.write("private.txt", "Safe content", stage=True)
            message = "Safe change" if private_content else self.private_name
            result = subprocess.run(
                ["git", "commit", "-m", message],
                cwd=self.repo,
                env=self.environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(BLOCKED.encode(), result.stderr)
            self.assertNotIn(self.private_name.encode(), result.stdout + result.stderr)
            refs = subprocess.run(
                ["git", "rev-parse", "--verify", "HEAD"],
                cwd=self.repo,
                env=self.environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertNotEqual(refs.returncode, 0)

    def test_pre_push_hook_blocks_before_remote_changes(self):
        hooks = GUARD.parents[1] / ".githooks"
        shutil.copytree(hooks, self.repo / ".githooks")
        self.git("add", ".githooks")
        self.write("private.txt", self.private_name, stage=True)
        self.commit()
        remote = self.root / "remote.git"
        subprocess.run(
            ["git", "init", "--bare", "--quiet", str(remote)],
            env=self.environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        self.git("remote", "add", "fixture", str(remote))
        self.git("config", "core.hooksPath", ".githooks")
        result = subprocess.run(
            ["git", "push", "fixture", "HEAD:refs/heads/review"],
            cwd=self.repo,
            env=self.environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(BLOCKED.encode(), result.stderr)
        self.assertNotIn(self.private_name.encode(), result.stdout + result.stderr)
        self.assertEqual(self.git("ls-remote", "fixture"), b"")


if __name__ == "__main__":
    unittest.main()
