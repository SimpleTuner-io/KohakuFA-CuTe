#!/usr/bin/env python3
"""Check exact publication inputs before committing or publishing."""

import argparse
import getpass
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

BLOCKED = "Blocked: local machine identity was found in public text."
SERVICE_IDENTITIES = {"root", "runner", "ubuntu", "user", "git", "nobody", "localhost"}
PATHS = re.compile(
    r"(?:(?<![\w:])|(?<=-I)|(?<=-L)|(?<=-isystem)|(?<=-iquote)|(?<=-idirafter)|(?<=-iprefix)|(?<=-isysroot))"
    r"/(?:Users|home|root|tmp|cache|workspace|workspaces|pod|pods|mnt|(?:private/)?var/(?:folders|tmp|cache))(?:/|\b)"
    r"|[A-Za-z]:[\\/](?:Users|Documents and Settings|Temp)[\\/]"
    r"|~[\\/](?:[^\s]*)",
    re.IGNORECASE,
)
HOSTS = re.compile(r"\b[\w.-]+\.(?:local|lan|internal)\b", re.IGNORECASE)
ATTRIBUTION = re.compile(r"(?im)^\s*(?:co-authored-by|reviewed-by|signed-off-by):[^\n]*")


class Blocked(Exception):
    pass


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise Blocked


def git(*args):
    return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL)


def identities():
    accounts = {getpass.getuser(), Path.home().name}
    accounts.update(os.environ.get(key, "") for key in ("USER", "LOGNAME", "USERNAME", "SUDO_USER"))
    values = {str(Path.home()), socket.gethostname(), socket.getfqdn(), socket.gethostname().split(".")[0]}
    values.update(os.environ.get(key, "") for key in ("HOSTNAME", "COMPUTERNAME"))
    result = subprocess.run(
        ["git", "config", "--get-all", "privacy.identity"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    if result.returncode not in (0, 1):
        raise Blocked
    accounts.update(result.stdout.decode("utf-8", "replace").splitlines())
    accounts = {value.strip() for value in accounts}
    values = {value.strip() for value in values} | accounts
    patterns = []
    for value in values:
        if len(value) <= 2 or value.lower() in SERVICE_IDENTITIES:
            continue
        pattern = re.escape(value)
        if value not in accounts or len(value) < 4:
            pattern = r"(?<![\w])" + pattern + r"(?![\w])"
        patterns.append(re.compile(pattern, re.IGNORECASE))
    return patterns


def scan(data, private):
    text = data.decode("utf-8", "replace") if isinstance(data, bytes) else data
    if PATHS.search(text) or HOSTS.search(text) or any(pattern.search(text) for pattern in private):
        raise Blocked
    # Attribution containing a personal email must not slip through under an
    # identity that differs from the current machine's configuration.
    if any("@" in match.group() for match in ATTRIBUTION.finditer(text)):
        raise Blocked


def scan_objects(objects, private):
    with subprocess.Popen(
        ["git", "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    ) as process:
        try:
            for oid in dict.fromkeys(objects):
                process.stdin.write(oid + b"\n")
                process.stdin.flush()
                header = process.stdout.readline().split()
                if len(header) != 3 or header[1] not in (b"blob", b"tree", b"commit", b"tag"):
                    raise Blocked
                size = int(header[2])
                data = process.stdout.read(size)
                if len(data) != size or process.stdout.read(1) != b"\n":
                    raise Blocked
                scan(data, private)
        finally:
            process.stdin.close()
            process.wait()
        if process.returncode:
            raise Blocked


def staged(private):
    objects = []
    for entry in git("ls-files", "--stage", "-z").split(b"\0"):
        if not entry:
            continue
        metadata, name = entry.split(b"\t", 1)
        mode, oid, stage = metadata.split()
        if stage != b"0":
            raise Blocked
        scan(name, private)
        if mode != b"160000":
            objects.append(oid)
    scan_objects(objects, private)


def history(private, push_refs):
    tips = []
    if push_refs:
        for line in sys.stdin.buffer:
            fields = line.split()
            if len(fields) != 4 or not re.fullmatch(rb"[0-9a-f]{40}(?:[0-9a-f]{24})?", fields[1]):
                raise Blocked
            if set(fields[1]) != {ord("0")}:
                tips.append(fields[1].decode("ascii"))
    output = git("rev-list", "--objects", "--all", *tips)
    scan_objects([line.split(b" ", 1)[0] for line in output.splitlines()], private)


def relative_file(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise Blocked
    return path.read_bytes()


def main():
    parser = Parser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="Scan every file in the complete index.")
    parser.add_argument("--metadata", action="store_true", help="Scan exact author and committer identities.")
    parser.add_argument("--history", action="store_true", help="Scan all objects reachable from local refs.")
    parser.add_argument("--push-refs", action="store_true", help="Also scan pre-push tips from standard input.")
    parser.add_argument("--message-stdin", action="store_true", help="Scan an exact commit message from standard input.")
    parser.add_argument("--message-file", action="append", default=[], metavar="FILE")
    parser.add_argument("--text-file", action="append", default=[], metavar="FILE")
    parser.add_argument("--file", action="append", default=[], metavar="FILE")
    args = parser.parse_args()
    if not (
        args.staged
        or args.metadata
        or args.history
        or args.message_stdin
        or args.message_file
        or args.text_file
        or args.file
    ):
        raise Blocked
    if args.push_refs and args.message_stdin:
        raise Blocked
    if args.push_refs and not args.history:
        raise Blocked
    private = identities()
    if args.staged:
        staged(private)
    if args.metadata:
        for role in ("GIT_AUTHOR_IDENT", "GIT_COMMITTER_IDENT"):
            scan(git("var", role), private)
    if args.history:
        history(private, args.push_refs)
    if args.message_stdin:
        scan(sys.stdin.buffer.read(), private)
    for value in args.message_file + args.text_file + args.file:
        scan(relative_file(value), private)


if __name__ == "__main__":
    try:
        main()
    except (Blocked, OSError, ValueError, subprocess.SubprocessError):
        print(BLOCKED, file=sys.stderr)
        sys.exit(1)
