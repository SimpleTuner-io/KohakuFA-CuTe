"""Retain compiler-returned code without ambiguous automatic dump filenames."""

import hashlib
import os
from pathlib import Path


def artifact_bytes(value, kind):
    """Accept actual code or an existing file; never serialize a missing path."""
    if isinstance(value, os.PathLike):
        data = Path(value).read_bytes()
    elif isinstance(value, (bytes, bytearray, str)):
        data = value.encode() if isinstance(value, str) else bytes(value)
        if not (data.startswith(b"\x7fELF") or b".version" in data):
            data = Path(os.fsdecode(data)).read_bytes()
    else:
        raise TypeError("The compiler did not expose code bytes or a file.")
    if kind == "cubin" and not data.startswith(b"\x7fELF"):
        raise ValueError("The compiler artifact is not ELF.")
    if kind == "ptx" and not (b".version" in data and b".entry" in data):
        raise ValueError("The compiler artifact lacks PTX entry points.")
    return data


def retain_code(function, directory):
    """Export by SHA; returned artifacts alone do not bind a loaded CUfunction."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for kind in ("cubin", "ptx"):
        try:
            data = artifact_bytes(getattr(function, f"__{kind}__", None), kind)
        except Exception as error:
            records.append({"kind": kind, "status": "unavailable", "exception_type": type(error).__name__})
            continue
        digest = hashlib.sha256(data).hexdigest()
        filename = f"{digest}.{kind}"
        path = directory / filename
        if path.exists():
            if path.read_bytes() != data:
                raise RuntimeError("A retained compiler artifact changed under its content hash.")
        else:
            path.write_bytes(data)
        records.append({"kind": kind, "status": "retained", "file": filename, "sha256": digest, "bytes": len(data)})
    return records
