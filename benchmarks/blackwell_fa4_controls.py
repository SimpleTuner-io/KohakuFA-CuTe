"""Explicit previous-native control after optional Blackwell dispatch is enabled."""

from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def previous_native_dispatch():
    """Keep the native CuTe baseline while explicit candidate factories bypass dispatch."""
    from kohakufa_cute import fa4_blackwell

    with patch.object(fa4_blackwell, "compatible", return_value=False):
        yield


def make_original_attention():
    """Select direct installed FA4 for independent ownership diagnostics."""
    from benchmarks.flashattention_backends import load_backend

    return load_backend("fa4")
