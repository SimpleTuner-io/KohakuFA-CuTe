import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch.utils.cpp_extension

import kohakufa_cute


class NativeBuildTests(unittest.TestCase):
    def test_changed_source_with_preserved_mtime_cannot_reuse_old_binary(self):
        source = Path(kohakufa_cute.__file__).with_name("native.py")
        spec = importlib.util.spec_from_file_location("kohakufa_cute._build_probe", source)
        module = importlib.util.module_from_spec(spec)
        binaries = {}

        def cached_load(*, name, **kwargs):
            return binaries.setdefault(name, SimpleNamespace(clear_cache=lambda: None))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            triton = ModuleType("triton")
            triton.__file__ = str(root / "triton.py")
            ffi = ModuleType("tvm_ffi")
            ffi.libinfo = SimpleNamespace(
                find_libtvm_ffi=lambda: str(root / "libtvm_ffi.so"),
                find_include_path=lambda: str(root),
                find_dlpack_include_path=lambda: str(root),
            )
            kernels = ModuleType("kohakufa_cute.kernels")
            with (
                patch.dict("sys.modules", {"triton": triton, "tvm_ffi": ffi, "kohakufa_cute.kernels": kernels}),
                patch.object(kohakufa_cute, "kernels", kernels, create=True),
                patch.object(torch.utils.cpp_extension, "load", side_effect=cached_load) as loader,
                patch("atexit.register"),
            ):
                spec.loader.exec_module(module)
                module.__file__ = str(root / "native.py")
                cpp = root / "native.cpp"
                cpp.write_text("first revision\n")
                os.utime(cpp, (1700000000, 1700000000))
                original = module.extension()
                self.assertIs(module.extension(), original)
                self.assertEqual(loader.call_count, 1)
                cpp.write_text("other revision\n")
                os.utime(cpp, (1700000000, 1700000000))
                module.extension.cache_clear()
                self.assertIsNot(module.extension(), original)
                self.assertEqual(len(binaries), 2)


if __name__ == "__main__":
    unittest.main()
