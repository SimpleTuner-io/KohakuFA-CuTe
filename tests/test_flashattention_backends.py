"""Check that direct FlashAttention comparisons preserve computation and layout."""

import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from benchmarks.flashattention_backends import Backend, UnsupportedCase, configuration_metadata, load_backend


class FlashAttentionBackendTests(unittest.TestCase):
    def test_bshd_adapter_preserves_outputs_gradients_scale_and_gqa(self):
        torch.manual_seed(209)
        for name in ("fa2", "fa3", "fa4"):
            with self.subTest(backend=name):
                calls = []

                def flash(q, k, v, **kwargs):
                    calls.append((q.stride(), kwargs))
                    output = torch.nn.functional.scaled_dot_product_attention(
                        *(t.transpose(1, 2) for t in (q, k, v)),
                        is_causal=kwargs["causal"],
                        scale=kwargs["softmax_scale"],
                        enable_gqa=True,
                    ).transpose(1, 2)
                    return (output, None) if name == "fa4" else output

                backend = Backend(name, flash, types.SimpleNamespace(), name)
                q = torch.randn(1, 4, 7, 8, dtype=torch.float64, requires_grad=True)
                k, v = [torch.randn(1, 2, 7, 8, dtype=q.dtype, requires_grad=True) for _ in range(2)]
                grad = torch.randn_like(q)
                actual = backend(q, k, v, is_causal=True, scale=0.25, enable_gqa=True)
                expected = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=True, scale=0.25, enable_gqa=True
                )
                for got, want in zip(
                    (actual, *torch.autograd.grad(actual, (q, k, v), grad)),
                    (expected, *torch.autograd.grad(expected, (q, k, v), grad)),
                ):
                    torch.testing.assert_close(got, want, atol=1e-12, rtol=1e-12)
                self.assertEqual(calls[0][0], q.transpose(1, 2).stride())
                self.assertEqual("dropout_p" in calls[0][1], name == "fa2")

    def test_rectangular_causal_and_d512_are_explicitly_unsupported(self):
        for name in ("fa2", "fa3", "fa4"):
            backend = Backend(name, lambda *a, **kw: self.fail("Unexpected backend launch"), None, name)
            with self.assertRaises(UnsupportedCase):
                backend(torch.empty(1, 2, 5, 64), torch.empty(1, 2, 7, 64), torch.empty(1, 2, 7, 64), is_causal=True)
        backend = Backend("fa2", lambda *a, **kw: self.fail("Unexpected backend launch"), None, "flash-attn")
        q = torch.empty(1, 2, 5, 512)
        with self.assertRaises(UnsupportedCase):
            backend(q, q, q)

    def test_metadata_identifies_loaded_extension_without_recording_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "wrapper.py"
            source.write_text("# selected entry point\n")
            binary = root / "extension.so"
            binary.write_bytes(b"\x7fELFselected binary")
            extension = types.ModuleType("flash_attn_test_extension")
            extension.__file__ = str(binary)
            entrypoint = types.ModuleType("flash_attn_test_entrypoint")
            entrypoint.__file__ = str(source)
            function = lambda *a, **kw: None
            function.__module__ = entrypoint.__name__
            with patch.dict(sys.modules, {extension.__name__: extension, entrypoint.__name__: entrypoint}):
                record = Backend("fa2", function, entrypoint, "absent-test-distribution").metadata()
            self.assertEqual(
                record["loaded_flash_extension_sha256"][extension.__name__], hashlib.sha256(binary.read_bytes()).hexdigest()
            )
            self.assertEqual(record["entrypoint_source_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertNotIn(directory, json.dumps(record))

    def test_external_configuration_paths_are_identified_by_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {"source": directory, "stages": 2, "options": [False, "shuffle", {"file": directory}]}
            metadata = configuration_metadata(config)
            self.assertEqual(metadata["stages"], 2)
            self.assertEqual(metadata["options"][:2], [False, "shuffle"])
            self.assertEqual(metadata["source"], {"external_path_sha256": hashlib.sha256(directory.encode()).hexdigest()})
            self.assertEqual(metadata["options"][2]["file"], metadata["source"])
            self.assertNotIn(directory, json.dumps(metadata))
            self.assertEqual(config["source"], directory)

    def test_missing_or_broken_standalone_backend_does_not_fall_back(self):
        for error in (ModuleNotFoundError("missing backend"), OSError("broken compiled extension")):
            with self.subTest(error=type(error).__name__):
                with patch("benchmarks.flashattention_backends.importlib.import_module", side_effect=error) as imported:
                    with self.assertRaises(type(error)):
                        load_backend("fa3")
                    imported.assert_called_once_with("flash_attn_3.flash_attn_interface")


if __name__ == "__main__":
    unittest.main()
