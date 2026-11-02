import unittest
from pathlib import Path

from benchmarks.diagnostics.tmem_init import compile_commands, native_architecture, validate_images


class TmemBuildTests(unittest.TestCase):
    def test_generic_architectures_are_not_silent_fallbacks(self):
        for value in ("sm_100", "sm_103", "sm_120", "native"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                native_architecture(value)

    def test_host_and_cubin_have_explicit_real_architecture(self):
        commands = compile_commands("nvcc", "sm_100a", "include", "lib64")
        for phase in ("host", "cubin"):
            self.assertIn("-arch=compute_100a", commands[phase])
            self.assertIn("-code=sm_100a", commands[phase])
        self.assertIn("-rpath=$ORIGIN/lib", commands["host"])
        self.assertIn("--cudart", commands["host"])
        self.assertEqual(commands["host"][commands["host"].index("--cudart") + 1], "shared")
        self.assertNotIn("-code=sm_100a", commands["ptx"])

    def test_no_ptx_message_is_not_an_embedded_ptx_image(self):
        self.assertEqual(
            validate_images("No PTX file found to extract", "ELF file    1: probe.sm_100a.cubin\n", "sm_100a"), 1
        )

    def test_positive_ptx_and_wrong_or_absent_native_images_are_rejected(self):
        cases = (
            ("PTX file 1: probe.sm_100a.ptx", "ELF file 1: probe.sm_100a.cubin"),
            ("", "ELF file 1: probe.sm_103a.cubin"),
            ("", "No ELF images found"),
        )
        for ptx, elf in cases:
            with self.subTest(ptx=ptx), self.assertRaises(ValueError):
                validate_images(ptx, elf, "sm_100a")

    def test_measured_cpp_source_stays_byte_identical(self):
        import hashlib

        path = Path(__file__).parents[1] / "benchmarks/diagnostics/tmem_init.cu"
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(), "be2b4e20e2dabe868f343e0464aaee5c6d557b802a62d1724086714bba6e3464"
        )


if __name__ == "__main__":
    unittest.main()
