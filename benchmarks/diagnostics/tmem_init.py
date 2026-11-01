"""Build the standalone allocator diagnostic without initializing CUDA."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_architecture(architecture):
    if architecture not in ("sm_100a", "sm_103a"):
        raise ValueError("Select a datacentre Blackwell architecture explicitly.")
    return architecture.replace("sm_", "compute_", 1)


def validate_images(ptx_listing, elf_listing, architecture):
    if re.search(r"^PTX file\s+\d+:", ptx_listing, re.MULTILINE | re.IGNORECASE):
        raise ValueError("The executable must contain native code only, without a PTX fallback.")
    images = re.findall(r"^ELF file\s+\d+:\s*(.+)$", elf_listing, re.MULTILINE)
    if not images or any(f".{architecture}." not in image for image in images):
        raise ValueError("Every embedded ELF must match the selected architecture.")
    return len(images)


def compile_commands(nvcc, architecture, include, library):
    base = [
        str(nvcc),
        "--cudart",
        "shared",
        "-std=c++17",
        "-O2",
        "-lineinfo",
        "-arch=" + native_architecture(architecture),
        "-I" + str(include),
        "tmem_init.cu",
    ]
    return {
        "host": base
        + ["-code=" + architecture, "-L" + str(library), "-Xlinker", "-rpath=$ORIGIN/lib", "-lcupti", "-o", "tmem_init"],
        "ptx": base + ["--ptx", "-o", "source.ptx"],
        "cubin": base + ["-code=" + architecture, "--cubin", "-o", "source.cubin"],
    }


def run(command, root, filename):
    with (root / filename).open("x") as output:
        subprocess.run(command, cwd=root, stdout=output, stderr=subprocess.STDOUT, check=True)


def copy_library(source, destination):
    actual = source.resolve(strict=True)
    shutil.copy2(actual, destination / actual.name)
    metadata = subprocess.check_output(["readelf", "-d", str(actual)], text=True)
    sonames = re.findall(r"SONAME.*?\[(.*?)\]", metadata)
    if len(sonames) != 1:
        raise ValueError("A unique CUDA library SONAME is required.")
    if sonames[0] != actual.name:
        (destination / sonames[0]).symlink_to(actual.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("sm_100a", "sm_103a"), required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh directory outside the checkout.")
    parser.add_argument(
        "--sanitizer-dir", type=Path, required=True, help="Checked newer tool directory containing compute-sanitizer."
    )
    parser.add_argument("--cupti-root", type=Path, help="Root containing include/cupti.h and lib64/libcupti.so.")
    args = parser.parse_args()
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        parser.error("Configure the CUDA toolkit in PATH before building.")
    nvcc = Path(nvcc).resolve()
    cuda = nvcc.parent.parent
    cupti = cuda if args.cupti_root is None else args.cupti_root.resolve()
    if not (cupti / "include/cupti.h").is_file() or not (cupti / "lib64/libcupti.so").is_file():
        parser.error("CUPTI development headers and library are required; set --cupti-root explicitly.")
    if not (args.sanitizer_dir / "compute-sanitizer").is_file():
        parser.error("The newer standalone sanitizer and its adjacent runtime libraries are required.")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_suffix(".cu")
    shutil.copy2(source, root / source.name)
    compiler = subprocess.check_output([str(nvcc), "--version"], text=True)
    commands = compile_commands(nvcc, args.arch, cupti / "include", cupti / "lib64")
    for phase, command in commands.items():
        run(command, root, f"build-{phase}.log")
    run(["cuobjdump", "--dump-sass", "source.cubin"], root, "source.sass")
    run(["nvdisasm", "-gi", "-hex", "-c", "source.cubin"], root, "source-inline.sass")
    run(["cuobjdump", "--dump-resource-usage", "source.cubin"], root, "resources.txt")
    run(["cuobjdump", "--list-ptx", "tmem_init"], root, "embedded-ptx.txt")
    run(["cuobjdump", "--list-elf", "tmem_init"], root, "embedded-elf.txt")
    images = validate_images((root / "embedded-ptx.txt").read_text(), (root / "embedded-elf.txt").read_text(), args.arch)
    dynamic = subprocess.check_output(["readelf", "-d", str(root / "tmem_init")], text=True)
    if "$ORIGIN/lib" not in dynamic:
        raise ValueError("The portable executable requires its origin-relative library search path.")
    (root / "dynamic.txt").write_text(dynamic)
    libraries = root / "lib"
    libraries.mkdir()
    copy_library(cupti / "lib64/libcupti.so", libraries)
    copy_library(cuda / "lib64/libcudart.so", libraries)
    needed = re.findall(r"NEEDED.*?\[(.*?)\]", dynamic)
    system_libraries = {
        "libstdc++.so.6",
        "libgcc_s.so.1",
        "libc.so.6",
        "libm.so.6",
        "libdl.so.2",
        "libpthread.so.0",
        "librt.so.1",
    }
    for name in needed:
        if name.startswith(("libcupti", "libcudart")):
            if not (libraries / name).is_file():
                raise ValueError("A required CUDA dependency is absent from the bundle.")
        elif name not in system_libraries:
            raise ValueError("Inspect the executable's new runtime dependency before running it.")
    loader = subprocess.check_output(
        ["/lib64/ld-linux-x86-64.so.2", "--list", str(root / "tmem_init")],
        text=True,
        env=dict(os.environ, LD_LIBRARY_PATH=str(libraries) + ":" + os.environ.get("LD_LIBRARY_PATH", "")),
    )
    for name in ("libcupti.so.13", "libcudart.so.13"):
        matches = [line for line in loader.splitlines() if name in line]
        if len(matches) != 1 or str(libraries) not in matches[0]:
            raise ValueError("The loader must resolve the bundled CUDA libraries.")
    (root / "loader-trace.txt").write_text(loader)
    shutil.copytree(args.sanitizer_dir, root / "sanitizer-new", symlinks=True)
    shutil.copy2(Path(__file__).with_name("run_tmem_init.sh"), root / "run_tmem_init.sh")
    report = {
        "scope": "CPU-only build and loader trace; executable main and CUDA APIs were not run",
        "architecture": args.arch,
        "compiler": compiler,
        "source_sha256": sha256(source),
        "embedded_native_images": images,
        "embedded_ptx": False,
        "needed": needed,
        "artifacts": {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()},
    }
    (root / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    (root / "SHA256SUMS").write_text(
        "".join(sha256(p) + "  " + str(p.relative_to(root)) + "\n" for p in sorted(root.rglob("*")) if p.is_file())
    )


if __name__ == "__main__":
    main()
