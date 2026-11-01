"""Build/run the standalone rank-4 TMA store diagnostic; retain loaded code."""

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sysconfig
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command, **kwargs):
    return subprocess.run(command, check=True, text=True, **kwargs)


def library_paths(cuda):
    """Prefer this toolkit's CUPTI; allow the pinned environment's split wheel."""
    roots = [cuda / "extras" / "CUPTI", cuda]
    site = Path(sysconfig.get_paths()["purelib"])
    roots.extend(sorted((site / "nvidia").glob("*")))
    for root in roots:
        headers = list(root.glob("include/cupti.h"))
        libraries = [*root.glob("lib64/libcupti.so*"), *root.glob("lib/libcupti.so*")]
        if headers and libraries:
            return headers[0].parent, sorted(libraries, key=lambda p: len(p.name))[0]
    raise RuntimeError("CUPTI headers/library are required for actual loaded-code retention")


def build(args):
    args.output.mkdir(parents=True, exist_ok=False)
    cuda = args.cuda.resolve()
    nvcc = cuda / "bin" / "nvcc"
    header, cupti = library_paths(cuda)
    source = Path(__file__).with_suffix(".cu")
    # Copy a source snapshot and compile relative to it. No machine directory
    # gets baked into line information or the retained source's metadata.
    shutil.copyfile(source, args.output / source.name)
    common = [str(nvcc), "-std=c++17", "-O2", "-lineinfo", "-arch=" + args.arch, source.name]
    binary = [
        *common,
        "-I" + str(header),
        "-L" + str(cupti.parent),
        "-L" + str(cuda / "lib64" / "stubs"),
        "-Xlinker=-rpath=" + str(cupti.parent),
        str(cupti),
        "-lcuda",
        "-o",
        "tma4d_store",
    ]
    with (args.output / "build.log").open("w") as log:
        run(binary, cwd=args.output, stdout=log, stderr=subprocess.STDOUT)
        run(
            [*common, "-I" + str(header), "--ptx", "-o", "source.ptx"],
            cwd=args.output,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    ptx = (args.output / "source.ptx").read_text()
    if not re.search(r"cp\.async\.bulk\.tensor\.4d\.global\.shared::cta", ptx):
        raise RuntimeError("Compilation did not retain the requested rank-4 S2G instruction")
    metadata = {
        "source_sha256": digest(source),
        "binary_sha256": digest(args.output / "tma4d_store"),
        "ptx_sha256": digest(args.output / "source.ptx"),
        "compiler": subprocess.check_output([str(nvcc), "--version"], text=True),
        "architecture": args.arch,
        "cupti_library": cupti.name,
        "cupti_library_sha256": digest(cupti),
        "compiler_options": ["-std=c++17", "-O2", "-lineinfo", "-arch=" + args.arch],
        "scope": "Source PTX is a compiler artifact; loaded ELF/SASS is retained separately on actual execution.",
    }
    (args.output / "build.json").write_text(json.dumps(metadata, indent=2) + "\n")


def function_body(sass, name):
    sections = re.split(r"(?m)^\s*Function\s*:\s*", sass)
    matches = [section for section in sections[1:] if section.splitlines()[0].strip() == name]
    if len(matches) != 1:
        raise RuntimeError("Selected runtime kernel must occur exactly once in the captured loaded ELF")
    return matches[0]


def inherited_lock_fds():
    """Keep the caller's GPU flock alive through this process's GPU children."""
    try:
        os.fstat(9)
    except OSError:
        return ()
    return (9,)


class Interrupted(Exception):
    def __init__(self, signum):
        self.signum = signum


class CaptureDisabled(Exception):
    """Keep unavailable instrumented capture distinct from a failed binding."""


def terminate_group(process):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        if sig == signal.SIGTERM:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
    # A sanitizer leader can exit before its target: SIGKILL above applies to
    # surviving group members even if waiting for the leader already succeeded.
    process.wait()


def execute_process(command, log, timeout, env=None):
    process = None

    def interrupted(signum, _frame):
        raise Interrupted(signum)

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
            pass_fds=inherited_lock_fds(),
            env=env,
        )
        return {"returncode": process.wait(timeout=timeout)}
    except (subprocess.TimeoutExpired, Interrupted) as error:
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        if process is not None:
            terminate_group(process)
        if isinstance(error, Interrupted):
            return {"returncode": 128 + error.signum, "signal": error.signum}
        return {"returncode": 124, "timeout_seconds": timeout}
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def validated_control(args, build_metadata):
    if args.capture != "disabled":
        return None
    if args.tool == "none" or args.control_result is None:
        raise RuntimeError("Disabled capture requires a sanitizer and an explicitly selected captured control")
    control = json.loads(args.control_result.read_text())
    for key in ("source_sha256", "binary_sha256", "ptx_sha256"):
        if control["build"][key] != build_metadata[key]:
            raise RuntimeError("Instrumented run must use the identical source and executable as its captured control")
    for key in ("mode", "initialization", "rows"):
        if control[key] != getattr(args, key):
            raise RuntimeError("Captured control must have matching producer, initialization and geometry")
    if (
        control.get("capture") != "enabled"
        or control.get("tool") != "none"
        or control.get("returncode") != 0
        or control.get("code_binding") != "passed"
        or control.get("physical_verification") != "passed"
        or not control.get("loaded_bindings")
    ):
        raise RuntimeError("The uninstrumented captured control has not passed exact loaded-code and physical checks")
    return {
        "result_sha256": digest(args.control_result),
        "binary_sha256": control["build"]["binary_sha256"],
        "source_sha256": control["build"]["source_sha256"],
        "loaded_bindings": control["loaded_bindings"],
        "scope": "Uninstrumented control only; instrumented loaded-code capture is disabled due to the CUPTI/sanitizer conflict.",
    }


def execute(args):
    args.output.mkdir(parents=True, exist_ok=False)
    binary = args.build / "tma4d_store"
    build_metadata = json.loads((args.build / "build.json").read_text())
    if digest(binary) != build_metadata["binary_sha256"]:
        raise RuntimeError("The executable changed after its recorded build")
    control = validated_control(args, build_metadata)
    command = [str(binary.resolve()), args.mode, args.initialization, str(args.rows), str(args.output.resolve())]
    if args.tool != "none":
        command = [
            "compute-sanitizer",
            "--tool",
            args.tool,
            "--error-exitcode",
            "99",
            "--print-limit",
            "12",
            *command,
        ]
    result = {
        "tool": args.tool,
        "mode": args.mode,
        "initialization": args.initialization,
        "rows": args.rows,
        "capture": args.capture,
        "execution_arguments": command,
        "runner_source_sha256": digest(Path(__file__)),
    }
    environment = dict(os.environ, TMA4D_CAPTURE=args.capture)
    try:
        with (args.output / "execution.log").open("w") as log:
            result.update(execute_process(command, log, args.timeout, env=environment))
    except OSError as error:
        result["returncode"] = 1
        result["launch_error"] = str(error)
    # Collect even failures. A sanitizer status 99 is not overwritten by matching
    # numerical values or successful code binding, and a timeout is not a pass.
    log = (args.output / "execution.log").read_text(errors="replace")
    result["sanitizer_error_summaries"] = re.findall(r"(?m)^.*ERROR SUMMARY:.*$", log)
    result["execution_log_sha256"] = digest(args.output / "execution.log")
    symbol = "tma4d_" + args.mode
    try:
        if control is not None:
            raise CaptureDisabled()
        launches = [json.loads(line) for line in (args.output / "launches.jsonl").read_text().splitlines()]
        if not any(row["symbol"] == symbol for row in launches):
            raise RuntimeError("CUPTI did not record the selected producer's eager launch")
        modules = sorted(args.output.glob("loaded-*.cubin"))
        bindings = []
        for cubin in modules:
            if not cubin.read_bytes().startswith(b"\x7fELF"):
                raise RuntimeError("CUPTI capture is not a loaded ELF image")
            sass = subprocess.check_output(["cuobjdump", "--dump-sass", str(cubin)], text=True)
            cubin.with_suffix(".sass").write_text(sass)
            body = function_body(sass, symbol)
            (args.output / (cubin.stem + "-selected.sass")).write_text(body)
            stores = re.findall(r"(?m)^.*UTMASTG[^;]*;.*$", body)
            if args.mode == "ordinary":
                if stores or not re.search(r"\bSTG\.", body):
                    raise RuntimeError("The selected ordinary-store control has an unexpected store family")
            elif not stores or any("UTMASTG.4D" not in line for line in stores):
                raise RuntimeError("The selected loaded producer is not exclusively UTMASTG.4D")
            bindings.append(
                {
                    "symbol": symbol,
                    "loaded_cubin": cubin.name,
                    "loaded_cubin_sha256": digest(cubin),
                    "selected_sass_sha256": digest(args.output / (cubin.stem + "-selected.sass")),
                    "tensor_store_lines": stores,
                    "disassembly_architectures": sorted(set(re.findall(r"code for (sm_[0-9a-z]+)", sass))),
                }
            )
        if len(bindings) != 1:
            raise RuntimeError("Expected one selected loaded module")
        result["loaded_bindings"] = bindings
        result["code_binding"] = "passed"
    except CaptureDisabled:
        result["code_binding"] = "unavailable_under_sanitizer"
        result["matching_uninstrumented_control"] = control
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        result["code_binding"] = "failed"
        result["code_binding_error"] = str(error)
    report_path = args.output / "report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text())
            result["physical_verification"] = "passed" if report["all_match"] else "failed"
            result["reported_capture_enabled"] = report["capture_enabled"]
            if report["capture_enabled"] != (args.capture == "enabled"):
                result["physical_verification"] = "failed_capture_configuration"
            result["capability"] = report["capability"]
            allocation = report["physical_allocation"]
            result["reported_read_addresses"] = []
            for address in dict.fromkeys(re.findall(r"Address (0x[0-9a-fA-F]+)", log)):
                value = int(address, 16)
                result["reported_read_addresses"].append(
                    {
                        "address": address,
                        "physical_output_offset": value - allocation["ptr"],
                        "within_output": allocation["ptr"] <= value < allocation["ptr"] + allocation["bytes"],
                    }
                )
            if not report["all_match"] and not result["returncode"]:
                result["returncode"] = 2
        except (ValueError, KeyError):
            result["physical_verification"] = "incomplete"
    else:
        result["physical_verification"] = "missing"
    result["build"] = build_metadata
    result["scope"] = "Functional instruction-family diagnostic on the recorded GPU, not H200 attention qualification."
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    binding_valid = result["code_binding"] == "passed" or (
        result["code_binding"] == "unavailable_under_sanitizer" and control is not None
    )
    if result["returncode"] or not binding_valid or result["physical_verification"] != "passed":
        raise SystemExit(result["returncode"] or 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    compiler = subparsers.add_parser("build")
    compiler.add_argument("--cuda", type=Path, default=Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")))
    compiler.add_argument("--arch", choices=["sm_90", "sm_120", "sm_121"], required=True)
    compiler.add_argument("--output", type=Path, required=True)
    compiler.set_defaults(function=build)
    runner = subparsers.add_parser("run")
    runner.add_argument("--build", type=Path, required=True)
    runner.add_argument("--mode", choices=["read", "full", "ordinary"], required=True)
    runner.add_argument("--initialization", choices=["uninitialized", "poison"], required=True)
    runner.add_argument("--rows", choices=[256, 257], type=int, required=True)
    runner.add_argument("--tool", choices=["none", "initcheck", "memcheck", "racecheck", "synccheck"], default="none")
    runner.add_argument("--capture", choices=["enabled", "disabled"], default="enabled")
    runner.add_argument("--control-result", type=Path)
    runner.add_argument("--timeout", type=int, default=45)
    runner.add_argument("--output", type=Path, required=True)
    runner.set_defaults(function=execute)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
