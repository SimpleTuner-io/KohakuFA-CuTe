"""Associate initcheck reports with one retained, uninstrumented SASS function.

Inline source information and equal ELF bytes strengthen source association;
they do not capture the instrumented image or establish an initialization waiver.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

MARKER = re.compile(r"^=+\s+Uninitialized __(shared|global)__ memory read of size (\d+) bytes\s*$", re.MULTILINE)
INSTRUCTION = re.compile(r"/\*([0-9a-fA-F]+)\*/\s+(.*?)\s*;", re.MULTILINE)


def selected_function(sass, symbol):
    functions = re.split(r"(?m)^\s*Function\s*:\s*", sass)
    matches = [part for part in functions[1:] if part.splitlines()[0].strip() == symbol]
    if len(matches) != 1:
        raise ValueError("The selected symbol must occur exactly once in the SASS.")
    return [(int(pc, 16), instruction) for pc, instruction in INSTRUCTION.findall(matches[0])]


def inline_location(sass, symbol, pc):
    selected = False
    pending, context = [], None
    for line in sass.splitlines():
        section = re.search(r"\.section\s+\.text\.([^,\s]+)", line)
        if section:
            selected = section.group(1) == symbol
            pending, context = [], None
        if not selected:
            continue
        if "//## File" in line:
            pending.append(line)
        instruction = INSTRUCTION.search(line)
        if instruction:
            if pending:
                inline = [re.search(r", line (\d+) inlined at .*?, line (\d+)", text) for text in pending]
                inline = [match for match in inline if match]
                context = None if not inline else tuple(map(int, inline[0].groups()))
                pending = []
            if int(instruction.group(1), 16) == pc:
                return context
    return None


def constant_address_candidate(instructions, selected):
    """Do not infer constants across register writes, including a cluster LEA."""
    address = re.search(r"\[((?:UR|R)\d+)(?:\+(0x[0-9a-fA-F]+))?\]", instructions[selected][1])
    if address is None:
        return None
    register, offset = address.groups()
    for _, earlier in reversed(instructions[:selected]):
        literal = re.fullmatch(r"(?:U?MOV) " + register + r", (0x[0-9a-fA-F]+|\d+)", earlier)
        if literal:
            return int(literal.group(1), 0) + int(offset or "0", 0)
        if re.match(r"(?:@!?\w+\s+)?\w+(?:\.\w+)* " + register + r"(?:,|\s)", earlier):
            return None
    return None


def associate(log, sass, symbol=None, inline_sass=None):
    starts = list(MARKER.finditer(log))
    if not starts:
        raise ValueError("No initialization report was present; absence is not a clean result.")
    reports = []
    for index, marker in enumerate(starts):
        block = log[marker.end() : starts[index + 1].start() if index + 1 < len(starts) else len(log)]
        location = re.search(r"^=+\s+at (.+)\+0x([0-9a-fA-F]+)(?: in .*?:(\d+))?\s*$", block, re.MULTILINE)
        address = re.search(r"^=+\s+Address (0x[0-9a-fA-F]+)\s*$", block, re.MULTILINE)
        if location is None or address is None:
            raise ValueError("Truncated or unsupported report; do not associate it with another error's address.")
        reported, pc_hex, source_line = location.groups()
        kernel = reported if symbol is None else symbol
        if kernel != reported:
            frame = re.search(r"Device Frame: (.+?)\+0x[0-9a-fA-F]+", block)
            if frame is None or frame.group(1) != kernel:
                raise ValueError("An inline report needs its explicitly selected containing-kernel frame.")
        pc = int(pc_hex, 16)
        instructions = selected_function(sass, kernel)
        selected = [i for i, item in enumerate(instructions) if item[0] == pc]
        if len(selected) != 1:
            raise ValueError("The reported PC must select exactly one instruction.")
        candidate = constant_address_candidate(instructions, selected[0])
        lineage = inline_location(inline_sass, kernel, pc) if inline_sass is not None else None
        reports.append(
            {
                "space": marker.group(1),
                "bytes": int(marker.group(2)),
                "reported_function": reported,
                "selected_kernel": kernel,
                "pc": hex(pc),
                "address": address.group(1),
                "source_line": None if source_line is None else int(source_line),
                "instruction": instructions[selected[0]][1],
                "nearest_literal_address_candidate": None if candidate is None else hex(candidate),
                "candidate_matches_report": None if candidate is None else candidate == int(address.group(1), 16),
                "is_shared_atomic_read_modify_write": instructions[selected[0]][1].startswith("ATOMS."),
                "inline_lineage": None if lineage is None else {"line": lineage[0], "call_site": lineage[1]},
            }
        )
    return {
        "association": "Uninstrumented artifact association; runtime loaded-code binding remains required.",
        "address_scope": "Literal register assignments only; cluster terms and branch dominance require review.",
        "instrumented_elf_captured": False,
        "log_was_terminated": "Terminated" in log,
        "reports": reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("log", "sass", "cubin", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--symbol", help="Explicit containing kernel for an inline report.")
    parser.add_argument("--inline-sass", type=Path, help="nvdisasm -gi output for that uninstrumented ELF.")
    parser.add_argument(
        "--loaded-cubin", type=Path, help="Independently captured uninstrumented ELF to compare byte for byte."
    )
    args = parser.parse_args()
    artifacts = {name: getattr(args, name).read_bytes() for name in ("log", "sass", "cubin")}
    if not artifacts["cubin"].startswith(b"\x7fELF"):
        raise ValueError("A real retained ELF is required.")
    inline = None
    if args.inline_sass is not None:
        artifacts["inline_sass"] = args.inline_sass.read_bytes()
        inline = artifacts["inline_sass"].decode()
    report = associate(artifacts["log"].decode("utf-8", "replace"), artifacts["sass"].decode(), args.symbol, inline)
    if args.loaded_cubin is not None:
        loaded = args.loaded_cubin.read_bytes()
        if loaded != artifacts["cubin"]:
            raise ValueError("The captured uninstrumented ELF differs from the compiler artifact.")
        report["provided_uninstrumented_elf_byte_equal"] = True
        artifacts["provided_uninstrumented_elf"] = loaded
    report["artifact_sha256"] = {name: hashlib.sha256(data).hexdigest() for name, data in artifacts.items()}
    report["tool_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
