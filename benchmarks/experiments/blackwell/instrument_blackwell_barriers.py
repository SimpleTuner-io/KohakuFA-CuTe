"""Generate a separate Blackwell kernel source with matrix-warp wait timers."""

import argparse
import hashlib
import json
from pathlib import Path

FIELDS = (
    "TMA right_ready",
    "pointwise p_ready",
    "PV completion p_free",
    "score read s_free",
    "dP read dp_free",
    "pointwise ds_ready",
    "dO TMA other_ready",
    "stationary TMA left_ready",
    "total MMA role",
)
BARRIERS = dict(
    zip(("right_ready", "p_ready", "p_free", "s_free", "dp_free", "ds_ready", "other_ready", "left_ready"), range(8))
)


def instrument(source):
    marker = source.index("class Backward(Forward):")
    prefix, body = source[:marker], source[marker:]
    call = "            q.shape[3] > 128,\n"
    signature = "args, streamed: cutlass.Constexpr\n"
    if body.count(call) != 1 or body.count(signature) != 1:
        raise ValueError("Backward entrypoint changed; inspect the source before instrumenting it.")
    body = body.replace(call, call + "            blocks,\n", 1)
    body = body.replace(signature, "args, streamed: cutlass.Constexpr, profile\n", 1)
    entry = "        tile, bh, wi = cute.arch.block_idx()\n"
    body = body.replace(entry, entry + '        profile_mode = 0 if cutlass.const_expr(self.mode == "q") else 1\n', 1)
    start = body.index("        elif tid >= 256 and tid < 288:")
    stop = body.index("        elif tid < 256 and cutlass.const_expr(self.columns == 128):", start)
    lines = body[start:stop].splitlines()
    additions = [f"            wait_{index} = cutlass.Int32(0)" for index in range(8)]
    section = "\n".join([lines[0], *additions, "            role_begin = cute.arch.clock()", *lines[1:]]) + "\n"
    output, position, sites = [], 0, {name: 0 for name in BARRIERS}
    while True:
        begin = section.find("cute.arch.mbarrier_wait(", position)
        if begin < 0:
            output.append(section[position:])
            break
        line = section.rfind("\n", 0, begin) + 1
        indent = section[line:begin]
        end = begin + len("cute.arch.mbarrier_wait(")
        argument_start, depth = end, 1
        while depth:
            if section[end] == "(":
                depth += 1
            elif section[end] == ")":
                depth -= 1
            end += 1
        barrier = section[argument_start : end - 1].split(",")[0].strip().split(" + ")[0]
        index = BARRIERS[barrier]
        sites[barrier] += 1
        output.extend(
            (
                section[position:line],
                indent + "wait_begin = cute.arch.clock()\n",
                section[line:end] + "\n",
                indent + f"wait_{index} += cute.arch.clock() - wait_begin\n",
            )
        )
        position = end + 1
    section = "".join(output)
    section += "            role_elapsed = cute.arch.clock() - role_begin\n            if tid == 256:\n"
    for index in range(9):
        value = f"wait_{index}" if index < 8 else "role_elapsed"
        section += f"                profile[profile_mode, bh, tile * cute.ceil_div(q.shape[3], self.width) + wi, {index}] = {value}\n"
    return prefix + body[:start] + section + body[stop:], sites


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("Write instrumentation to a separate source file, preserving the production kernel.")
    source = args.source.read_text()
    result, sites = instrument(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result)
    metadata = {
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "instrumented_sha256": hashlib.sha256(result.encode()).hexdigest(),
        "fields": FIELDS,
        "static_wait_sites": sites,
        "scope": "Matrix warp only; clock32 cycle deltas include instrumentation overhead. These are not hardware stall counters or whole-kernel utilization.",
    }
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
