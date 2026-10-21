"""Generate a separate Blackwell source with sampled, shared-buffered role traces.

Each selected CTA records inner steps N and N+1 using globaltimer's low 32 bits
(nanoseconds). Only warp leaders write, into 20 KiB of shared memory; a coalesced
copy exports records after the kernel's final CTA join. This changes scheduling
and resources: compare against an uninstrumented control before interpreting it.
MMA issue intervals are issuing-thread time, not tensor-core execution intervals.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path

ROLES = ("pointwise half 0", "pointwise half 1", "MMA issuer", "TMA loader", "statistics loader")
RECORDS_PER_STEP = 256
CAPACITY = 512


def instrument(source, sample_step=8, tile_stride=8):
    if "def trace_event(" in source or "def global_timer(" in source:
        raise ValueError("Use an uninstrumented source snapshot.")
    original = source
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    cls = next((n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "Backward"))
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    changes, events = ([], {})
    eid = 0

    def replace(start, end, text, priority=20):
        changes.append((start, end, text, priority))

    def insert(line, text, priority=20):
        replace(offsets[line], offsets[line], text, priority)

    call = source.index("            q.shape[3] > 128,", offsets[methods["__call__"].lineno - 1])
    insert(source[:call].count("\n") + 1, "            blocks,\n")
    k = methods["kernel"]
    pos = source.index("streamed: cutlass.Constexpr", offsets[k.lineno - 1])
    replace(pos, pos + len("streamed: cutlass.Constexpr"), "streamed: cutlass.Constexpr, profile")
    line = (
        next(
            (
                n
                for n in k.body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and (n.targets[0].id == "tmem")
            )
        ).lineno
        - 1
    )
    trace_tuple = f"(trace_buffer, bh == 0 and wi == 0 and tile % {tile_stride} == 0 and STEP, tid, tid // 128 if tid < 256 else 2 if tid < 288 else 3 if tid < 320 else 4)"
    insert(
        line,
        """        trace_buffer = allocator.allocate_tensor(cutlass.Int32, cute.make_layout((5,512,2)), byte_alignment=128)
        for i in cutlass.range(cute.ceil_div(5120,384),unroll=1):
            index = i * 384 + tid
            if index < 5120:
                trace_buffer[index // 1024, (index // 2) % 512, index % 2] = cutlass.Int32(0)
        trace = """
        + trace_tuple.replace("STEP", "True")
        + "\n        trace_count = cutlass.Int32(0)\n",
    )
    for n in ast.walk(k):
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and (n.target.id == "step"):
            indent = " " * (n.col_offset + 4)
            insert(
                n.body[0].lineno - 1,
                indent
                + "trace_count = cutlass.Int32(step * 256)\n"
                + indent
                + "trace_count = self.trace_event(trace, trace_count, 0)\n",
                10,
            )
            insert(n.end_lineno, indent + "trace_count = self.trace_event(trace, trace_count, 1)\n", 10)
    selected = {
        "mbarrier_wait",
        "issue",
        "issue_pv",
        "barrier",
        "read_matrix",
        "write_matrix",
        "reconstruct_probabilities",
        "stream_stats",
        "store_output",
        "box_rows",
    }
    for name in ("kernel", "stream_pair", "stream_right", "stream_stats"):
        fn = methods[name]
        if name != "kernel":
            last = fn.args.args[-1]
            end = offsets[last.end_lineno - 1] + last.end_col_offset
            replace(end, end, ", trace, trace_count")
            insert(fn.end_lineno, "        return trace_count\n")
        for n in ast.walk(fn):
            if not isinstance(n, (ast.Assign, ast.Expr)):
                continue
            value = n.value
            if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Attribute):
                continue
            attr = value.func.attr
            if attr in ("stream_pair", "stream_right", "stream_stats") and name == "kernel":
                start = offsets[n.lineno - 1] + n.col_offset
                end = offsets[n.end_lineno - 1] + n.end_col_offset
                seg = source[start:end]
                seg = seg[: seg.rfind(")")].rstrip().rstrip(",") + ", trace, trace_count)"
                replace(start, end, "trace_count = " + seg)
                continue
            if attr not in selected:
                continue
            eid += 1
            events[eid] = {
                "function": name,
                "line": n.lineno,
                "operation": ast.get_source_segment(source, value.func),
                "code": ast.get_source_segment(source, n),
            }
            indent = " " * n.col_offset
            insert(n.lineno - 1, indent + f"trace_count = self.trace_event(trace, trace_count, {eid * 2})\n")
            insert(n.end_lineno, indent + f"trace_count = self.trace_event(trace, trace_count, {eid * 2 + 1})\n", 0)
    line = next(
        (
            n
            for n in reversed(k.body)
            if isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Attribute)
            and (n.value.func.attr == "sync_threads")
        )
    ).lineno
    insert(
        line,
        """        profile_mode = 0 if cutlass.const_expr(self.mode == "q") else 1
        if bh == 0 and wi == 0 and tile % TILE_STRIDE == 0:
            for i in cutlass.range(cute.ceil_div(5120,384),unroll=1):
                index = i * 384 + tid
                if index < 5120:
                    role, item, field = index // 1024, (index // 2) % 512, index % 2
                    profile[profile_mode, tile // TILE_STRIDE, role, item, field] = trace_buffer[role,item,field]
""".replace(
            "TILE_STRIDE", str(tile_stride)
        ),
    )
    helper = """
    @cute.jit
    def trace_event(self, trace, count, event: cutlass.Constexpr):
        buffer, enabled, tid, role = trace
        if enabled and count >= SAMPLE_BEGIN and count < SAMPLE_END and (tid == 0 or tid == 128 or tid == 256 or tid == 288 or tid == 320):
            buffer[role,count - SAMPLE_BEGIN,0] = cutlass.Int32(global_timer())
            buffer[role,count - SAMPLE_BEGIN,1] = cutlass.Int32(event)
        return count + 1
"""
    helper = helper.replace("SAMPLE_BEGIN", str(sample_step * 256)).replace("SAMPLE_END", str((sample_step + 2) * 256))
    insert(cls.end_lineno, helper)
    stats = methods["stream_stats"]
    eid += 1
    events[eid] = {
        "function": "stream_stats",
        "line": stats.lineno,
        "operation": "statistics payload",
        "code": "Load four scalar statistics per query row, sync warp, and publish if applicable.",
    }
    load_loop = next((n for n in stats.body if isinstance(n, ast.For)))
    insert(load_loop.lineno - 1, "        trace_count = self.trace_event(trace, trace_count, " + str(eid * 2) + ")\n")
    insert(stats.end_lineno, "        trace_count = self.trace_event(trace, trace_count, " + str(eid * 2 + 1) + ")\n", 0)
    for start, end, text, priority in sorted(changes, key=lambda x: (x[0], x[1], x[3]), reverse=True):
        source = source[:start] + text + source[end:]
    imports = """from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

@dsl_user_op
def global_timer(*, loc=None, ip=None):
    return cutlass.Uint64(llvm.inline_asm(T.i64(), [], "mov.u64 $0, %globaltimer;", "=l", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))

"""
    source = source.replace("class Forward(NativeForward):", imports + "class Forward(NativeForward):", 1)
    ast.parse(source)
    return (
        source,
        {
            "source_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "instrumented_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "sample_step": sample_step,
            "sample_steps": 2,
            "tile_stride": tile_stride,
            "sample_head": 0,
            "sample_output_slice": 0,
            "roles": ROLES,
            "capacity": CAPACITY,
            "records_per_step": RECORDS_PER_STEP,
            "shared_trace_bytes": 20480,
            "clock": "globaltimer low 32 bits, nanoseconds, wraps every 4.295 seconds",
            "events": events,
            "scope": "Instrumented thread intervals, not hardware stalls or tensor-core utilization. Includes observer overhead; operands remain asynchronous.",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-step", type=int, default=8)
    parser.add_argument("--tile-stride", type=int, default=8)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("Preserve production source; write to a separate file.")
    if args.sample_step < 1 or args.tile_stride < 1:
        parser.error("Use positive sample-step and tile-stride values.")
    result, metadata = instrument(args.source.read_text(), args.sample_step, args.tile_stride)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result)
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
