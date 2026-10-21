"""Generate diagnostic KV role delays in a frozen register-auxiliary source.

Only use these modules with measure_blackwell_sensitivity: they read a delay
and write a sampled duration through its four-element native workspace argument.
They are not production kernels. Each generated role has a runtime delay input,
so all delay levels execute the same compiled instruction stream.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    r = args.output
    r.mkdir(parents=True, exist_ok=True)
    s = args.source.read_text()
    if "def busy_wait(" in s:
        parser.error("Use an uninstrumented source snapshot.")

    imports = """from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

@dsl_user_op
def busy_wait(ns, *, loc=None, ip=None):
    return cutlass.Int32(llvm.inline_asm(
        T.i32(), [cutlass.Uint32(ns).ir_value(loc=loc, ip=ip)],
        "{ .reg .u64 begin, now, elapsed, target; .reg .pred again; "
        "cvt.u64.u32 target, $1; mov.u64 begin, %globaltimer; "
        "repeat_delay: mov.u64 now, %globaltimer; sub.u64 elapsed, now, begin; "
        "setp.lt.u64 again, elapsed, target; @again bra repeat_delay; "
        "cvt.u32.u64 $0, elapsed; }", "=r,r", has_side_effects=True,
        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))

"""
    s = s.replace("class RegisterAux(Backward):", imports + "class RegisterAux(Backward):", 1)
    s = s.replace(
        "def __call__(self, q, k, v, maximum, logsum, dout, delta, base, dq, dk, dv, scale, stream):",
        "def __call__(self, q, k, v, maximum, logsum, dout, delta, base, dq, dk, dv, scale, stream, profile):",
        1,
    )
    s = s.replace("(la, lt, oa, ot, ra, rt, da, dt)).launch(", "(la, lt, oa, ot, ra, rt, da, dt), profile).launch(", 1)
    s = s.replace(
        "def kernel(self, q, k, maximum, logsum, dout, delta, base, dq, dk, dv, scale, args):",
        "def kernel(self, q, k, maximum, logsum, dout, delta, base, dq, dk, dv, scale, args, profile):",
        1,
    )
    s = s.replace(
        "self.q(q, k, v, maximum, logsum, dout, delta, base, dq, dk, dv, scale, stream)",
        "self.q(q, k, v, maximum, logsum, dout, delta, base, dq, dk, dv, scale, stream, workspace)",
        1,
    )
    s = s.replace(
        "self.kv(q, k, v, maximum, logsum, dout, delta, base, dq, dk, dv, scale, stream)",
        "self.kv(q, k, v, maximum, logsum, dout, delta, base, dq, dk, dv, scale, stream, workspace)",
        1,
    )
    sites = {
        "loader": (
            """                    for c in cutlass.range(nch, unroll=1):
                        self.stream_pair(""",
            """                    if cutlass.const_expr(self.mode == "kv"):
                        if tid == 416:
                            elapsed = busy_wait(profile[0, 0, 0, 0])
                            if tile == 0 and bh == 0 and step == 0:
                                profile[0, 0, 0, 1] = elapsed
                    for c in cutlass.range(nch, unroll=1):
                        self.stream_pair(""",
        ),
        "mma": (
            """                    for c in cutlass.range(nch, unroll=1):
                        slot = cursor % 5""",
            """                    if cutlass.const_expr(self.mode == "kv"):
                        if tid == 384:
                            elapsed = busy_wait(profile[0, 0, 0, 0])
                            if tile == 0 and bh == 0 and step == 0:
                                profile[0, 0, 0, 1] = elapsed
                    for c in cutlass.range(nch, unroll=1):
                        slot = cursor % 5""",
        ),
        "pointwise": (
            """                if local == 0:
                    cute.arch.mbarrier_arrive(wr)""",
            """                if local == 0:
                    if cutlass.const_expr(self.mode == "kv"):
                        elapsed = busy_wait(profile[0, 0, 0, 0])
                        if tile == 0 and bh == 0 and step == 0:
                            profile[0, 0, 0, 1] = elapsed
                    cute.arch.mbarrier_arrive(wr)""",
        ),
        "auxiliary": (
            """                if local == 0:
                    cute.arch.mbarrier_arrive(af)""",
            """                if local == 0:
                    if cutlass.const_expr(self.mode == "kv"):
                        elapsed = busy_wait(profile[0, 0, 0, 0])
                        if tile == 0 and bh == 0 and step == 0 and half == 0:
                            profile[0, 0, 0, 1] = elapsed
                    cute.arch.mbarrier_arrive(af)""",
        ),
    }
    metadata = {
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "scope": "Runtime delay in one KV role per module; driver-only workspace contract.",
        "roles": {},
    }
    for role, (old, new) in sites.items():
        assert old in s, role
        text = s.replace(old, new, 1)
        ast.parse(text)
        p = r / ("delay-" + role + ".py")
        p.write_text(text)
        metadata["roles"][role] = {"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "placement": new}
    (r / "delay-sites.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
