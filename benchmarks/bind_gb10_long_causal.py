"""Retain actual GB10 module-load images and associate exact graph resources."""

import argparse
import ctypes
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.parent
    root.mkdir(parents=True, exist_ok=True)
    dump = root / "compiler"
    dump.mkdir(exist_ok=True)
    os.environ.update(CUTE_DSL_KEEP="ptx,cubin", CUTE_DSL_DUMP_DIR=str(dump), CUTE_DSL_ARCH="sm_121a")
    cupti = ctypes.CDLL("/usr/local/cuda-13.2/targets/sbsa-linux/lib/libcupti.so")

    class Resource(ctypes.Structure):
        _fields_ = [("context", ctypes.c_void_p), ("handle", ctypes.c_void_p), ("descriptor", ctypes.c_void_p)]

    class Module(ctypes.Structure):
        _fields_ = [("id", ctypes.c_uint32), ("size", ctypes.c_size_t), ("cubin", ctypes.c_void_p)]

    callback_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32, ctypes.c_void_p)
    loaded = []
    callback_errors = []
    scope = "initialization"

    @callback_type
    def callback(userdata, domain, cbid, info):
        if domain != 3 or cbid != 6:
            return
        try:
            resource = ctypes.cast(info, ctypes.POINTER(Resource)).contents
            module = ctypes.cast(resource.descriptor, ctypes.POINTER(Module)).contents
            data = ctypes.string_at(module.cubin, module.size)
            digest = hashlib.sha256(data).hexdigest()
            assert data[:4] == b"\x7fELF", "Module resource payload is not ELF."
            (root / (digest + ".cubin")).write_bytes(data)
            loaded.append({"scope": scope, "module_id": module.id, "bytes": len(data), "sha256": digest})
        except Exception as error:
            callback_errors.append(type(error).__name__ + ": " + str(error))

    subscriber = ctypes.c_void_p()
    cupti.cuptiSubscribe.argtypes = [ctypes.POINTER(ctypes.c_void_p), callback_type, ctypes.c_void_p]
    cupti.cuptiEnableCallback.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32]
    status = cupti.cuptiSubscribe(ctypes.byref(subscriber), callback, None)
    assert status == 0, ("cuptiSubscribe", status)
    status = cupti.cuptiEnableCallback(1, subscriber, 3, 6)
    assert status == 0, ("cuptiEnableCallback", status)

    import torch

    from benchmarks.compare_ada import capture
    from benchmarks.compare_l40s import graph_kernels
    from benchmarks.flashattention_backends import load_backend
    from benchmarks.gb10_native_head_groups import make_long_public

    torch.manual_seed(2031)
    q, k, v, dout = [torch.randn(1, 4, 4096, 128, device="cuda", dtype=torch.float16) for _ in range(4)]
    functions = {"before": make_long_public(previous=True), "selected": make_long_public(), "fa2": load_backend("fa2")}
    graphs, owners, nodes = {}, {}, {}
    for name, function in functions.items():
        scope = name
        graphs[name], owners[name] = capture(function, (q, k, v), dout, True, "joint", keep_graph=True)
        nodes[name] = graph_kernels(graphs[name])
    cupti.cuptiUnsubscribe.argtypes = [ctypes.c_void_p]
    assert cupti.cuptiUnsubscribe(subscriber) == 0
    assert not callback_errors, callback_errors
    wanted = {r["symbol"] for rows in nodes.values() for r in rows}
    records = []
    for digest in sorted({r["sha256"] for r in loaded}):
        path = root / (digest + ".cubin")
        resource = subprocess.run(
            ["cuobjdump", "--dump-resource-usage", str(path)], capture_output=True, text=True, check=True
        ).stdout
        regs = {name: int(number) for name, number in re.findall(r"Function (\S+):\s+REG:(\d+)", resource)}
        if not (wanted & regs.keys()):
            continue
        (root / (digest + ".resources")).write_text(resource)
        sass = subprocess.run(
            ["cuobjdump", "--dump-sass", str(path)], capture_output=True, text=True, check=True
        ).stdout
        for part in re.split(r"Function :\s+", sass)[1:]:
            symbol = part.splitlines()[0].strip()
            if symbol not in wanted:
                continue
            text = "Function : " + part
            sass_sha = hashlib.sha256(text.encode()).hexdigest()
            (root / (sass_sha + ".sass")).write_text(text)
            ops = re.findall(r"/\*[0-9a-f]+\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Za-z0-9_.]+)", part)
            records.append(
                {
                    "symbol": symbol,
                    "registers": regs[symbol],
                    "loaded_cubin_sha256": digest,
                    "sass_sha256": sass_sha,
                    "whole_function_opcodes": {op: ops.count(op) for op in sorted(set(ops))},
                }
            )
    bindings = {}
    for name, rows in nodes.items():
        bindings[name] = []
        for node in rows:
            candidates = [r for r in records if r["symbol"] == node["symbol"] and r["registers"] == node["NUM_REGS"]]
            bindings[name].append(
                {
                    "graph": node,
                    "status": "unique" if len(candidates) == 1 else "ambiguous_or_missing",
                    "candidates": candidates,
                }
            )
    timing_matches = {}
    compiler_hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p in dump.rglob("*.cubin")}
    headers = Path("/usr/local/cuda-13.2/targets/sbsa-linux/include/cupti_callbacks.h")
    report = {
        "scope": "Actual CUPTI MODULE_LOADED payload hashes. Graph symbols and register resources matched against loaded ELF; ambiguity is explicit. Whole-function instruction counts are static and are not normalized runtime percentages.",
        "callback_header_sha256": hashlib.sha256(headers.read_bytes()).hexdigest(),
        "callback_structure_sizes": [ctypes.sizeof(Resource), ctypes.sizeof(Module)],
        "loaded": loaded,
        "bindings": bindings,
        "timing_graph_matches": timing_matches,
        "compiler_export_loaded_hash_matches": sorted(compiler_hashes & {r["sha256"] for r in loaded}),
        "status": "pass" if all(r["status"] == "unique" for rows in bindings.values() for r in rows) else "incomplete",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print({name: [r["status"] for r in rows] for name, rows in bindings.items()}, flush=True)


if __name__ == "__main__":
    main()
