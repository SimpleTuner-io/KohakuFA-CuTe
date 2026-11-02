"""Bind exact A100 public group graph functions to CUPTI-loaded ELF images."""

import argparse
import ctypes
import ctypes.util
import hashlib
import json
import os
import re
import subprocess
from dataclasses import asdict
from pathlib import Path


def associate(records, loaded, name, node):
    candidates = [row for row in records if row["symbol"] == node["symbol"] and row["registers"] == node["NUM_REGS"]]
    if len(candidates) == 1:
        return candidates, "unique loaded ELF symbol and register resources"
    phase = {row["sha256"] for row in loaded if row["scope"] == name}
    selected = [row for row in candidates if row["loaded_cubin_sha256"] in phase]
    return selected, "loaded ELF symbol/resources disambiguated by load phase"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.parent
    root.mkdir(parents=True, exist_ok=True)
    library = ctypes.util.find_library("cupti")
    if not library:
        candidates = [
            *Path("/usr/local").glob("cuda*/targets/x86_64-linux/lib/libcupti.so*"),
            *Path("/usr/local").glob("cuda*/extras/CUPTI/lib64/libcupti.so*"),
        ]
        assert candidates, "CUPTI unavailable"
        library = str(candidates[0])
    cupti = ctypes.CDLL(library)

    class Resource(ctypes.Structure):
        _fields_ = [("context", ctypes.c_void_p), ("handle", ctypes.c_void_p), ("descriptor", ctypes.c_void_p)]

    class Module(ctypes.Structure):
        _fields_ = [("id", ctypes.c_uint32), ("size", ctypes.c_size_t), ("cubin", ctypes.c_void_p)]

    callback_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32, ctypes.c_void_p)
    loaded, errors = [], []
    scope = "initialization"

    @callback_type
    def callback(userdata, domain, cbid, info):
        if domain != 3 or cbid != 6:
            return
        try:
            resource = ctypes.cast(info, ctypes.POINTER(Resource)).contents
            module = ctypes.cast(resource.descriptor, ctypes.POINTER(Module)).contents
            data = ctypes.string_at(module.cubin, module.size)
            assert data[:4] == b"\x7fELF", "Loaded resource is not ELF"
            digest = hashlib.sha256(data).hexdigest()
            (root / (digest + ".cubin")).write_bytes(data)
            loaded.append(
                {
                    "scope": scope,
                    "module_id": module.id,
                    "bytes": len(data),
                    "sha256": digest,
                }
            )
        except Exception as error:
            errors.append(type(error).__name__)

    subscriber = ctypes.c_void_p()
    cupti.cuptiSubscribe.argtypes = [ctypes.POINTER(ctypes.c_void_p), callback_type, ctypes.c_void_p]
    cupti.cuptiEnableCallback.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32]
    assert cupti.cuptiSubscribe(ctypes.byref(subscriber), callback, None) == 0
    assert cupti.cuptiEnableCallback(1, subscriber, 3, 6) == 0
    import cuda.bindings.driver as cuda
    import torch

    from benchmarks.a100_d256_api_controls import make_attention
    from benchmarks.compare_ada import capture
    from benchmarks.compare_l40s import graph_kernels

    nodes, owners, node_modules = {}, [], {}
    functions = {"previous": make_attention(previous=True), "candidate": make_attention(previous=False)}
    torch.manual_seed(4127)
    for dtype, heads, dimension in ((torch.bfloat16, 8, 256), (torch.float16, 16, 256)):
        inputs = [torch.randn(1, heads, 1024, dimension, device="cuda", dtype=dtype) for _ in range(3)]
        dout = torch.randn_like(inputs[0])
        for name, function in functions.items():
            scope = name + "-h" + str(heads)
            graph, storage = capture(function, [t.detach() for t in inputs], dout, True, "joint", keep_graph=True)
            graph.replay()
            torch.cuda.synchronize()
            assert all(torch.isfinite(t).all() for t in storage[-1])
            nodes[scope] = graph_kernels(graph)
            status, graph_nodes, count = cuda.cuGraphGetNodes(cuda.CUgraph(graph.raw_cuda_graph()), 256)
            assert int(status) == 0
            handles = []
            for node in graph_nodes[:count]:
                status, kind = cuda.cuGraphNodeGetType(node)
                assert int(status) == 0
                if kind != cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
                    continue
                status, params = cuda.cuGraphKernelNodeGetParams(node)
                assert int(status) == 0
                if hasattr(cuda, "cuFuncGetModule"):
                    status, module = cuda.cuFuncGetModule(params.func)
                    handles.append(
                        {
                            "status": int(status),
                            "module": int(module) if not int(status) else None,
                            "function": int(params.func),
                        }
                    )
                else:
                    handles.append({"status": "cuFuncGetModule unavailable"})
            node_modules[scope] = handles
            owners.append((graph, storage, inputs, dout))
    cupti.cuptiUnsubscribe.argtypes = [ctypes.c_void_p]
    assert cupti.cuptiUnsubscribe(subscriber) == 0
    assert not errors, errors
    records = []
    for digest in sorted({row["sha256"] for row in loaded}):
        image = root / (digest + ".cubin")
        resources = subprocess.check_output(["cuobjdump", "--dump-resource-usage", str(image)], text=True)
        regs = {name: int(number) for name, number in re.findall(r"Function (\S+):\s+REG:(\d+)", resources)}
        wanted = {node["symbol"] for group in nodes.values() for node in group}
        if not wanted.intersection(regs):
            continue
        (root / (digest + ".resources")).write_text(resources)
        sass = subprocess.check_output(["cuobjdump", "--dump-sass", str(image)], text=True)
        ptx = subprocess.check_output(["cuobjdump", "--dump-ptx", str(image)], text=True)
        ptx_available = ".entry" in ptx
        if ptx_available:
            (root / (digest + ".ptx")).write_text(ptx)
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
                    "ptx": "retained" if ptx_available else "not packaged in loaded ELF",
                    "whole_function_opcodes": {op: ops.count(op) for op in sorted(set(ops))},
                }
            )
    bindings = {}
    for name, group in nodes.items():
        bindings[name] = []
        for node, handle in zip(group, node_modules[name]):
            candidates, method = associate(records, loaded, name, node)
            bindings[name].append(
                {
                    "graph": node,
                    "driver_handles": handle,
                    "association": method,
                    "status": "unique" if len(candidates) == 1 else "ambiguous_or_missing",
                    "candidates": candidates,
                }
            )
    report = {
        "scope": "CUPTI MODULE_LOADED ELF bytes matched by unique graph symbol/resources and loading phase. Driver module queries are retained separately; the callback resourceHandle is stream-only on this SDK. Static instruction counts are not runtime percentages.",
        "loaded": loaded,
        "bindings": bindings,
        "callback_errors": errors,
        "status": "pass" if all(row["status"] == "unique" for group in bindings.values() for row in group) else "incomplete",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print({name: [row["status"] for row in group] for name, group in bindings.items()}, flush=True)
    raise SystemExit(0 if report["status"] == "pass" else 2)


if __name__ == "__main__":
    main()
