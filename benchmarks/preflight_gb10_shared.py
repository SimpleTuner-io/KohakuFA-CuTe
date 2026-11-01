"""Compile actual GB10 storage variants, inspect resources, and launch no kernels."""

import argparse
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import torch

from benchmarks.compiled_artifacts import retain_code
from kohakufa_cute.fa4_sm80._attention import Schedule
from kohakufa_cute.fa4_sm80._backward import FlashAttentionBackwardSm80
from kohakufa_cute.fa4_sm80._native import compile_kernel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    props = torch.cuda.get_device_properties(0)
    assert (props.major, props.minor, props.multi_processor_count) == (12, 1, 48)
    q = [torch.empty(1, 16, 4096, 128, device="cuda", dtype=torch.float16) for _ in range(8)]
    stats = [torch.empty(1, 16, 4096, device="cuda", dtype=torch.float32) for _ in range(4)]
    scratch = torch.empty(1, 16, 4096 * 128, device="cuda", dtype=torch.float32)
    inputs = [q[0], q[1], q[2], q[0], q[3], stats[0], stats[1], q[4], stats[2], q[5], q[6], q[7], scratch, stats[3]]
    records = []
    original = FlashAttentionBackwardSm80._get_shared_storage_cls
    observed = []

    def inspect(self):
        storage = original(self)
        observed.append(storage.size_in_bytes())
        assert storage.size_in_bytes() <= props.shared_memory_per_block_optin
        return storage

    FlashAttentionBackwardSm80._get_shared_storage_cls = inspect
    try:
        for variant in ("q2-register", "q1-shared", "q2-shared"):
            for group in (1, 4, 16):
                schedule = Schedule(
                    forward_m=64,
                    backward_n=64,
                    query_stages=1 if variant == "q1-shared" else 2,
                    dout_stages=1,
                    score_m_warps=4,
                    kv_m_warps=2,
                    dq_m_warps=2,
                    v_in_regs=variant == "q2-register",
                    causal_head_group=group,
                )
                observed.clear()
                function = compile_kernel("backward", inputs, True, asdict(schedule))
                directory = args.output.parent / (variant + "-g" + str(group))
                artifacts = retain_code(function, directory)
                cubins = [
                    directory / a["file"] for a in artifacts if a["kind"] == "cubin" and a["status"] == "retained"
                ]
                usage = (
                    subprocess.check_output(["cuobjdump", "--dump-resource-usage", str(cubins[0])], text=True)
                    if cubins
                    else "Compiler FFI wrapper exposes no standalone code; actual graph resources are collected by the paired driver."
                )
                (directory / "resources.txt").write_text(usage)
                record = {
                    "variant": variant,
                    "group": group,
                    "actual_storage_bytes": list(observed),
                    "capacity": props.shared_memory_per_block_optin,
                    "artifacts": artifacts,
                    "resource_usage": usage,
                }
                assert observed, "Actual storage instantiation was not observed."
                records.append(record)
                args.output.write_text(json.dumps({"no_kernel_launch": True, "rows": records}, indent=2) + "\n")
                print(variant, group, observed, flush=True)
    finally:
        FlashAttentionBackwardSm80._get_shared_storage_cls = original
    args.output.write_text(json.dumps({"no_kernel_launch": True, "rows": records, "status": "pass"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
