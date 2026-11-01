"""Bounded resident-V, warp ownership, and causal forward experiments.

These controls preserve the public default schedule.  Qualify each selected
configuration before interpreting its timing or considering dispatch changes.
"""

import argparse
import hashlib
import json
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

PRESETS = (
    "default",
    "d64-do1",
    "d64-do1-kv4dq4",
    "d64-do1-resident",
    "d64-do1-kv4dq4-resident",
    "d64-do1-kv4dq2",
    "d64-fa2-tile",
    "d64-fa2-tile-resident",
    "causal-m64",
)


def schedule_for(preset, dimension, causal):
    from kohakufa_cute.fa4_sm80._attention import Schedule

    if preset not in PRESETS:
        raise ValueError("Unknown explicit SM80 schedule.")
    schedule = Schedule()
    if dimension == 64 and preset.startswith("d64-"):
        schedule = replace(schedule, dout_stages=1, v_in_regs=preset.endswith("resident"))
        if "kv4dq4" in preset:
            schedule = replace(schedule, kv_m_warps=4, dq_m_warps=4)
        elif "kv4dq2" in preset:
            schedule = replace(schedule, kv_m_warps=4)
        if "fa2-tile" in preset:
            schedule = replace(schedule, backward_m=128, query_stages=1, score_m_warps=4, kv_m_warps=4, dq_m_warps=4)
    elif preset == "causal-m64" and dimension == 128 and causal:
        schedule = replace(schedule, forward_m=64)
    return schedule


@contextmanager
def schedule_control(preset="default"):
    from kohakufa_cute import fa4_sm80
    from kohakufa_cute.fa4_sm80._attention import attention

    def selected(q, k, v, *, causal=False, scale=None):
        return attention(q, k, v, causal=causal, scale=scale, schedule=schedule_for(preset, q.shape[-1], causal))

    with patch.object(fa4_sm80, "attention", selected):
        yield


def make_attention(preset="default"):
    import torch

    from benchmarks.static_sm80_experiments import _select_native_control
    from kohakufa_cute.fa4_sm80 import available
    from kohakufa_cute.fa4_sm80._attention import attention

    if torch.cuda.get_device_capability() != (8, 0) or not available():
        raise ValueError("The explicit candidate requires the qualified SM80 stack.")
    schedule_for(preset, 64, False)
    _select_native_control()

    def selected(q, k, v, *, is_causal=False, scale=None, enable_gqa=False):
        if (
            q.shape != k.shape
            or q.shape != v.shape
            or q.shape[-1] not in (64, 128)
            or not all(t.is_contiguous() and t.data_ptr() % 16 == 0 for t in (q, k, v))
            or (scale is not None and scale <= 0)
        ):
            raise ValueError("The candidate requires aligned square positive-scale D64/D128 MHA.")
        return attention(q, k, v, causal=is_causal, scale=scale, schedule=schedule_for(preset, q.shape[-1], is_causal))

    return selected


def main():
    import sys

    from benchmarks.validate_sm80_static import main as qualify

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    sys.argv = [sys.argv[0], "--direct", "--output", str(args.output)]
    if args.smoke:
        sys.argv.append("--smoke")
    with schedule_control(args.preset):
        qualify()
    report = json.loads(args.output.read_text())
    report["preset"] = args.preset
    report["schedules"] = {
        f"d{dimension}-{visibility}": asdict(schedule_for(args.preset, dimension, visibility == "causal"))
        for dimension in (64, 128)
        for visibility in ("dense", "causal")
    }
    report["candidate_driver_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
