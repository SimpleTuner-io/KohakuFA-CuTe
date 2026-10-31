"""One-factor dO-stage experiment for the shared static warp-MMA schedule."""

import argparse
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch


@contextmanager
def schedule_control(dout_stages=1):
    """Qualify a different schedule through the real public dispatch boundary."""
    from kohakufa_cute import fa4_sm80
    from kohakufa_cute.fa4_sm80._attention import Schedule, attention

    schedule = Schedule(dout_stages=int(dout_stages))

    def selected(q, k, v, *, causal=False, scale=None):
        return attention(q, k, v, causal=causal, scale=scale, schedule=schedule)

    with patch.object(fa4_sm80, "attention", selected):
        yield


def make_attention(dout_stages=1):
    """Return the selected schedule and retain native CuTe as the control."""
    import torch

    from benchmarks.static_sm80_experiments import _select_native_control
    from kohakufa_cute.fa4_sm80 import available
    from kohakufa_cute.fa4_sm80._attention import Schedule, attention

    if torch.cuda.get_device_capability() != (8, 0) or not available():
        raise ValueError("The explicit schedule requires the qualified SM80 runtime.")
    if dout_stages not in (1, 2):
        raise ValueError("Select one or two dO stages.")
    schedule = Schedule(dout_stages=dout_stages)
    _select_native_control()

    def selected(q, k, v, *, is_causal=False, scale=None, enable_gqa=False):
        if q.shape != k.shape or q.shape != v.shape or q.shape[-1] not in (64, 128):
            raise ValueError("The schedule experiment requires square MHA D64/D128.")
        return attention(q, k, v, causal=is_causal, scale=scale, schedule=schedule)

    return selected


def main():
    import sys

    from benchmarks.validate_sm80_static import main as qualify

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dout-stages", type=int, choices=(1, 2), default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    sys.argv = [sys.argv[0], "--direct", "--output", str(args.output)]
    if args.smoke:
        sys.argv.append("--smoke")
    with schedule_control(args.dout_stages):
        qualify()
    report = json.loads(args.output.read_text())
    report["schedule_configuration"] = {"dout_stages": args.dout_stages}
    report["schedule_driver_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
