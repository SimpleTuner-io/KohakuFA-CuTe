"""Tune installed upstream Gluon kernels on the comparison's BF16 BHSD shapes."""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--seq", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--modes", nargs="+", choices=("dense", "causal"), default=["dense", "causal"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(*args.dim, *args.seq, args.heads, args.rounds) < 1:
        parser.error("Shapes and rounds must be positive.")
    if not torch.cuda.is_available():
        parser.error("Upstream tuning requires CUDA.")
    from kohakufa.device import Device
    from kohakufa.sm100 import fwd, tune

    # Preserve upstream's planner, candidate families and interleaved timings.
    # Its default input factory uses FP16 BSHD storage; match this comparison's
    # BF16 contiguous BHSD inputs instead, including the backward's saved output.
    def problem(kind, shape, head_dim, elem_bytes):
        if elem_bytes != 2:
            raise ValueError("This comparison tunes two-byte BF16 operands.")
        batch, heads, queries, keys, block = shape
        generator = torch.Generator(device="cuda").manual_seed(0)

        def make(sequence):
            return torch.randn(batch, heads, sequence, head_dim, device="cuda", dtype=torch.bfloat16, generator=generator)

        q, k, v = make(queries), make(keys), make(keys)
        scale = head_dim**-0.5
        if kind == "forward":
            return q, k, v, scale, block
        out, lse = fwd.attention_forward(q, k, v, scale, block)
        return q, k, v, out, torch.randn_like(out), lse, scale, block

    device = Device.query()
    original_problem = tune._problem
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    settings = dict(
        device=dataclasses.asdict(device),
        dtype="bf16",
        layout="bhsd",
        heads=args.heads,
        dimensions=args.dim,
        sequences=args.seq,
        rounds=args.rounds,
        modes=args.modes,
        method="upstream planner/shortlist, interleaved minimum over rounds; BF16 BHSD input factory",
    )
    (args.output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    try:
        tune._problem = problem
        for dimension in args.dim:
            tune.clear()
            for sequence in args.seq:
                for causal in (False, True):
                    if ("causal" if causal else "dense") not in args.modes:
                        continue
                    for kind in ("forward", "backward"):
                        print(f"START D={dimension} S={sequence} causal={causal} {kind}", flush=True)
                        started = time.monotonic()
                        selected = tune.tune(
                            kind,
                            dimension,
                            elem_bytes=2,
                            causal=causal,
                            seq=sequence,
                            batch_heads=args.heads,
                            dev=device,
                            rounds=args.rounds,
                            verbose=True,
                            shapes=[(1, args.heads, sequence, sequence, int(causal))],
                        )
                        records.append(
                            dict(
                                dimension=dimension,
                                sequence=sequence,
                                causal=causal,
                                kind=kind,
                                seconds=time.monotonic() - started,
                                selected=dataclasses.asdict(selected),
                            )
                        )
                        (args.output / "phases.json").write_text(json.dumps(records, indent=2) + "\n")
                        tune.save(
                            str(
                                args.output
                                / f"sm{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}-d{dimension}.json"
                            )
                        )
                        torch.cuda.empty_cache()
    finally:
        tune._problem = original_problem
    (args.output / "complete").touch()


if __name__ == "__main__":
    main()
