"""Check changed graph inputs and tail ownership for explicit SM80 schedules."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import torch

from benchmarks.compare_ada import capture
from benchmarks.sm80_schedule_candidates import PRESETS, make_attention
from benchmarks.validate_sm80_static import assess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    function = make_attention(args.preset)
    report = {
        "preset": args.preset,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__).resolve().parents[1] / "src/kohakufa_cute/fa4_sm80").glob("*.py")
        },
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sequences = (129,) if args.smoke else (129, 256, 257)
    for dtype, sequence, causal in itertools.product((torch.bfloat16, torch.float16), sequences, (False, True)):
        torch.manual_seed(853)
        inputs = [torch.randn(1, 3, sequence, 64, device="cuda", dtype=dtype) for _ in range(3)]
        dout = torch.randn_like(inputs[0])
        graph, storage = capture(function, inputs, dout, causal, "joint", keep_graph=True)
        operands, _, _, actual = storage
        for seed in (857, 859):
            torch.manual_seed(seed)
            with torch.no_grad():
                for tensor in (*operands, dout):
                    tensor.copy_(torch.randn_like(tensor))
                for tensor in actual:
                    tensor.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            row = {
                "case": "changed-poisoned-graph",
                "dtype": str(dtype),
                "sequence": sequence,
                "causal": causal,
                "seed": seed,
                "errors": assess(actual, operands, dout, causal=causal, scale=None, directed=False, pointwise=False),
            }
            report["rows"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print({key: value for key, value in row.items() if key != "errors"}, flush=True)
    report["status"] = "pass"
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
