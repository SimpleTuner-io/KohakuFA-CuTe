"""Check the same-process previous-policy control against a preceding checkout."""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--previous", action="store_true")
    args = parser.parse_args()
    from benchmarks.experimental_precision_control import previous_precision
    from kohakufa_cute import scaled_dot_product_attention

    args.fixtures.mkdir(parents=True, exist_ok=True)
    cases = (
        (16, 128, False, torch.bfloat16),
        (64, 1024, True, torch.bfloat16),
        (128, 128, False, torch.float16),
        (128, 1024, True, torch.bfloat16),
        (256, 128, False, torch.float16),
        (256, 4096, False, torch.bfloat16),
        (512, 128, True, torch.bfloat16),
        (512, 1024, False, torch.float16),
        (512, 1024, False, torch.bfloat16),
    )
    rows = []
    for dimension, sequence, causal, dtype in cases:
        torch.manual_seed(2029)
        inputs = [torch.randn(1, 16, sequence, dimension, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)]
        dout = torch.randn_like(inputs[0])
        with previous_precision() if args.previous else nullcontext():
            output = scaled_dot_product_attention(*inputs, is_causal=causal)
            actual = [x.detach().cpu() for x in (output, *torch.autograd.grad(output, inputs, dout))]
        name = f'{dimension}-{sequence}-{int(causal)}-{str(dtype).split(".")[-1]}'
        filename = args.fixtures / ("control-" + name + ".pt")
        if not args.previous:
            torch.save(actual, filename)
        else:
            expected = torch.load(filename, weights_only=True)
            equal = [torch.equal(x, y) for x, y in zip(actual, expected)]
            if not all(equal):
                raise AssertionError(f"Previous-policy reconstruction differs for {name}: {equal}")
            rows.append({"case": name, "exact_equal": equal})
            print(rows[-1], flush=True)
    if args.previous:
        (args.fixtures / "control-verification.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
