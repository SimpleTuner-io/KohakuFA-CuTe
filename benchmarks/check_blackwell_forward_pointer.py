"""One eager allocation-pointer control with an independent FP64 forward check."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    from benchmarks.diagnose_blackwell_allocator_pointer import make_pointer_poisoned_attention
    from benchmarks.reference import metrics

    torch.manual_seed(2231)
    q, k, v = [torch.randn(1, 3, 129, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    out = make_pointer_poisoned_attention()(q, k, v, is_causal=True)
    scores = (q.double() @ k.double().transpose(-1, -2)) * 0.125
    scores.masked_fill_(torch.ones(129, 129, device="cuda", dtype=torch.bool).triu(1), -torch.inf)
    expected = scores.softmax(-1) @ v.double()
    errors = [metrics(out[:, h : h + 1], expected[:, h : h + 1]) for h in range(3)]
    if not all(e["finite"] and e["relative_error"] < 0.01 for e in errors):
        raise AssertionError("The poisoned allocation-pointer control failed its unchanged BF16 norm gate.")
    args.output.write_text(
        json.dumps(
            dict(
                scope="Diagnostic only; allocator slot poisoned before tcgen05 allocation, actual values printed after its barrier.",
                passed=True,
                errors=errors,
            ),
            indent=2,
        )
        + "\n"
    )
    print("forward allocation pointer and FP64 output check passed", flush=True)


if __name__ == "__main__":
    main()
