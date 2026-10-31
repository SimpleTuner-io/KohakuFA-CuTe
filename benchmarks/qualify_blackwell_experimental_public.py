"""Explicitly enable experimental public dispatch for selection diagnostics only."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from benchmarks.qualify_blackwell_public_core import main as verify
    from kohakufa_cute import fa4_blackwell

    if fa4_blackwell._AUTOMATIC_DISPATCH_QUALIFIED:
        raise ValueError("This diagnostic expects automatic experimental selection to remain disabled.")
    with patch.object(fa4_blackwell, "_AUTOMATIC_DISPATCH_QUALIFIED", True):
        verify()
    report = json.loads(args.output.read_text())
    report["experimental_dispatch_override"] = True
    report["automatic_dispatch_enabled"] = False
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
