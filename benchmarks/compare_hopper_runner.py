"""Compare the native Hopper host against the same interleaved Python policy."""

import hashlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch


def main():
    # CuTe reads its keep options when imported. The matched control imports it
    # before driver.main, so configure retention before constructing that control.
    if "--export-code" in sys.argv:
        directory = Path(sys.argv[sys.argv.index("--export-code") + 1]).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        os.environ.update(CUTE_DSL_KEEP="ptx,cubin", CUTE_DSL_DUMP_DIR=str(directory))
    from benchmarks import compare_flashattention as driver
    from benchmarks import experimental_hopper_lpt as experiment
    from benchmarks.flashattention_backends import Backend

    control = Backend("cute", experiment.make_attention(), experiment, "kohakufa-cute")
    original = driver.load_backend
    with patch.object(driver, "load_backend", lambda name: control if name == "cute" else original(name)):
        driver.main()
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    report = json.loads(output.read_text())

    def identify(value):
        if isinstance(value, dict):
            return {"python_lpt_control" if key == "cute" else key: identify(item) for key, item in value.items()}
        if isinstance(value, list):
            return [identify(item) for item in value]
        return "python_lpt_control" if value == "cute" else value

    report = identify(report)
    report["control"] = {
        "factory": "benchmarks.experimental_hopper_lpt:make_attention",
        "scope": "Matched interleaved policy through the Python API; includes its existing maximum padding and preparation.",
    }
    report["wrapper_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
