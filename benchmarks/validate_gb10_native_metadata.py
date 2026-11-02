"""Compare live native metadata decisions with the complete published parent."""

import argparse
import json
import runpy
from pathlib import Path
from unittest.mock import patch

import torch

from benchmarks.gb10_metadata_controls import parent as load_parent
from kohakufa_cute.fa4_sm80 import gb10

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
# Capture the actually loaded native host module before any timed stage.
import hashlib
import shutil

from torch.utils.cpp_extension import _get_build_directory

from kohakufa_cute.native import extension

source_path = Path(__file__).resolve().parents[1] / "src/kohakufa_cute/native.cpp"
source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
module_name = "kohakufa_cute_native_" + source_digest[:16]
cache_dir = Path(_get_build_directory(module_name, verbose=False))
host = args.output.parent / "host"
host.mkdir(exist_ok=False)
prior_cache = cache_dir / (module_name + ".so")
if prior_cache.is_file():
    shutil.copyfile(prior_cache, host / "preceding-cache.so")
    (host / "preceding-cache.json").write_text(
        json.dumps(
            {
                "scope": "Pre-load existing source-keyed cache, not original pilot mapping proof",
                "sha256": hashlib.sha256(prior_cache.read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
module = extension()
library = Path(module.__file__).resolve()
mapped = {
    Path(parts[5]).resolve()
    for line in Path("/proc/self/maps").read_text().splitlines()
    if len(parts := line.split(maxsplit=5)) == 6 and parts[5].startswith("/")
}
assert library in mapped, "Native host module is not mapped in this qualifying process"
shutil.copyfile(library, host / library.name)
build = library.parent / "build.ninja"
assert build.is_file()
shutil.copyfile(build, host / "build.ninja")
host_record = {
    "status": "actual_loaded_mapping_verified",
    "file": library.name,
    "bytes": library.stat().st_size,
    "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    "source_sha256": source_digest,
    "build_ninja_sha256": hashlib.sha256(build.read_bytes()).hexdigest(),
    "scope": "Actual production qualifier extension.__file__ matched /proc/self/maps; native module and exact compiler/include configuration retained before timing.",
}
(host / "mapping.json").write_text(json.dumps(host_record, indent=2) + "\n")

parent = load_parent()
decisions = []
assert gb10._qualified_device(torch.cuda.current_device())


def check(values, expected_override=None, **kwargs):
    expected = parent.inference_compatible(*values, **kwargs)
    if expected_override is not None:
        expected = expected_override
    actual = gb10.inference_compatible(*values, **kwargs)
    decisions.append(
        {
            "shape": list(values[0].shape),
            "dtype": str(values[0].dtype),
            "options": {k: str(v) for k, v in kwargs.items()},
            "selected": actual,
        }
    )

    (args.output.parent / "metadata-decisions.json").write_text(json.dumps(decisions, indent=2) + "\n")
    assert actual == expected, (list(values[0].shape), values[0].dtype, kwargs, expected, actual)


for dtype in (torch.bfloat16, torch.float16):
    for heads in (4, 16):
        for length in (128, 512, 1024, 4096):
            values = [torch.randn(1, heads, length, 128, device="cuda", dtype=dtype) for _ in range(3)]
            for causal in (False, True):
                check(values, causal=causal)
    for shape in (
        (2, 16, 128, 128),
        (1, 8, 128, 128),
        (1, 16, 129, 128),
        (1, 16, 2048, 128),
        (1, 16, 128, 16),
        (1, 16, 128, 64),
        (1, 16, 128, 256),
        (1, 16, 128, 512),
    ):
        check(
            [torch.randn(*shape, device="cuda", dtype=dtype) for _ in range(3)],
            expected_override=True if shape in ((1, 16, 128, 16), (1, 16, 128, 64), (1, 16, 128, 256)) else None,
        )
    values = [torch.randn(1, 16, 128, 128, device="cuda", dtype=dtype) for _ in range(3)]
    for options in (
        {"scale": 0.0},
        {"scale": -0.125},
        {"scale": 1e-40},
        {"scale": float("inf")},
        {"scale": float("nan")},
        {"mask": object()},
        {"dropout": 0.1},
        {"gqa": True},
    ):
        check(values, **options)
    for position in range(3):
        for variant in ("negative", "strided", "unaligned", "requires_grad"):
            modified = [t.detach() for t in values]
            if variant == "negative":
                modified[position] = modified[position]._neg_view()
            elif variant == "strided":
                modified[position] = torch.randn(1, 16, 128, 256, device="cuda", dtype=dtype)[..., ::2]
            elif variant == "unaligned":
                modified[position] = torch.randn(values[position].numel() + 1, device="cuda", dtype=dtype)[1:].view_as(
                    values[position]
                )
            else:
                modified[position].requires_grad_(True)
            check(modified)
            with torch.no_grad():
                check(modified)
    with patch.object(torch.compiler, "is_compiling", return_value=True):
        check(values)
    with patch.object(torch, "are_deterministic_algorithms_enabled", return_value=True):
        check(values)

for dtype in (torch.bfloat16, torch.float16):
    values = [torch.randn(1, 16, 128, 128, device="cuda", dtype=dtype) for _ in range(3)]
    values[0] = torch.nn.Parameter(values[0])
    check(values)
    with torch.no_grad():
        check(values)
    with torch.inference_mode():
        frozen = [torch.randn_like(value) for value in values]
        check(frozen)
    check(frozen)


class IdentityTensor(torch.Tensor):
    pass


for dtype in (torch.bfloat16, torch.float16):
    values = [torch.randn(1, 16, 128, 128, device="cuda", dtype=dtype) for _ in range(3)]
    subclass = [value.as_subclass(IdentityTensor) for value in values]
    check(subclass)
    with torch.no_grad():
        check(subclass)
    check(values)
    values[0].transpose_(2, 3)
    check(values)
    values[0].transpose_(2, 3)
    check(values)
    values[0].requires_grad_(True)
    check(values)
    with torch.no_grad():
        check(values)
    values[0].requires_grad_(False)
    check(values)
    values[0].resize_(2, 16, 128, 128)
    check(values)
    values[0].resize_(1, 16, 128, 128)
    check(values)

values = [torch.randn(1, 16, 128, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
with patch.object(gb10, "_inference_metadata_predicate", side_effect=RuntimeError("Native adapter initialization failed")):
    try:
        gb10.scaled_dot_product_attention(*values)
    except RuntimeError as error:
        assert "Native adapter initialization failed" in str(error)
    else:
        raise AssertionError("Selected initialization failure fell through")


for item in (None, object(), torch.randn(1, 16, 128, 128)):
    assert not gb10._inference_metadata_predicate()(item, item, item, False)

runpy.run_module("benchmarks.validate_gb10_inference", run_name="__main__")
import sys

output = Path(sys.argv[sys.argv.index("--output") + 1])
record = json.loads(output.read_text())
# Preserve the preceding H4 FP16 long-causal training schedule and gradients.
from benchmarks.reference import metrics

torch.manual_seed(419431)
training = [torch.randn(1, 4, 4096, 128, device="cuda", dtype=torch.float16, requires_grad=True) for _ in range(3)]
dout = torch.randn_like(training[0])
torch.save(
    {"inputs": [value.detach().cpu() for value in training], "dout": dout.cpu()},
    args.output.parent / "h4-long-training.pt",
)
compatibility = []
for operation in (parent.scaled_dot_product_attention, gb10.scaled_dot_product_attention):
    out = operation(*training, is_causal=True)
    assert out.requires_grad and not gb10.inference_compatible(*training, causal=True)
    compatibility.append(tuple(value.detach().cpu() for value in (out, *torch.autograd.grad(out, training, dout))))
training_errors = []
for before, after in zip(*compatibility):
    component = []
    for head in range(4):
        error = metrics(after[:, head : head + 1], before[:, head : head + 1])
        assert error["finite"] and error["relative_error"] < 0.005, error
        component.append(error)
    training_errors.append(component)
record["h4_long_causal_training"] = {
    "scope": "Compatibility with unchanged preceding training implementation; independent FP64 forward gates remain in the forward-only matrix",
    "status": "pass",
    "per_component_per_head": training_errors,
}

record["native_metadata_equivalence"] = decisions
record["host_mapping"] = host_record
record["selected_initialization_failure"] = "propagated"
record["native_metadata_equivalence_count"] = len(decisions)
record["selector_scope"] = (
    "Production native metadata selector; complete unmodified PR72 parent retained independently. AttentionRunner and GPU bodies unchanged."
)
output.write_text(json.dumps(record, indent=2) + "\n")
print("Native metadata equivalence passed", len(decisions), flush=True)
