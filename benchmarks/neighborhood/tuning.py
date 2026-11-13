"""Explicit, serialized tile experiments for the default neighborhood arithmetic.

Use this context around the complete forward-plus-backward call, every eager
measurement, or CUDA graph warmup/capture. A graph records the chosen kernels;
replay does not consult the process-wide Python experiment setting. Re-enter
the matching context when comparing changed graph inputs against fresh eager
execution. No automatic dispatch preset is changed by this helper.
"""

from contextlib import contextmanager

from kohakufa_cute.neighborhood_geometry import normalize_exact_contiguous

BASELINE_COLUMNS = (32, 32, 32)
COLUMN_EXPERIMENTS = {
    "baseline": BASELINE_COLUMNS,
    "forward64": (64, 32, 32),
    "query64": (32, 64, 32),
    "keyvalue64": (32, 32, 64),
    "columns64": (64, 64, 64),
}
SECONDARY_COLUMN_EXPERIMENTS = {
    "forward128": (128, 32, 32),
    "query128": (32, 128, 32),
    "keyvalue128": (32, 32, 128),
    "columns128": (128, 128, 128),
}


def normalize_columns(columns):
    if columns is None:
        return None
    selected = tuple(columns)
    if len(selected) != 3 or any(type(value) is not int or value not in (16, 32, 64, 128) for value in selected):
        raise ValueError("Expected forward/query/KV column widths from 16, 32, 64, or 128.")
    return selected


def describe_columns(columns):
    """Describe the changed factor without inventing compiler resource usage."""
    columns = normalize_columns(columns)
    if columns is None:
        return {
            "columns": "automatic",
            "owner_rows": 64,
            "head_dimension": 64,
            "arithmetic_contract": "docs/precision.md",
            "precision_policy_changed": False,
            "selection_scope": "Production dispatch; each row records its effective schedule.",
            "resource_scope": "Resolve concrete columns before estimating staging; retain actual compiler resources.",
        }
    return {
        "columns": dict(zip(("forward", "query", "keyvalue"), columns)),
        "owner_rows": 64,
        "head_dimension": 64,
        "warps": 4,
        "width": 64,
        "arithmetic_contract": "docs/precision.md",
        "precision_policy_changed": False,
        "staging_estimate_bytes": {
            mode: 2 * 64 * ((64 if mode == "forward" else 128) + 2 * value)
            for mode, value in zip(("forward", "query", "keyvalue"), columns)
        },
        "resource_scope": "Staging estimate only; retain actual cubin register, shared-memory and spill metadata.",
        "selection_scope": "Serialized explicit experiment; no automatic architecture preset.",
    }


@contextmanager
def neighborhood_columns(columns, *, exact_contiguous=False):
    """Select one schedule around an entire serialized call or graph capture."""
    from kohakufa_cute.neighborhood_kernels import set_experiment_configuration, set_experiment_exact_contiguous

    selected_traversal = None if exact_contiguous is None else normalize_exact_contiguous(exact_contiguous)
    previous = set_experiment_configuration(normalize_columns(columns))
    previous_traversal = set_experiment_exact_contiguous(selected_traversal)
    try:
        yield
    finally:
        set_experiment_exact_contiguous(previous_traversal)
        set_experiment_configuration(previous)
