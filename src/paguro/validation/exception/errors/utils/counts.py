from __future__ import annotations

from typing import Any


def drop_zeros_errors(
    mapping: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    # keep only keys with nonzero values
    out = {}
    for k, v in mapping.items():
        v = _drop_zeros_errors(v)
        if v:
            out[k] = v
    return out


def _drop_zeros_errors(mapping: dict[str, int]) -> dict[str, int]:
    # keep only keys with nonzero values
    return {k: v for k, v in mapping.items() if v != 0}


def clean_and_group_error_counts_by_type(
    d: dict[str, dict[str, Any]],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, Any]] = {
        "schema": {}, "data": {}
    }

    for key, mapping in d.items():
        cleaned = _drop_zeros_errors(mapping)
        if cleaned:  # only keep if not empty
            if key in ("dtype", "required"):
                result["schema"][key] = cleaned
            else:
                result["data"][key] = cleaned

    # Drop empty groups if no items remain
    return {k: v for k, v in result.items() if v}
