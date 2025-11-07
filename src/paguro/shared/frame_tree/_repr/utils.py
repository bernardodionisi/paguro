from __future__ import annotations

from typing import Any

import polars as pl


def replace_dict_keys(
        data: Any,
        key_mapping: dict[str, str],
) -> Any:
    """
    Recursively replace keys in a deeply nested dictionary.

    Parameters
    ----------
    data : Any
        The dictionary, list, or value to process. Can be nested to any depth.
    key_mapping : dict[str, str]
        Dictionary mapping old keys to new keys. Keys not in this mapping
        will remain unchanged.
    """
    if isinstance(data, dict):
        new_dict: dict[str, Any] = {}
        for key, value in data.items():
            # Replace the key if it exists in mapping, otherwise keep original
            new_key: str = key_mapping.get(key, key)  # type: ignore[assignment]
            # Recursively process the value
            new_dict[new_key] = replace_dict_keys(value, key_mapping)
        return new_dict

    elif isinstance(data, list):
        # Process each item in the list
        return [replace_dict_keys(item, key_mapping) for item in data]

    else:
        # Return primitive values as-is
        return data


def calculate_width_with_depth(data_dict: dict[str, Any]) -> int:
    """Calculate width based on DataFrames found, with depth adjustment."""
    max_columns = 0
    max_depth = 0
    found_dataframes = False

    def _traverse(obj: Any, depth: int = 1):
        nonlocal max_columns, max_depth, found_dataframes

        if isinstance(obj, dict):
            for value in obj.values():
                _traverse(value, depth + 1)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _traverse(item, depth + 1)
        else:
            # Check if this leaf is a DataFrame
            if isinstance(obj, pl.DataFrame):
                found_dataframes = True
                df = obj  # type: pl.DataFrame
                col_count = len(df.columns)
                if col_count > max_columns:
                    max_columns = col_count
                    max_depth = depth

    _traverse(data_dict)

    # If no DataFrames found, return 80
    if not found_dataframes:
        return 80

    # Calculate width with depth adjustment
    depth_adjustment = (
                               max_depth - 1
                       ) * 5  # level 1 = +0, level 2 = +5, etc.
    return 45 + max_columns * 5 + depth_adjustment
