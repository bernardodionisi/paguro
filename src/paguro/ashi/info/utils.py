from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


def is_json_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False


def rename_dict_keys(original_dict: dict, mapping: Mapping[str, str]):
    # check for conflicts:
    # if a new key is in the original keys and not meant to be renamed
    for key, new_key in mapping.items():
        if (
                new_key in original_dict
                and key != new_key
                and new_key not in mapping
        ):
            raise ValueError(
                f"trying to rename the dictionary key '{key}' to '{new_key}'\n"
                f"but '{new_key}' is already a key in the dictionary "
                f"(and is not being renamed)"
            )

    # create a temporary dictionary to store the new key-value pairs
    out: dict = {}

    for key, value in original_dict.items():
        if key in mapping:
            # rename key if it's in the key_mapping
            new_key = mapping[key]
            out[new_key] = value
        elif key not in mapping.values():
            # keep the original key-value pair if the key is not supposed to be renamed
            out[key] = value

    return out


def make_nested_dict_json_serializable(obj):
    """
    Recursively converts keys and values of any nested dictionaries or lists
    within 'obj' to ensure JSON serializability. Non-JSON serializable objects
    are converted to their string representation. Non-string keys are also
    converted to strings.

    Parameters
    ----------
    obj
        The original dictionary, list, or any object to make JSON serializable.
    """
    if isinstance(obj, dict):
        return {
            str(k): make_nested_dict_json_serializable(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [make_nested_dict_json_serializable(item) for item in obj]
    elif not isinstance(obj, (int, float, str, bool, type(None))):
        return str(obj)
    else:
        return obj
