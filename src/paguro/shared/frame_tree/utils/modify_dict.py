from __future__ import annotations

from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

from paguro.utils.dependencies import copy

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


def _transform_pairs(
        root: MutableMapping[Any, Any],
        items: Iterable[tuple[Sequence[Any], Any]],  # stream of (path, obj)
        *,
        deepcopy: bool,
) -> MutableMapping[Any, Any] | None:
    target = copy.deepcopy(root) if deepcopy else root

    for path, obj in items:
        if not path:
            msg = "Encountered an empty path."
            raise ValueError(msg)

        parent = get_parent_at_path(target, path)
        _key = path[-1]

        if _key not in parent:
            msg = f"Leaf key {_key!r} does not exist at parent path {tuple(path[:-1])!r}."
            raise KeyError(msg)

        parent[_key] = obj

    return target if deepcopy else None


def _transform_pairs_with_errors_renaming(
        root: MutableMapping[Any, Any],
        items: Iterable[tuple[Sequence[Any], Any]],  # stream of (path, obj)
        *,
        on_conflict: Literal["overwrite", "skip", "raise"],
        deepcopy: bool,
        # only for errors dicts, probably we should have two separate funcs
        is_errors_mapping: bool,
        is_errors_limit: bool,
        is_errors_row_counts: bool,
) -> MutableMapping[Any, Any] | None:
    target = copy.deepcopy(root) if deepcopy else root

    for path, obj in items:
        if not path:
            msg = "Encountered an empty path."
            raise ValueError(msg)

        parent = get_parent_at_path(target, path)
        old_key = path[-1]

        if old_key not in parent:
            msg = f"Leaf key {old_key!r} does not exist at parent path {tuple(path[:-1])!r}."
            raise KeyError(msg)

        # ---------
        new_key = old_key
        if is_errors_mapping:
            new_value = _replace_errors_if_passed(
                obj=obj,
                ok_value="PASSED",
                row_counts=is_errors_row_counts,
            )
            if isinstance(new_value, str) and new_value == "PASSED":
                new_key = "no_errors"
            elif isinstance(new_value, pl.DataFrame):
                if is_errors_limit:
                    new_key = "errors_limited"
                else:
                    new_key = "errors"
            elif isinstance(new_value, int):
                new_key = "errors_count"
            elif isinstance(new_value, Exception):
                new_key = "exception"
            elif isinstance(new_value, pl.LazyFrame):
                pass
            else:
                msg = (
                    f"Invalid type for new_value ({new_key}): {new_value}"
                )
                raise TypeError(msg)
                # new_key = "errors"
        else:
            new_value = obj

        # new_key = replace_key_fn(obj, "no_errors", old_key)
        # if replace_key_fn is not None else old_key
        # ---------

        if new_key == old_key:
            parent[old_key] = new_value
            continue

        renamed = rename_key_in(
            parent=parent,
            old_key=old_key,
            new_key=new_key,
            on_conflict=on_conflict,
        )
        # If rename was skipped (conflict+skip), keep original entry unchanged.
        if renamed:
            parent[new_key] = new_value

    return target if deepcopy else None


def _replace_errors_if_passed(
        *,
        obj: Any,
        row_counts: bool,
        ok_value: Literal["PASSED"]
) -> pl.DataFrame | int | Literal["PASSED"]:
    if obj is None:
        return ok_value
    elif isinstance(obj, pl.DataFrame):

        if row_counts:
            count = obj.item(0, 0)
            if count:
                return count
            else:  # 0
                return ok_value

        if obj.shape[0] == 0:
            return ok_value

    return obj


# Path / rename utilities


def get_parent_at_path(
        root: MutableMapping[Any, Any],
        path: Sequence[Any],
) -> MutableMapping[Any, Any]:
    """
    Return the parent mapping for a given full path.

    Example:
        path = ("a", "b", "c") -> returns node at root["a"]["b"].

    Raises
    ------
    ValueError
        If `path` is empty.
    KeyError
        If any segment is missing.
    TypeError
        If an intermediate node is not a MutableMapping.
    """
    if not path:
        msg = "Path must be non-empty (needs a leaf key)."
        raise ValueError(msg)

    cur: Any = root
    for key in path[:-1]:
        if not isinstance(cur, MutableMapping):
            msg = (
                f"Expected a mapping before "
                f"segment {key!r}, found {type(cur).__name__}."
            )
            raise TypeError(msg)
        if key not in cur:
            msg = f"Missing path segment {key!r}."
            raise KeyError(msg)
        cur = cur[key]

    if not isinstance(cur, MutableMapping):
        msg = (
            f"Parent at {tuple(path[:-1])!r} is "
            f"not a mapping (got {type(cur).__name__})."
        )
        raise TypeError(msg)
    return cur


def rename_key_in(
        *,
        parent: MutableMapping[Any, Any],
        old_key: Any,
        new_key: Any,
        on_conflict: Literal["overwrite", "skip", "raise"] = "raise",
) -> bool:
    """
    Rename `old_key` -> `new_key` inside `parent` under a conflict policy.

    Returns
    -------
    bool
        True if a rename occurred, False if it was skipped
        (e.g., conflict+skip or same key).

    Raises
    ------
    KeyError
        If `old_key` is missing, or if a conflict happens and on_conflict="error".
    """
    if old_key == new_key:
        return False

    if new_key in parent:
        if on_conflict == "skip":
            return False
        if on_conflict == "raise":
            msg = f"Key conflict: {new_key!r} already exists."
            raise KeyError(msg)

    # Pop will raise KeyError if old_key is missing (desired behavior).
    val = parent.pop(old_key)
    parent[new_key] = val
    return True


# ----------------------------------------------------------------------


def prune_on_leaf_pair(
        tree: dict[str, object],
        allowed_keys: Iterable[str],
        allowed_values: Iterable[object],
) -> dict[str, object]:
    ak: set[str] = set(allowed_keys)
    av: set[object] = set(allowed_values)

    def _prune(d: dict[str, object]) -> dict[str, object]:
        out: dict[str, object] = {}
        for k, v in d.items():
            if isinstance(v, dict):
                child = _prune(v)
                if child:
                    out[k] = child
            else:
                if (k in ak) and (v in av):
                    out[k] = v
        return out

    return _prune(tree)


def prune_on_leaf_pair_by_type(
        tree: dict[str, Any],
        allowed_keys: str | Iterable[str],
        allowed_value_types: type[Any] | Iterable[type[Any]] | None,
) -> dict[str, Any]:
    """
    Keep leaves (k, v).

    Where
      1) k ∈ allowed_keys, AND
      2) (allowed_value_types is None) OR isinstance(v, any of allowed_value_types).

    `allowed_value_types` may be:
      - None (no type filtering)
      - a single type (e.g., `str`)
      - an iterable of types (e.g., `[str, int]`)

    Returns a NEW pruned dict and does not mutate `tree`.
    """
    # Normalize allowed keys
    if isinstance(allowed_keys, str):
        ak: set[str] = {allowed_keys}
    else:
        ak = set(allowed_keys)

    # Normalize allowed value types
    if allowed_value_types is None:
        av_types: tuple[type[Any], ...] | None = None
    elif isinstance(allowed_value_types, type):
        av_types = (allowed_value_types,)
    else:
        av_types = tuple(allowed_value_types)

    def _type_ok(v: Any) -> bool:
        return True if av_types is None else isinstance(v, av_types)

    def _prune(d: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in d.items():
            if isinstance(v, dict):
                child = _prune(v)
                if child:
                    out[k] = child
            else:
                if (k in ak) and _type_ok(v):
                    out[k] = v
        return out

    return _prune(tree)
