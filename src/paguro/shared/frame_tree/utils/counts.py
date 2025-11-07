from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


def count_leaf_keys(tree: Mapping[Any, Any], keys: Iterable[Any]) -> dict[Any, int]:
    """
    Count how many dicts contain each key as a *'leaf'* (& value is NOT a dict).

    - Traverses dicts only (no lists/tuples).
    - Returns counts in the same order as `keys` (duplicates collapsed).
    """
    # Order-preserving unique targets
    targets_list = list(dict.fromkeys(keys))
    counts = dict.fromkeys(targets_list, 0)
    targets_set = set(targets_list)

    if not isinstance(tree, dict):
        return counts

    stack = [tree]

    while stack:
        d = stack.pop()
        for k, v in d.items():
            if k in targets_set and not isinstance(v, dict):
                counts[k] += 1
            if isinstance(v, dict):
                stack.append(v)

    return counts


def count_keys_per_ancestor(
    tree: Mapping[str, Any],
    keys: Iterable[Any],
    ancestor_keys: Iterable[Any],
) -> dict[str, dict[str, int]]:
    """
    Count how many times each `keys` leaf occurs *under* each `ancestor_keys`.

     (ancestor may be at any depth).

        {
            ancestor_key: { target_key: count, ... },
            ...
        }
    """
    targets = list(dict.fromkeys(keys))
    ancestors = list(dict.fromkeys(ancestor_keys))

    # Initialize result with zeros so absent keys show up explicitly
    result: dict[str, dict[str, int]] = {
        a: dict.fromkeys(targets, 0) for a in ancestors
    }

    if not isinstance(tree, dict):
        return result

    # Stack holds: (current_dict, path_of_keys_to_here_as_tuple)
    # stack = [(tree, tuple())]
    stack: list[tuple[dict[str, object], tuple[str, ...]]] = [
        (tree, tuple())
    ]

    while stack:
        d, path = stack.pop()

        # Which ancestor keys are currently in the path?
        active_ancestors = set(path) & set(ancestors)

        for k, v in d.items():
            if k in targets and not isinstance(v, dict):
                for a in active_ancestors:
                    result[a][k] += 1
            if isinstance(v, dict):
                stack.append((v, path + (k,)))

    return result


# def count_leaf_key(tree: dict, target: Any) -> int:
#     """Count how many dicts have `target` as a leaf key (value is NOT a dict)."""
#     if not isinstance(tree, dict):
#         return 0
#
#     count = 0
#     stack = [tree]
#
#     while stack:
#         d = stack.pop()
#
#         # count once per dict
#         if target in d and not isinstance(d[target], dict):
#             count += 1
#
#         # descend into child dicts
#         for v in d.values():
#             if isinstance(v, dict):
#                 stack.append(v)
#
#     return count
