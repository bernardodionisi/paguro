# At every node in the nested mapping,
# the function checks each key (k) of the current dictionary.
# If that key equals the target key, it inspects the associated value:
# If the value is of the desired want type,
# the current path + (k,) is recorded as a match.
# If the value is itself a mapping, it keeps walking inside that value as well.

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator


def find_keyed_typed_values(
    root: Mapping[str, Any],
    want: type | tuple[type, ...],
    key: str,
) -> list[tuple[tuple[Any, ...], Any]]:
    return list(_iter_keyed_typed_matches_fast(root, want, key))


def find_keyed_typed_lists(
    root: Mapping[str, Any],
    want: type | tuple[type, ...],
    key: str,
) -> tuple[list[tuple[Any, ...]], list[Any]]:
    paths: list[tuple[Any, ...]] = []
    values: list[Any] = []
    for p, v in _iter_keyed_typed_matches_fast(root, want, key):
        paths.append(p)
        values.append(v)
    return paths, values


def _iter_keyed_typed_matches_fast(
    root: Mapping[str, Any],
    want: type | tuple[type, ...],
    key: str,
) -> Iterator[tuple[tuple[Any, ...], Any]]:
    """
    Depth-first traversal over nested dicts (or any Mapping).

    Yields (path_tuple, value) for every match of key==`key`
    where value is an instance of `want`.
    """
    seen: set[int] = set()
    stack: list[tuple[Mapping[str, Any], Any, int]] = []
    path: list[Any] = []

    if not isinstance(root, Mapping):
        return
    seen.add(id(root))
    stack.append((root, iter(root.items()), 0))

    while stack:
        current, it, pushed = stack[-1]
        try:
            k, v = next(it)
        except StopIteration:
            stack.pop()
            if pushed:
                path.pop()
            continue

        if k != key:
            if isinstance(v, Mapping):
                oid = id(v)
                if oid not in seen:
                    seen.add(oid)
                    path.append(k)
                    stack.append((v, iter(v.items()), 1))
            continue

        # k == key
        if isinstance(v, want):
            yield (tuple(path) + (k,), v)
        elif isinstance(v, Mapping):
            oid = id(v)
            if oid not in seen:
                seen.add(oid)
                path.append(k)
                stack.append((v, iter(v.items()), 1))
