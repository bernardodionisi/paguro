from __future__ import annotations

from typing import Any


def find_nested_keys_dict(
    data: dict[str, Any], parent: str, max_depth: int = 0
) -> set[str]:
    if max_depth < 0:
        msg = "max_depth must be >= 0"
        raise ValueError(msg)

    result: set[str] = set()
    stack = [data]

    # fast references
    result_add = result.add
    get_parent = dict.get

    while stack:
        node = stack.pop()
        if type(node) is not dict:
            continue

        # 1) Handle this node's parent hit (if any) once
        v = get_parent(node, parent)
        if type(v) is dict:
            if max_depth == 0:
                # immediate children only (fast path)
                for k in v:
                    result_add(k)
            else:
                # depth-limited single-pass collect
                work = [(v, 0)]
                work_pop = work.pop
                work_append = work.append
                while work:
                    d, depth = work_pop()
                    # add all keys at this depth
                    for ck, cv in d.items():
                        result_add(ck)
                        if depth < max_depth and type(cv) is dict:
                            work_append((cv, depth + 1))
            # Note: do NOT push `v` to stack—already handled fully.

        # 2) Keep searching the rest of the subtree for more parent hits
        for child in node.values():
            if type(child) is dict:
                stack.append(child)

    return result


def find_nested_keys_paths(
    data: dict[str, Any],
    parent: str,
    max_depth: int = 0,
) -> set[tuple[str, ...]]:
    """
    Find keys nested under any occurrence of `parent` in a nested dict.

    returning the full path from the root as tuples of strings.

    - parent: key whose nested descendants we want to collect
    - max_depth: depth under `parent` to include (0 = only immediate children)
    """
    if max_depth < 0:
        msg = "max_depth must be >= 0"
        raise ValueError(msg)
    if type(data) is not dict:
        return set()

    result: set[tuple[str, ...]] = set()
    stack: list[tuple[dict[str, Any], tuple[str, ...]]] = [(data, ())]

    while stack:
        node, path = stack.pop()
        if type(node) is not dict:
            continue

        # Handle hits of `parent`
        v = node.get(parent)
        if isinstance(v, dict):
            if max_depth == 0:
                for ck in v:
                    result.add(path + (parent, ck))
            else:
                work: list[tuple[dict[str, Any], int, tuple[str, ...]]] = [
                    (v, 0, (parent,))
                ]
                while work:
                    d, depth, rel = work.pop()
                    for ck, cv in d.items():
                        full_path = path + rel + (ck,)
                        result.add(full_path)
                        if depth < max_depth and isinstance(cv, dict):
                            work.append((cv, depth + 1, rel + (ck,)))

        # Continue traversing the tree
        for k, child in node.items():
            if isinstance(child, dict):
                stack.append((child, path + (k,)))

    return result
