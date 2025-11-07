from __future__ import annotations

from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable


def topological_order(
    edges: Iterable[tuple[str, str]],
) -> list[str]:
    """
    Topologically order nodes for directed edges (child -> parent).

    - Parents appear before children if the graph is acyclic.
    """
    # Collect nodes and build
    # parent->children (for Kahn) + indegrees (parents of each node)
    parent_to_children: defaultdict[Any, set] = defaultdict(set)
    indeg: defaultdict[Any, int] = defaultdict(int)
    nodes: set[str] = set()

    for child, parent in edges:
        nodes.add(child)
        nodes.add(parent)
        if (
            child not in parent_to_children[parent]
        ):  # avoid double counting multi-edges
            parent_to_children[parent].add(child)
            indeg[child] += 1
        indeg.setdefault(parent, indeg.get(parent, 0))

    # Start with nodes that have no incoming edges (true roots/ultimate parents)
    q = deque([n for n in nodes if indeg[n] == 0])
    order: list[str] = []

    while q:
        v = q.popleft()
        order.append(v)
        for child in parent_to_children[v]:
            indeg[child] -= 1
            if indeg[child] == 0:
                q.append(child)

    has_cycle = len(order) < len(nodes)
    # leftover = nodes - set(order)

    if has_cycle:
        msg = (
            "Cycle detected: no full topological order exists.\n"
            # "Use SCC/fixed-point for cyclic components."
        )
        raise ValueError(msg)

    return order


def _relation_counts(
    edges: Iterable[tuple[str, str]],
) -> dict[str, dict[str, int]]:
    """Count how many relations each table participates in, as child and as parent."""
    counts: defaultdict[Any, dict[str, int]] = defaultdict(
        lambda: {
            "in": 0,
            "out": 0,
            # "total": 0
        }
    )
    for child, parent in edges:
        counts[child]["out"] += 1
        # counts[child]["total"] += 1
        counts[parent]["in"] += 1
        # counts[parent]["total"] += 1
    return counts
