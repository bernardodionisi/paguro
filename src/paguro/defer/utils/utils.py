from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final, TypeAlias, cast

DEFAULT_STEP_FLAG: Final = "__has_dp__"

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from paguro.defer.frames import LazyFrameExpr

# A step is a single-key mapping:
# {"method": {"args": tuple, "kwargs": dict, optional flag}}
Step: TypeAlias = dict[str, dict[str, Any]]


# accessors
# (fix IDE tuple/dict confusion, keep checks in one place)


def _step_input_data(step: Step) -> dict[str, Any]:
    # Optional debug-time shape check (no cost in -O mode)
    if __debug__:
        if not isinstance(step, dict) or len(step) != 1:
            msg = f"Invalid step shape: {step!r}"
            raise ValueError(msg)
        input_data = next(iter(step.values()))
        if not isinstance(input_data, dict):
            msg = f"Invalid step input_data: {input_data!r}"
            raise ValueError(msg)
        # don't *require* keys; some steps may omit 'kwargs' if empty
    return next(iter(step.values()))


def _step_method(step: Step) -> str:
    return next(iter(step.keys()))


# ----------------------------------------------------------------------


def step_has_refs(
        step: Step, *, step_flag: str = DEFAULT_STEP_FLAG
) -> bool:
    input_data = _step_input_data(step)
    return bool(input_data.get(step_flag, False))


def is_self_contained(
        p: LazyFrameExpr,
        *,
        step_flag: str = DEFAULT_STEP_FLAG,
) -> bool:
    return all(not step_has_refs(s, step_flag=step_flag) for s in p._steps)


def _find_deferred(
        x: Any, is_dp: Callable[[Any], bool]
) -> list[LazyFrameExpr]:
    out: list[LazyFrameExpr] = []
    stack: list[Any] = [x]
    while stack:
        cur = stack.pop()
        if is_dp(cur):
            out.append(cast(LazyFrameExpr, cur))
            continue
        if isinstance(cur, tuple):
            stack.extend(cur)
        elif isinstance(cur, list):
            stack.extend(cur)
        elif isinstance(cur, Mapping):
            stack.extend(cur.values())
    return out


def direct_referenced_pipelines(
        p: LazyFrameExpr,
        *,
        is_dp: Callable[[Any], bool],
        step_flag: str = DEFAULT_STEP_FLAG,
) -> set[LazyFrameExpr]:
    refs: set[LazyFrameExpr] = set()
    for step in p._steps:
        if not step_has_refs(step, step_flag=step_flag):
            continue
        input_data = _step_input_data(step)
        for obj in _find_deferred(input_data.get("args", ()), is_dp):
            refs.add(obj)
        for obj in _find_deferred(input_data.get("kwargs", {}), is_dp):
            refs.add(obj)
    return refs


def referenced_pipelines(
        p: LazyFrameExpr,
        *,
        is_dp: Callable[[Any], bool],
        step_flag: str = DEFAULT_STEP_FLAG,
        include_self: bool = False,
) -> set[LazyFrameExpr]:
    out: set[LazyFrameExpr] = set()  # set["DeferredPipeline"]
    seen: set[int] = set()

    def walk(node: LazyFrameExpr) -> None:
        nid = id(node)
        if nid in seen:
            return
        seen.add(nid)
        if include_self:
            out.add(node)
        for q in direct_referenced_pipelines(
                node, is_dp=is_dp, step_flag=step_flag
        ):
            out.add(q)
            walk(q)

    walk(p)
    if not include_self and p in out:
        out.remove(p)
    return out


def all_names(
        p: LazyFrameExpr,
        *,
        is_dp: Callable[[Any], bool],
        name_of: Callable[[Any], str | None] = lambda x: getattr(
            x, "_name", None
        ),
        step_flag: str = DEFAULT_STEP_FLAG,
        include_self: bool = True,
) -> set[str]:
    names: set[str] = set()
    if include_self and p._name is not None:
        names.add(p._name)
    for q in referenced_pipelines(p, is_dp=is_dp, step_flag=step_flag):
        nm = name_of(q)
        if nm is not None:
            names.add(nm)
    return names


def dag_edges(
        p: LazyFrameExpr,
        *,
        is_dp: Callable[[Any], bool],
        name_of: Callable[[Any], str | None] = lambda x: getattr(
            x, "_name", None
        ),
        step_flag: str = DEFAULT_STEP_FLAG,
        unnamed_fmt: Callable[[Any], str] = lambda x: f"<unnamed@{id(x):x}>",
) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    seen: set[int] = set()

    def label(x: Any) -> str:
        return name_of(x) or unnamed_fmt(x)

    def walk(node: LazyFrameExpr) -> None:
        nid = id(node)
        if nid in seen:
            return
        seen.add(nid)
        u = label(node)
        for q in direct_referenced_pipelines(
                node, is_dp=is_dp, step_flag=step_flag
        ):
            edges.add((u, label(q)))
            walk(q)

    walk(p)
    return edges


def topo_order(
        p: LazyFrameExpr,
        *,
        is_dp: Callable[[Any], bool],
        name_of: Callable[[Any], str | None] = lambda x: getattr(
            x, "_name", None
        ),
        step_flag: str = DEFAULT_STEP_FLAG,
        strict: bool = False,
) -> list[str]:
    edges = dag_edges(p, is_dp=is_dp, name_of=name_of, step_flag=step_flag)
    nodes: set[str] = set()
    for u, v in edges:
        nodes.add(u)
        nodes.add(v)
    if not nodes:
        return [name_of(p) or f"<unnamed@{id(p):x}>"]
    indeg = dict.fromkeys(nodes, 0)
    adj: dict[str, set[str]] = {n: set() for n in nodes}
    for u, v in edges:
        if v not in adj[u]:
            adj[u].add(v)
            indeg[v] += 1
    q = [n for n in nodes if indeg[n] == 0]
    order: list[str] = []
    while q:
        u = q.pop()
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if strict and len(order) != len(nodes):
        msg = "Cycle detected in pipeline DAG."
        raise RuntimeError(msg)
    return order


def require_data_keys(
        p: LazyFrameExpr,
        data_keys: Iterable[str],
        *,
        is_dp: Callable[[Any], bool],
        name_of: Callable[[Any], str | None] = lambda x: getattr(
            x, "_name", None
        ),
        step_flag: str = DEFAULT_STEP_FLAG,
) -> None:
    need = all_names(
        p,
        is_dp=is_dp,
        name_of=name_of,
        step_flag=step_flag,
        include_self=True,
    )
    missing = sorted(k for k in need if k not in set(data_keys))
    if missing:
        msg = f"Data dict missing required keys: {missing}"
        raise KeyError(msg)
