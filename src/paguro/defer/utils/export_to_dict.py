from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from paguro.defer.utils.utils import DEFAULT_STEP_FLAG

if TYPE_CHECKING:
    from paguro.defer.frames import LazyFrameExpr

_DP_TAG = "__dls__"  # marker key for nested pipelines


def export_lfe_to_dict(
    p: LazyFrameExpr, _seen: set[int] | None = None
) -> dict[str, Any]:
    """Export a pipeline to a JSON-friendly nested dict (no ids/registry)."""
    if _seen is None:
        _seen = set()
    pid = id(p)
    if pid in _seen:
        # This simple dict format can't represent cycles;
        # a graph format with ids is required for that.
        msg = "Cycle detected: simple export does not support cyclic pipelines."
        raise ValueError(msg)
    _seen.add(pid)

    out_steps: list[dict[str, Any]] = []
    for step in p._steps:
        method, params = next(iter(step.items()))
        out_params = {
            "args": _export_any(params.get("args", ()), _seen),
            "kwargs": _export_any(params.get("kwargs", {}), _seen),
            # NOTE: the execution flag is recomputed during import
        }
        out_steps.append({method: out_params})

    return {"name": p._name, "steps": out_steps}


def _export_any(x: Any, _seen: set[int]) -> Any:
    """Recursively replace DeferredLazyFrame objects with {_DP_TAG: <exported dict>}."""
    from paguro.defer.frames import LazyFrameExpr

    if isinstance(x, LazyFrameExpr):
        return {_DP_TAG: export_lfe_to_dict(x, _seen)}
    if isinstance(x, tuple):
        # JSON will emit lists; that is acceptable—we convert back to tuple on import.
        return tuple(_export_any(v, _seen) for v in x)
    if isinstance(x, list):
        return [_export_any(v, _seen) for v in x]
    if isinstance(x, Mapping):
        return {k: _export_any(v, _seen) for k, v in x.items()}
    # Everything else passes through; json.dumps
    # may still fail if it's not serializable.
    return x


def import_lfe_dict(
    cls: type[LazyFrameExpr],
    d: dict[str, Any],
) -> LazyFrameExpr:
    """Construct a DeferredLazyFrame from a dict produced by _export_pipeline_dict."""
    from paguro.defer.frames import _contains_lfe

    name = d.get("name")
    ser_steps = d.get("steps", [])
    steps: list[dict[str, Any]] = []

    for ser_step in ser_steps:
        method, step_dict = next(iter(ser_step.items()))
        args = _import_any(cls, step_dict.get("args", []))
        kwargs = _import_any(cls, step_dict.get("kwargs", {}))

        # Normalize shapes for internal storage
        if isinstance(args, list):
            args = tuple(args)
        elif not isinstance(args, tuple):
            args = (args,)

        if not isinstance(kwargs, dict):
            msg = f"Imported 'kwargs' must be a dict; got {type(kwargs)!r}"
            raise TypeError(msg)

        step_content: dict[str, Any] = {"args": args, "kwargs": kwargs}

        # Recompute the fast-execution flag
        if _contains_lfe(args) or _contains_lfe(kwargs):
            step_content[DEFAULT_STEP_FLAG] = True

        steps.append({method: step_content})

    # Use the internal fast path
    return cls(name=name, _steps=tuple(steps))


def _import_any(cls: type[LazyFrameExpr], x: Any) -> Any:
    """Recursively rebuild nested pipelines from {_DP_TAG: <dict>}."""
    if isinstance(x, dict) and set(x.keys()) == {_DP_TAG}:
        return import_lfe_dict(cls, x[_DP_TAG])
    if isinstance(x, list):
        return [_import_any(cls, v) for v in x]
    if isinstance(x, tuple):
        return tuple(_import_any(cls, v) for v in x)
    if isinstance(x, Mapping):
        return {k: _import_any(cls, v) for k, v in x.items()}
    return x
