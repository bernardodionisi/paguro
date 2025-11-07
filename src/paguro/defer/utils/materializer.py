from __future__ import annotations

from typing import TYPE_CHECKING, Any

from paguro.defer.utils.utils import DEFAULT_STEP_FLAG

if TYPE_CHECKING:
    from collections.abc import Mapping

    import polars as pl

    from paguro.defer.frames import LazyFrameExpr


class _Materializer:
    """
    Materializer.

    Resolves DeferredPipelines embedded in step args/kwargs to LazyFrames,
    using only a data map (name -> LazyFrame). Memoized & cycle-safe.
    Steps: {method: {"args": tuple, "kwargs": dict, "__has_dp__"?: bool}}.
    """

    def __init__(self, data: dict[str, pl.LazyFrame] | None) -> None:
        self._data = data
        self._cache: dict[int, pl.LazyFrame] = {}
        self._stack: set[str] = set()

    def _resolve_step_input_data(
            self,
            input_data: Mapping[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if not input_data.get(DEFAULT_STEP_FLAG, False):
            return input_data["args"], input_data["kwargs"]

        from paguro.defer.frames import _resolve_any

        args = _resolve_any(
            input_data["args"],
            materialize=self.materialize,
        )

        kwargs = _resolve_any(
            input_data["kwargs"],
            materialize=self.materialize,
        )
        return args, kwargs

    def materialize(self, p: LazyFrameExpr) -> pl.LazyFrame:
        pid = id(p)

        # cache still by object identity (cheap + correct)
        if pid in self._cache:
            return self._cache[pid]

        # guards
        if self._data is None:
            msg = f"Pipeline {p._name!r} references other pipelines but no data dict was provided."
            raise RuntimeError(msg)
        if p._name is None:
            msg = (
                "Referenced pipeline must have a name to run in dict-mode."
            )
            raise RuntimeError(msg)

        pname = p._name

        # NAME-BASED cycle detection
        if pname in self._stack:
            msg = f"Cycle detected involving pipeline {pname!r}."
            raise RuntimeError(msg)

        if pname not in self._data:
            msg = (
                f"Data dict is missing key '{pname}' required by pipeline."
            )
            raise KeyError(msg)

        self._stack.add(pname)
        try:
            lf = self._data[pname]  # LazyFrame by contract
            for step in p._steps:
                method, input_data = next(iter(step.items()))
                if input_data.get(DEFAULT_STEP_FLAG, False):
                    args, kwargs = self._resolve_step_input_data(input_data)
                else:
                    args, kwargs = input_data["args"], input_data["kwargs"]
                try:
                    lf = getattr(lf, method)(*args, **kwargs)
                except Exception as exc:
                    msg = (
                        f"[pipeline {pname!r}] error applying '{method}' "
                        f"with args={args} kwargs={kwargs}: {exc}"
                    )
                    raise RuntimeError(msg) from exc
            self._cache[pid] = lf
            return lf
        finally:
            self._stack.remove(pname)
