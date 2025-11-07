from __future__ import annotations

import json
import sys

import warnings
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any, TypeVar

import polars as pl

from paguro.defer.utils.materializer import _Materializer
from paguro.defer.utils.utils import DEFAULT_STEP_FLAG
from paguro.utils.dependencies import inspect

if sys.version_info >= (3, 10):
    from typing import Concatenate, ParamSpec
else:
    from typing import Concatenate
    from typing_extensions import ParamSpec

if TYPE_CHECKING:
    import sys
    from collections.abc import Callable, Iterable

    import numpy as np
    from polars import DataFrame
    from polars._typing import (
        ColumnNameOrSelector,
        IntoExpr,
        IntoExprColumn,
    )

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

T = TypeVar("T")
P = ParamSpec("P")


# https://github.com/pola-rs/polars/issues/18284

class LazyFrameExpr:
    __slots__ = ("_name", "_steps")

    def __init__(
            self, name: str | None = None, _steps: tuple | None = None
    ) -> None:
        self._name = name
        self._steps: tuple

        if _steps is not None:
            # Internal fast path (immutably store tuple)
            if not isinstance(_steps, tuple):
                msg = "`_steps` must be a tuple (internal use only)."
                raise TypeError(msg)
            self._steps = _steps
            return
        else:
            self._steps = ()

    # ---------

    def _to_dict(self) -> dict[str, Any]:
        return {"name": self._name, "steps": self._steps}
        # # simple nested format; raises on cycles
        # return export_pipeline_dict(self)

    @classmethod
    def _from_dict(cls, source: dict[str, Any]) -> LazyFrameExpr:
        # only called when the object is  from serializable
        return cls(name=source["name"], _steps=source["steps"])
        # return import_pipeline_dict(cls, source)

    def serialize(self) -> str:
        from paguro.shared.serialize import CustomJSONEncoder

        return json.dumps(self, cls=CustomJSONEncoder)

    @classmethod
    def deserialize(self, source: str) -> LazyFrameExpr:
        from paguro.shared.serialize import CustomJSONDecoder
        return json.loads(source, cls=CustomJSONDecoder)

    # ---------

    def __copy__(self) -> Self:
        return self.__class__(name=self._name, _steps=self._steps)

    def __deepcopy__(self, memo) -> Self:
        return self.__class__(name=self._name, _steps=self._steps)

    def __repr__(self) -> str:
        name = f"{self._name!r}" if isinstance(self._name, str) else self._name
        return f"{self.__class__.__name__}(name={name}, steps={len(self._steps)})"

    def __len__(self) -> int:
        return len(self._steps)

    def __iter__(self) -> Iterator:
        return iter(self._steps)

    # def __call__(self, data: pl.LazyFrame) -> pl.LazyFrame:
    #     for i, step in enumerate(self._steps):
    #         try:
    #             data = _bind_from_step(data, step)
    #         except Exception as exc:
    #             raise RuntimeError(f"[step {i}] {step.keys()} failed: {exc}") from exc
    #     return data

    def _is_self_contained(self) -> bool:
        """Fast O(#steps) check based on the per-step flag."""
        for step in self._steps:
            input_data = next(iter(step.values()))
            if input_data.get(DEFAULT_STEP_FLAG, False):
                return False
        return True

    def __call__(
            self,
            data: pl.LazyFrame | dict[str, pl.LazyFrame],
    ) -> pl.LazyFrame:
        # ---------
        if not isinstance(
                data, dict
        ):  # single-frame fast path (only for linear/self-contained plans)
            if not self._is_self_contained():
                msg = (
                    "This pipeline references other LazyFrameExpr; "
                    "call with a data dict (name -> LazyFrame)."
                )
                raise RuntimeError(msg)

            lf: pl.LazyFrame | None = None

            for i, step in enumerate(self._steps):
                method, input_data = next(iter(step.items()))
                try:
                    if lf is None:
                        lf = data

                    lf = getattr(lf, method)(
                        *input_data["args"],
                        **input_data["kwargs"],
                    )
                except Exception as exc:
                    msg = f"[step {i}] '{method}' failed: {exc}"
                    raise RuntimeError(msg) from exc

            if lf is None:
                return data
            return lf
        # ---------

        # dict-mode: supports cross-pipeline refs
        if self._name is None:
            msg = "Root pipeline must have a name to run on a data dict."
            raise RuntimeError(msg)

        single_lf: pl.LazyFrame | None = data.get(
            self._name
        )  # LazyFrame by contract

        if single_lf is None:
            msg = (
                f"Data dict is missing key '{self._name}' "
                f"required by the root pipeline."
            )
            raise KeyError(msg)
        else:
            mat = _Materializer(data=data)

            for i, step in enumerate(self._steps):
                method, params = next(iter(step.items()))

                if params.get(DEFAULT_STEP_FLAG, False):
                    args, kwargs = mat._resolve_step_input_data(params)
                else:
                    args, kwargs = params["args"], params["kwargs"]

                try:
                    single_lf = getattr(single_lf, method)(*args, **kwargs)
                except Exception as exc:
                    msg = f"[root {self._name!r} step {i}] '{method}' failed: {exc}"
                    raise RuntimeError(msg) from exc

            return single_lf

    def __getattr__(self, attr: str):
        """
        Get attribute.

        Dynamically expose pl.LazyFrame methods whose return type annotation
        indicates they return a LazyFrame (or Self/Union/Optional containing it).

        Explicit methods (like .select, .filter, .group_by, .collect) are defined
        on this class and will not reach here.
        """
        lf_attr = getattr(pl.LazyFrame, attr, None)
        if lf_attr is None:
            msg = f"{self.__class__.__name__}(pl.LazyFrame) has no attribute {attr!r}"
            raise AttributeError(msg)

        elif not callable(lf_attr):
            msg = f"{self.__class__.__name__}(pl.LazyFrame).{attr} is not a callable. "
            raise AttributeError(msg)

        if not _returns_lazyframe(lf_attr):
            msg = (
                f"pl.LazyFrame.{attr} is not annotated to return a LazyFrame; "
                f"wrap explicitly if you intend to support it in the pipeline."
            )
            raise AttributeError(msg)

        def _method(
                *args,
                **kwargs: Any,
        ) -> LazyFrameExpr:
            return self._append(attr, args, kwargs)

        _method.__name__ = attr
        _method.__doc__ = (
            f"Pipeline step proxy for pl.LazyFrame.{attr}(...)."
        )
        return _method

    def _append(
            self, method: str, args: Iterable[Any], kwargs: dict[str, Any]
    ) -> Self:
        # do not check that LazyFrame has attribute here
        # because we call a ppend from GroupBy etc objects
        # if not hasattr(pl.LazyFrame, method):
        #     raise AttributeError(f"pl.LazyFrame has no method '{method}'")

        # store deterministic shapes: args as tuple, kwargs as plain dict

        a = tuple(args)
        k = dict(kwargs)
        step: dict = {method: {"args": a, "kwargs": k}}

        # FIX: set the flag when any arg/kwarg contains a LazyFrameExpr
        if _contains_lfe(a) or _contains_lfe(k):
            step[method][DEFAULT_STEP_FLAG] = True

        return self.__class__(
            name=self._name, _steps=self._steps + (step,)
        )

    # ---------

    def select(
            self,
            *exprs: IntoExpr | Iterable[IntoExpr],
            **named_exprs: IntoExpr,
    ) -> Self:
        return self._append("select", exprs, named_exprs)

    def filter(
            self,
            *predicates: IntoExprColumn
                         | Iterable[IntoExprColumn]
                         | bool
                         | list[bool]
                         | np.ndarray[Any, Any],
            **constraints: Any,
    ) -> Self:
        return self._append("filter", predicates, constraints)

    def with_columns(
            self,
            *exprs: IntoExpr | Iterable[IntoExpr],
            **named_exprs: IntoExpr,
    ) -> Self:
        return self._append("with_columns", exprs, named_exprs)

    def explode(
            self,
            columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
            *more_columns: ColumnNameOrSelector,
    ) -> Self:
        return self._append("explode", (columns, *more_columns), {})

    def pipe(
            self,
            function: Callable[Concatenate[pl.LazyFrame, P], pl.LazyFrame],
            *args: P.args,
            **kwargs: P.kwargs,
    ) -> Self:
        # note: storing callables is fine in-process, but not JSON-serializable
        # note unlike pipe on a DataFrame the function here must return a DataFrame,
        # not a generic T
        return self._append("pipe", (function, *args), kwargs)

    # ------------------------------------------------------------------

    def group_by(
            self,
            *by: IntoExpr | Iterable[IntoExpr],
            maintain_order: bool = False,
            **named_by: IntoExpr,
    ) -> _GroupByExpr:
        kwargs = {"maintain_order": maintain_order, **named_by}
        new_pipe = self._append("group_by", by, kwargs)
        return _GroupByExpr(pipeline=new_pipe)

    def collect(
            self,
            *args,
            **kwargs: Any,
    ) -> DataFrameExpr:
        new_pipe = self._append("collect", args, kwargs)
        return DataFrameExpr(pipeline=new_pipe)


class _GroupByExpr:
    __slots__ = ("_name", "_pipeline")

    def __init__(self, pipeline: LazyFrameExpr) -> None:
        self._pipeline = pipeline

    def __copy__(self) -> Self:
        return self.__class__(pipeline=self._pipeline)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(pipeline={self._pipeline})"

    def agg(
            self,
            *aggs: IntoExpr | Iterable[IntoExpr],
            **named_aggs: IntoExpr,
    ) -> LazyFrameExpr:
        return self._pipeline._append("agg", aggs, named_aggs)


class DataFrameExpr:
    __slots__ = ("_pipeline",)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._pipeline})"

    def __init__(self, pipeline: LazyFrameExpr) -> None:
        self._pipeline = pipeline

    def __copy__(self) -> Self:
        return self.__class__(pipeline=self._pipeline)

    def __getattr__(self, name: str):
        df_attr = getattr(pl.DataFrame, name, None)
        if df_attr is None or not callable(df_attr):
            msg = (
                f"{self.__class__.__name__} has no attribute {name!r} "
                f"and pl.LazyFrame.{name} is not a callable."
            )
            raise AttributeError(msg)

        if not _returns_dataframe(df_attr):
            msg = (
                f"pl.DataFrame.{name} is not annotated to return a DataFrame; "
                f"wrap explicitly if you intend to support it in the pipeline."
            )
            raise AttributeError(msg)

        def _method(
                *args,
                **kwargs: Any,
        ) -> DataFrameExpr:
            self._pipeline = self._pipeline._append(name, args, kwargs)
            return self

        _method.__name__ = name
        _method.__doc__ = (
            f"Pipeline step proxy for pl.DataFrame.{name}(...)."
        )
        return _method

    def pipe(
            self,
            function: Callable[Concatenate[DataFrame, P], DataFrame],
            *args: P.args,
            **kwargs: P.kwargs,
    ) -> Self:
        # note unlike pipe on a DataFrame the function
        # here must return a DataFrame, not a generic T
        self._pipeline = self._pipeline._append(
            "pipe", (function, *args), kwargs
        )
        return self

    def pivot(
            self,
            *args,
            **kwargs: Any,
    ) -> Self:
        self._pipeline = self._pipeline._append("pivot", args, kwargs)
        return self

    # -----------

    def lazy(self) -> LazyFrameExpr:
        return self._pipeline._append("lazy", (), {})


# -------------------


def _bind_from_step(obj: Any, step, ):
    ((method, input_data),) = step.items()  # exact one key per step
    try:
        return getattr(obj, method)(
            *input_data["args"],
            **input_data["kwargs"],
        )
    except Exception as exc:
        msg = f"Error applying step '{method}': {exc}"
        raise RuntimeError(msg) from exc


# -------------------


def inspect_transform_data_function(
        func: Callable,
) -> Callable:
    if not callable(func):
        msg = f"'{func}' must be a callable!"
        raise TypeError(msg)

    signature = inspect.signature(func)

    # Get the first parameter name
    parameters = signature.parameters

    # Ensure there is at least one parameter
    if len(parameters) == 0:
        msg = "The function must have at least one parameter."
        raise TypeError(msg)
    # elif len(parameters) > 1:
    #     raise TypeError("The function must have at most one parameter.")

    first_param_name = next(iter(parameters))
    type_hints = get_type_hints(func)

    # Check if type hints are missing and issue a warning
    if first_param_name not in type_hints:
        warnings.warn(
            f"The function's parameter "
            f"'{first_param_name}' is missing type hints.\n"
            f"Note that the function's parameter "
            f"'{first_param_name}' type should be"
            "\n\t- polars.LazyFrame",
            UserWarning,
            stacklevel=2,
        )
    elif type_hints[first_param_name] is not pl.LazyFrame:
        msg = (
            f"The function's parameter '{first_param_name}' "
            f"type should be\n\t- polars.LazyFrame"
        )
        raise TypeError(msg)

    if "return" not in type_hints:
        warnings.warn(
            "Missing a return typehint.\n"
            "The function's return type should be"
            "\n\t- polars.LazyFrame"
            "\n\t- polars.DataFrame",
            stacklevel=2,
        )

    elif type_hints["return"] not in [
        pl.DataFrame,
        pl.LazyFrame,
        # Dataset
    ]:
        msg = (
            "The return type must be one of:\n\t"
            "- polars.LazyFrame\n\t- polars.DataFrame"
            # "\n\t- paguro.Dataset"
        )
        raise TypeError(msg)

    return func


# ----------------------------------------------------------------------

from typing import get_type_hints


# get_type_hints(pl.DataFrame.pivot)
# ann = getattr(pl.DataFrame.pivot, "__annotations__", {}) or {}


def _annotation_matches(tp: Any, target: type) -> bool:
    """
    Annotation matches.

    Return True if the given annotation is 'exactly' target
    or a container (Union/Optional) that includes target.
    """
    if tp is None:
        return False
    if tp is target:
        return True

    # origin = get_origin(tp)
    # args = get_args(tp)

    # # Handle Union / Optional
    # if origin is Union:
    #     return any(_annotation_matches(a, target) for a in args)

    # Handle forward-referenced string hints
    if isinstance(tp, str):
        # e.g. "LazyFrame", "Self"
        if tp == "Self":
            return issubclass(
                target, target
            )  # always True: Self maps to the class
        return tp == target.__name__

    # PEP 673 `Self`
    if (
            getattr(tp, "__module__", "") == "typing"
            and getattr(tp, "__qualname__", "") == "Self"
    ):
        return True if target is not None else False

    return False


def returns_type(
        func: Any, target: type, *, globalns=None, localns=None
) -> bool:
    """
    Inspect callable's return type annotation.

    Returns True if the annotation is exactly `target` or
    a Union/Optional containing it. False otherwise.
    """
    if not callable(func):
        return False

    try:
        hints = get_type_hints(func, globalns=globalns, localns=localns)
    except Exception:
        # fallback: raw annotations (may contain strings/ForwardRef)
        ann = getattr(func, "__annotations__", {}) or {}
        return _annotation_matches(dict(ann).get("return", None), target)

    ret = hints.get("return", None)
    return _annotation_matches(ret, target)


def _returns_lazyframe(func: Any) -> bool:
    return returns_type(
        func,
        pl.LazyFrame,
        globalns={"pl": pl, "polars": pl, **vars(pl)},
        localns={"LazyFrame": pl.LazyFrame, "Self": pl.LazyFrame},
    )


def _returns_dataframe(func: Any) -> bool:
    return returns_type(
        func,
        pl.DataFrame,
        globalns={"pl": pl, "polars": pl, **vars(pl)},
        localns={"DataFrame": pl.DataFrame, "Self": pl.DataFrame},
    )


# ----------------------------------------------------------------------


def _contains_lfe(x: Any) -> bool:
    """Return True iff `x` (recursively) contains a LazyFrameExpr object."""
    if isinstance(x, LazyFrameExpr):
        return True
    if isinstance(x, tuple):
        return any(_contains_lfe(v) for v in x)
    if isinstance(x, list):
        return any(_contains_lfe(v) for v in x)
    if isinstance(x, Mapping):
        return any(_contains_lfe(v) for v in x.values())
    return False


# ----------------------------------------------------------------------


def _resolve_any(
        x: Any,
        materialize: Callable[[LazyFrameExpr], pl.LazyFrame],
) -> Any:
    """
    Recursive resolver that replaces any LazyFrameExpr with its LazyFrame.

    Using the provided `materialize` callback. Never mutates inputs.
    """
    if isinstance(x, LazyFrameExpr):
        return materialize(x)

    if isinstance(x, tuple):
        if not x:
            return x
        return tuple(_resolve_any(v, materialize) for v in x)

    if isinstance(x, list):
        if not x:
            return x
        return [_resolve_any(v, materialize) for v in x]

    if isinstance(x, Mapping):
        if not x:
            return dict(x)
        return {k: _resolve_any(v, materialize) for k, v in x.items()}

    return x
