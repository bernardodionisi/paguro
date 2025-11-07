from __future__ import annotations
import sys

import warnings
from typing import Any, Callable, Iterable, cast, TypeAlias, Union

import polars as pl
import polars.selectors as cs
from polars.exceptions import ComputeError

# A single column spec users can pass (string name, expression, or selector)
ColumnSpec: TypeAlias = Union[str, pl.Expr, cs.Selector]
ColumnsList: TypeAlias = list[ColumnSpec]

# Metric can be:
# - a string like "mean" or "str.len_chars.min"
# - a ready-made pl.Expr
# - a tuple:
#     * (expr,)                         -> single expr
#     * (name, expr)                    -> label + expr
#     * (name, arg1, arg2, ...)         -> label + call on column with args
MetricTupleSingleExpr: TypeAlias = tuple[pl.Expr]
MetricTupleNameExpr: TypeAlias = tuple[str, pl.Expr]

# MetricVariadic: TypeAlias = tuple[str, *tuple[Any, ...]]
MetricVariadic: TypeAlias = tuple[str, ...]

MetricTuple: TypeAlias = (
    Union[MetricTupleSingleExpr, MetricTupleNameExpr, MetricVariadic]
)
Metric: TypeAlias = Union[str, pl.Expr, MetricTuple]

# A single config: (title, columns, metrics)
# columns / metrics can be a single item or a list; we'll normalize to lists
SingleConfig: TypeAlias = tuple[
    str, Union[ColumnSpec, ColumnsList], Union[Metric, list[Metric]]
]


# TODO
class SingleSkimConfig:
    def __init__(self, config: Any) -> None: ...


def unpack_single_config_and_get_exprs(
        data: pl.LazyFrame,
        single_config: SingleConfig,
        by: list[str] | None,
) -> tuple[
    str,
    ColumnsList,
    dict[str, list[str]],
    list[pl.Expr],
]:
    title, columns, metrics = _unpack_single_config(
        single_config=single_config, by=by
    )
    out_names, exprs = _get_column_metrics_expressions(
        data=data, columns=columns, metrics=metrics
    )
    return title, columns, out_names, exprs


def _unpack_single_config(
        single_config: SingleConfig, by: list[str] | None
) -> tuple[str, ColumnsList, list[Metric]]:
    if len(single_config) != 3:
        msg = (
            "Config must have 3 elements:"
            "\n\t- title: str"
            "\n\t- columns: str | pl.Expr | cs.Selector | list[str | pl.Expr | cs.Selector]"
            "\n\t- metrics: str | pl.Expr | tuple | list[str | pl.Expr | tuple]"
        )
        raise ValueError(msg)

    title, columns, metrics = single_config

    # Normalize columns to list
    col_list: ColumnsList = (
        columns if isinstance(columns, list) else [columns]
    )

    if by is not None:
        col_list = _remove_by_from_columns(by=by, columns=col_list)

    # Normalize metrics to list
    met_list: list[Metric] = (
        metrics if isinstance(metrics, list) else [metrics]
    )

    return title, col_list, met_list


def _get_column_metrics_expressions(
        data: pl.LazyFrame,
        *,
        columns: ColumnsList,
        metrics: list[Metric],
) -> tuple[dict[str, list[str]], list[pl.Expr]]:
    metrics_names = [_get_metric_name(m) for m in metrics]
    columns_names = _get_column_names(data=data, columns=columns)

    names = {
        c: [f"{i}{c}" for i in metrics_names]
        for c in columns_names  # columns do not include 'by'
    }

    exprs: list[pl.Expr] = []
    for c in columns:
        # selector_proxy is a pl.Expr too; ensure we have an Expr
        col_expr = c if isinstance(c, pl.Expr) else pl.col(c)

        # all expressions for set of column(s)
        exprs.extend(
            _get_single_expr(column=col_expr, metric=m) for m in metrics
        )

    return names, exprs


def _get_single_expr(
        column: pl.Expr,
        metric: Metric,
) -> pl.Expr:
    if isinstance(metric, str):
        name = _get_metric_name(m=metric)
        func = _get_nested_attr(column, metric)
        expr = cast(Callable[..., pl.Expr], func)()

    elif isinstance(metric, pl.Expr):
        name = _get_metric_name(m=metric)
        expr = metric

    else:  # tuple variants
        name, expr = _dispatch_metric_tuple(metric, col=column)

    return expr.name.prefix(name)


def _dispatch_metric_tuple(
        metric: MetricTuple,
        col: pl.Expr,
) -> tuple[str, pl.Expr]:
    # (name, expr)
    if len(metric) == 2 and isinstance(metric[1], pl.Expr):
        name = _get_metric_name(metric[0])
        expr = metric[1]
        return name, expr

    # (expr,)
    if len(metric) == 1 and isinstance(metric[0], pl.Expr):
        expr = metric[0]
        name = _get_metric_name(m=expr)
        return name, expr

    # (name,) or (name, *args)
    if isinstance(metric[0], str):
        name = _get_metric_name(m=metric)
        func = _get_nested_attr(col, metric[0])
        expr = cast(Callable[..., pl.Expr], func)(*metric[1:])
        return name, expr

    msg = "When metric is a tuple its element(s) must be either str or pl.Expr"
    raise ValueError(msg)


def _get_nested_attr(obj: Any, attr_string: str) -> Any:
    for attr in attr_string.split("."):
        obj = getattr(obj, attr)
    return obj


# ----------------------------------------------------------------------


def _get_metric_name(m: str | pl.Expr | MetricTuple) -> str:
    if isinstance(m, tuple):
        if len(m) == 2 and isinstance(m[1], pl.Expr):
            return f"{m[0]!s}::"
        return f"{m[0]}({', '.join(str(a) for a in m[1:])})::"
    return f"{m!s}::"


def _get_column_names(
        data: pl.LazyFrame,
        columns: ColumnSpec | ColumnsList,
) -> list[str]:
    cols: ColumnsList = columns if isinstance(columns, list) else [columns]

    has_selector = any(isinstance(m, cs.Selector) for m in cols)

    if not has_selector:
        try:
            return [
                c.meta.output_name()
                if isinstance(c, pl.Expr)
                else cast(str, c)
                for c in cols
            ]
        except ComputeError as e:  # e.g. pl.col(["a", "b"]) is passed
            warnings.warn(str(e), stacklevel=2)

    # Fallback: let Polars resolve selectors/exprs to concrete names
    # (avoid using cs.expand_selector since we want to keep deps minimal)
    return list(data.select(cols).collect_schema().names())


# ----------------------------------------------------------------------


def _remove_by_from_columns(
        by: list[str],
        columns: ColumnsList,
) -> ColumnsList:
    # TODO: allow `by` to include Expressions: just remove if string,
    #  otherwise return as is (the default Polars error about including alias is clear)

    out: ColumnsList = []
    for col in columns:
        if isinstance(col, str):
            if col not in by:
                out.append(col)

        elif isinstance(col, cs.Selector):
            sel = col
            for i in by:
                sel -= cs.matches(i)
            out.append(sel)

        elif isinstance(col, pl.Expr):
            root_names = col.meta.root_names()

            if len(root_names) > 1:
                overlap = [i for i in by if i in root_names]
                if overlap:
                    msg = (
                        f"The column '{overlap}', used in the argument 'by', "
                        f"cannot be included among the columns to 'skim': {root_names}."
                    )
                    raise ValueError(msg)

            out.append(col)
        else:
            msg = f"Unsupported type: {type(col)}"
            raise TypeError(msg)

    return out
