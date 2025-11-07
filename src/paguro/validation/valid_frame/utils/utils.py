from __future__ import annotations

import os
from typing import TYPE_CHECKING

import polars as pl

from paguro.validation.shared.keep_columns import _select_keep_columns

if TYPE_CHECKING:
    from collections.abc import Collection

    from paguro.typing import IntoKeepColumns


def _negate_filter_from_expr_unique_by(
    frame: pl.LazyFrame,
    unique_by: Collection[str],
    keep_columns: IntoKeepColumns,
    *,
    with_row_index: bool | str,
    sort: bool,
) -> pl.LazyFrame:
    frame = _select_keep_columns(
        frame=frame,
        column_names=unique_by,
        keep_columns=keep_columns,
        with_row_index=with_row_index,
    )

    frame = frame.filter(pl.struct(unique_by).is_duplicated())

    if sort:
        frame = frame.sort(unique_by)

    return frame


def _negate_filter_from_expr(
    frame: pl.LazyFrame,
    expr: pl.Expr,
    *,
    keep_columns: IntoKeepColumns,
    with_row_index: bool | str,
) -> pl.LazyFrame:
    _root_names: list[str] = expr.meta.root_names()

    if not _root_names:
        # pl.all(), ...
        # watch: pl.element().meta.root_names() -> ['']
        # https://github.com/pola-rs/polars/issues/23553
        if (
            isinstance(keep_columns, bool) and not keep_columns
            # unless directly specified. This could lead to an error.
            # If expression refers to columns not in keep_columns
        ):
            # warnings.warn(
            #     f"The root names of the expression is empty (expr: str({expr}))."
            #     f"Automatically setting keep_columns=True. "
            #     f"Use 'keep_columns' to specify which columns to keep.",
            # )
            keep_columns = True

        # NOTE: when not _root_names, for example,
        # pl.all_horizontal(pl.col("a", "b").ge(0))
        # the errors frames columns will not be
        # reordered to show the involved columns first,
        # since we do not know which columns to place first

    # root names may be repeated, let's drop repetitions
    # do not call set, otherwise we lose
    # the order of columns which we use when selecting
    root_names = list(dict.fromkeys(_root_names))

    frame = _select_keep_columns(
        frame=frame,
        column_names=root_names,
        keep_columns=keep_columns,
        # TODO: either set this to True or have two selection
        #  pre post filtering because root names may be empty
        with_row_index=with_row_index,
    )

    if bool(
        int(os.environ.get("PAGURO_INVERTED_VALIDATION_FILTER", True))
    ):  # inverted=True
        frame = frame.filter(~(expr))
    else:
        frame = frame.filter(expr)

    # # if we select after
    # filtering we don't have to worry
    # about missing the columns in the expression
    # frame = _select_keep_columns(
    # frame=frame,
    # column_names=root_names,
    # keep_columns=keep_columns,
    # with_row_index=with_row_index)

    return frame
