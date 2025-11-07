from __future__ import annotations

import os
from typing import TYPE_CHECKING

import polars as pl

from paguro.validation.shared.keep_columns import _select_keep_columns

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paguro.typing import IntoKeepColumns


def _null_counts(data: pl.LazyFrame, column_name: str) -> pl.LazyFrame:
    # if the count is 0, the collected dataframe will be empty
    new_column_name = f"{column_name}[null_count]"
    return (
        data
        .select(pl.col(column_name).alias(new_column_name))
        .null_count()
        .filter(pl.col(new_column_name).ge(1))
    )


def _get_duplicates(
        data: pl.LazyFrame,
        column_name: str,
        keep_columns: IntoKeepColumns,
        *,
        with_row_index: bool | str,
        sort: bool = True,
) -> pl.LazyFrame:
    # if there are no duplicates, the collected dataframe will be empty
    data = _select_keep_columns(
        frame=data,
        column_names=column_name,
        keep_columns=keep_columns,
        with_row_index=with_row_index,
    )
    data = data.filter(pl.col(column_name).is_duplicated())
    if sort:
        return data.sort(column_name)
    return data


def _get_nulls(
        data: pl.LazyFrame,
        *,
        column_name: str,
        keep_columns: IntoKeepColumns,
        with_row_index: bool | str,
) -> pl.LazyFrame:
    # if there are no duplicates, the collected dataframe will be empty
    data = _select_keep_columns(
        frame=data,
        column_names=column_name,
        keep_columns=keep_columns,
        with_row_index=with_row_index,
    )
    return data.filter(pl.col(column_name).is_null())


def _has_additional_columns(
        *,
        keep_columns: IntoKeepColumns,
        with_row_index: bool | str,
) -> bool:
    keep_columns = bool(
        not isinstance(
            keep_columns, bool
        )  # str, expression, selector, list...
        or keep_columns  # bool: True
    )
    return keep_columns or bool(with_row_index)


def _select_and_filter(
        data: pl.LazyFrame,
        *,
        column_name: str,
        keep_columns: IntoKeepColumns,
        with_row_index: bool | str,
        expr: pl.Expr,
) -> pl.LazyFrame:
    data = _select_keep_columns(
        frame=data,
        column_names=column_name,
        keep_columns=keep_columns,
        with_row_index=with_row_index,
    )

    # TODO: move inversion outside in expr definition
    if bool(
            int(os.environ.get("PAGURO_INVERTED_VALIDATION_FILTER", True))
    ):  # inverted=True
        data = data.filter(~expr)
    else:
        data = data.filter(expr)

    return data


# ----------------------------------------------------------------------

# old function, probably best to use collected schemas
def _get_renamed_columns_in_join(
        left_columns: list[str],
        right_columns: list[str],
        suffix: str,
        on: str | pl.Expr | Sequence[str | pl.Expr] | None = None,
        left_on: str | pl.Expr | Sequence[str | pl.Expr] | None = None,
        right_on: str | pl.Expr | Sequence[str | pl.Expr] | None = None,
) -> dict[str, str]:
    def extract_name(key: str | pl.Expr) -> str:
        if isinstance(key, str):
            return key
        elif isinstance(key, pl.Expr):
            return key.meta.output_name()
        else:  # Fallback: convert to string.
            return str(key)

    def normalize_keys(
            keys: str | pl.Expr | Sequence[str | pl.Expr] | None,
    ) -> set:
        """
        Converts keys into a set of output column names.

        keys which may be a string, iterable, or Polars expressions.
        """
        if keys is None:
            return set()
        if isinstance(keys, (str, pl.Expr)):
            return {extract_name(keys)}
        try:
            return {extract_name(key) for key in keys}
        except TypeError as e:
            raise e

    # Determine effective right join keys.
    if on is not None:
        effective_right_keys = normalize_keys(on)
    elif left_on is not None and right_on is not None:
        effective_right_keys = normalize_keys(left_on) & normalize_keys(
            right_on
        )
    elif right_on is not None:
        effective_right_keys = normalize_keys(right_on)
    else:
        effective_right_keys = set()

    left_set = set(left_columns)
    rename_mapping = {}

    for col in right_columns:
        if col in left_set and col not in effective_right_keys:
            rename_mapping[col] = f"{col}{suffix}"
    return rename_mapping

# ----------------------------------------------------------------------
