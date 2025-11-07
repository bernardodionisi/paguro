from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from polars import selectors as cs

from paguro.utils.dependencies import re

if TYPE_CHECKING:
    from collections.abc import Collection

    import polars as pl
    from polars._typing import ColumnNameOrSelector

    from paguro.typing import IntoKeepColumns


def _select_keep_columns(
    frame: pl.LazyFrame,
    *,
    column_names: ColumnNameOrSelector | Collection[str],
    keep_columns: IntoKeepColumns,
    with_row_index: bool | str,
) -> pl.LazyFrame:
    # -------
    if not isinstance(column_names, cs.Selector):
        column_selector: cs.Selector = cs.by_name(column_names)
    else:
        column_selector = column_names

    # if _is_str_or_iterable_str(column_names):
    #     column_names = cs.by_name(column_names)
    #     # column_names = cs.matches(_exact_match_regex(column_names))

    # -------

    if _is_str_or_iterable_str(keep_columns):
        # keep_columns = cs.by_name(keep_columns)
        keep_columns = cs.matches(_exact_match_regex(keep_columns))  # type: ignore[arg-type]

    # -------

    if isinstance(keep_columns, bool):
        if not keep_columns:
            frame = frame.select(column_selector)
        else:
            frame = frame.select(
                column_selector, (cs.all() - column_selector)
            )

    elif isinstance(keep_columns, cs.Selector):
        frame = frame.select(
            column_selector, (keep_columns - column_selector)
        )

    else:  # keep_columns is expression
        if isinstance(keep_columns, Iterable):
            adjusted_keep_columns: list[str | pl.Expr] = []
            for i in keep_columns:
                if isinstance(i, cs.Selector):
                    adjusted_keep_columns.append(i - column_selector)
                else:  # expression/str
                    adjusted_keep_columns.append(i)

            frame = frame.select(column_selector, *adjusted_keep_columns)
        else:
            frame = frame.select(column_selector, keep_columns)

    if with_row_index:
        if isinstance(with_row_index, bool):
            frame = frame.with_row_index()
        else:
            frame = frame.with_row_index(name=with_row_index)

    return frame


def _exact_match_regex(strings: str | Iterable[str]) -> str:
    """Return a regex pattern that matches exactly the given string(s)."""
    if isinstance(strings, str):
        choices = [re.escape(strings)]
    elif isinstance(strings, Iterable):
        choices = [re.escape(s) for s in strings]
    else:
        msg = "Input must be a string or iterable of strings."
        raise TypeError(msg)

    pattern = "^(" + "|".join(choices) + ")$"
    return pattern


def _is_str_or_iterable_str(input_: str | Iterable[str] | Any) -> bool:
    return isinstance(input_, str) or (
        isinstance(input_, Iterable)
        and all(isinstance(i, str) for i in input_)
    )
