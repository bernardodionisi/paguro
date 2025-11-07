from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import polars.selectors as cs

from paguro.shared.various import pl_schema

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence
    from polars._typing import ColumnNameOrSelector


def _get_join_columns(
    on: str | pl.Expr | Sequence[str | pl.Expr] | None = None,
    left_on: str | pl.Expr | Sequence[str | pl.Expr] | None = None,
    right_on: str | pl.Expr | Sequence[str | pl.Expr] | None = None,
) -> tuple[set[str], set[str]]:
    """Get the columns used in the join for both left and right datasets."""

    def _to_col_names(keys) -> set[str]:
        if keys is None:
            return set()
        if isinstance(keys, (str, pl.Expr)):
            keys = [keys]

        # TODO: fix expression to return column name?
        return {str(k) if isinstance(k, pl.Expr) else k for k in keys}

    if on is not None:
        join_cols = _to_col_names(on)
        return join_cols, join_cols
    return _to_col_names(left_on), _to_col_names(right_on)


def _unnest(
    data: pl.DataFrame | pl.LazyFrame,
    columns: ColumnNameOrSelector | None = None,
    separator: str = "_",
) -> pl.DataFrame | pl.LazyFrame:
    # Support adding prefixes to
    # DataFrame.unnest: https://github.com/pola-rs/polars/issues/9790

    if columns is None:
        more_structs = True

        while more_structs:
            cols = []
            for k, v in pl_schema(data).items():
                if isinstance(v, pl.Struct):
                    cols.append(k)

            if not cols:
                break

            data = unnest_with_prefix(
                data=data, columns=cols, separator=separator
            )
        return data

    _columns: Sequence[str] | cs.Selector
    if isinstance(columns, cs.Selector):
        # Cannot select struct columns by
        # dtype: https://github.com/pola-rs/polars/issues/11067
        _columns = cs.expand_selector(data, columns)
    elif isinstance(columns, str):
        _columns = (columns,)
    else:
        _columns = columns

    data = unnest_with_prefix(
        data=data, columns=_columns, separator=separator
    )
    return data


def unnest_with_prefix(
    data: pl.DataFrame | pl.LazyFrame,
    columns: Collection[str],
    separator: str,
) -> pl.DataFrame | pl.LazyFrame:
    return data.with_columns(
        pl.col(struct).name.prefix_fields(f"{struct}{separator}")
        for struct in columns
    ).unnest(columns)


# ----------------------------------------------------------------------

