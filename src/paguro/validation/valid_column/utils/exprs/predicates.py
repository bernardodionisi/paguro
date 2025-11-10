from __future__ import annotations

import polars as pl
from typing import TYPE_CHECKING


def get_allow_nulls_predicate(
        column_name: str,
        root_down: tuple[str, ...] | None,
) -> pl.Expr:
    if not root_down:
        return pl.col(column_name).is_not_null()
    # inverted if requested? pl.col(column_name).is_null()

    return get_struct_expr(root_down).is_not_null()


def get_unique_predicate(
        column_name: str,
        root_down: tuple[str, ...] | None,
) -> pl.Expr:
    if not root_down:
        return pl.col(column_name).is_unique()
    # inverted if requested? pl.col(column_name).is_duplicated()

    return get_struct_expr(root_down).is_unique()


def get_struct_expr(root_down: tuple[str, ...]) -> pl.Expr:
    expr = pl.col(root_down[0])
    for f in root_down[1:]:
        expr = expr.struct.field(f)
    return expr
