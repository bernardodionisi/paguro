from __future__ import annotations

import polars as pl
from typing import TYPE_CHECKING


def get_allow_nulls_predicate(
        column_name: str,
        struct_fields: tuple[str, ...] | None,
) -> pl.Expr:
    if not struct_fields:
        return pl.col(column_name).is_not_null()
    # inverted if requested? pl.col(column_name).is_null()

    return get_struct_expr(struct_fields).is_not_null()


def get_unique_predicate(
        column_name: str,
        struct_fields: tuple[str, ...] | None,
) -> pl.Expr:
    if not struct_fields:
        return pl.col(column_name).is_unique()
    # inverted if requested? pl.col(column_name).is_duplicated()

    return get_struct_expr(struct_fields).is_unique()


def get_struct_expr(struct_fields: tuple[str, ...]) -> pl.Expr:
    expr = pl.col(struct_fields[0])
    for f in struct_fields[1:]:
        expr = expr.struct.field(f)
    return expr
