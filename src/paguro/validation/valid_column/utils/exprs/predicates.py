from __future__ import annotations

import polars as pl


def get_allow_nulls_predicate(
        column_name: str,
        _new_root: pl.Expr | None,
) -> pl.Expr:
    if _new_root is None:
        return pl.col(column_name).is_not_null()
    # inverted if requested? pl.col(column_name).is_null()

    return _new_root.is_not_null()


def get_unique_predicate(
        column_name: str,
        _new_root: pl.Expr | None,
) -> pl.Expr:
    if _new_root is None:
        return pl.col(column_name).is_unique()
    # inverted if requested? pl.col(column_name).is_duplicated()

    return _new_root.is_unique()


def get_new_root(  # get_struct_expr
        _root_down: tuple[str, ...],
) -> pl.Expr:
    expr = pl.col(_root_down[0])
    for f in _root_down[1:]:
        expr = expr.struct.field(f)
    return expr
