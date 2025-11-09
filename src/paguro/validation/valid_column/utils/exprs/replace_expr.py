from __future__ import annotations

import polars as pl
import os
import warnings

from typing import TYPE_CHECKING, Any, Callable
from paguro.validation.valid_column.utils.exprs.json_expr import replace_expression_root
from paguro.validation.valid_column.utils.exprs.predicates import get_struct_expr

if TYPE_CHECKING:
    from paguro.typing import IntoKeepColumns


def replace_predicate(
        *,
        expr: pl.Expr,
        struct_fields: tuple[str, ...],
) -> pl.Expr:
    try:
        return replace_expression_root(
            expr=expr,
            new_root=get_struct_expr(struct_fields=struct_fields),
        )
    except Exception as e:
        msg = f"unable to determine predicate for {struct_fields}"
        raise type(e)(msg) from e


def replace_expr(
        expr: pl.Expr,
        column_name: str,
        *,
        keep_columns: IntoKeepColumns,
        with_row_index: bool | str,
) -> tuple[IntoKeepColumns, bool | str, pl.Expr]:
    try:
        if (
                # if _PAGURO_JSON_REPLACEMENT_WORKING is False
                # we have already set
                # keep_columns, with_row_index to False
                bool(
                    int(
                        os.environ.get(
                            "_PAGURO_JSON_REPLACEMENT_WORKING", True
                        )
                    )
                ) and
                not expr.meta.root_names()  # pl.all().
                # Users could also have passed other expr that have
                # no root_names pl.col("a", "b") but that is discouraged for vcol.
        ):
            # going from pl.all() to pl.col(name)

            # used to do this only if keep_columns/with_row_index were set
            # but its useful to have it as fully specified predicate
            expr = replace_expression_root(
                expr=expr,
                new_root=column_name,
            )
            # TODO: maybe allow replace_expression_root
            #  to a parameter that can be changed from
            #  self._validate? similar to get_expr?

    except (
            Exception
    ) as e:  # Tried replacing within json but failed.
        # We would not be unable to use pl.all if additional columns
        keep_columns = False
        with_row_index = False

        # we need to set a global variable to be able to
        # determine if we can filter the data once we set up the ValidationError
        os.environ["_PAGURO_JSON_REPLACEMENT_WORKING"] = str(int(False))

        warnings.warn(
            f"{e}\n"
            f"- Unable to process pl.all() [{expr}]. "
            f"If you need to filter the data please use other ways to specify the expressions"
            f"Setting keep_columns=False",
            stacklevel=2,
        )

    return keep_columns, with_row_index, expr


# ----------------------------------------------------------------------


def _dispatch_expr_args_for_errors(
        *,
        column_name: str,
        value: Any,
        attr: str,
        keep_columns: IntoKeepColumns,
        with_row_index: bool | str,
        _struct_fields: tuple[str, ...] | None,
        get_expr: Callable,
) -> tuple[IntoKeepColumns, bool | str, pl.Expr, pl.Expr]:
    if isinstance(value, pl.Expr):
        keep_columns, with_row_index, expr = replace_expr(
            expr=value,
            column_name=column_name,
            keep_columns=keep_columns,
            with_row_index=with_row_index,
        )
        if _struct_fields:
            predicate = replace_predicate(
                expr=value,
                struct_fields=_struct_fields,
            )
        else:
            predicate = expr
    else:
        expr = get_expr(attr, value, column_name)
        if _struct_fields:
            predicate = get_expr(
                attr,
                value,
                get_struct_expr(struct_fields=_struct_fields)
            )
        else:
            predicate = expr

    return keep_columns, with_row_index, expr, predicate


def _dispatch_expr_args_for_predicates(
        *,
        column_name: str,
        value: Any,
        attr: str,
        _struct_fields: tuple[str, ...] | None,
        get_expr: Callable,
) -> pl.Expr:
    if isinstance(value, pl.Expr):
        _, _, expr = replace_expr(
            expr=value,
            column_name=column_name,
            keep_columns=False,
            with_row_index=False,
        )
        if _struct_fields:
            # replace_predicate will raise if we are unable to replace root
            predicate: pl.Expr = replace_predicate(
                expr=value,
                struct_fields=_struct_fields,
            )
        else:
            predicate = expr
    else:
        expr = get_expr(attr, value, column_name)
        if _struct_fields:
            predicate = get_expr(
                attr,
                value,
                get_struct_expr(struct_fields=_struct_fields)
            )
        else:
            predicate = expr

    return predicate
