from __future__ import annotations

from typing import Any

import polars as pl

from paguro.utils.dependencies import inspect

NAMESPACES = {
    "str",
    "list",
    "struct",
    "arr",
    "bin",
    "cat",
    "dt",
}


def _build_expr(
        attr: str,
        value: Any,
        expr: str | None | pl.Expr,
) -> pl.Expr:
    if isinstance(value, pl.Expr):
        return value

    expr = _build_base_expr(expr=expr)
    return _build_expr_from_key_value(attr, value, expr)


def _build_base_expr(expr: str | None | pl.Expr) -> pl.Expr:
    if expr is None:
        expr = pl.all()
    elif isinstance(expr, str):
        expr = pl.col(expr)
    else:
        if not isinstance(expr, pl.Expr):
            msg = f"expr must be an Expr or str, instead got {type(expr)}"
            raise ValueError(msg)
    return expr


def _build_expr_from_key_value(
        attr: str,
        value: Any,
        expr: str | None | pl.Expr,
) -> pl.Expr:
    components = attr.split("_")
    i = 0
    n = len(components)

    while i < n:
        matched_attr = None
        matched_j = None

        # Try all combinations starting from i
        for j in range(i + 1, n + 1):
            candidate = "_".join(components[i:j])
            if hasattr(expr, candidate):
                matched_attr = candidate
                matched_j = j

        if matched_attr is None or matched_j is None:
            current_input = components[i]
            suggestions = [
                attr
                for attr in dir(expr)
                if attr.startswith(current_input[0])
            ]

            message = f"Invalid attribute starting at: '{'_'.join(components[i:])}'"
            if suggestions:
                message += "\nMaybe you meant one of:\n  " + "\n  ".join(
                    sorted(suggestions)
                )

            raise AttributeError(message)

        try:
            attr_obj = getattr(expr, matched_attr)
        except AttributeError as e:
            msg = f"Attribute '{matched_attr}' not found on {type(expr).__name__}"
            raise AttributeError(msg) from e

        is_last = matched_j == n

        if is_last:
            if isinstance(value, dict):
                expr = attr_obj(**value)

            elif isinstance(value, bool):
                if not inspect.signature(attr_obj).parameters:
                    expr = attr_obj() if value else ~attr_obj()
                else:
                    expr = attr_obj(value)
            else:
                expr = attr_obj(value)
        else:
            if matched_attr not in NAMESPACES:
                expr = attr_obj()
            else:
                expr = attr_obj

        i = matched_j

    assert isinstance(expr, pl.Expr)  # should always be Expr
    return expr
