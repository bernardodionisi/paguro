from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import polars as pl

from paguro.utils.dependencies import inspect

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence




def required_params(func: Callable) -> list[str]:
    """
    Return the names of parameters that must still be provided to call `func`.

    Works for plain functions, bound methods, functools.partial, and callable objects.
    """
    sig = inspect.signature(func, follow_wrapped=True)
    out: list[str] = []
    for name, p in sig.parameters.items():
        # *args/**kwargs are never strictly "required" by name
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        # Required if it has no default
        if p.default is inspect._empty:
            out.append(name)
    return out


# -------


def _on_to_expr(on: str | Sequence[str]) -> pl.Expr:
    if isinstance(on, str):
        return pl.col(on)
    return pl.struct(on)


def _duplicated(on: str | Sequence[str]) -> pl.Expr:
    # duplicates on single or composite key
    return _on_to_expr(on).is_duplicated()


def _any_null(on: str | Sequence[str]) -> pl.Expr:
    # any null among key columns (single or composite)
    if isinstance(on, str):
        return pl.col(on).is_null()
    return pl.any_horizontal([pl.col(c).is_null() for c in on])


def _tag(frame: pl.LazyFrame, flag: str | None) -> pl.LazyFrame:
    if flag:
        return frame.with_columns(__flag__=pl.lit(flag))
    return frame


def _one(
        data: pl.LazyFrame, on: str | Sequence[str], flag: str | None
) -> pl.LazyFrame:
    # rows that violate "unique" (duplicates)
    return _tag(data.filter(_duplicated(on)), flag)


def _pk(
        data: pl.LazyFrame, on: str | Sequence[str], flag: str | None
) -> pl.LazyFrame:
    # rows that violate "primary key" (duplicates OR null in key)
    return _tag(data.filter(_duplicated(on) | _any_null(on)), flag)


def _fk(
        left: pl.LazyFrame,
        right: pl.LazyFrame,
        left_on: str | Sequence[str],
        right_on: str | Sequence[str],
        *,
        flag: str | None,
) -> pl.LazyFrame:
    # FK semantics: NULL on the *left* is not a violation (no constraint)
    left_non_null = left.filter(~_any_null(left_on))
    miss = left_non_null.join(
        right,
        left_on=left_on,
        right_on=right_on,
        how="anti",
        nulls_equal=True,  # allow matching NULLs (for completeness on right)
    )
    return _tag(miss, flag)


def _concat(items: Iterable[pl.LazyFrame]) -> pl.LazyFrame:
    items = [it for it in items if it is not None]
    if not items:
        return pl.LazyFrame()

    elif len(items) == 1:
        return items[0]

    return pl.concat(items, how="diagonal_relaxed")
