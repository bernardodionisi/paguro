from __future__ import annotations

from typing import Protocol

from typing import TYPE_CHECKING

import polars as pl

from paguro.defer import LazyFrameExpr
from paguro.validation.valid_frame.valid_frame import ValidFrame

if TYPE_CHECKING:
    import typing

    from polars import Expr

    from paguro.typing import (
        IntoValidators,
    )

__all__ = [
    "VFrame"
]


class ValidFrameFactory(Protocol):
    def __call__(
            self,
            *validators: IntoValidators | typing.Collection[IntoValidators],
            name: str | None = None,
            transform: LazyFrameExpr | pl.Expr | None = None,
            unique: str | typing.Collection[str] | None = None,
            **constraints: pl.Expr,
    ) -> ValidFrame: ...


class VFrame:
    """
    ValidFrame constructor
    """

    def __call__(
            self,
            *validators: IntoValidators | typing.Collection[IntoValidators],
            transform: LazyFrameExpr | pl.Expr | None = None,
            name: str | None = None,
            unique: str | typing.Collection[str] | None = None,
            **constraints: pl.Expr,
    ) -> ValidFrame:
        return ValidFrame(
            *validators,
            transform=transform,
            name=name,
            unique=unique,
            **constraints,
        )

    @classmethod
    def _(
            cls,
            *validators: IntoValidators | typing.Collection[IntoValidators],
            name: str | None = None,
            transform: LazyFrameExpr | pl.Expr | None = None,
            unique: str | typing.Collection[str] | None = None,
            constraints: dict[str, pl.Expr] | None = None,
    ) -> ValidFrame:
        return ValidFrame._(
            *validators,
            name=name,
            transform=transform,
            unique=unique,
            constraints=constraints,
        )

    def __getattr__(self, attr: str) -> ValidFrameFactory:
        if attr.startswith("__"):
            raise AttributeError(attr)

        def _valid_frame_factory(
                *validators: IntoValidators | typing.Collection[IntoValidators],
                name: str | None = None,
                transform: LazyFrameExpr | pl.Expr | None = None,
                unique: str | typing.Collection[str] | None = None,
                **constraints: pl.Expr,
        ) -> ValidFrame:
            return ValidFrame(
                *validators,
                name=name,
                transform=transform,
                unique=unique,
                **constraints,
            )

        return _valid_frame_factory  # type: ignore
