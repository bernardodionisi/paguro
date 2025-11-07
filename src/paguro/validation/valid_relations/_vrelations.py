from __future__ import annotations

from typing import Protocol

from typing import TYPE_CHECKING

import polars as pl

from paguro.validation.valid_relations.valid_relations import ValidRelations

if TYPE_CHECKING:
    import typing
    import sys

    from paguro.validation.valid_frame.valid_frame import ValidFrame
    from paguro.validation.valid_relations.valid_relations import IntoRelations

    if sys.version_info >= (3, 11):
        from typing import Unpack
    else:
        from typing_extensions import Unpack

    CustomConstraint = typing.Callable[
        [Unpack[tuple[pl.LazyFrame, ...]]], pl.LazyFrame
    ]

__all__ = [
    "VRelations"
]


class ValidRelationsFactory(Protocol):
    def __call__(
            self,
            *validators: ValidFrame | typing.Iterable[ValidFrame],
            relations: IntoRelations | None = None,
            **constraints: CustomConstraint,
    ) -> ValidRelations: ...


class VRelations:
    """
    ValidFrame constructor
    """

    def __call__(
            self,
            *validators: ValidFrame | typing.Iterable[ValidFrame],
            relations: IntoRelations | None = None,
            **constraints: CustomConstraint,
    ) -> ValidRelations:
        return ValidRelations(
            *validators,
            relations=relations,
            **constraints
        )

    @classmethod
    def _(
            cls,
            *validators: ValidFrame | typing.Iterable[ValidFrame],
            relations: IntoRelations | None = None,
            constraints: dict[str, CustomConstraint] | None = None,
    ) -> ValidRelations:
        return ValidRelations._(
            *validators,
            relations=relations,
            constraints=constraints,
        )

    def __getattr__(self, attr: str) -> ValidRelationsFactory:
        if attr.startswith("__"):
            raise AttributeError(attr)

        def _valid_relations_factory(
                *validators: ValidFrame | typing.Iterable[ValidFrame],
                relations: IntoRelations | None = None,
                **constraints: CustomConstraint,
        ) -> ValidRelations:
            return ValidRelations(
                *validators,
                relations=relations,
                **constraints
            )

        return _valid_relations_factory  # type: ignore
