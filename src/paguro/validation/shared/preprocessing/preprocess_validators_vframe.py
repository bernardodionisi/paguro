from __future__ import annotations

from typing import Iterable, TYPE_CHECKING, Collection

import polars as pl

from paguro.validation.shared.preprocessing.utils import _parse_validators_as_iterable

if TYPE_CHECKING:
    from paguro.typing import IntoValidators
    from paguro.validation.validation import Validation
    from paguro.validation.valid_column.valid_column import ValidColumn
    from paguro.validation.valid_frame.valid_frame import ValidFrame


def split_validators_into_validation_and_constraints(
        validators: tuple[IntoValidators | Collection[IntoValidators], ...],
        constraints: dict[str, pl.Expr],
) -> tuple[Validation | None, dict[str, pl.Expr]]:
    iter_validators: Iterable[IntoValidators] = (
        _parse_validators_as_iterable(
            inputs=validators,
        )
    )

    list_validators, constraints = _assign_expressions_to_constraints(
        validators=iter_validators,
        constraints=constraints,
    )

    if list_validators:
        from paguro.validation.validation import Validation
        return Validation(*list_validators), constraints
    return None, constraints


def _assign_expressions_to_constraints(
        validators: Iterable[IntoValidators],
        constraints: dict[str, pl.Expr],
) -> tuple[
    list[ValidColumn | ValidFrame | Validation],
    dict[str, pl.Expr]
]:
    validators_expr: dict[str, pl.Expr] = {}
    validators_no_expr: list[ValidColumn | ValidFrame | Validation] = []

    counter = 0
    for v in validators:
        if isinstance(v, pl.Expr):
            validators_expr[f"constraint_{counter}"] = v
            # use counter (instead of enumerating) otherwise we count also non-pl.Expr
            counter += 1
        else:
            validators_no_expr.append(v)

    duplicate_constraint_names = set(validators_expr.keys()) & set(
        constraints.keys()
    )
    if duplicate_constraint_names:
        msg = f"Duplicate constraint names: {duplicate_constraint_names}"
        raise ValueError(msg)

    validators_expr.update(constraints)

    return validators_no_expr, validators_expr
