from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from typing import TypedDict, Any

if TYPE_CHECKING:
    from polars import DataFrame, LazyFrame


# ----------------------------------------------------------------------

class BaseErrors(TypedDict, total=False):
    maybe_errors: pl.LazyFrame
    no_errors: str
    errors_limited: pl.DataFrame
    errors_count: int
    errors: pl.DataFrame | str


class Errors(BaseErrors, total=False):
    predicate: pl.Expr | None


class ConstraintsErrors(Errors, total=False):
    value: Any
    info: Any


# ----------------------------------------------------------------------

class ValidColumnSchemaErrors(TypedDict, total=False):
    dtype: dict[str, Any]
    required: BaseErrors


class ValidColumnDataErrors(TypedDict, total=False):
    allow_nulls: Errors
    unique: Errors
    constraints: dict[str, ConstraintsErrors]


class ValidColumnFieldsErrors(TypedDict, total=False):
    fields: ValidationErrors


class ValidColumnErrors(
    ValidColumnSchemaErrors,
    ValidColumnDataErrors,
    ValidColumnFieldsErrors,
    total=False
):
    ...


# ----------------------------------------------------------------------

class ValidFrameSchemaErrors(TypedDict, total=False):
    columns_policy: dict[str, Any]


class ValidFrameDataErrors(TypedDict, total=False):
    transform: dict[str, Any]
    unique: Errors
    constraints: dict[str, ConstraintsErrors]


class ValidFrameValidatorsErrors(TypedDict, total=False):
    validators: ValidationErrors


class ValidFrameErrors(
    ValidColumnSchemaErrors,
    ValidColumnDataErrors,
    ValidFrameValidatorsErrors,
    total=False
):
    ...


# ----------------------------------------------------------------------

class ValidationErrors(TypedDict, total=False):
    valid_column_list: dict[str, ValidColumnErrors]
    valid_frame_list: dict[str, ValidFrameErrors]
