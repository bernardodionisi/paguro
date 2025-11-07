from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal, TypeAlias, cast

import polars as pl
from polars._typing import PythonDataType
from polars.datatypes.classes import (
    Array,
    Categorical,
    DataType,
    DataTypeClass,
    Enum,
    String,
    Struct,
)
from polars.datatypes.group import (
    DATETIME_DTYPES,
    DURATION_DTYPES,
    FLOAT_DTYPES,
    NESTED_DTYPES,
    NUMERIC_DTYPES,
    SIGNED_INTEGER_DTYPES,
    TEMPORAL_DTYPES,
    UNSIGNED_INTEGER_DTYPES,
)

IntoDataType: TypeAlias = (
        DataType
        | DataTypeClass
        | PythonDataType
        | Iterable[DataType | DataTypeClass | str]
        | dict
        | Literal[
            "numeric",
            "uint",
            "nested",
            "array",
            "categorical",
            "temporal",
            "datetime",
            "duration",
        ]
        | None
)


def parse_dtype_into_frozenset(
        dtype: IntoDataType,
) -> frozenset[DataTypeClass | DataType] | None:
    # --- instances / trivial cases
    if dtype is None:
        return None

    elif isinstance(dtype, (list, set, tuple)):
        if all(
                isinstance(i, str) for i in dtype
        ):  # a tuple/list of strings -> Enum
            return frozenset([Enum(dtype)])  # runtime unchanged
        elif all(isinstance(i, dict) for i in dtype):
            # keep existing behavior; _to_dtype handles list specs (may raise if len!=1)
            return frozenset([_to_dtype(dtype)])  # type: ignore[arg-type]
        else:
            # accumulate nested unions safely (handle None)
            out: set[DataType | DataTypeClass] = set()
            for i in dtype:
                sub = parse_dtype_into_frozenset(i)  # type: ignore[arg-type]
                if sub is not None:
                    out |= sub
            return frozenset(out)

    elif isinstance(dtype, (DataType, DataTypeClass)):
        return frozenset([dtype])  # type: ignore[list-item]

    # --- strings/pydtypes (class objects)
    elif dtype is str:
        return frozenset([String])

    elif dtype is int:
        return frozenset(SIGNED_INTEGER_DTYPES)

    elif dtype is float:
        return frozenset(FLOAT_DTYPES)

    elif dtype == "uint":
        return frozenset(UNSIGNED_INTEGER_DTYPES)

    elif dtype == "numeric":
        return frozenset(NUMERIC_DTYPES)

    # --- nested / structural
    elif isinstance(dtype, dict):
        return frozenset([_nested_struct(dtype)])

    elif dtype is dict:
        return frozenset([Struct])

    elif dtype == "nested":
        return frozenset(NESTED_DTYPES)

    elif dtype == "array":
        return frozenset([Array])

    elif dtype == "categorical":
        return frozenset([Categorical])

    # --- temporal
    elif dtype == "datetime":
        return frozenset(DATETIME_DTYPES)

    elif dtype == "duration":
        return frozenset(DURATION_DTYPES)

    elif dtype == "temporal":
        return frozenset(TEMPORAL_DTYPES)

    # --- fallback to Polars parser
    try:
        parsed = pl.datatypes.parse_into_dtype(
            dtype
        )  # runtime stays the same
        # Help mypy: Polars’ stub may widen this; cast it back to dtype-like.
        return frozenset([cast("DataType | DataTypeClass", parsed)])
    except TypeError:
        # Re-raise unchanged to preserve runtime behavior
        raise


def _to_dtype(spec: Any) -> pl.DataType | DataTypeClass:
    # Already a Polars dtype
    if isinstance(spec, pl.DataType):
        return spec

    # Structs from nested mappings
    if isinstance(spec, Mapping):
        return pl.Struct({k: _to_dtype(v) for k, v in spec.items()})

    # Variable-length lists: require exactly one element to define inner dtype
    if isinstance(spec, list):
        if len(spec) != 1:
            msg = "List specs must contain exactly one element, e.g. [int] or [{'a': int}]"
            raise ValueError(msg)
        inner = _to_dtype(spec[0])
        return pl.List(inner)

    # Fixed-size arrays: require a 2-tuple (inner_spec, size:int)
    if isinstance(spec, tuple):
        if len(spec) == 2 and isinstance(spec[1], int):
            inner = _to_dtype(spec[0])
            size = spec[1]
            if size < 0:
                msg = "Array size must be >= 0"
                raise ValueError(msg)
            return pl.Array(inner, size)
        msg = "Tuple specs must be (inner_spec, size:int), e.g. (int, 3) or ((float, 2), 4)"
        raise ValueError(msg)

    # Fallback: let Polars parse builtins / numpy dtypes, etc.
    return pl.datatypes.parse_into_dtype(spec)


def _nested_struct(spec: Mapping) -> pl.DataType:
    if not isinstance(spec, Mapping):
        msg = "Top-level spec must be a mapping (dict-like)"
        raise TypeError(msg)
    return pl.Struct({k: _to_dtype(v) for k, v in spec.items()})
