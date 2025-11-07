from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from polars.datatypes import DataTypeClass


# make sure one can pass DataTypeClass, or not
def find_supertype_multiple(
    data_types: frozenset[pl.DataType | DataTypeClass] | None,
) -> pl.DataType | DataTypeClass:
    """Find the supertype for multiple data types."""
    # data_types = _parse_inputs_as_iterable(data_types)
    if not data_types:
        msg = "dtype is None. Please provide at least one data type to determine supertype."
        raise ValueError(msg)

    dtypes: tuple[pl.DataType | DataTypeClass, ...] = tuple(data_types)

    if len(data_types) == 1:
        return dtypes[0]

    result: pl.DataType | DataTypeClass = dtypes[0]

    if len(dtypes) == 1:
        return result
    else:
        if isinstance(result, pl.DataType):
            msg = (
                f"Unable to determine supertype for: {result} in {dtypes}"
            )
            raise ValueError(msg)
        else:
            for dt in dtypes[1:]:
                if isinstance(dt, pl.DataType):
                    msg = f"Unable to determine supertype for: {dt} in {dtypes}"
                    raise TypeError(msg)

                temp: DataTypeClass | None = _get_supertype(result, dt)
                if temp is None:
                    # we tried to determine but were not able:
                    # [pl.Struct, pl.Enum] no supertype
                    msg = f"Unable to determine supertype for: {dtypes}"
                    raise ValueError(msg)
                result = temp
        return result


# ----------------------------------------------------------------------

UNSIGNED_INTEGERS: list[DataTypeClass] = [
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.Int128,
]

SIGNED_INTEGERS: list[DataTypeClass] = [
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
]

FLOATS: list[DataTypeClass] = [pl.Float32, pl.Float64]

INTEGER_LIKES: list[DataTypeClass] = (
    [pl.Boolean] + UNSIGNED_INTEGERS + SIGNED_INTEGERS
)

# Mixed signed/unsigned integer promotions
MIXED_PROMOTIONS: dict[
    tuple[DataTypeClass, DataTypeClass], DataTypeClass
] = {
    (pl.Int8, pl.UInt8): pl.Int16,
    (pl.Int8, pl.UInt16): pl.Int32,
    (pl.Int8, pl.UInt32): pl.Int64,
    (pl.Int8, pl.UInt64): pl.Float64,  # Follow numpy
    (pl.Int16, pl.UInt8): pl.Int16,
    (pl.Int16, pl.UInt16): pl.Int32,
    (pl.Int16, pl.UInt32): pl.Int64,
    (pl.Int16, pl.UInt64): pl.Float64,  # Follow numpy
    (pl.Int32, pl.UInt8): pl.Int32,
    (pl.Int32, pl.UInt16): pl.Int32,
    (pl.Int32, pl.UInt32): pl.Int64,
    (pl.Int32, pl.UInt64): pl.Float64,  # Follow numpy
    (pl.Int64, pl.UInt8): pl.Int64,
    (pl.Int64, pl.UInt16): pl.Int64,
    (pl.Int64, pl.UInt32): pl.Int64,
    (pl.Int64, pl.UInt64): pl.Float64,  # Follow numpy
}


def _get_supertype(
    left: DataTypeClass,  # | pl.DataType
    right: DataTypeClass,  # | pl.DataType
) -> DataTypeClass | None:
    """
    Determine the supertype that both data types can safely be cast to.

    Based on Polars' internal supertype logic.

    A simple approach for https://docs.pola.rs/api/rust/dev/src/polars_core/utils/supertype.rs.html
    """
    if left == right:
        return left

    # Handle null types
    if left == pl.Null:
        return right
    if right == pl.Null:
        return left

    # Boolean promotions
    if left == pl.Boolean:
        if right in UNSIGNED_INTEGERS:
            return right
        elif right in SIGNED_INTEGERS:
            return right
        elif right in FLOATS:
            return right
    if right == pl.Boolean:
        if left in UNSIGNED_INTEGERS:
            return left
        elif left in SIGNED_INTEGERS:
            return left
        elif left in FLOATS:
            return left

    # Integer promotions (signed)
    if left in UNSIGNED_INTEGERS and right in UNSIGNED_INTEGERS:
        left_idx: int = (
            UNSIGNED_INTEGERS.index(left)
            if left in UNSIGNED_INTEGERS
            else -1
        )
        right_idx: int = (
            UNSIGNED_INTEGERS.index(right)
            if right in UNSIGNED_INTEGERS
            else -1
        )
        return UNSIGNED_INTEGERS[max(left_idx, right_idx)]

    # Integer promotions (unsigned)
    if left in SIGNED_INTEGERS and right in SIGNED_INTEGERS:
        left_idx = (
            SIGNED_INTEGERS.index(left) if left in SIGNED_INTEGERS else -1
        )
        right_idx = (
            SIGNED_INTEGERS.index(right)
            if right in SIGNED_INTEGERS
            else -1
        )
        return SIGNED_INTEGERS[max(left_idx, right_idx)]

    if (left, right) in MIXED_PROMOTIONS:
        return MIXED_PROMOTIONS[(left, right)]
    if (right, left) in MIXED_PROMOTIONS:
        return MIXED_PROMOTIONS[(right, left)]

    # Float promotions
    if left in [pl.Float32, pl.Float64] and right in [
        pl.Float32,
        pl.Float64,
    ]:
        return pl.Float64 if pl.Float64 in [left, right] else pl.Float32

    # Integer to float promotions
    if left in FLOATS and right in INTEGER_LIKES:
        if left == pl.Float64 or right in [
            pl.Int32,
            pl.UInt32,
            pl.Int64,
            pl.UInt64,
        ]:
            return pl.Float64
        return pl.Float32

    if right in FLOATS and left in INTEGER_LIKES:
        if right == pl.Float64 or left in [
            pl.Int32,
            pl.UInt32,
            pl.Int64,
            pl.UInt64,
        ]:
            return pl.Float64
        return pl.Float32

    # String promotions - most types can be cast to string
    if left == pl.String and right != pl.Binary:
        return pl.String
    if right == pl.String and left != pl.Binary:
        return pl.String

    # Date/time promotions (simplified)
    if left == pl.Date and right == pl.Datetime:
        return right
    if right == pl.Date and left == pl.Datetime:
        return left

    # No supertype found
    return None
