from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polars import DataType
    from polars.datatypes import DataTypeClass


def dtype_errors(
        column_dtype: DataType | DataTypeClass | None,
        valid_dtypes: frozenset[DataType | DataTypeClass],
) -> dict[str, str | DataType | DataTypeClass | list[DataType | DataTypeClass]] | None:
    # TODO: if DataType passed just compare the two for equality before
    if column_dtype is None:
        return None

    match_found = False

    for vdt in valid_dtypes:
        # pl.List(pl.String) == pl.List(pl.String) # -> True
        # pl.List(pl.String) == pl.List # -> True
        # pl.List(pl.String) == pl.List(pl.Int64) # -> False

        match_found = column_dtype == vdt

        if match_found:
            break

    if not match_found:
        return {
            "expected": "\n".join(str(i) for i in valid_dtypes),
            "errors": column_dtype,
        }
    return None
