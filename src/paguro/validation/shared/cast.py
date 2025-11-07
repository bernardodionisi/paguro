from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, overload

if TYPE_CHECKING:
    from polars import DataFrame, LazyFrame, Schema

    from paguro import Dataset, LazyDataset


@overload
def cast_frame(frame: LazyFrame, schema: Schema) -> LazyFrame: ...


@overload
def cast_frame(frame: DataFrame, schema: Schema) -> DataFrame: ...


@overload
def cast_frame(frame: Dataset, schema: Schema) -> Dataset: ...


@overload
def cast_frame(frame: LazyDataset, schema: Schema) -> LazyDataset: ...


def cast_frame(
    frame: LazyFrame | DataFrame | Dataset | LazyDataset,
    schema: Schema,
) -> LazyFrame | DataFrame | Dataset | LazyDataset:
    # We keep schema typed as Schema publicly; tell mypy the call is fine.
    return frame.cast(cast("Any", schema))
