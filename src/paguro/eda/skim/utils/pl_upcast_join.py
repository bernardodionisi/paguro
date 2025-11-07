from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import polars as pl

# from polars._typing import JoinStrategy

if TYPE_CHECKING:
    from polars._typing import JoinStrategy

T = TypeVar("T", pl.DataFrame, pl.LazyFrame)


def join_frames_with_upcast(
    frames: list[T], on: str | list[str], how: JoinStrategy
) -> T:
    if len(frames) == 1:
        return frames[0]

    frames = upcast_integers_for_joins(frames=frames, on=on)

    data: T = frames[0]
    for f in frames[1:]:
        data = data.join(f, on=on, how=how)

    return data


def upcast_integers_for_joins(
    frames: list[T], on: str | list[str]
) -> list[T]:
    if isinstance(on, str):
        on = [on]

    for i in on:
        dt = find_upcast_integer_type(
            *(d.collect_schema()[i] for d in frames)
        )

        if dt is None:  # if the column is not an integer then leave as is
            continue

        for idx, df in enumerate(frames):
            # upcast the integer dtype for each frame
            frames[idx] = df.with_columns(pl.col(i).cast(dt))

    return frames


def find_upcast_integer_type(*data_types):
    # Define the hierarchy of data types
    hierarchy = {
        pl.UInt8: 1,
        pl.Int8: 2,
        pl.UInt16: 3,
        pl.Int16: 4,
        pl.UInt32: 5,
        pl.Int32: 6,
        pl.UInt64: 7,
        pl.Int64: 8,
    }

    if not all(dt in hierarchy for dt in data_types):
        return pl.String

    max_type_value = max(hierarchy[dt] for dt in data_types)

    # Find and return the corresponding data type
    for dtype, value in hierarchy.items():
        if value == max_type_value:
            return dtype


def find_upcast_type(*data_types):
    # Define the hierarchy of data types
    hierarchy = {
        pl.Boolean: 1,
        pl.UInt8: 2,
        pl.Int8: 3,
        pl.UInt16: 4,
        pl.Int16: 5,
        pl.UInt32: 6,
        pl.Int32: 7,
        pl.UInt64: 8,
        pl.Int64: 9,
        pl.Float32: 10,
        pl.Float64: 11,
        # pl.Date: 12,
        # pl.Datetime: 13,
        pl.String: 12,
    }

    if not all(dt in hierarchy for dt in data_types):
        return None

    max_type_value = max(hierarchy[dt] for dt in data_types)

    # Find and return the corresponding data type
    for dtype, value in hierarchy.items():
        if value == max_type_value:
            return dtype
