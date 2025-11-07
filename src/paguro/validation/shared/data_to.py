from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from paguro.collection.collection import Collection
    from paguro.collection.lazycollection import LazyCollection


def _data_to_dict_lazyframes(
        data: dict | Collection | LazyCollection,
) -> dict[str, pl.LazyFrame]:
    from paguro.dataset.dataset import Dataset
    from paguro.dataset.lazydataset import LazyDataset

    out = {}
    for k, v in data.items():
        if isinstance(v, (pl.DataFrame, pl.LazyFrame)):
            out[k] = v.lazy()
        elif isinstance(v, (Dataset, LazyDataset)):
            out[k] = v.lazy().to_polars()
        else:
            out[k] = pl.LazyFrame(v)
            # raise TypeError(f"Invalid type {type(v)}")

    return out


def _data_to_collection(data: dict) -> Collection | LazyCollection:
    if all(isinstance(v, pl.DataFrame) for v in data.values()):
        from paguro.collection.collection import Collection
        return Collection(data)
    elif all(isinstance(v, pl.LazyFrame) for v in data.values()):
        from paguro.collection.lazycollection import LazyCollection
        return LazyCollection(data)
    else:
        msg = f"Unsupported data types in values: {data}"
        raise TypeError(msg)
