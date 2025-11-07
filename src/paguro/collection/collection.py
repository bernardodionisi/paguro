from __future__ import annotations

from typing import TYPE_CHECKING

from paguro.collection._collection import _Collection
from paguro.dataset.dataset import Dataset
from paguro.shared._getattr._paguro._dataset_mixin import _DatasetMixin

if TYPE_CHECKING:
    import polars as pl
    from polars._typing import FrameInitTypes

    from paguro.collection.lazycollection import LazyCollection


class Collection(_Collection[Dataset], _DatasetMixin):
    """
    A collection of Datasets.
    """

    def __init__(
            self,
            data: dict[str, pl.DataFrame | Dataset | FrameInitTypes],
            name: str | None = None,
    ) -> None:
        _data: dict[str, Dataset] = preprocess_data(data)
        super().__init__(_data, name)

    def __getitem__(self, key: str) -> Dataset:
        return self._data[key]

    def lazy(self) -> LazyCollection:
        from paguro.collection.lazycollection import LazyCollection
        if self._scope is not None:
            data = {k: self._data[k].lazy() for k in self._scope}
        else:
            data = {k: v.lazy() for k, v in self._data.items()}
        return LazyCollection._from_object(base_object=self, data=data)  # type: ignore


def preprocess_data(data: dict) -> dict[str, Dataset]:
    out: dict[str, Dataset] = {}
    for k, v in data.items():
        if not isinstance(v, Dataset):
            out[k] = Dataset(v)
        else:
            out[k] = v
    return out
