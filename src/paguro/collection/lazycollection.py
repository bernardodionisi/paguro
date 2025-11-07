from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

from paguro.collection._collection import _Collection
from paguro.dataset.lazydataset import LazyDataset
from paguro.shared._getattr._paguro._lazydataset_mixin import _LazyDatasetMixin

from paguro.shared.functions import collect_all

if TYPE_CHECKING:
    from collections.abc import Mapping

    import polars as pl
    from polars._typing import FrameInitTypes
    from paguro.dataset.dataset import Dataset

    from paguro.collection.collection import Collection


class LazyCollection(_Collection[LazyDataset], _LazyDatasetMixin):  # type: ignore[misc]
    """A collection of LazyDatasets."""

    def __init__(
            self,
            data: dict[str, pl.LazyFrame | LazyDataset | FrameInitTypes],
            name: str | None = None,
    ) -> None:
        ldict = preprocess_data(data)
        super().__init__(ldict, name)

    def __getitem__(self, key: str) -> LazyDataset:
        return self._data[key]

    def collect(
            self,
            **kwargs: Any,
    ) -> Collection:
        return self._collect(sequentially=True, **kwargs)

    def _collect(
            self,
            sequentially: bool | Literal["skip", "skip-silent"] = False,
            **kwargs: Any
    ) -> Collection:

        if self._scope is None:
            in_data: dict[str, LazyDataset] = self._data
        else:
            in_data = {k: self._data[k] for k in self._scope}

        if not sequentially:
            datasets: list[Dataset] = collect_all(in_data.values(), **kwargs)
            out_data = dict(zip(in_data.keys(), datasets))

        elif sequentially:
            datasets = []
            keys = []
            for k, lds in in_data.items():
                try:
                    datasets.append(lds.collect(**kwargs))
                    keys.append(k)
                except Exception as e:
                    if isinstance(sequentially, str):
                        if sequentially.startswith("skip"):
                            if not sequentially.endswith("silent"):
                                warnings.warn(
                                    f"Skipping {k!r}: {e}",
                                    stacklevel=2,
                                )
                            continue
                    raise e
            if datasets:
                out_data = dict(zip(keys, datasets))
            else:
                msg = "No datasets collected, all were skipped."
                raise ValueError(msg)
        else:
            msg = f"Unknown collect_mode: {sequentially}"
            raise NotImplementedError(msg)

        from paguro.collection.collection import Collection
        return Collection._from_object(base_object=self, data=out_data)  # type: ignore



def preprocess_data(data: Mapping) -> dict[str, LazyDataset]:
    out: dict[str, LazyDataset] = {}
    for k, v in data.items():
        if not isinstance(v, LazyDataset):
            out[k] = LazyDataset(v)
        else:
            out[k] = v
    return out
