from __future__ import annotations

from typing import TYPE_CHECKING, Any

from paguro.collection.io.utils import _scan_parquet
from paguro.collection.lazycollection import LazyCollection

if TYPE_CHECKING:
    from pathlib import Path


def scan_parquet(
    path: Path | str,
    *,
    all_independent: bool = True,
    **kwargs: Any,
) -> LazyCollection:
    # TODO: scan as dataset to include validation?
    data = _scan_parquet(
        path=path, all_independent=all_independent, **kwargs
    )
    return LazyCollection(data=data)
