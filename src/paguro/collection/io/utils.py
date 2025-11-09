from __future__ import annotations

from typing import Any, TYPE_CHECKING, Mapping

from paguro.dataset.io.read_source import scan_parquet
from paguro.utils.dependencies import pathlib

if TYPE_CHECKING:
    from paguro.dataset.lazydataset import LazyDataset


def _scan_parquet(
        path: pathlib.Path | str,
        *,
        all_independent: bool = True,
        **kwargs: Any,
) -> dict[str, LazyDataset]:
    if isinstance(path, str):
        path = pathlib.Path(path)
    out = {}

    if all_independent:
        for d in {p for p in path.rglob("*.parquet")}:
            lf = scan_parquet(d, **kwargs)
            out[str(d.relative_to(path).with_suffix(""))] = lf
    else:
        # iterate over all leaf directories that contain parquet files
        for d in {
            p.parent for p in path.rglob("*.parquet")
        }:  # unique set of directories
            # create a LazyFrame that scans all parquet
            # files under this directory (nested)
            lf = scan_parquet(
                d / "*.parquet",
                hive_partitioning=True,  # include hive parts if
                # directory names are like key=value
                glob=True,
                **kwargs,
            )
            out[d.name] = lf

    return out
