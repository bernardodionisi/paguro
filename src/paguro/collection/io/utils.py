from __future__ import annotations

from typing import Any

import polars as pl

from paguro.utils.dependencies import pathlib


def _scan_parquet(
    path: pathlib.Path | str,
    *,
    all_independent: bool = True,
    **kwargs: Any,
) -> dict[str, pl.LazyFrame]:
    if isinstance(path, str):
        path = pathlib.Path(path)
    out = {}

    if all_independent:
        for d in {p for p in path.rglob("*.parquet")}:
            lf = pl.scan_parquet(d, **kwargs)
            out[str(d.relative_to(path).with_suffix(""))] = lf
    else:
        # iterate over all leaf directories that contain parquet files
        for d in {
            p.parent for p in path.rglob("*.parquet")
        }:  # unique set of directories
            # create a LazyFrame that scans all parquet
            # files under this directory (nested)
            lf = pl.scan_parquet(
                d / "*.parquet",
                hive_partitioning=True,  # include hive parts if
                # directory names are like key=value
                glob=True,
                **kwargs,
            )
            out[d.name] = lf

    return out
