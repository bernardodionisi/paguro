from __future__ import annotations

import pytest
import paguro as pg
import polars as pl
from pathlib import Path
from typing import Callable, TypeAlias, Union

DatasetLike: TypeAlias = Union[pg.Dataset, pg.LazyDataset]


def _assert_same_validation_and_info(original: DatasetLike, new: DatasetLike, ) -> None:
    assert (
            original.validation._fingerprint(include_info=True)
            == new.validation._fingerprint(include_info=True)
    )
    assert original._info == new._info


@pytest.fixture(params=[False, True], ids=["eager", "lazy"])
def lazy(request) -> bool:
    return request.param


@pytest.fixture
def example_dataset(lazy: bool) -> DatasetLike:
    data = {"a": [1, 2], "b": ["x", "y"]}
    ds: DatasetLike = pg.LazyDataset(data) if lazy else pg.Dataset(data)
    ds = (
        ds
        .with_info("desc", a="1")
        .with_validation(
            pg.vcol("a", ge=1).with_info(title="validator for a"),
            pg.vframe(
                pg.vcol("a", ge=1).with_info(title="validator for a"),
                pl.col("a") != pl.col("b"),
            ).with_info(title="validator for frame"),
        )
    )
    return ds


@pytest.fixture
def parquet_writer() -> Callable[[DatasetLike, Path], None]:
    """
    Write a dataset with metadata:
      - pg.Dataset   -> write_parquet
      - pg.LazyDataset -> sink_parquet
    """

    def write(ds: DatasetLike, path: Path) -> None:
        if isinstance(ds, pg.LazyDataset):
            ds.sink_parquet(path, write_paguro_metadata=True)
        else:
            ds.write_parquet(file=path, write_paguro_metadata=True)

    return write


@pytest.fixture(params=["scan_parquet", "read_parquet"], ids=["scan", "read"])
def parquet_reader(request) -> Callable[[Path], DatasetLike]:
    if request.param == "scan_parquet":
        def scan(path: Path) -> DatasetLike:
            return pg.scan_parquet(path, paguro_metadata=True)

        return scan
    else:
        def read(path: Path) -> DatasetLike:
            return pg.read_parquet(path, paguro_metadata=True)

        return read


# -------------------------- tests -------------------------------------

def test_parquet_roundtrip(
        tmp_path: Path,
        example_dataset: DatasetLike,
        parquet_writer: Callable[[DatasetLike, Path], None],
        parquet_reader: Callable[[Path], DatasetLike],
) -> None:
    path = tmp_path / "dataset-io-test.parquet"

    parquet_writer(example_dataset, path)
    assert path.exists()

    loaded = parquet_reader(path)
    _assert_same_validation_and_info(example_dataset, loaded)
