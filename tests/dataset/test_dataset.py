from __future__ import annotations

import polars as pl
import paguro as pg

from paguro.ashi.info.info_collection import InfoCollection


def test_dataset_info_basic():
    ds = pg.Dataset({"a": [1, 2, 3]})
    assert isinstance(ds.with_info("outer-name", a="column a")._info, InfoCollection)
    assert ds._info is None

