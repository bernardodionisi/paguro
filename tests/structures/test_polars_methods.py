# The tests in these module are based on various examples from the Polars
# documentation page. Polars is distributed with the following license.
#
# '''
# Copyright (c) 2025 Ritchie Vink
# Some portions Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# '''
from __future__ import annotations

from datetime import date
from datetime import datetime

import polars as pl
import paguro as pg


def test_concat():
    paguro_object = pg.Dataset
    ds1 = paguro_object({"a": [1], "b": [3]})
    ds2 = paguro_object({"a": [2], "b": [4]})
    out = pg.concat([ds1, ds2])
    assert isinstance(
        out,
        paguro_object,
    ), f"Expected {type(paguro_object)}, got {type(out)}"

    ds1 = paguro_object({"a": [1], "b": [3]})
    ds2 = paguro_object({"a": [2.5], "b": [4]})
    out = pg.concat([ds1, ds2], how="vertical_relaxed")
    assert isinstance(
        out,
        paguro_object,
    ), f"Expected {type(paguro_object)}, got {type(out)}"

    ds_h1 = paguro_object({"l1": [1, 2], "l2": [3, 4]})
    ds_h2 = paguro_object({"r1": [5, 6], "r2": [7, 8], "r3": [9, 10]})
    out = pg.concat([ds_h1, ds_h2], how="horizontal")
    assert isinstance(
        out,
        paguro_object,
    ), f"Expected {type(paguro_object)}, got {type(out)}"

    ds_a1 = paguro_object({"id": [1, 2], "x": [3, 4]})
    ds_a2 = paguro_object({"id": [2, 3], "y": [5, 6]})
    ds_a3 = paguro_object({"id": [1, 3], "z": [7, 8]})

    for a in ["align", "align_left", "align_right", "align_inner"]:
        out = pg.concat([ds_a1, ds_a2, ds_a3], how=a)
        assert isinstance(
            out,
            paguro_object,
        ), f"Expected {type(paguro_object)}, got {type(out)}"


def test_join():
    for paguro_object in [pg.Dataset, pg.LazyDataset]:
        ds = paguro_object(
            {
                "foo": [1, 2, 3],
                "bar": [6.0, 7.0, 8.0],
                "ham": ["a", "b", "c"],
            }
        )
        other_ds = paguro_object(
            {
                "apple": ["x", "y", "z"],
                "ham": ["a", "b", "d"],
            }
        )
        out = ds.join(other_ds, on="ham")

        assert isinstance(
            out,
            paguro_object,
        ), f"Expected {type(paguro_object)}, got {type(out)}"


def test_join_where():
    for paguro_object in [pg.Dataset, pg.LazyDataset]:
        east = paguro_object(
            {
                "id": [100, 101, 102],
                "dur": [120, 140, 160],
                "rev": [12, 14, 16],
                "cores": [2, 8, 4],
            }
        )
        west = paguro_object(
            {
                "t_id": [404, 498, 676, 742],
                "time": [90, 130, 150, 170],
                "cost": [9, 13, 15, 16],
                "cores": [4, 2, 1, 4],
            }
        )
        out = east.join_where(
            west,
            pl.col("dur") < pl.col("time"),
            pl.col("rev") < pl.col("cost"),
        )

        assert isinstance(
            out,
            paguro_object,
        ), f"Expected {type(paguro_object)}, got {type(out)}"


def test_join_asof():
    for paguro_object in [pg.Dataset, pg.LazyDataset]:
        gdp = paguro_object(
            {
                "date": pl.date_range(
                    date(2016, 1, 1),
                    date(2020, 1, 1),
                    "1y",
                    eager=True,
                ),
                "gdp": [4164, 4411, 4566, 4696, 4827],
            }
        )
        population = paguro_object(
            {
                "date": [
                    date(2016, 3, 1),
                    date(2018, 8, 1),
                    date(2019, 1, 1)
                ],
                "population": [82.19, 82.66, 83.12],
            }
        ).sort("date")
        out = population.join_asof(gdp, on="date", strategy="backward")
        assert isinstance(
            out,
            paguro_object,
        ), f"Expected {type(paguro_object)}, got {type(out)}"


# ----------- group_by

def test_group_by():
    for paguro_object in [pg.Dataset, pg.LazyDataset]:
        ds = paguro_object(
            {
                "a": ["a", "b", "a", "b", "c"],
                "b": [1, 2, 1, 3, 3],
                "c": [5, 4, 3, 2, 1],
            }
        )
        out = ds.group_by("a").agg(pl.col("b").sum())

        assert isinstance(
            out,
            paguro_object,
        ), f"Expected {type(paguro_object)}, got {type(out)}"

        out = ds.group_by("a", maintain_order=True).agg(pl.col("c"))

        assert isinstance(
            out,
            paguro_object,
        ), f"Expected {type(paguro_object)}, got {type(out)}"


def test_group_by_dynamic():
    for paguro_object in [pg.Dataset, pg.LazyDataset]:
        ds = paguro_object(
            {
                "time": pl.datetime_range(
                    start=datetime(2021, 12, 16),
                    end=datetime(2021, 12, 16, 3),
                    interval="30m",
                    eager=True,
                ),
                "n": range(7),
            }
        )

        out = (
            ds
            .group_by_dynamic(
                "time", every="1h", closed="right", )
            .agg(pl.col("n"))
        )

        assert isinstance(
            out,
            paguro_object,
        ), f"Expected {type(paguro_object)}, got {type(out)}"

        out = (
            ds
            .group_by_dynamic(
                "time",
                every="1h",
                include_boundaries=True,
                closed="right",
            )
            .agg(pl.col("n").mean())
        )

        assert isinstance(
            out,
            paguro_object,
        ), f"Expected {type(paguro_object)}, got {type(out)}"

        out = (
            ds
            .group_by_dynamic(
                "time",
                every="1h",
                closed="left",
            )
            .agg(pl.col("n"))
        )

        assert isinstance(
            out,
            paguro_object,
        ), f"Expected {type(paguro_object)}, got {type(out)}"

        out = (
            ds
            .group_by_dynamic(
                "time",
                every="1h",
                closed="both", )
            .agg(pl.col("n"))
        )

        assert isinstance(
            out,
            paguro_object,
        ), f"Expected {type(paguro_object)}, got {type(out)}"

        ds = (
            ds
            .with_columns(
                groups=pl.Series(["a", "a", "a", "b", "b", "a", "a"])
            )
        )
        out = (
            ds
            .group_by_dynamic(
                "time",
                every="1h",
                closed="both",
                group_by="groups",
                include_boundaries=True,
            )
            .agg(pl.col("n"))
        )

        assert isinstance(
            out,
            paguro_object,
        ), f"Expected {type(paguro_object)}, got {type(out)}"

        ds = paguro_object(
            {
                "idx": pl.int_range(0, 6, eager=True),
                "A": ["A", "A", "B", "B", "B", "C"],
            }
        )

        out = (
            ds.group_by_dynamic(
                "idx",
                every="2i",
                period="3i",
                include_boundaries=True,
                closed="right",
            ).agg(pl.col("A").alias("A_agg_list"))
        )

        assert isinstance(
            out,
            paguro_object,
        ), f"Expected {type(paguro_object)}, got {type(out)}"


def test_rolling():
    for paguro_object in [pg.Dataset, pg.LazyDataset]:
        dates = [
            "2020-01-01 13:45:48",
            "2020-01-01 16:42:13",
            "2020-01-01 16:45:09",
            "2020-01-02 18:12:48",
            "2020-01-03 19:45:32",
            "2020-01-08 23:16:43",
        ]
        ds = (
            paguro_object(
                {"dt": dates, "a": [3, 7, 5, 9, 2, 1]}
            )
            .with_columns(
                pl.col("dt").str.strptime(pl.Datetime).set_sorted()
            )
        )

        out = (
            ds
            .rolling(index_column="dt", period="2d")
            .agg(
                pl.sum("a").alias("sum_a"),
                pl.min("a").alias("min_a"),
                pl.max("a").alias("max_a"),
            )
        )

        assert isinstance(
            out,
            paguro_object,
        ), f"Expected {type(paguro_object)}, got {type(out)}"
