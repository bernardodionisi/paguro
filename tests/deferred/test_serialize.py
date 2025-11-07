from __future__ import annotations

import json

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import paguro as pg
from paguro.shared.serialize import CustomJSONDecoder, CustomJSONEncoder


@pytest.fixture
def single_lf_data() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "id": [1, 2, 3, 4],
            "group": ["a", "a", "b", "b"],
            "value": [10, 5, 2, 8],
        }
    )


@pytest.fixture
def ms_data() -> dict[str, pl.LazyFrame]:
    customers = pl.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "name": ["Alice", "Bob", "Caro"],
        }
    ).lazy()

    orders = pl.DataFrame(
        {
            "order_id": [100, 101, 102, 103],
            "customer_id": [1, 2, 1, 3],
            "amount": [20.0, 35.5, 12.0, 99.0],
        }
    ).lazy()
    return {"customers": customers, "orders": orders}


def _roundtrip(obj):
    """Helper: JSON round-trip using your custom encoder/decoder."""
    return json.loads(json.dumps(obj, cls=CustomJSONEncoder), cls=CustomJSONDecoder)


# --- 1) Single-LF aggregate pipeline

@pytest.fixture
def dlf_agg() -> pg.LazyFrameExpr:
    return (
        pg.deferred()
        .filter(pl.col("value") > 5)
        .group_by("group")
        .agg(pl.col("value").sum().alias("value_sum"))
        .collect()
        .sort("group")
        .lazy()
        .filter()
    )


@pytest.mark.parametrize("via_json", [False, True], ids=["direct", "json"])
def test_single_lf_aggregate_roundtrip(dlf_agg, single_lf_data, via_json):
    deferred = _roundtrip(dlf_agg) if via_json else dlf_agg
    out = deferred(single_lf_data).collect()

    expected = pl.from_repr(
        """
shape: (2, 2)
┌───────┬───────────┐
│ group ┆ value_sum │
│ ---   ┆ ---       │
│ str   ┆ i64       │
╞═══════╪═══════════╡
│ a     ┆ 10        │
│ b     ┆ 8         │
└───────┴───────────┘
"""
    )
    assert_frame_equal(out, expected)


# --- 2) Multi-source join + transform pipeline

@pytest.fixture
def dlf_join_select() -> pg.LazyFrameExpr:
    return (
        pg.deferred("orders")
        .join(pg.deferred("customers").select(pl.all()), on="customer_id", how="inner")
        .with_columns(pl.col("amount").round(0).alias("amount_rounded"))
        .select("order_id", "name", "amount_rounded")
        .collect()
        .sort("order_id")
        .lazy()
    )


@pytest.mark.parametrize("via_json", [False, True], ids=["direct", "json"])
def test_ms_join_transform_roundtrip(dlf_join_select, ms_data, via_json):
    deferred = _roundtrip(dlf_join_select) if via_json else dlf_join_select
    out = deferred(ms_data).collect()

    expected = pl.from_repr(
        """
shape: (4, 3)
┌──────────┬───────┬────────────────┐
│ order_id ┆ name  ┆ amount_rounded │
│ ---      ┆ ---   ┆ ---            │
│ i64      ┆ str   ┆ f64            │
╞══════════╪═══════╪════════════════╡
│ 100      ┆ Alice ┆ 20.0           │
│ 101      ┆ Bob   ┆ 36.0           │
│ 102      ┆ Alice ┆ 12.0           │
│ 103      ┆ Caro  ┆ 99.0           │
└──────────┴───────┴────────────────┘
"""
    )
    assert_frame_equal(out, expected)


# --- 3) Self-join on the same source name

@pytest.fixture
def dlf_self_join() -> pg.LazyFrameExpr:
    return pg.deferred("orders").join(
        pg.deferred("orders").select(pl.all()),
        on="order_id",
        how="inner",
    )


@pytest.mark.parametrize("via_json", [False, True], ids=["direct", "json"])
def test_self_join_roundtrip(dlf_self_join, ms_data, via_json):
    deferred = _roundtrip(dlf_self_join) if via_json else dlf_self_join
    out = deferred(ms_data).collect()

    expected = pl.from_repr(
        """
shape: (4, 5)
┌──────────┬─────────────┬────────┬───────────────────┬──────────────┐
│ order_id ┆ customer_id ┆ amount ┆ customer_id_right ┆ amount_right │
│ ---      ┆ ---         ┆ ---    ┆ ---               ┆ ---          │
│ i64      ┆ i64         ┆ f64    ┆ i64               ┆ f64          │
╞══════════╪═════════════╪════════╪═══════════════════╪══════════════╡
│ 100      ┆ 1           ┆ 20.0   ┆ 1                 ┆ 20.0         │
│ 101      ┆ 2           ┆ 35.5   ┆ 2                 ┆ 35.5         │
│ 102      ┆ 1           ┆ 12.0   ┆ 1                 ┆ 12.0         │
│ 103      ┆ 3           ┆ 99.0   ┆ 3                 ┆ 99.0         │
└──────────┴─────────────┴────────┴───────────────────┴──────────────┘
"""
    )
    # ensure deterministic row order just in case
    assert_frame_equal(out.sort("order_id"), expected)
