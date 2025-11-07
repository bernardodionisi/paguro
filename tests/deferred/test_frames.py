from __future__ import annotations

import polars as pl
import paguro as pg

import pytest
from polars.testing import assert_frame_equal

from paguro.defer import (
    LazyFrameExpr,
    deferred,
)

from paguro.defer.frames import (
    _contains_lfe,
    _resolve_any,
)



# ---------- basics & happy paths ----------

def test_self_contained_pipeline_basic_select_filter_groupby():
    lf = pl.DataFrame(
        {"id": [1, 2, 3, 4], "g": ["a", "a", "b", "b"], "v": [10, 5, 2, 8]}
    ).lazy()

    pipe = (
        pg.deferred()
        .filter(pl.col("v") > 5)
        .group_by("g")
        .agg(pl.col("v").sum().alias("sum_v"))
        .collect()
        .sort("g")
        .lazy()
    )
    out = pipe(lf).collect().sort("g")
    expected = pl.DataFrame({"g": ["a", "b"], "sum_v": [10, 8]})
    assert_frame_equal(out, expected)


def test_cross_pipeline_join_with_dict_mode_args():
    customers_lf = pl.DataFrame(
        {"customer_id": [1, 2, 3], "name": ["Alice", "Bob", "Caro"]}
    ).lazy()

    orders_lf = pl.DataFrame(
        {"order_id": [100, 101, 102, 103], "customer_id": [1, 2, 1, 3],
         "amount": [20.0, 35.5, 12.0, 99.0]}
    ).lazy()

    customers = deferred("customers").select(pl.all())
    orders_joined = (
        pg.deferred("orders")
        .join(customers, on="customer_id", how="inner")
        .select("order_id", "name", "amount")
    )

    out = orders_joined(
        {"orders": orders_lf, "customers": customers_lf}).collect().sort(
        "order_id")
    expected = pl.DataFrame(
        {"order_id": [100, 101, 102, 103], "name": ["Alice", "Bob", "Alice", "Caro"],
         "amount": [20.0, 35.5, 12.0, 99.0]}
    )
    assert_frame_equal(out, expected)


def test_cross_pipeline_join_with_dict_mode_kwargs_other():
    """Also verify the DP flagging works when the other pipeline is passed via kwarg."""
    left_lf = pl.DataFrame({"id": [1, 2], "val": [10, 20]}).lazy()
    right_lf = pl.DataFrame({"id": [1, 2], "name": ["A", "B"]}).lazy()

    right = deferred("right").select(pl.all())
    left = deferred("left").join(other=right, on="id").select("id", "name", "val")

    out = left({"left": left_lf, "right": right_lf}).collect().sort("id")
    expected = pl.DataFrame({"id": [1, 2], "name": ["A", "B"], "val": [10, 20]})
    assert_frame_equal(out, expected)


# ---------- error guards & constraints ----------

def test_cross_pipeline_requires_dict_raises():
    left_lf = pl.DataFrame({"id": [1]}).lazy()
    right = deferred("right").select(pl.all())
    left_plan = deferred("left").join(right, on="id")
    with pytest.raises(RuntimeError):
        _ = left_plan(left_lf)


def test_missing_root_key_in_data_dict_raises_keyerror():
    customers_lf = pl.DataFrame({"customer_id": [1], "name": ["Alice"]}).lazy()
    orders_join = deferred("orders").join(deferred("customers"), on="customer_id").select(
        pl.all())
    with pytest.raises(KeyError):
        _ = orders_join({"customers": customers_lf})


def test_nonexistent_method_attr_raises_on_lazy_proxy():
    dp = deferred()
    with pytest.raises(AttributeError):
        _ = dp.totally_not_a_method


def test_eager_proxy_rejects_non_dataframe_returning_methods():
    lf = pl.DataFrame({"x": [1, 1, 2]}).lazy()
    dp = deferred().collect()
    # DataFrame.group_by returns a GroupBy, not a DataFrame → should raise
    with pytest.raises(AttributeError):
        dp.group_by("x")  # type: ignore[attr-defined]
    # Make sure we can still continue with a valid DF method afterwards
    _ = dp.sort("x")


# ---------- dynamic method proxying ----------

def test_dynamic_lazy_method_rename_and_sort():
    lf = pl.DataFrame({"a": [2, 1]}).lazy()
    dp = deferred().rename({"a": "A"}).sort("A")  # rename is proxied dynamically
    out = dp(lf).collect()
    assert out.columns == ["A"]
    assert_frame_equal(out, pl.DataFrame({"A": [1, 2]}))


# ---------- eager <-> lazy hops ----------

def test_eager_hop_with_columns_and_back_to_lazy():
    lf = pl.DataFrame({"x": [1, 2]}).lazy()

    dp = (
        deferred()
        .select(pl.col("x"))
        .collect()  # hop to eager
        .with_columns((pl.col("x") * 10).alias("y"))  # DataFrame method
        .lazy()  # back to lazy
        .select("y")
    )

    out = dp(lf).collect()
    assert_frame_equal(out, pl.DataFrame({"y": [10, 20]}))


# ---------- helpers & internals (simple, deterministic) ----------

def test_contains_dp_and_self_contained_flag():
    a = deferred("a").select(pl.all())
    b = deferred("b").filter(pl.col("x").is_not_null())
    c = deferred("c").join(a, on="id")  # contains another DP

    # _contains_dp smoke tests
    assert _contains_lfe(a) is True
    assert _contains_lfe(("x", {"k": b})) is True
    assert _contains_lfe(42) is False

    # self-contained checks
    assert a._is_self_contained() is True
    assert b._is_self_contained() is True
    assert c._is_self_contained() is False


def test_resolve_any_replaces_nested_pipelines():
    # Build small graph
    p1 = deferred("p1").select(pl.all())
    p2 = deferred("p2").filter(pl.col("x") > 0)

    def materialize(pipe: LazyFrameExpr) -> str:
        # return a recognizable token based on name
        return f"LF<{pipe._name}>"

    nested = {
        "left": (p1, [1, {"mid": p2}], ()),
        "right": "keep-me",
    }
    resolved = _resolve_any(nested, materialize)

    assert resolved["left"][0] == "LF<p1>"
    assert resolved["left"][1][1]["mid"] == "LF<p2>"
    assert resolved["right"] == "keep-me"


def test_empty_pipeline_is_noop_on_single_lazyframe():
    lf = pl.DataFrame({"x": [1, 2]}).lazy()
    dp = deferred()  # no steps
    out = dp(lf).collect()
    assert_frame_equal(out, pl.DataFrame({"x": [1, 2]}))


# ---------- pipe (both lazy and eager) ----------

def test_lazy_pipe_function_annotation_and_result():
    lf = pl.DataFrame({"x": [1, 2]}).lazy()

    # simple pipe that doubles x
    def double(lf_in: pl.LazyFrame) -> pl.LazyFrame:
        return lf_in.with_columns((pl.col("x") * 2).alias("x2"))

    dp = deferred().pipe(double).select("x2")
    out = dp(lf).collect()
    assert_frame_equal(out, pl.DataFrame({"x2": [2, 4]}))


def test_eager_pipe_function_annotation_and_result():
    lf = pl.DataFrame({"x": [1, 2]}).lazy()

    def add_one(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns((pl.col("x") + 1).alias("x_plus_one"))

    dp = deferred().collect().pipe(add_one).lazy().select("x_plus_one")
    out = dp(lf).collect()
    assert_frame_equal(out, pl.DataFrame({"x_plus_one": [2, 3]}))
