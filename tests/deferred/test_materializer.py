from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from paguro.defer import LazyFrameExpr, deferred
from paguro.defer.utils.materializer import _Materializer
from paguro.defer.utils.utils import DEFAULT_STEP_FLAG

# ----------------------------------------------------------------------
# fixtures (tiny in-memory frames; keep tests DRY and deterministic)
# ----------------------------------------------------------------------

@pytest.fixture
def customers_lf():
    return pl.DataFrame(
        {"customer_id": [1, 2, 3], "name": ["Alice", "Bob", "Caro"]}
    ).lazy()

@pytest.fixture
def orders_lf():
    return pl.DataFrame(
        {"order_id": [10, 11, 12], "customer_id": [1, 2, 1], "amount": [20.0, 35.5, 12.0]}
    ).lazy()

@pytest.fixture
def data_xy():
    return {
        "t": pl.DataFrame({"x": [1]}).lazy(),
        "root": pl.DataFrame({"x": [1, 2]}).lazy(),
        "sub": pl.DataFrame({"x": [1, 2]}).lazy(),
        "a": pl.DataFrame({"k": [1]}).lazy(),
        "b": pl.DataFrame({"k": [1]}).lazy(),
        "c": pl.DataFrame({"k": [1]}).lazy(),
    }


# ----------------------------------------------------------------------
# happy paths
# ----------------------------------------------------------------------

def test_materialize_resolves_named_dependency_and_joins(customers_lf, orders_lf):
    customers = deferred("customers").select(pl.all())
    orders = (
        deferred("orders")
        .join(customers, on="customer_id", how="inner")   # cross-pipeline ref (in args)
        .select("order_id", "name", "amount")
    )

    mat = _Materializer({"customers": customers_lf, "orders": orders_lf})
    out = mat.materialize(orders).collect().sort("order_id")
    expected = pl.DataFrame(
        {"order_id": [10, 11, 12], "name": ["Alice", "Bob", "Alice"], "amount": [20.0, 35.5, 12.0]}
    )
    assert_frame_equal(out, expected)


def test_materialize_resolves_when_other_pipeline_passed_via_kwargs(customers_lf, orders_lf):
    customers = deferred("customers").select(pl.all())
    orders = (
        deferred("orders")
        .join(other=customers, on="customer_id", how="inner")  # cross-pipeline ref (in kwargs)
        .select("order_id", "name", "amount")
    )

    mat = _Materializer({"customers": customers_lf, "orders": orders_lf})
    out = mat.materialize(orders).collect().sort("order_id")
    expected = pl.DataFrame(
        {"order_id": [10, 11, 12], "name": ["Alice", "Bob", "Alice"], "amount": [20.0, 35.5, 12.0]}
    )
    assert_frame_equal(out, expected)


def test_materialize_memoizes_by_pipeline_identity():
    base = deferred("t").select(pl.all())
    t_lf = pl.DataFrame({"x": [1]}).lazy()

    mat = _Materializer({"t": t_lf})
    a = mat.materialize(base)
    b = mat.materialize(base)
    assert a is b  # cached by id(pipeline)


# deep nesting (no cycle): a -> b -> c
def test_materialize_resolves_deep_nesting_no_cycle():
    c = deferred("c").with_columns(pl.lit(1).alias("c1"))
    b = deferred("b").join(c, how="cross").with_columns(pl.lit(2).alias("b2"))
    a = deferred("a").join(b, how="cross").select(pl.all())

    data = {
        "a": pl.DataFrame({"xa": [1]}).lazy(),
        "b": pl.DataFrame({"xb": [1]}).lazy(),
        "c": pl.DataFrame({"xc": [1]}).lazy(),
    }
    out = _Materializer(data).materialize(a).collect()
    assert {"xa", "xb", "xc", "c1", "b2"}.issubset(set(out.columns))



def test_resolve_step_input_flagged_args_and_kwargs_are_resolved(
        data_xy,
):
    sub = deferred("sub").select(pl.all())

    # synthetic input_ with a DP
    # in args *and* kwargs and the flag set
    input_ = {
        "args": (sub, 42),
        "kwargs": {"other": sub, "value": "x"},
        DEFAULT_STEP_FLAG: True,
    }
    mat = _Materializer(data_xy)

    args, kwargs = mat._resolve_step_input_data(input_)
    # DPs should be replaced by LazyFrames
    assert hasattr(args[0], "collect") and hasattr(kwargs["other"], "collect")
    assert args[1] == 42 and kwargs["value"] == "x"


def test_resolve_step_input_unflagged_passes_through_even_if_it_contains_dp(data_xy):
    sub = deferred("sub").select(pl.all())

    # Note: this shape should never occur from production code (flag is set automatically),
    # but we test the guard: without the flag, input_ is passed through unchanged.
    input_ = {
        "args": (sub, 99),
        "kwargs": {"other": sub},
        # no DEFAULT_STEP_FLAG
    }
    mat = _Materializer(data_xy)
    args, kwargs = mat._resolve_step_input_data(input_)

    from paguro.defer import LazyFrameExpr as DP
    assert isinstance(args[0], DP) and isinstance(kwargs["other"], DP)
    assert args[1] == 99


def test_resolve_step_input_unflagged_plain_pass_through():
    input_ = {"args": (1, 2), "kwargs": {"a": 3}}
    mat = _Materializer({})
    args, kwargs = mat._resolve_step_input_data(input_)
    assert args == (1, 2) and kwargs == {"a": 3}


# ----------------------------------------------------------------------
# error conditions: missing data / unnamed / missing key
# ----------------------------------------------------------------------

def test_materialize_raises_without_data_dict_for_cross_refs():
    sub = deferred("sub").select(pl.all())
    root = deferred("root").join(sub, on="x")

    with pytest.raises(RuntimeError, match="no data dict was provided"):
        _ = _Materializer(None).materialize(root)


def test_materialize_raises_when_referenced_pipeline_has_no_name():
    unnamed = deferred().select(pl.all())
    root = deferred("root").join(unnamed, on="x")

    with pytest.raises(RuntimeError, match="must have a name"):
        _ = _Materializer({"root": pl.DataFrame({"x": [1]}).lazy()}).materialize(root)


def test_materialize_raises_keyerror_when_missing_required_key():
    sub = deferred("sub").select(pl.all())
    root = deferred("root").join(sub, on="x")

    with pytest.raises(KeyError, match="missing key 'sub'"):
        _ = _Materializer({"root": pl.DataFrame({"x": [1]}).lazy()}).materialize(root)


# ----------------------------------------------------------------------
# cycle detection (name-based)
# ----------------------------------------------------------------------

def test_materialize_detects_cycles_between_pipelines():
    a0 = deferred("a")
    b0 = deferred("b")

    a1 = a0.join(b0, on="k")   # a1 -> b0
    b1 = b0.join(a1, on="k")   # b1 -> a1
    a2 = a1.join(b1, on="k")   # a2 -> b1 -> a1 -> b0  (now causes name re-entry of "a")

    data = {
        "a": pl.DataFrame({"a": [1], "k": [1]}).lazy(),
        "b": pl.DataFrame({"b": [1], "k": [1]}).lazy(),
    }
    mat = _Materializer(data)

    with pytest.raises(RuntimeError, match="Cycle detected"):
        _ = mat.materialize(a2)


def test_materialize_stack_is_cleared_after_error_allows_next_materialization(customers_lf, orders_lf):
    # Trigger an error on first materialize, then ensure a fresh pipeline still works
    bad_steps = ({"no_such_method": {"args": (), "kwargs": {}}},)
    bad = LazyFrameExpr(name="orders", _steps=bad_steps)  # same name as a valid entry in the dict

    mat = _Materializer({"customers": customers_lf, "orders": orders_lf})

    with pytest.raises(RuntimeError):
        _ = mat.materialize(bad)  # raises; stack should be cleared in finally

    # Now try a valid pipeline with the same materializer; should work if the stack was cleared
    customers = deferred("customers").select(pl.all())
    orders = deferred("orders").join(customers, on="customer_id").select("order_id", "name", "amount")
    out = mat.materialize(orders).collect().sort("order_id")
    expected = pl.DataFrame(
        {"order_id": [10, 11, 12], "name": ["Alice", "Bob", "Alice"], "amount": [20.0, 35.5, 12.0]}
    )
    assert_frame_equal(out, expected)


# ----------------------------------------------------------------------
# error wrapping during step application
# ----------------------------------------------------------------------

def test_materialize_wraps_step_errors_with_context():
    bad_steps = ({"no_such_method": {"args": (), "kwargs": {}}},)
    bad = LazyFrameExpr(name="bad", _steps=bad_steps)

    data = {"bad": pl.DataFrame({"x": [1]}).lazy()}
    mat = _Materializer(data)

    with pytest.raises(RuntimeError) as exc:
        _ = mat.materialize(bad)
    msg = str(exc.value)
    assert "no_such_method" in msg
    assert "pipeline 'bad'" in msg
