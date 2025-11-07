from __future__ import annotations

import copy

import polars as pl
import pytest
from polars.testing import assert_frame_equal
import paguro as pg
from paguro.defer import (
    deferred,
LazyFrameExpr
)
from paguro.defer.frames import (
    _bind_from_step,
    _returns_dataframe,
    _returns_lazyframe,
    inspect_transform_data_function,
)


# ----------------------------------------------------------------------
# Fixtures (tiny + optional, just to keep the tests DRY)
# ----------------------------------------------------------------------

@pytest.fixture
def lf_xy():
    return pl.DataFrame({"x": [0, 1, 1], "y": [10, 20, 30]}).lazy()


@pytest.fixture
def lf_nested():
    return pl.DataFrame({"id": [1, 2], "vals": [[1, 2], [3]]}).lazy()


# ---- serialization / construction ----

def test_to_dict_nonserializable_contains_steps_and_name():
    p = (
        pg.deferred("root")
        .select(pl.all())
        .filter(pl.col("x") > 0)
    )
    d = p._to_dict()
    assert d["name"] == "root"
    assert isinstance(d["steps"], tuple)
    assert len(d["steps"]) == 2


def test_constructor_steps_type_guard():
    with pytest.raises(TypeError):
        # the internal fast-path enforces tuple; simulate misuse to hit the guard
        LazyFrameExpr(name="x", _steps=[])


# ---- dict-mode guard when unnamed root ----

def test_dict_mode_requires_root_name_runtimeerror():
    # cross-pipeline reference forces dict-mode; unnamed root should refuse
    other = pg.deferred("other").select(pl.all())
    unnamed = pg.deferred().join(other, on="id")
    with pytest.raises(RuntimeError):
        _ = unnamed(
            {"other": pl.DataFrame({"id": [1]}).lazy(),
             "": pl.DataFrame({"id": [1]}).lazy()}
        )


# ---- _bind_from_step error wrapping ----

def test_bind_from_step_wraps_errors():
    lf = pl.DataFrame({"x": [1]}).lazy()
    # bogus method name triggers AttributeError, which should be wrapped as RuntimeError
    bad_step = {"nope": {"args": (), "kwargs": {}}}
    with pytest.raises(RuntimeError):
        _ = _bind_from_step(lf, bad_step)


# ---- explicit wrapper coverage (explode) ----

def test_explode_wrapper_smoke(lf_nested):
    p = deferred().explode("vals").select("id", "vals")
    out = p(lf_nested).collect().sort("id", "vals")
    assert_frame_equal(out, pl.DataFrame({"id": [1, 1, 2], "vals": [1, 2, 3]}))


# ---- type-hint utilities & inspect function ----

def test_returns_helpers_with_local_functions():
    def f_lazy(x: pl.LazyFrame) -> pl.LazyFrame:
        return x

    def f_df(x: pl.DataFrame) -> pl.DataFrame:
        return x

    def f_none(x: pl.LazyFrame) -> None:
        return None  # pragma: no cover

    assert _returns_lazyframe(f_lazy) is True
    assert _returns_dataframe(f_lazy) is False
    assert _returns_dataframe(f_df) is True
    assert _returns_lazyframe(f_df) is False
    assert _returns_lazyframe(f_none) is False
    assert _returns_dataframe(f_none) is False


def test_inspect_transform_data_function_warnings_and_errors():
    # ok: param LazyFrame, return LazyFrame
    def ok(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    assert inspect_transform_data_function(ok) is ok

    # warn: missing param type
    def missing_param_type(lf):  # type: ignore[no-untyped-def]
        return lf

    with pytest.warns(UserWarning):
        inspect_transform_data_function(missing_param_type)

    # warn: missing return type
    def missing_ret(lf: pl.LazyFrame):  # type: ignore[no-untyped-def]
        return lf

    with pytest.warns(UserWarning):
        inspect_transform_data_function(missing_ret)

    # error: wrong param type
    def bad_param(df: pl.DataFrame) -> pl.LazyFrame:
        return df.lazy()  # type: ignore[return-value]

    with pytest.raises(TypeError):
        inspect_transform_data_function(bad_param)

    # error: wrong return type
    def bad_return(lf: pl.LazyFrame) -> int:
        return 1  # type: ignore[return-value]

    with pytest.raises(TypeError):
        inspect_transform_data_function(bad_return)


# ---- __getattr__ branches ----

def test_lazy_proxy_attr_exists_but_not_callable_raises():
    dp = deferred()
    # LazyFrame.schema is an attribute (not callable) → should raise the "not callable" branch
    with pytest.raises(AttributeError):
        _ = dp.schema


def test_lazy_proxy_callable_but_not_lazyframe_return_raises():
    dp = deferred()
    # `explain` returns a string; callable but not LazyFrame-returning → must raise
    with pytest.raises(AttributeError):
        _ = dp.explain


# ---- misc small edges: copy/len/iter/repr/deepcopy ----

def test_copy_len_iter_repr_are_reasonable():
    p = deferred("n").select(pl.all()).filter(pl.col("x") > 0)
    p2 = copy.copy(p)
    p3 = copy.deepcopy(p)  # relies on a correct __deepcopy__(memo) signature
    assert len(p) == 2 == len(tuple(iter(p)))
    assert repr(p).startswith("LazyFrameExpr(")
    assert repr(p2) == repr(p3) == repr(p)


def test_deepcopy_pipeline_roundtrip(lf_xy):
    p = (
        deferred("root")
        .select(pl.all())
        .filter(pl.col("x") > 0)
        .group_by("x")
        .agg(pl.len().alias("n"))
    )
    q = copy.deepcopy(p)
    assert p is not q
    out_p = p({"root": lf_xy}).collect().sort("x")
    out_q = q({"root": lf_xy}).collect().sort("x")
    assert_frame_equal(out_p, out_q)


# ---- tiny extras to bump coverage without complexity ----

def test_noop_pipeline_is_identity_on_single_lf(lf_xy):
    dp = deferred()  # no steps
    out = dp(lf_xy).collect()
    assert_frame_equal(out, pl.DataFrame({"x": [0, 1, 1], "y": [10, 20, 30]}))


def test_dynamic_lazy_method_rename_and_sort():
    lf = pl.DataFrame({"a": [2, 1]}).lazy()
    dp = deferred().rename({"a": "A"}).sort("A")  # dynamic proxy path
    out = dp(lf).collect()
    assert out.columns == ["A"]
    assert_frame_equal(out, pl.DataFrame({"A": [1, 2]}))


def test_eager_proxy_rejects_non_dataframe_returning_methods():
    dp = deferred().collect()
    # DataFrame.group_by returns a GroupBy object → eager proxy must reject
    with pytest.raises(AttributeError):
        dp.group_by("x")  # type: ignore[attr-defined]
    # but a real DF method should still be available afterwards
    _ = dp.sort("x")


def test_pipe_lazy_and_eager():
    lf = pl.DataFrame({"x": [1, 2]}).lazy()

    # lazy pipe: doubles x
    def double(lf_in: pl.LazyFrame) -> pl.LazyFrame:
        return lf_in.with_columns((pl.col("x") * 2).alias("x2"))

    out_lazy = deferred().pipe(double).select("x2")(lf).collect()
    assert_frame_equal(out_lazy, pl.DataFrame({"x2": [2, 4]}))

    # eager pipe: add 1
    def add_one(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns((pl.col("x") + 1).alias("x_plus_one"))

    out_eager = deferred().collect().pipe(add_one).lazy().select("x_plus_one")(
        lf).collect()
    assert_frame_equal(out_eager, pl.DataFrame({"x_plus_one": [2, 3]}))
