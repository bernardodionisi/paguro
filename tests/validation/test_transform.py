from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import paguro as pg
from paguro.defer import LazyFrameExpr
from paguro.validation.valid_frame.utils.transform import TransformFrameTree


# ---------- Fixtures ----------

@pytest.fixture
def sample_data():
    sample_data = {"a": [1, 2, 3]}
    return sample_data


@pytest.fixture
def vf():
    vf = pg.vframe(name="test", transform=pl.col("a").mean())
    return vf


@pytest.fixture
def tr(vf, sample_data):
    tr = vf.transform(sample_data, collect=True)
    return tr


# ---------- Helpers ----------

def _unnest_to_type(tr: TransformFrameTree, to_lazyframe: bool):
    out = tr._to_type_dict(type_=pl.DataFrame, key="frame", to_lazyframe=to_lazyframe)
    return out


# ---------- Tests ----------

def test_returns_transform_tree(tr):
    assert isinstance(tr, TransformFrameTree)


@pytest.mark.parametrize(
    "to_lazyframe, expected_type",
    [(True, pl.LazyFrame), (False, pl.DataFrame)],
)
def test_to_type_dict_contains_named_frame(tr, to_lazyframe, expected_type):
    out = _unnest_to_type(tr, to_lazyframe)
    assert set(out) == {"test"}
    assert isinstance(out["test"], expected_type)


def test_transform_mapping_structure(tr):
    transform_dict = tr._mapping.get("test").get("transform")
    assert isinstance(transform_dict, dict)
    assert isinstance(transform_dict.get("pipeline"), LazyFrameExpr)


def test_transform_output_matches_expected(tr, sample_data):
    # Build expected using Polars instead of from_repr for clarity/robustness
    expected = pl.DataFrame(sample_data).select(pl.col("a").mean())
    transform_dict = tr._mapping.get("test").get("transform")
    assert_frame_equal(transform_dict.get("frame"), expected)
