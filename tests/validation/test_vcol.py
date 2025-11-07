from __future__ import annotations

import polars as pl
import polars.selectors as cs
import pytest

import paguro as pg


# Fixtures

@pytest.fixture(scope="module")
def sample_data() -> dict:
    # a has a None to exercise required/allow_nulls checks
    # b,c used for composite-unique checks
    sample_data = {
        "a": [1, 2, None],
        "b": [4, 5, 5],
        "c": ["a", "b", "b"],
        "d": [{"D": 1}, {"D": 2}, {"D": 3}],
    }
    return sample_data


def test_vcol_to_schema():
    with pytest.raises(TypeError):
        pg.vcol("a").to_schema()

    assert pg.vcol("a", dtype=int).to_schema() == pl.Schema({"a": pl.Int128})
    assert pg.vcol("a", dtype=pl.Int32).to_schema() == pl.Schema({"a": pl.Int32})


# Helpers

def counts_by_group(error_obj) -> dict:
    """Return error counts by group, or {} if None."""
    return {} if error_obj is None else error_obj._get_error_counts_by_group()


# Construction/defaults

def test_vcol_construction_defaults():
    vc = pg.vcol(name="name")

    assert vc._name == "name"

    # Defaults
    assert vc._required is True
    assert vc._allow_drop is True
    assert vc._allow_rename is True
    assert vc._allow_nulls is False
    assert vc._fields is None


# Required / presence checks

def test_required_missing_column_returns_error(sample_data):
    vc = pg.vcol("F", required=True)
    err = vc.validate(sample_data, on_failure="return_error")
    assert counts_by_group(err).get("required", {}).get("errors") == 1
    err = pg.Validation(vc).validate(sample_data, on_failure="return_error")
    assert counts_by_group(err).get("required", {}).get("errors") == 1


def test_required_present_with_success_modes(sample_data):
    vc = pg.vcol("a", required=True, allow_nulls=True)

    # return_none: success → None
    assert vc.validate(sample_data, on_success="return_none") is None
    assert pg.Validation(vc).validate(sample_data, on_success="return_none") is None

    # return_data: success → DataFrame
    df = vc.validate(sample_data, on_success="return_data")
    assert isinstance(df, pl.DataFrame)
    df = pg.Validation(vc).validate(sample_data, on_success="return_data")
    assert isinstance(df, pl.DataFrame)


def test_required_raises_when_missing_column():
    with pytest.raises(pg.exceptions.ValidationError):
        (
            pg.vcol("a", required=True)
            .validate(data={"b": [1]})  # missing 'a'
        )

    with pytest.raises(pg.exceptions.ValidationError):
        (
            pg.Validation(pg.vcol("a", required=True))
            .validate(data={"b": [1]})  # missing 'a'
        )


# Dtype

def test_dtype_single_and_multiple_and_mapping():
    vc1 = pg.vcol("a", dtype=pl.Int64)
    assert vc1._dtype == frozenset([pl.Int64])

    vc2 = pg.vcol("a", dtype=[pl.Int64, pl.String])
    assert vc2._dtype == frozenset([pl.Int64, pl.String])

    # Mapping types (e.g., dict) should map to Struct
    vc3 = pg.vcol(name="a", dtype=dict)
    assert vc3._dtype == frozenset([pl.Struct])


def test_dtype_and_allow_nulls_produce_both_error_groups(sample_data):
    vc = pg.vcol(name="a", dtype=pl.Float32, allow_nulls=False)
    err = vc.validate(sample_data, on_failure="return_error")
    # 'a' contains None → allow_nulls error; and dtype mismatch vs Float32 → dtype error
    assert counts_by_group(err) == {"dtype": {"errors": 1},
                                    "allow_nulls": {"errors": 1}}

    err = pg.Validation(vc).validate(sample_data, on_failure="return_error")
    assert counts_by_group(err) == {"dtype": {"errors": 1},
                                    "allow_nulls": {"errors": 1}}


# Constraints

def test_constraints_with_selector_and_limits(sample_data):
    # Apply constraints to all numeric columns
    vc = pg.vcol(cs.numeric(), c0=pl.all().le(0), allow_nulls=True, le=0)
    err = vc.validate(sample_data, on_failure="return_error")
    # Expect 4 constraint errors across numeric columns (based on sample_data)
    assert counts_by_group(err) == {"constraints": {"errors": 4}}

    err = pg.Validation(vc).validate(sample_data, on_failure="return_error")
    assert counts_by_group(err) == {"constraints": {"errors": 4}}


def test_constraints_with_none_name_and_required(sample_data):
    vc = pg.vcol(None, required=True, allow_nulls=True, c0=pl.all().le(0), le=0)
    err = vc.validate(sample_data, on_failure="return_error")
    # Expect both errors and exceptions tallied under constraints
    assert counts_by_group(err) == {"constraints": {"errors": 4, "exception": 4}}

    err = pg.Validation(vc).validate(sample_data, on_failure="return_error")
    assert counts_by_group(err) == {"constraints": {"errors": 4, "exception": 4}}


# Uniqueness

def test_unique_single_column_ok(sample_data):
    # 'a' values are [1,2,None]; None does not violate uniqueness here
    vc = pg.vcol("a", unique=True, allow_nulls=True)
    err = vc.validate(sample_data, on_failure="return_error")
    assert err is None
    err = pg.Validation(vc).validate(sample_data, on_failure="return_error")
    assert err is None


def test_unique_composite_column_fails(sample_data):
    # ('b','c') has two identical pairs: (5,'b') appears twice
    vc = pg.vcol(("b", "c"), unique=True, allow_nulls=True)
    err = vc.validate(sample_data, on_failure="return_error")
    assert counts_by_group(err) == {"unique": {"errors": 2}}
    err = pg.Validation(vc).validate(sample_data, on_failure="return_error")
    assert counts_by_group(err) == {"unique": {"errors": 2}}


# foelds

def test_fields_ok(sample_data):
    # 'a' values are [1,2,None]; None does not violate uniqueness here
    vc = pg.vcol.Struct(pg.vcol("D", dtype=int), name="d")
    err = vc.validate(sample_data, on_failure="return_error")
    assert err is None
    err = pg.Validation(vc).validate(sample_data, on_failure="return_error")
    assert err is None


def test_unique_fileds_fails(sample_data):
    # ('b','c') has two identical pairs: (5,'b') appears twice
    vc = pg.vcol.Struct(
        pg.vcol("D", dtype=int, allow_nulls=True, ge=10),
        pg.vcol("D1", dtype=int, required=True, allow_nulls=True, ge=10)
        , name="d"
    )

    errors_counts = {'required': {'errors': 1},
                     'allow_nulls': {'no_errors': 1},
                     'constraints': {'errors': 1}}
    err = vc.validate(sample_data, on_failure="return_error")
    assert counts_by_group(err) == errors_counts
    err = pg.Validation(vc).validate(sample_data, on_failure="return_error")
    assert counts_by_group(err) == errors_counts
