from __future__ import annotations

import polars as pl
import pytest

from paguro.shared.dtypes.dtype_serialize import (
    DTypeDict,
    compare_dicts,
    dict_to_dtype_list,
    dict_to_pl_dtype,
    dtype_to_dict_list,
    pl_dtype_to_dict,
)


# ------- helpers

def roundtrip_serde(dtype) -> DTypeDict:
    """Serialize -> deserialize -> serialize (normalized)"""
    d = pl_dtype_to_dict(dtype)
    rt = dict_to_pl_dtype(d)
    return pl_dtype_to_dict(rt)


def _maybe(name: str):
    return getattr(pl, name) if hasattr(pl, name) else None


def _make_instance(dtype_cls):
    """Attempt dtype_cls(); if not possible, return the class itself (DataTypeClass)."""
    try:
        return dtype_cls()
    except Exception:
        return dtype_cls


# ------- numerics / string / bool / misc

PRIMITIVE_NAMES = [
    # integers
    "Int8", "Int16", "Int32", "Int64", "Int128",
    "UInt8", "UInt16", "UInt32", "UInt64", "UInt128",
    # floats
    "Float32", "Float64",
    # string + alias
    "String", "Utf8",
    # misc
    "Boolean", "Binary", "Null", "Object", "Unknown", "Categorical",
]


@pytest.mark.parametrize("name", PRIMITIVE_NAMES)
def test_primitives_class_and_instance_roundtrip(name: str):
    dtype_cls = _maybe(name)
    if dtype_cls is None:
        pytest.skip(f"pl.{name} not available in this Polars version")
    # class form
    assert pl_dtype_to_dict(dtype_cls) == roundtrip_serde(dtype_cls)
    # instance form (if constructible)
    inst = _make_instance(dtype_cls)
    assert pl_dtype_to_dict(inst) == roundtrip_serde(inst)


def test_utf8_alias_matches_string():
    if not hasattr(pl, "Utf8"):
        pytest.skip("Utf8 alias not present")
    d_utf8 = pl_dtype_to_dict(pl.Utf8)
    d_str = pl_dtype_to_dict(pl.String)
    assert d_utf8["_dtype"] == d_str["_dtype"]


# ------- temporal

def test_temporal_classes_roundtrip():
    for name in ["Date", "Time", "Datetime", "Duration"]:
        dtype_cls = _maybe(name)
        if dtype_cls is None:
            continue
        assert pl_dtype_to_dict(dtype_cls) == roundtrip_serde(dtype_cls)


@pytest.mark.parametrize("time_unit", ["ns", "us", "ms"])
@pytest.mark.parametrize("tz", [None, "UTC", "Europe/Berlin"])
def test_datetime_with_attrs_roundtrip(time_unit: str, tz: str | None):
    if not hasattr(pl, "Datetime"):
        pytest.skip("pl.Datetime not available")
    dt = pl.Datetime(time_unit=time_unit, time_zone=tz)
    assert pl_dtype_to_dict(dt) == roundtrip_serde(dt)


@pytest.mark.parametrize("time_unit", ["ns", "us", "ms"])
def test_duration_with_attrs_roundtrip(time_unit: str):
    if not hasattr(pl, "Duration"):
        pytest.skip("pl.Duration not available")
    dt = pl.Duration(time_unit=time_unit)
    assert pl_dtype_to_dict(dt) == roundtrip_serde(dt)


# ------- Decimal / Enum

@pytest.mark.parametrize("prec_scale", [(10, 0), (10, 2), (38, 10)])
def test_decimal_roundtrip(prec_scale: tuple[int, int]):
    if not hasattr(pl, "Decimal"):
        pytest.skip("pl.Decimal not available")
    precision, scale = prec_scale
    dt = pl.Decimal(precision=precision, scale=scale)
    d1 = pl_dtype_to_dict(dt)
    d2 = roundtrip_serde(dt)
    assert d1 == d2
    assert d1["_attrs"]["precision"] == precision
    assert d1["_attrs"]["scale"] == scale


def test_enum_roundtrip():
    if not hasattr(pl, "Enum"):
        pytest.skip("pl.Enum not available")
    dt = pl.Enum(["x", "y", "z"])
    assert pl_dtype_to_dict(dt) == roundtrip_serde(dt)


# ------- containers

def test_list_roundtrip():
    dt = pl.List(pl.Int32)
    assert pl_dtype_to_dict(dt) == roundtrip_serde(dt)


def test_array_roundtrip_canonical():
    dt = pl.Array(pl.Int16, (2, 3))
    d1 = pl_dtype_to_dict(dt)
    d2 = roundtrip_serde(dt)
    assert d1 == d2
    # canonical: top-level shape only; inner is base element dtype (not nested Array)
    assert tuple(d1["_attrs"]["shape"]) == (2, 3)
    assert d1["_attrs"]["inner"]["_dtype"] == pl_dtype_to_dict(pl.Int16)["_dtype"]


def test_array_nested_canonicalization():
    # Some internal reps look like Array(inner=Array(Int16, (3,)), shape=(2,))
    inner = pl.Array(pl.Int16, (3,))
    outer = pl.Array(inner, (2,))
    d = pl_dtype_to_dict(outer)
    assert tuple(d["_attrs"]["shape"]) == (2, 3)
    assert d["_attrs"]["inner"]["_dtype"] == pl_dtype_to_dict(pl.Int16)["_dtype"]
    assert d == roundtrip_serde(outer)


def test_struct_roundtrip_simple():
    fields = [pl.Field("a", pl.Int64), pl.Field("b", pl.List(pl.Utf8))]
    dt = pl.Struct(fields)
    assert pl_dtype_to_dict(dt) == roundtrip_serde(dt)


def test_struct_deep_combo():
    # Handles both with/without Decimal/Enum available
    score_dt = pl.Decimal(precision=9, scale=3) if hasattr(pl,
                                                           "Decimal") else pl.Float64
    labels_dt = pl.Enum(["a", "b"]) if hasattr(pl, "Enum") else pl.Categorical
    inner_struct = pl.Struct([
        pl.Field("id", pl.Int64),
        pl.Field("score", score_dt),
        pl.Field("labels", labels_dt),
    ])
    complex_dt = pl.List(pl.Struct([pl.Field("input_", inner_struct)]))
    assert pl_dtype_to_dict(complex_dt) == roundtrip_serde(complex_dt)


# ------- list/array/struct in dtype_to_dict_list + dict_to_dtype_list

def test_dtype_to_dict_list_and_back():
    types = [
        pl.Int8,
        pl.List(pl.Utf8),
        pl.Array(pl.Int16, (3,)),
        pl.Struct([pl.Field("x", pl.Int32)]),
    ]
    dumped = dtype_to_dict_list(types)
    assert isinstance(dumped, list) and all(isinstance(x, dict) for x in dumped)
    loaded = dict_to_dtype_list(dumped)  # type: ignore[arg-type]
    dumped2 = [pl_dtype_to_dict(dt) for dt in loaded]
    assert dumped == dumped2


def test_dtype_to_dict_list_none():
    assert dtype_to_dict_list(None) is None


# ------- class-level containers (no _attrs)

def test_class_level_containers_without_instantiate():
    d_struct = pl_dtype_to_dict(pl.Struct)
    d_list = pl_dtype_to_dict(pl.List)
    d_array = pl_dtype_to_dict(pl.Array)

    assert d_struct["_dtype"] == "Struct" and "_attrs" not in d_struct
    assert d_list["_dtype"] == "List" and "_attrs" not in d_list
    assert d_array["_dtype"] == "Array" and "_attrs" not in d_array

    # deserializing class-level dicts returns the class objects
    assert dict_to_pl_dtype(d_struct) is pl.Struct
    assert dict_to_pl_dtype(d_list) is pl.List
    assert dict_to_pl_dtype(d_array) is pl.Array


# ------- compare_dicts behavior

def test_compare_dicts_equal_and_subset():
    d1 = pl_dtype_to_dict(pl.Struct([pl.Field("a", pl.Int8), pl.Field("b", pl.Utf8)]))
    d2 = pl_dtype_to_dict(pl.Struct([pl.Field("a", pl.Int8), pl.Field("b", pl.Utf8)]))
    eq, subset = compare_dicts(d1, d2)
    assert eq is True and subset is False

    # Make d1 a superset via an extra param under _attrs
    d1b = dict(d1)
    d1b["_attrs"] = dict(d1b.get("_attrs", {}))
    d1b["_attrs"]["extra_param"] = 123
    eq2, subset2 = compare_dicts(d1b, d1)
    assert eq2 is False and subset2 is True

    # fields keyset mismatch -> not subset
    d3 = pl_dtype_to_dict(pl.Struct([pl.Field("a", pl.Int8)]))
    eq3, subset3 = compare_dicts(d1, d3)
    assert eq3 is False and subset3 is False
