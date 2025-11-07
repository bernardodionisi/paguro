from __future__ import annotations

import polars as pl
import pytest

from paguro.shared.dtypes.into_dtypes import parse_dtype_into_frozenset


def _dtype_tag(x: pl.DataType | pl.datatypes.DataTypeClass) -> str:
    # Turn dtype (class or instance) into a stable string tag
    # Polars exposes .base_type() for both classes and instances
    return str(x.base_type())  # type: ignore[arg-type]


def _tags_from_result(fr: frozenset[pl.DataType | pl.datatypes.DataTypeClass]) -> set[str]:
    return {_dtype_tag(x) for x in fr}


# -------------------- happy paths --------------------

def test_none_returns_none():
    assert parse_dtype_into_frozenset(None) is None


@pytest.mark.parametrize(
    "inp, expected_tag",
    [
        (str, "String"),
        (int, "Int64"),
        # group choice is handled by constants; but single 'int' should parse as signed group later
        (float, "Float64"),  # see fallback behavior; we validate groups separately below
        (dict, "Struct"),
    ],
)
def test_class_inputs_map_to_known_tags(inp, expected_tag):
    # For class-typed inputs, we only check that the tag-set includes the expected tag
    fr = parse_dtype_into_frozenset(inp)
    assert fr is not None
    tags = _tags_from_result(fr)
    assert expected_tag in tags


def test_string_literal_groups_numeric_uint():
    numeric = parse_dtype_into_frozenset("numeric")
    assert numeric is not None
    tags = _tags_from_result(numeric)
    # Spot-check a few members
    assert "Int64" in tags or "Int32" in tags
    assert "UInt64" in tags or "UInt32" in tags
    assert "Float64" in tags

    uint = parse_dtype_into_frozenset("uint")
    assert uint is not None
    utags = _tags_from_result(uint)
    assert any(t.startswith("UInt") for t in utags)
    assert all(not t.startswith("Int") or t.startswith("UInt") for t in utags)


@pytest.mark.parametrize("lit, expect_member", [
    ("nested", "List"),  # group contains nested types incl. List/Array/Struct
    ("array", "Array"),  # specific to Array
    ("categorical", "Categorical"),
    ("temporal", "Datetime"),  # temporal group includes Datetime/Duration/Date/Time
    ("datetime", "Datetime"),
    ("duration", "Duration"),
])
def test_literal_groups_and_specifics(lit, expect_member):
    fr = parse_dtype_into_frozenset(lit)
    assert fr is not None
    tags = _tags_from_result(fr)
    assert expect_member in tags


def test_enum_from_list_of_strings():
    fr = parse_dtype_into_frozenset(["a", "b", "c"])
    assert fr is not None and len(fr) == 1
    (dt,) = tuple(fr)
    # dt should be an Enum (instance)
    assert isinstance(dt, pl.Enum) or _dtype_tag(dt) == "Enum"


def test_fallback_bool_goes_to_boolean():
    # bool is not handled explicitly; falls back to polars parse_into_dtype
    fr = parse_dtype_into_frozenset(bool)
    assert fr is not None
    assert "Boolean" in _tags_from_result(fr)


def test_nested_struct_dict_only_scalars():
    spec = {"a": int, "b": float, "c": str}
    fr = parse_dtype_into_frozenset(spec)
    assert fr is not None and len(fr) == 1
    (dt,) = tuple(fr)
    assert isinstance(dt, pl.Struct)
    # Inspect fields: names and inner tags
    fields = {f.name: _dtype_tag(f.dtype) for f in dt.fields}
    assert fields["a"].startswith("Int")
    assert fields["b"].startswith("Float")
    assert fields["c"] == "String"


def test_nested_struct_with_list_and_tuple_array():
    spec = {
        "tags": [str],  # -> List(String)
        "vec3": (int, 3),  # -> Array(Int*, 3)
        "input_": {"x": float}  # -> Struct(x=Float*)
    }
    fr = parse_dtype_into_frozenset(spec)
    assert fr is not None and len(fr) == 1
    (dt,) = tuple(fr)
    assert isinstance(dt, pl.Struct)

    # tags -> List(String)
    tags_field = next(f for f in dt.fields if f.name == "tags")
    assert isinstance(tags_field.dtype, pl.List)
    assert _dtype_tag(tags_field.dtype.inner) == "String"

    # vec3 -> Array(Int*, 3)
    vec_field = next(f for f in dt.fields if f.name == "vec3")
    assert isinstance(vec_field.dtype, pl.Array)
    assert _dtype_tag(vec_field.dtype.inner).startswith("Int")
    assert tuple(vec_field.dtype.shape) == (3,)

    input_ = next(f for f in dt.fields if f.name == "input_")
    assert isinstance(input_.dtype, pl.Struct)
    inner_fields = {f.name: _dtype_tag(f.dtype) for f in input_.dtype.fields}
    assert inner_fields["x"].startswith("Float")


def test_iterable_mixed_accumulates_union():
    # A nested iterable mixing types should union all parse results
    fr = parse_dtype_into_frozenset([int, float, "categorical", {"x": int}])
    assert fr is not None
    tags = _tags_from_result(fr)
    # Expect presence of int/float/categorical/struct in the union
    assert any(t.startswith("Int") for t in tags)
    assert any(t.startswith("Float") for t in tags)
    assert "Categorical" in tags
    assert "Struct" in tags


def test_struct_class_and_instance_results():
    # dict -> Struct(instance); dict (type) -> Struct (class)
    fr_instance = parse_dtype_into_frozenset({"a": int})
    fr_class = parse_dtype_into_frozenset(dict)
    assert fr_instance is not None and fr_class is not None
    (dti,) = tuple(fr_instance)
    (dtc,) = tuple(fr_class)
    assert isinstance(dti, pl.Struct)
    assert _dtype_tag(dtc) == "Struct"


# -------------------- edge cases / error cases --------------------

def test_iterable_of_dicts_more_than_one_raises():
    # Your current implementation sends the list-of-dicts directly to _to_dtype,
    # which requires a single element; >1 raises ValueError.
    with pytest.raises(ValueError):
        parse_dtype_into_frozenset([{"a": int}, {"b": float}])


def test_list_spec_with_wrong_arity_in_nested():
    # Inside the mapping, _to_dtype enforces exactly one element in list specs.
    with pytest.raises(ValueError):
        parse_dtype_into_frozenset({"bad_list": [int, float]})
