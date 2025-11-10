from __future__ import annotations

import json
import pytest

from paguro.ashi.info.info import Info
from paguro.ashi.info.info_collection import (
    InfoCollection,
    MissingInfoNameException,
)


def names(ic: InfoCollection) -> list[str]:
    return ic.names


def body(ic: InfoCollection) -> dict[str, dict]:
    return ic.to_dict(keep_empty=True)


# ---------------------------------------------------------------------
# Construction & basic behavior
# ---------------------------------------------------------------------

def test_empty_init_and_repr_str():
    ic = InfoCollection()
    assert len(ic) == 0
    assert not ic  # falsy
    str(ic)
    repr(ic)


def test_append_is_immutable_and_classifies_schema_member():
    ic = InfoCollection().set_schema(mode="enforced", keys={"a", "b"})
    ic2 = ic.append("ONLY", a=1)

    # immutability
    assert len(ic) == 0
    assert len(ic2) == 1
    assert names(ic2) == ["ONLY"]

    only = ic2["ONLY"]
    assert only._is_schema_level is True
    assert only.to_dict(include_name=False) == {"a": 1}


def test_update_named_create_and_update_is_immutable():
    ic = InfoCollection().set_schema(mode="infer", keys={"x", "y"})
    ic2 = ic.update("meta", title="T1")
    ic3 = ic2.update("meta", desc="D2")

    assert len(ic) == 0
    assert body(ic2) == {"meta": {"title": "T1"}}
    assert body(ic3) == {"meta": {"title": "T1", "desc": "D2"}}


# ---------------------------------------------------------------------
# Bare mapping routing
# ---------------------------------------------------------------------

def test_bare_mapping_routes_to_single_schema_level_member():
    ic = InfoCollection().set_schema(mode="enforced", keys={"a"})
    ic = ic.append("A", a=1)   # classified as schema-level by collection view
    ic2 = ic.update(a=7)       # bare mapping → routes to A
    assert body(ic2) == {"A": {"a": 7}}


def test_bare_mapping_raises_if_not_uniquely_resolvable():
    ic = InfoCollection().set_schema(mode="enforced", keys={"a"})
    ic = ic.append("A", a=1)
    ic = ic.append("B", a=2)  # both schema-level; ambiguous
    with pytest.raises(MissingInfoNameException):
        _ = ic.update(a=9)


def test_bare_mapping_fallback_to_single_member_schema():
    """
    Collection schema is empty/irrelevant, but *one* member is enforced with a schema.
    Fallback should target that single enforced member.
    """
    ic = InfoCollection()
    enforced = Info("ONLY").update(k=1).set_schema(mode="enforced", keys={"k"})
    free = Info("FREE").update(z=0)  # off
    ic = ic.append(enforced).append(free)

    ic2 = ic.update(k=7)  # should route to "ONLY"
    assert body(ic2) == {"ONLY": {"k": 7}, "FREE": {"z": 0}}


# ---------------------------------------------------------------------
# Sync behavior
# ---------------------------------------------------------------------

def test_sync_schema_enforced_only_and_drop_empty_variants():
    # One schema-level, one non-schema
    ic = InfoCollection().set_schema(mode="enforced", keys={"a", "b"})
    ic = ic.append("S", a=1)       # schema-level
    ic = ic.append("F", note="x")  # free-form

    # Sync to new schema that excludes 'a' so S becomes empty
    ic.sync_schema(keys={"b"}, drop_empty="schema")
    # drop_empty="schema" → drop empty schema-level infos only; free-form stays
    assert body(ic) == {"F": {"note": "x"}}

    # Now check drop_empty="all": drop empty infos across the collection (schema + non-schema)
    ic = InfoCollection().set_schema(mode="enforced", keys={"a", "b"})
    ic = ic.append("S", a=1)       # schema-level
    ic = ic.append("F", note="x")  # non-schema (remains non-empty)
    ic.sync_schema(keys=set(), drop_empty="all")
    # S normalized to empty → dropped; F remains because it's NOT empty.
    assert body(ic) == {"F": {"note": "x"}}


def test_non_schema_members_untouched_on_sync():
    ic = InfoCollection().set_schema(mode="enforced", keys={"a"})
    ic = ic.append("FREE", x=1)  # non-schema
    ic.sync_schema(keys={"b"})   # sync shouldn't touch FREE
    assert body(ic) == {"FREE": {"x": 1}}


# ---------------------------------------------------------------------
# Rename behavior
# ---------------------------------------------------------------------

def test_rename_schema_keys_returns_new_and_respects_apply_rules():
    ic = InfoCollection().set_schema(mode="enforced", keys={"a", "b"})
    ic = ic.append("I1", a=1)            # effective schema != collection
    ic = ic.append("I2", a=10, b=20)     # effective schema == collection

    ic2 = ic.rename_schema_keys({"a": "A"}, apply_to_members="enforced")

    # original unchanged
    assert body(ic) == {"I1": {"a": 1}, "I2": {"a": 10, "b": 20}}
    # new collection schema updated
    assert ic2._schema_keys == {"A", "b"}
    # I1 data key renamed (member-specific); I2 retains keys
    assert body(ic2) == {"I1": {"A": 1}, "I2": {"a": 10, "b": 20}}

    # with apply_to_members="all", both get data renamed
    ic3 = ic.rename_schema_keys({"a": "A"}, apply_to_members="all")
    assert body(ic3) == {"I1": {"A": 1}, "I2": {"A": 10, "b": 20}}


def test_rename_schema_keys_validates_collisions():
    ic = InfoCollection().set_schema(mode="enforced", keys={"a", "b"})
    ic = ic.append("I", a=1, b=2)
    with pytest.raises(ValueError, match="collides"):
        _ = ic.rename_schema_keys({"a": "b"})


# ---------------------------------------------------------------------
# Drop utilities
# ---------------------------------------------------------------------

def test_drop_and_drop_where_are_immutable():
    ic = InfoCollection().append("X", a=1).append("Y", a=2).append("Z", a=3)

    ic2 = ic.drop("Y")
    assert names(ic) == ["X", "Y", "Z"]
    assert names(ic2) == ["X", "Z"]

    ic3 = ic.drop_where(lambda info: "a" in info.info and info.info["a"] >= 2)
    assert names(ic3) == ["X"]

    ic4 = ic.drop(["NOPE"], missing="ignore")
    assert names(ic4) == ["X", "Y", "Z"]

    with pytest.raises(KeyError):
        _ = ic.drop(["NOPE"], missing="error")


# ---------------------------------------------------------------------
# (De)serialization of data-only view
# ---------------------------------------------------------------------

def test_serialize_and_deserialize_data_view():
    ic = InfoCollection().append("A", a=1).append("B", x="y")
    s = ic._serialize()
    data = json.loads(s)
    assert data == {"A": {"a": 1}, "B": {"x": "y"}}

    ic2 = InfoCollection._deserialize(s)
    assert body(ic2) == {"A": {"a": 1}, "B": {"x": "y"}}
    # data-only (de)serialization does not carry attrs
    assert ic2["A"]._is_schema_level is False
    assert ic2["B"]._is_schema_level is False


# ---------------------------------------------------------------------
# Snapshot API (attrs-preserving round-trip)
# ---------------------------------------------------------------------

def test_snapshot_roundtrip_preserves_member_attrs():
    ic = InfoCollection().set_schema(mode="enforced", keys={"a", "b"})
    ic = ic.append("S", a=1)
    ic = ic.append("F", note="x")

    snap = ic._serialize_info_collection_snapshot()
    ic2 = InfoCollection._deserialize_info_collection_snapshot(snap)

    assert body(ic2) == {"S": {"a": 1}, "F": {"note": "x"}}

    s2 = ic2["S"]
    f2 = ic2["F"]
    assert s2._is_schema_level is True
    assert f2._is_schema_level is False


# ---------------------------------------------------------------------
# In-place vs immutable: set_schema is in-place; append/update return new
# ---------------------------------------------------------------------

def test_set_schema_is_in_place():
    ic = InfoCollection()
    ic2 = ic.set_schema(mode="enforced", keys={"k"})
    assert ic2 is ic
    assert ic._schema_keys == {"k"}

    ic3 = ic.append("K", k=1)
    assert ic3 is not ic
    assert body(ic3) == {"K": {"k": 1}}
