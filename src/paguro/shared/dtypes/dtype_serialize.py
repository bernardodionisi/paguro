from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, TypedDict, cast, overload, TypeAlias, Union

import polars as pl
from polars.datatypes import DataTypeClass, Field

if TYPE_CHECKING:
    import sys

    from polars import DataType

    if sys.version_info >= (3, 11):
        from typing import NotRequired
    else:
        from typing_extensions import NotRequired


# -------------------- Serialized dtype payloads (open-ended) --------------------


class DTypeDict(TypedDict, total=True):
    """
    DTypeDict.

    _dtype: human-readable tag (eg "Int64", "Struct", "MyExtType")
    _path:  fully-qualified class path for the *base type* (eg "polars.datatypes.Int64")
            used to re-import user/extension types. Optional, but recommended.
    _attrs: per-dtype attributes (eg fields/inner/shape/precision/scale/categories,
            plus any public, JSON-serializable
             or nested-dtype attributes for extension types).
    """

    _dtype: str
    _path: NotRequired[str]
    _attrs: NotRequired[dict[str, Any]]


# -------------------- Helpers --------------------


def _class_path(cls: type[Any]) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _import_by_path(path: str) -> Any:
    mod_name, _, qual = path.rpartition(".")
    if not mod_name:
        msg = f"Invalid path: {path!r}"
        raise ImportError(msg)
    mod = importlib.import_module(mod_name)
    obj: Any = mod
    for part in qual.split("."):
        obj = getattr(obj, part)
    return obj


def _dtype_base_class(dtype: DataType | DataTypeClass) -> type[Any]:
    """
    Returns the class object representing the base type.

    Works for both DataTypeClass and instantiated DataType.
    """
    base = dtype.base_type()  # type: ignore[call-arg]
    # base can be a class or an instance; normalize to a class
    return base if isinstance(base, type) else type(base)


# ---- JSON / attr (de)serialization helpers for extension attrs ----

JSONScalar: TypeAlias = Union[int, float, bool, str, None]
JSONLike: TypeAlias = Union[JSONScalar, list["JSONLike"], dict[str, "JSONLike"],]


def _to_jsonlike(v: Any) -> JSONLike | None:
    """Conservatively convert v into JSON-like data; return None if unsupported."""
    if isinstance(v, (int, float, bool, str)) or v is None:
        return v  # type: ignore[return-value]
    if isinstance(v, (list, tuple)):
        out_list: list[JSONLike] = []
        for item in v:
            j = _to_jsonlike(item)
            if j is None:
                return None
            out_list.append(j)
        return out_list
    if isinstance(v, dict):
        out_map: dict[str, JSONLike] = {}
        for k, item in v.items():
            if not isinstance(k, str):
                return None
            j = _to_jsonlike(item)
            if j is None:
                return None
            out_map[k] = j
        return out_map
    if isinstance(v, pl.Series):
        # Make stable/JSON
        return _to_jsonlike(v.to_list())
    return None


def _serialize_attr_value(v: Any, *, instantiate: bool) -> Any:
    """
    Serialize attr value.

    Accept JSON-like values; convert polars dtypes (or collections containing them)
    into nested DTypeDicts; drop truly unserializable objects.
    """
    if isinstance(v, (pl.DataType, DataTypeClass)):
        return pl_dtype_to_dict(v, instantiate=instantiate)

    j = _to_jsonlike(v)
    if j is not None:
        return j

    if isinstance(v, (list, tuple)):
        return [
            _serialize_attr_value(x, instantiate=instantiate) for x in v
        ]

    if isinstance(v, dict):
        out_map: dict[str, Any] = {}
        for k, x in v.items():
            if not isinstance(k, str):
                continue
            out_map[k] = _serialize_attr_value(x, instantiate=instantiate)
        return out_map

    # Unknown/unserializable -> drop; or replace with str(v) if desired
    return None


def _convert_nested_attrs(
        attrs: dict[str, Any],
) -> dict[str, Any]:
    """Turn any embedded DTypeDicts back into DataTypes, recursing lists/dicts."""

    def conv(v: Any) -> Any:
        if isinstance(v, dict):
            if "_dtype" in v:
                return dict_to_pl_dtype(cast("DTypeDict", v))
            return {k: conv(x) for k, x in v.items()}
        if isinstance(v, list):
            return [conv(x) for x in v]
        return v

    return {k: conv(v) for k, v in attrs.items()}


# -------------------- Public helpers --------------------


def dtype_to_dict_list(
        dtypes: list[DataType | DataTypeClass] | None,
) -> list[DTypeDict] | None:
    if dtypes is None:
        return None
    return [pl_dtype_to_dict(dt) for dt in dtypes]


def dict_to_dtype_list(
        items: list[DTypeDict],
) -> list[DataType | DataTypeClass]:
    return [dict_to_pl_dtype(item) for item in items]


# ---- Core (serialize)


def pl_dtype_to_dict(
        dtype: DataType | DataTypeClass,
        *,
        instantiate: bool = False,
) -> DTypeDict:
    """
    Convert a Polars DataType/DataTypeClass into a minimal, stable dict.

    Never returns None. No Field-at-top-level handling (Field is not a dtype).
    """
    if instantiate:
        # Instantiate DataTypeClass so attributes like .inner/.fields are present
        if isinstance(dtype, DataTypeClass):
            try:
                dtype = dtype()
            except TypeError:
                # Non-instantiable without params; leave as class-level behavior
                pass

    # Human-readable tag and fully-qualified base class path
    base_cls = _dtype_base_class(dtype)
    tag = str(dtype.base_type())  # type: ignore[call-arg]
    out: DTypeDict = {"_dtype": tag, "_path": _class_path(base_cls)}

    # ---- Explicit branches for containers ----
    if isinstance(dtype, pl.Struct):
        if not hasattr(dtype, "fields"):
            return out
        fields: dict[str, DTypeDict] = {
            f.name: pl_dtype_to_dict(f.dtype, instantiate=instantiate)
            for f in dtype.fields
        }
        out["_attrs"] = {"fields": fields}
        return out

    if isinstance(dtype, pl.List):
        if not hasattr(dtype, "inner"):
            return out
        out["_attrs"] = {
            "inner": pl_dtype_to_dict(dtype.inner, instantiate=instantiate)
        }
        return out

    if isinstance(dtype, pl.Array):
        if not (hasattr(dtype, "inner") and hasattr(dtype, "shape")):
            return out
        # Canonicalize: keep the top-level shape only and strip nested Array wrappers.
        base_inner = dtype.inner
        while isinstance(base_inner, pl.Array):
            if not hasattr(base_inner, "inner"):
                break
            base_inner = base_inner.inner

        out["_attrs"] = {
            "inner": pl_dtype_to_dict(base_inner, instantiate=instantiate),
            "shape": tuple(dtype.shape),
        }
        return out

    # ---- Explicit branches for Decimal and Enum ----
    if tag == "Decimal":
        dec_attrs: dict[str, Any] = {}
        precision = getattr(dtype, "precision", None)
        scale = getattr(dtype, "scale", None)
        if precision is not None:
            dec_attrs["precision"] = int(precision)
        if scale is not None:
            dec_attrs["scale"] = int(scale)
        if dec_attrs:
            out["_attrs"] = dec_attrs
        # continue to generic sweep

    if tag == "Enum":
        enum_attrs = out.get("_attrs", {}) or {}
        cats = getattr(dtype, "categories", None)
        if cats is not None:
            if isinstance(cats, pl.Series):
                enum_attrs["categories"] = cats.to_list()
            elif isinstance(cats, (list, tuple)):
                enum_attrs["categories"] = list(cats)
            out["_attrs"] = enum_attrs
        # continue to generic sweep

    # ---- Generic: gather additional public attrs (extensions/others) ----
    attrs: dict[str, Any] = dict(
        out.get("_attrs", {}) or {}
    )  # force dict type for mypy

    for name in dir(dtype):
        if name.startswith("_"):
            continue
        try:
            val = getattr(dtype, name)
        except Exception:
            continue
        if callable(val):
            continue

        ser = _serialize_attr_value(val, instantiate=instantiate)
        if ser is not None:
            attrs[name] = ser

    if attrs:
        out["_attrs"] = attrs

    return out


# -------------------- Core (deserialize) --------------------


@overload
def dict_to_pl_dtype(
        data: DTypeDict,
) -> DataType | DataTypeClass: ...


@overload
def dict_to_pl_dtype(data: str, ) -> DataType | DataTypeClass: ...


def dict_to_pl_dtype(data: DTypeDict | str, ) -> DataType | DataTypeClass:
    """
    Reconstruct a dtype strictly from our serialized input data.

    Deserialization order:
      1) If `data` is str, try pl.<name>() / pl.<name> (strict).
      2) If dict with `_path`, try importing that path (handles user/ext types).
      3) Fallback to pl.<_dtype>.
      4) Otherwise, raise TypeError.

    For nested types (Struct/List/Array), also reconstruct children from `_attrs`.
    """

    if isinstance(data, (pl.DataType, DataTypeClass)):
        return data

    if isinstance(data, str):
        if hasattr(pl, data):
            ctor = getattr(pl, data)
            try:
                return ctor()  # instance
            except TypeError:
                return ctor  # class
        msg = f"Invalid dtype tag: {data!r}"
        raise TypeError(msg)

    tag: str = data["_dtype"]
    path: str | None = data.get("_path")
    attrs: dict[str, Any] = data.get("_attrs", {}) or {}

    # ---- Containers ----
    if tag == "Struct":
        if not attrs:
            if path:
                try:
                    return _import_by_path(path)
                except Exception:
                    pass
            return pl.Struct
        fields_map = attrs.get("fields", {})
        if not isinstance(fields_map, dict):
            msg = "Struct._attrs['fields'] must be a mapping of name -> dtype"
            raise TypeError(msg)
        fields = [
            Field(name, dict_to_pl_dtype(field_dtype))
            for name, field_dtype in fields_map.items()
        ]
        return pl.Struct(fields)

    if tag == "List":
        if not attrs:
            if path:
                try:
                    return _import_by_path(path)
                except Exception:
                    pass
            return pl.List
        inner = attrs.get("inner")
        if inner is None:
            msg = "List._attrs['inner'] is required"
            raise TypeError(msg)
        return pl.List(dict_to_pl_dtype(inner))

    if tag == "Array":
        if not attrs:
            if path:
                try:
                    return _import_by_path(path)
                except Exception:
                    pass
            return pl.Array

        inner = attrs.get("inner")
        shape = attrs.get("shape")
        if inner is None or shape is None:
            msg = "Array requires both 'inner' and 'shape' in _attrs"
            raise TypeError(msg)

        inner_dt = dict_to_pl_dtype(inner)

        # Canonicalize: if inner_dt is an Array, strip nested Array wrappers.
        while isinstance(inner_dt, pl.Array):
            inner_dt = inner_dt.inner

        return pl.Array(inner_dt, tuple(shape))

    # ---- Decimal and Enum explicit handling ----
    if tag == "Decimal":
        if not attrs:
            if path:
                try:
                    return _import_by_path(path)
                except Exception:
                    pass
            return pl.Decimal
        fixed = _convert_nested_attrs(attrs)
        if path:
            try:
                ctor = _import_by_path(path)
                return ctor(**fixed)  # type: ignore[misc]
            except Exception:
                pass
        return pl.Decimal(**fixed)

    if tag == "Enum":
        if not attrs:
            if path:
                try:
                    return _import_by_path(path)
                except Exception:
                    pass
            return pl.Enum
        fixed = _convert_nested_attrs(attrs)
        if path:
            try:
                ctor = _import_by_path(path)
                return ctor(**fixed)  # type: ignore[misc]
            except Exception:
                pass
        return pl.Enum(**fixed)

    # ---- Generic (non-container): try by _path first, then pl.<_dtype> ----
    if path:
        try:
            ctor = _import_by_path(path)
            if attrs:
                return ctor(**_convert_nested_attrs(attrs))  # type: ignore[misc]
            return ctor  # preserve class form when no attrs
        except Exception:
            pass

    if hasattr(pl, tag):
        ctor = getattr(pl, tag)
        if attrs:
            return ctor(**_convert_nested_attrs(attrs))
        return ctor  # preserve class form when no attrs

    msg = f"Invalid dtype: {_class_path(type(tag)) if not isinstance(tag, str) else tag!r}"
    raise TypeError(msg)


# -------------------- Compare helper (unchanged API) --------------------


def compare_dicts(
        dict1: dict[str, Any], dict2: dict[str, Any]
) -> tuple[bool, bool]:
    """
    Returns (equal, subset).

    Special-case: if comparing under '_attrs' -> 'fields',
    keys must match exactly.
    """
    if dict1 == dict2:
        return True, False

    def is_subset(d1: Any, d2: Any, parent_key: str = "") -> bool:
        if not isinstance(d1, dict) or not isinstance(d2, dict):
            return False
        for key, value in d2.items():
            if key not in d1:
                return False
            if isinstance(value, dict):
                if key == "fields" and parent_key == "_attrs":
                    if set(d1[key].keys()) != set(value.keys()):
                        return False
                if not is_subset(d1[key], value, key):
                    return False
            elif d1[key] != value:
                return False
        return True

    return False, is_subset(dict1, dict2)
