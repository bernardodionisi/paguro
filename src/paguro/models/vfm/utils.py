from __future__ import annotations

import dataclasses
import inspect
import sys
import types
import typing
from typing import Any, Mapping, get_type_hints, get_origin, get_args

import polars as pl

from paguro.validation.validation import Validation
from paguro.validation.valid_column.utils._vdtypes import ValidStruct

from paguro.validation.valid_column.valid_column import ValidColumn
from paguro.validation.valid_frame.valid_frame import ValidFrame


@dataclasses.dataclass(slots=True)
class VFrameModelConfig:
    """
    Configuration for VFrameModel subclasses.

    Set it on the model as a class attribute:
        class MyModel(VFrameModel):
            _model_config = VFrameModelConfig(
                frame_name="my_frame",
                prefer_default_name=True,
                allow_attribute_override=False,
            )

    You may also assign a plain dict with the same keys.
    """
    # Name for the focal ValidFrame (defaults to class name when None/empty)
    frame_name: str | None = None

    # When a ValidColumn default carries a name, prefer it over the attribute name.
    prefer_default_name: bool = True

    # Permit non-string names on ValidColumn defaults (advanced)
    allow_non_string_default_name: bool = False

    # (Reserved for future) Collect constraint methods automatically
    collect_constraint_methods: bool = True

    # Extend the set of forbidden public attribute names (avoid overshadowing API)
    forbidden_names: set[str] | list[str] | tuple[str, ...] | None = None

    # Allow focal class to override inherited/include columns/constraints by name
    allow_attribute_override: bool = True

    unknown_annotation: typing.Literal["error", "warn", "ignore"] = "error"


_DEFAULT_FORBIDDEN_ATTRS = frozenset(
    {
        # internal machinery/state
        "_valid_frame",
        "_valid_columns_tuple",
        "_valid_frames_tuple",
        # common class-level attributes
        "__include__",
        # NOTE: 'model_config' | '_model_config' is intentionally NOT forbidden.
    }
) | frozenset(
    # Forbid all non-dunder attribute names of ValidColumn
    # to avoid overshadowing the API
    a for a in dir(ValidColumn)
    if not
    (a.startswith("__") and a.endswith("__"))
    # and a != "validate"
)

_DEFAULT_CONFIG = {
    "frame_name": None,
    "prefer_default_name": True,
    "allow_non_string_default_name": False,
    "collect_constraint_methods": True,
    "forbidden_names": None,
    "allow_attribute_override": True,
    "unknown_annotation": "error",
}


def _to_dtype_frozenset(dtype: Any) -> frozenset[pl.DataType] | None:
    """
    Normalize user/internal dtype into a frozenset of pl.DataType.
    Accepts:
      - None -> returns None
      - a single pl.DataType or DataTypeClass -> frozenset({pl_dtype})
      - an iterable of the above -> frozenset({...})
    """
    if dtype is None:
        return None

    # Single dtype (DataType or DataTypeClass)
    # TODO: fix DataTypeClass
    if isinstance(dtype, pl.DataType) or (
            getattr(dtype, "__name__", None) == "DataTypeClass"
    ):
        return frozenset({dtype})

    # Try iterable
    try:
        items = list(dtype)
    except TypeError:
        # Not iterable; treat as single
        return frozenset({dtype})

    out: set[pl.DataType] = set()
    for d in items:
        if d is None:
            continue
        out.add(d)
    return frozenset(out)


def _ensure_col_name_dtype(
        vcol: ValidColumn,
        *,
        name: Any,
        dtype: Any,
        allow_rename: bool = True,
        owner: type | None = None,
        attr: str | None = None,
) -> ValidColumn:
    _current = vcol._name
    _same_name = _names_equal(_current, name)

    incoming = _to_dtype_frozenset(dtype)
    existing = _to_dtype_frozenset(vcol._dtype)

    if not _same_name:
        if not allow_rename:
            _who = f"{owner.__name__}.{attr}" if owner and attr else "column"
            raise ValueError(
                f"{_who}: default ValidColumn name {(_current)!r} "
                f"does not match expected {name!r}. "
                "Either rename the default to match, omit its name, "
                "or enable model_config.prefer_default_name."
            )
        vcol._name = name

    if dtype is not None and incoming != existing:
        vcol._dtype = incoming
    elif (
            dtype is None
            and vcol._dtype is not None
            and existing != vcol._dtype
            # todo: fix this is should be a frozenset so compare wisely
    ):
        vcol._dtype = existing

    return vcol


def _with_struct_fields(
        vcol: ValidColumn,
        nested_frame: ValidFrame,
        *,
        name: Any,
        allow_rename: bool = True,
        owner: type | None = None,
        attr: str | None = None,
) -> ValidColumn:
    vcol = _ensure_col_name_dtype(
        vcol=vcol,
        name=name,
        dtype=pl.Struct,
        allow_rename=allow_rename,
        owner=owner,
        attr=attr,
    )
    vcol._fields = Validation(nested_frame)
    return vcol


def _names_equal(a: object, b: object) -> bool:
    """Safe equality for names; never triggers Polars Expr truthiness."""
    if a is b:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    return False


def _synth_struct_column(
        name: str,
        nested_frame: ValidFrame,
) -> ValidColumn:
    return _with_struct_fields(
        ValidStruct(name=name),
        nested_frame,
        name=name,
    )


def _strict_align_name(
        *,
        owner: type,
        attr: str,
        vcol: ValidColumn,
        expected_name: str,
) -> ValidColumn:
    _internal = vcol._name
    if _internal is not None and not _names_equal(_internal, expected_name):
        msg = (
            f"{owner.__name__}.{attr}: column name mismatch. "
            f"Attribute name is '{expected_name}', "
            f"but ValidColumn carries name='{_internal}'."
        )
        raise ValueError(msg)
    return _ensure_col_name_dtype(
        vcol,
        name=expected_name,
        dtype=vcol._dtype,
    )


def _clean_doc(obj: object) -> str | None:
    """Return a cleaned docstring or None (never raises)."""
    try:
        return inspect.getdoc(obj) or None
    except Exception:
        return None


def _read_config(cls: type) -> dict[str, Any]:
    """
    Load config with the following precedence:
      1) cls._model_config if it's a VFrameModelConfig instance
      2) cls._model_config if it's a dict of known keys
      4) defaults
    """
    cfg: dict[str, Any] = dict(_DEFAULT_CONFIG)

    # 1/2) model_config preferred
    mc = getattr(cls, "_model_config", None)
    if mc is None:
        mc = getattr(cls, "model_config", None)

    if mc is None:
        return cfg

    if isinstance(mc, VFrameModelConfig):
        for field in dataclasses.fields(VFrameModelConfig):
            cfg[field.name] = getattr(mc, field.name)
        return cfg
    if isinstance(mc, dict):
        for k in _DEFAULT_CONFIG:
            if k in mc:
                cfg[k] = mc[k]
        return cfg

    # 4) defaults
    return cfg


def _get_frame_name(cls: type, cfg: Mapping[str, Any]) -> str:
    _cfg_name = cfg.get("frame_name")
    if isinstance(_cfg_name, str) and _cfg_name:
        return _cfg_name
    return cls.__name__


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _skip_attr(name: str) -> bool:
    return _is_dunder(
        name) or name in _DEFAULT_FORBIDDEN_ATTRS or name == "_model_config"


def _get_forbidden_set(cfg: Mapping[str, Any]) -> set[str]:
    extra = cfg.get("forbidden_names")
    out = set(_DEFAULT_FORBIDDEN_ATTRS)
    if extra:
        out |= set(extra)
    return out


def _is_public_name(name: str) -> bool:
    # allow any dunder and any private (single leading underscore)
    return not name.startswith("_")


def _validate_forbidden_attribute_names(
        owner_cls: type,
        ns: Mapping[str, object],
        forbidden: set[str],
) -> None:
    """
    Forbid declaring PUBLIC attributes whose names collide with forbidden set.

    - Dunder (e.g., __repr__) and private (leading underscore) are allowed.
    - This runs before we process annotations/defaults to fail fast.
    """
    for attr in ns.keys():
        if not isinstance(attr, str):
            continue
        if not _is_public_name(attr):
            # Allow __magic__ and _private
            continue
        if attr in forbidden:
            raise ValueError(
                f"{owner_cls.__name__}: "
                f"attribute name '{attr}' is reserved/forbidden.\n"
                "Choose a different name to avoid "
                "shadowing core API or ValidColumn attributes."
            )


def _resolve_annotations_preserving_order(cls: type, ns: dict[str, Any]) -> list[
    tuple[str, Any]]:
    _raw = ns.get("__annotations__", {}) or {}
    try:
        _mod = sys.modules.get(cls.__module__)
        _resolved = get_type_hints(
            cls, globalns=getattr(_mod, "__dict__", {}), localns=None,
            include_extras=True
        )
    except Exception:
        _resolved = {}
    return [(n, _resolved.get(n, _raw[n])) for n in _raw.keys()]


def _unwrap_annotation(annot: Any) -> Any:
    _origin = get_origin(annot)
    if _origin is None:
        return annot

    if ((_origin is getattr(typing, "Annotated", None)) or (
            str(_origin) == "typing.Annotated")):
        _args = get_args(annot)
        return _unwrap_annotation(_args[0]) if _args else annot
    if ((_origin is typing.Union) or (str(_origin) == "typing.Union")):
        _args = tuple(get_args(annot))
        _non_none = [a for a in _args if a is not type(None)]  # noqa: E721
        if len(_non_none) == 1:
            return _unwrap_annotation(_non_none[0])
    return annot


def _is_validcolumn_annotation(annot: Any) -> bool:
    return (
            (annot is ValidColumn)
            or isinstance(annot, ValidColumn)
            or (annot == "ValidColumn")
    )  # todo: support all dtypes


def _return_is_pl_expr(fn: types.FunctionType) -> bool:
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = getattr(fn, "__annotations__", {}) or {}
    return_ann = hints.get("return")
    return (return_ann is pl.Expr) or (str(return_ann).endswith("Expr"))
