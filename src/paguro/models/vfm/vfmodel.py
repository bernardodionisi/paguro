from __future__ import annotations

import warnings
import inspect
import types
from typing import (
    Mapping, ClassVar, Any,
    Sequence, Union, TYPE_CHECKING
)

import polars as pl

from paguro.models.vfm.decorators.constraint import _collect_constraints_from_namespace
from paguro.models.vfm.decorators.transformed import _collect_transforms_from_namespace
from paguro.models.vfm.utils import (
    _ensure_col_name_dtype,
    _with_struct_fields,
    _synth_struct_column,
    _strict_align_name,
    _read_config,
    _get_frame_name,
    _skip_attr,
    _get_forbidden_set,
    _validate_forbidden_attribute_names,
    _resolve_annotations_preserving_order,
    _unwrap_annotation,
    _is_validcolumn_annotation
)

from paguro.validation.valid_column.valid_column import ValidColumn
from paguro.validation.valid_frame.valid_frame import ValidFrame

if TYPE_CHECKING:
    from paguro.models.vfm.utils import VFrameModelConfig

IncludeItem = Union[type, tuple[type, str]]

__all__ = [
    "VFrameModel",
]


def _choose_name_from_default(
        *,
        owner: type,
        attr: str,
        default_col: ValidColumn,
        cfg: Mapping[str, Any],
) -> tuple[str, Any, bool]:
    _def_name = default_col._name
    _prefer_default = bool(cfg.get("prefer_default_name", False))
    _allow_non_str = bool(cfg.get("allow_non_string_default_name", False))

    if _def_name is None:
        return attr, attr, True

    if not isinstance(_def_name, str):
        if not _allow_non_str:
            msg = (
                f"{owner.__name__}.{attr}: "
                f"default ValidColumn has non-string name {_def_name!r}; "
                "set model_config.allow_non_string_default_name = True "
                "to allow this."
            )
            raise TypeError(msg)
        return attr, _def_name, False

    if not _prefer_default:
        if _def_name == attr:
            return attr, attr, False
        msg = (
            f"{owner.__name__}.{attr}: "
            f"default ValidColumn name '{_def_name}' "
            f"does not match attribute name '{attr}'. "
            "Either rename the default to match, omit its name, "
            "or set model_config.prefer_default_name = True."
        )
        raise ValueError(msg)

    if _def_name:
        return _def_name, _def_name, True
    return attr, attr, True


# =============================================================================
# INHERITANCE / INCLUDES / CONSTRAINTS / TRANSFORMED
# =============================================================================

def _collect_inherited_state(
        cls: type,
) -> tuple[
    dict[str, ValidColumn],
    dict[str, pl.Expr],
    dict[str, str],
    tuple[ValidFrame, ...],
]:
    cols: dict[str, ValidColumn] = {}
    cons: dict[str, pl.Expr] = {}
    origins: dict[str, str] = {}
    frames_accum: list[ValidFrame] = []

    # base: VFrameModel
    for base in reversed(cls.__mro__[1:]):
        base_cols = getattr(base, "_valid_columns_tuple", None)
        if base_cols:
            for c in base_cols:
                cols[c._name] = c
                origins[c._name] = base.__name__

        valid_frame = getattr(base, "_valid_frame", None)
        if valid_frame is not None:
            base_cons = getattr(valid_frame, "_constraints", None)
            if isinstance(base_cons, dict) and base_cons:
                cons.update(base_cons)

        base_frames = getattr(base, "_valid_frames_tuple", None)
        if base_frames:
            frames_accum.extend(base_frames)

    return cols, cons, origins, tuple(frames_accum)


def _apply_includes(
        cls: type[VFrameModel],
        *,
        base_cols: Mapping[str, ValidColumn],
        base_origins: Mapping[str, str],
) -> tuple[
    dict[str, ValidColumn],
    dict[str, str]
]:
    merged: dict[str, ValidColumn] = dict(base_cols)
    origins: dict[str, str] = dict(base_origins)

    raw_include = getattr(cls, "__include__", ()) or ()
    if not isinstance(raw_include, (tuple, list)):
        msg = (
            f"{cls.__name__}.__include__ must be a sequence of VFrameModel types "
            "or (VFrameModel, prefix) tuples"
        )
        raise TypeError(msg)

    # Normalize to a list of (model, prefix)
    normalized: list[tuple[type[VFrameModel], str]] = []
    for item in raw_include:
        if isinstance(item, tuple):
            if len(item) != 2:
                msg = (
                    f"{cls.__name__}.__include__: "
                    f"tuple items must be (Model, 'prefix')"
                )
                raise TypeError(msg)
            model, prefix = item
            if not (isinstance(model, type) and issubclass(model, VFrameModel)):
                msg = (
                    f"{cls.__name__}.__include__: "
                    f"first element must be a VFrameModel subclass, got {model!r}"
                )
                raise TypeError(msg)
            if not isinstance(prefix, str):
                msg = (
                    f"{cls.__name__}.__include__: "
                    f"prefix must be a str, got {type(prefix).__name__}"
                )
                raise TypeError(msg)
            normalized.append((model, prefix))
        else:
            # plain model, empty prefix
            model = item
            if not (isinstance(model, type) and issubclass(model, VFrameModel)):
                msg = (
                    f"{cls.__name__}.__include__: "
                    f"items must be VFrameModel subclasses "
                    f"or (VFrameModel, str) tuples; got {item!r}"
                )
                raise TypeError(msg)
            normalized.append((model, ""))

    # Copy columns with collision checks
    for inc_model, pref in normalized:
        for c in inc_model._valid_columns_tuple:
            new_name = f"{pref}{c._name}"
            if new_name in merged:
                prev = origins.get(new_name, "unknown origin")
                msg = (
                    f"{cls.__name__}: include collision for column '{new_name}'. "
                    f"Already from {prev}, also from include {inc_model.__name__}."
                )
                raise ValueError(msg)
            merged[new_name] = _strict_align_name(
                owner=cls,
                attr=new_name,
                vcol=c,
                expected_name=new_name,
            )
            origins[new_name] = f"include:{inc_model.__name__}"

    return merged, origins


# CLASS-BODY PROCESSING (COLUMNS)

# --- VCol-bound method → ValidColumn

def _validcolumn_from_vcol_method(annot: Any) -> ValidColumn | None:
    """
    If `annot` is a bound method owned b
    y class `VCol` (or an instance of VCol),
    call it and return the resulting ValidColumn;
    else return None.
    """
    if not inspect.ismethod(annot):
        return None
    owner = getattr(annot, "__self__", None)
    if owner is None:
        return None

    owner_type = owner if isinstance(owner, type) else type(owner)
    # If can import VCol, replace the name check with:
    #   from paguro.validation.valid_column.vcol import VCol
    #   if not (owner is VCol or isinstance(owner, VCol)):
    if getattr(owner_type, "__name__", "") != "VCol":
        return None

    result = annot()
    if not isinstance(result, ValidColumn):
        msg = (
            f"VCol-bound annotation must return a ValidColumn, "
            f"got {type(result).__name__}."
        )
        raise TypeError(msg)
    return result


# --- Builders: produce (col, key)

def _build_from_validcolumn_default(
        *,
        owner_cls: type,
        attr: str,
        default_col: ValidColumn,
        cfg: Mapping[str, Any],
) -> tuple[ValidColumn, str]:
    """Normalize a ValidColumn default into (column, external_key)."""
    ext, internal_target, allow_rename = _choose_name_from_default(
        owner=owner_cls,
        attr=attr,
        default_col=default_col,
        cfg=cfg,
    )

    col = _ensure_col_name_dtype(
        default_col,
        name=internal_target,
        dtype=getattr(default_col, "_dtype", None),
        allow_rename=allow_rename,
        owner=owner_cls,
        attr=attr,
    )
    return col, ext


def _build_from_struct_annotation(
        *,
        owner_cls: type,
        attr: str,
        model_cls: type[VFrameModel],
        default: ValidColumn | None,
        cfg: Mapping[str, Any],
) -> tuple[ValidColumn, str]:
    """Annotation is a VFrameModel subclass → struct column."""
    nested = model_cls._valid_frame
    if default is not None and not isinstance(default, ValidColumn):
        msg = (
            f"{owner_cls.__name__}.{attr}: "
            f"struct columns may default to a ValidColumn or None; "
            f"got {type(default)!r}."
        )
        raise TypeError(msg)

    if isinstance(default, ValidColumn):
        col, ext = _build_from_validcolumn_default(
            owner_cls=owner_cls,
            attr=attr,
            default_col=default,
            cfg=cfg,
        )
        col = _with_struct_fields(
            col, nested,
            name=getattr(col, "_name", None),
            allow_rename=False,
            owner=owner_cls,
            attr=attr,
        )
        return col, ext

    # no default → synthesize struct column
    col = _synth_struct_column(attr, nested)
    return col, attr


def _build_from_plain_annotation(
        *,
        owner_cls: type,
        attr: str,
        default: ValidColumn | None,
) -> tuple[ValidColumn, str]:
    """Annotation is ValidColumn (type) → must not have default."""
    if isinstance(default, ValidColumn):
        msg = (
            f"{owner_cls.__name__}.{attr}: "
            f"provide either an annotation (type[ValidColumn]) "
            f"or a default (ValidColumn instance), not both."
        )
        raise TypeError(msg)
    return ValidColumn(name=attr), attr


def _insert_column(
        *,
        owner_cls: type,
        attr: str,
        key: str,
        col: ValidColumn,
        merged_cols: dict[str, ValidColumn],
        origins: dict[str, str],
        seen_internal_names: dict[str, str],
        allow_override: bool,
) -> None:
    """Apply override policy, maintain internal-name registry, assign attr, fill maps."""
    if key in merged_cols:
        if not allow_override:
            prev = origins.get(key, "inherited/include")
            msg = (
                f"{owner_cls.__name__}: column name '{key}' collides with {prev}, "
                f"and model_config.allow_attribute_override is False."
            )
            raise ValueError(msg)
        _prev_internal = getattr(merged_cols[key], "_name", None)
        if isinstance(_prev_internal, str):
            seen_internal_names.pop(_prev_internal, None)

    internal = getattr(col, "_name", None)
    if internal is not None:
        prev_attr = seen_internal_names.get(internal)
        if prev_attr is not None and prev_attr != attr:
            msg = (
                f"{owner_cls.__name__}: "
                f"internal column name '{internal}' used by multiple attributes "
                f"('{prev_attr}' and '{attr}')."
            )
            raise ValueError(msg)
        seen_internal_names[internal] = attr

    merged_cols[key] = col
    origins[key] = f"class:{owner_cls.__name__}"
    setattr(owner_cls, attr, col)


def _process_one_annotation(
        *,
        owner_cls: type,
        attr: str,
        raw_annot: Any,
        default: Any,
        merged_cols: dict[str, ValidColumn],
        origins: dict[str, str],
        seen_attr_names: set[str],
        seen_internal_names: dict[str, str],
        cfg: Mapping[str, Any],
) -> None:
    if _skip_attr(attr):
        return

    if attr in seen_attr_names:
        msg = (
            f"{owner_cls.__name__}: "
            f"duplicate attribute name '{attr}' in the same class body."
        )
        raise ValueError(msg)
    seen_attr_names.add(attr)

    annot = _unwrap_annotation(raw_annot)
    allow_override = bool(cfg.get("allow_attribute_override", True))

    # 1) Struct column (annotation is VFrameModel subclass)
    if inspect.isclass(annot) and issubclass(annot, VFrameModel):
        col, key = _build_from_struct_annotation(
            owner_cls=owner_cls, attr=attr, model_cls=annot, default=default, cfg=cfg
        )
        _insert_column(
            owner_cls=owner_cls,
            attr=attr,
            key=key,
            col=col,
            merged_cols=merged_cols,
            origins=origins,
            seen_internal_names=seen_internal_names,
            allow_override=allow_override,
        )
        return

    # 2) Plain column via annotation (type[ValidColumn]) — MUST NOT have default
    if _is_validcolumn_annotation(annot):
        col, key = _build_from_plain_annotation(
            owner_cls=owner_cls,
            attr=attr,
            default=default,
        )

        _insert_column(
            owner_cls=owner_cls,
            attr=attr,
            key=key,
            col=col,
            merged_cols=merged_cols, origins=origins,
            seen_internal_names=seen_internal_names,
            allow_override=allow_override,
        )
        return

    # 0) VCol-bound method case → call it → ValidColumn default path
    try:
        vc_from_method = _validcolumn_from_vcol_method(annot)
    except Exception as e:
        msg = (

            f"{owner_cls.__name__}.{attr}:"
            f" failed evaluating vcol-bound annotation."
        )
        raise TypeError(msg) from e

    if vc_from_method is not None:
        if default is not None:
            msg = (
                f"{owner_cls.__name__}.{attr}: "
                f"Please either provide an annotation or a default value, "
                f"not both. "
            )
            raise TypeError(msg)
        col, key = _build_from_validcolumn_default(
            owner_cls=owner_cls, attr=attr, default_col=vc_from_method, cfg=cfg
        )
        _insert_column(
            owner_cls=owner_cls, attr=attr, key=key, col=col,
            merged_cols=merged_cols, origins=origins,
            seen_internal_names=seen_internal_names,
            allow_override=allow_override,
        )
        return

    # 3) Unsupported

    # 3) Unsupported
    _handle_unknown_annotation(
        owner_cls=owner_cls,
        attr=attr,
        raw_annot=raw_annot,
        cfg=cfg,
    )
    return  # for "warn"/"ignore"


def _handle_unknown_annotation(
        *,
        owner_cls: type,
        attr: str,
        raw_annot: Any,
        cfg: Mapping[str, Any],
) -> None:
    policy = (cfg.get("unknown_annotation") or "error").lower()
    if policy == "ignore":
        return
    if policy == "warn":
        warnings.warn(
            f"{owner_cls.__name__}.{attr}: "
            f"unsupported annotation {raw_annot!r}; ignoring.",
            category=UserWarning,
            stacklevel=3,
        )
        return
    # "error" (default)
    msg = (
        f"{owner_cls.__name__}.{attr}: unsupported annotation {raw_annot!r}. "
        "Allowed: `ValidColumn`, "
        "a `VFrameModel` subclass, "
        "a VCol-bound method returning ValidColumn, "
        "or no annotation with a ValidColumn default."
    )
    if isinstance(raw_annot, str):
        msg += (
            "\nHint: avoid string annotations or disable "
            "`from __future__ import annotations` here."
        )
    raise TypeError(msg)


# ----------------------------------------------------------------------


def _process_unannotated_defaults(
        *,
        owner_cls: type,
        ns: dict[str, Any],
        merged_cols: dict[str, ValidColumn],
        origins: dict[str, str],
        already_annotated: set[str],
        seen_internal_names: dict[str, str],
        cfg: Mapping[str, Any],
) -> None:
    """
    Handle *unannotated* class attributes that specify columns.

    Accepted shapes:
      - A ValidColumn instance
      - A bound method on VCol that returns a ValidColumn

    Everything else is ignored (optionally warn/error via config).
    """
    allow_override = bool(cfg.get("allow_attribute_override", True))
    anns: dict[str, Any] = ns.get("__annotations__", {}) or {}

    # preserve class-body order
    for attr in ns.keys():
        if _skip_attr(attr) or attr in already_annotated:
            continue

        val = ns.get(attr, None)

        # --- Case A: plain ValidColumn instance default
        if isinstance(val, ValidColumn):
            if attr in anns:
                raise TypeError(
                    f"{owner_cls.__name__}.{attr}: provide either an annotation "
                    f"(type[ValidColumn]) or a default (ValidColumn instance), not both."
                )

            col, key = _build_from_validcolumn_default(
                owner_cls=owner_cls, attr=attr, default_col=val, cfg=cfg
            )
            _insert_column(
                owner_cls=owner_cls, attr=attr, key=key, col=col,
                merged_cols=merged_cols, origins=origins,
                seen_internal_names=seen_internal_names,
                allow_override=allow_override,
            )
            continue

        # --- Case B: bound VCol method returning a ValidColumn
        try:
            vc_from_method = _validcolumn_from_vcol_method(val)
        except Exception as e:
            msg = (f"{owner_cls.__name__}.{attr}: "
                   f"failed evaluating VCol-bound default.")
            raise TypeError(msg) from e

        if vc_from_method is not None:
            if attr in anns:
                msg = (
                    f"{owner_cls.__name__}.{attr}: "
                    f"VCol-bound default returns a ValidColumn, "
                    "so no annotation may be provided as well."
                )
                raise TypeError(msg)

            col, key = _build_from_validcolumn_default(
                owner_cls=owner_cls,
                attr=attr,
                default_col=vc_from_method,
                cfg=cfg,
            )
            _insert_column(
                owner_cls=owner_cls, attr=attr, key=key, col=col,
                merged_cols=merged_cols, origins=origins,
                seen_internal_names=seen_internal_names,
                allow_override=allow_override,
            )
            continue

        # _maybe_report_ignored_default(
        #     owner_cls=owner_cls,
        #     attr=attr,
        #     val=val,
        #     cfg=cfg,
        # )


def _is_probably_columnish(val: object) -> bool:
    """
    Heuristic: return True
    if this unannotated default *looks* like the user
    intended a column but we ignored it.
    Keep conservative to avoid noise.
    """
    if isinstance(val, ValidColumn):
        return True
    # bound VCol methods are handled elsewhere;
    # warn only for *unbound* VCol callables
    if isinstance(val, types.FunctionType):
        qn = getattr(val, "__qualname__", "")
        # e.g., "VCol.some_factory" — user forgot to bind/call?
        if qn.startswith("VCol.") or "VCol." in qn:
            return True
    # common footgun: a ValidColumn *class* (type), not instance
    if val is ValidColumn:
        return True
    return False


def _maybe_report_ignored_default(
        *,
        owner_cls: type,
        attr: str,
        val: object,
        cfg: Mapping[str, object],
) -> None:
    """Warn or error if configured and the value looks like a mistaken column."""
    if not _is_probably_columnish(val):
        return

    if bool(cfg.get("error_on_ignored_defaults", False)):
        msg = (
            f"{owner_cls.__name__}.{attr}: "
            f"unannotated default was ignored but looks like a column "
            f"or VCol factory; add a proper annotation or bind/call it."
        )
        raise TypeError(msg)

    if bool(cfg.get("warn_on_ignored_defaults", False)):
        warnings.warn(
            f"{owner_cls.__name__}.{attr}: "
            f"unannotated default ignored. "
            f"If this was intended as a column, "
            f"annotate it or bind/call it.",
            category=UserWarning,
            stacklevel=3,
        )


def _process_local_annotations(
        *,
        owner_cls: type,
        ns: dict[str, Any],
        merged_cols: dict[str, ValidColumn],
        origins: dict[str, str],
        cfg: Mapping[str, Any],
) -> None:
    ordered = _resolve_annotations_preserving_order(owner_cls, ns)
    seen_attrs: set[str] = set()

    # PRE-SEED internal-name tracker with inherited/include internals
    seen_internal: dict[str, str] = {}
    for _k, _c in merged_cols.items():
        _iname = getattr(_c, "_name", None)
        if isinstance(_iname, str):
            seen_internal[_iname] = f"inherited:{_k}"

    for attr, raw_annot in ordered:
        default = ns.get(attr, None)
        _process_one_annotation(
            owner_cls=owner_cls,
            attr=attr,
            raw_annot=raw_annot,
            default=default,
            merged_cols=merged_cols,
            origins=origins,
            seen_attr_names=seen_attrs,
            seen_internal_names=seen_internal,
            cfg=cfg,
        )
    _process_unannotated_defaults(
        owner_cls=owner_cls,
        ns=ns,
        merged_cols=merged_cols,
        origins=origins,
        already_annotated=seen_attrs,
        seen_internal_names=seen_internal,
        cfg=cfg,
    )


# Build focal ValidFrame (finalization)

def _clean_local_class_doc(cls: type) -> str | None:
    """
    Return the docstring.

    Docstring declared *on this class body*
    (not inherited), cleaned, or None
    ."""
    raw = cls.__dict__.get("__doc__", None)  # only local; ignores bases
    if raw is None:
        return None
    try:
        doc = inspect.cleandoc(raw)
        return doc or None
    except Exception:
        return None


def _finalize_valid_frame_and_columns(
        *,
        owner_cls: type[VFrameModel],
        merged_cols: dict[str, ValidColumn],
        constraints: Mapping[str, pl.Expr],
        transformed_frames: tuple[ValidFrame, ...],
        cfg: Mapping[str, Any],
        constraint_docs: Mapping[str, str],
) -> None:
    valid_frame_name = _get_frame_name(owner_cls, cfg)
    cols_tuple = tuple(merged_cols[k] for k in merged_cols.keys())

    owner_cls._valid_columns_tuple = cols_tuple
    owner_cls._valid_frames_tuple = transformed_frames

    # Build the focal valid_frame
    vf = ValidFrame(
        *cols_tuple,
        *transformed_frames,
        transform=None,
        name=valid_frame_name,
        unique=None,
        **constraints
    )

    # Title always; description only if locally specified on this class.
    class_title = owner_cls.__name__
    class_desc = _clean_local_class_doc(owner_cls)  # <-- use local-only doc

    if class_desc is not None:
        vf = vf.with_info(
            title=class_title,
            description=class_desc,
            constraints=dict(constraint_docs) if constraint_docs else None,
        )
    else:
        vf = vf.with_info(
            title=class_title,
            constraints=dict(constraint_docs) if constraint_docs else None,
        )

    owner_cls._valid_frame = vf


# ----------------------------------------------------------------------

# todo: remove this, we might just delete the include, unclear if very useful
def _normalize_include(cls) -> list[tuple[type, str]]:
    raw = getattr(cls, "__include__", ()) or ()
    out: list[tuple[type, str]] = []
    for item in raw:
        if isinstance(item, tuple):
            model, prefix = item
            if not (isinstance(model, type) and issubclass(model, VFrameModel)):
                continue
            if not isinstance(prefix, str):
                continue
            out.append((model, prefix))
        else:
            model = item
            if not (isinstance(model, type) and issubclass(model, VFrameModel)):
                continue
            out.append((model, ""))
    return out


# =============================================================================

class _VFrameMeta(type):
    def __new__(mcls, name, bases, ns, **kw):
        # no locals here
        return _assemble_vc_class(
            super().__new__(mcls, name, bases, dict(ns)),
            ns,
        )

    def __str__(cls) -> str:
        lines: list[str] = [f"{cls.__name__}"]

        # Real attributes (incl. inherited)
        real_attrs = inspect.getmembers(
            cls, lambda o: isinstance(o, ValidColumn)
        )
        for attr, col in real_attrs:
            internal = col._name
            if isinstance(internal, str) and internal != attr:
                lines.append(f"  {attr} ({internal!r})")
            else:
                lines.append(f"  {attr}")

        # Included block
        includes = _normalize_include(cls)
        if includes:
            lines.append("")
            lines.append("  included")
            real_attr_names = {name for name, _ in real_attrs}

            for model, prefix in includes:
                lines.append(f"    {model.__name__}:")
                for inc_attr, inc_col in inspect.getmembers(
                        model,
                        lambda o: isinstance(o, ValidColumn)
                ):
                    internal = getattr(inc_col, "_name", None)
                    if not isinstance(internal, str):
                        continue
                    external = f"{prefix}{internal}"
                    if external in real_attr_names:
                        continue  # overridden locally
                    if prefix:
                        lines.append(f"      {prefix}{inc_attr} ({internal!r})")
                    else:
                        lines.append(f"      {inc_attr} ({internal!r})")

        return "\n".join(lines)


def _assemble_vc_class(cls: type, ns: dict[str, object], ) -> type:
    cfg = _read_config(cls)

    # Fail fast on forbidden public names (dunders/private allowed)
    forbidden = _get_forbidden_set(cfg)
    _validate_forbidden_attribute_names(cls, ns, forbidden)

    (inherited_cols,
     inherited_cons,
     origins,
     inherited_frames) = _collect_inherited_state(cls=cls)

    merged_cols, origins = _apply_includes(
        cls=cls,
        base_cols=inherited_cols,
        base_origins=origins,
    )

    _process_local_annotations(
        owner_cls=cls,
        ns=ns,
        merged_cols=merged_cols,
        origins=origins,
        cfg=cfg,
    )

    # Constraints: gather locals
    local_exprs, local_docs = _collect_constraints_from_namespace(
        ns,
        owner_cls=cls,
    )

    # Enforce override policy for constraints
    if not bool(cfg.get("allow_attribute_override", True)):
        _overlap = set(inherited_cons).intersection(local_exprs)
        if _overlap:
            names = sorted(_overlap)
            msg = (
                f"{cls.__name__}: constraint(s) {names} would override inherited ones, "
                f"set _model_config = VFrameModelConfig(allow_attribute_override=True) "
                f"to allow attribute override."
            )
            raise ValueError(msg)

    # Merge constraints (locals take precedence when allowed)
    constraints = dict(inherited_cons)
    constraints.update(local_exprs)

    local_frames = _collect_transforms_from_namespace(
        ns, owner_cls=cls,
    )
    all_frames = inherited_frames + local_frames

    _finalize_valid_frame_and_columns(
        owner_cls=cls,
        merged_cols=merged_cols,
        constraints=constraints,
        transformed_frames=all_frames,
        cfg=cfg,
        constraint_docs=local_docs,
    )
    return cls


class VFrameModel(metaclass=_VFrameMeta):
    """
    Base model for specifying a ValidFrame validator.

    A model-based alternative to :obj:`vframe <paguro.validation.vframe>`
    for constructing a :obj:`ValidFrame <paguro.validation.ValidFrame>`
    validator.

    Defining a model
    ================

    Define a model similarly to how you would in `pydantic`
    or any other validation library that supports a model based API.

    The `base model` for your tabular data is
    :obj:`VFrameModel <paguro.models.vfm.VFrameModel>`,
    which allows you to represents:

    - column validators using :obj:`vcol <paguro.validation.vcol>`
    - cross-column validation rules using the `@constraint` decorator
    - validation rules over transformation of your data using the `@transformed` decorator

    Example
    =======

    .. paguro-example:: Simple Example: OrdersModel

        .. ipython:: python

            import paguro as pg
            from paguro.models import vfm

        .. paguro-example:: Failing Validation
            :collapsible: open

            .. ipython:: python

                class OrdersModel(vfm.VFrameModel):
                    customer_id = pg.vcol.String(str_starts_with="CUSTOMER")
                    order_date = pg.vcol.Date()
                    total_amount = pg.vcol(ge=0)

            .. ipython:: python

                print(orders) # orders is defined in the tutorials page

            .. ipython:: python

                try:
                    orders = pg.Dataset(orders).with_model(OrdersModel)
                except pg.exceptions.ValidationError as e:
                    print(e)
    """

    # Plain columns:
    #   - EITHER annotate as `ValidColumn` (type) with no default
    #       → instance uses the attribute name;
    #   - OR omit annotation and provide a `ValidColumn(...)`
    #       default instance (external name chosen by Config);
    #   - Providing both is an error.
    #
    # Struct columns:
    #   - Annotate with another `VFrameModel` subclass.
    #   - Default may be a ValidColumn or None;
    #       naming follows the same rules for the outer field.
    #
    # Notes:
    #   - Public attribute names that collide with core/machinery
    #       or any non-dunder attribute of
    #       `ValidColumn` are forbidden and will raise at class creation time.
    #   - Dunder methods (e.g., __repr__, __str__)
    #       and private names (_something) are allowed.
    #   - The decorator names 'constraint' and 'transformed'
    #       are intentionally not forbidden.

    __include__: ClassVar[Sequence[IncludeItem]] = ()
    # currently we do not bind included members
    # to class attributes, think it over first

    # Users may override on subclasses:
    _model_config: ClassVar[VFrameModelConfig | dict[str, Any] | None] = None

    _valid_columns_tuple: ClassVar[tuple[ValidColumn, ...]]
    _valid_frames_tuple: ClassVar[tuple[ValidFrame, ...]]
    _valid_frame: ClassVar[ValidFrame]

    def __call__(self) -> pl.Expr:  # type: ignore[empty-body]
        # this has no effect and it is defined here for typing purposes
        pass
