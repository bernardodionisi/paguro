from __future__ import annotations

import keyword
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TYPE_CHECKING, Literal

from paguro.shared.extra_utilities import _unnest_schema

if TYPE_CHECKING:
    from polars import DataFrame, LazyFrame, Field, DataType
    from paguro import Dataset, LazyDataset
    from paguro.shared.extra_utilities import SchemaTree

__all__ = ["collect_model_blueprint"]


def collect_model_blueprint(
        data: (
                DataFrame
                | LazyFrame
                | Dataset
                | LazyDataset
                | Mapping[str, DataFrame | LazyFrame | Dataset | LazyDataset]
        ),
        *,
        path: str | Path | None = None,
        root_class_name: str = "DatasetModel",
        include_dtypes: bool | Literal["as_values"] = False,
        allow_nulls: bool | None = False,
        print_usage_suggestion: bool = True,
) -> str | None:
    """
    Build a model template (Python module string) from one or many dataset-like objects.

    Naming rules:
      - Single dataset (or 1-element dict):
        no suffixing; only numeric dedupe for conflicts.
      - Multiple datasets (>1):
        suffix ALL child (struct) classes with the focal root name,
        concatenated in PascalCase (e.g., AddressCustomers).
        Numeric dedupe if needed.
    """
    # normalize into {root_name: SchemaTree}
    if isinstance(data, Mapping):

        # treat 1-element dict as single dataset per spec
        items = list(data.items())
        multi_roots = len(items) > 1
        schemas_by_root: dict[str, SchemaTree] = {}

        for k, v in items:
            schema = v.collect_schema()
            unnested: SchemaTree = _unnest_schema(schema=schema)
            root = _to_class_name(str(k))
            schemas_by_root[root] = unnested

    else:
        multi_roots = False
        schema = data.collect_schema()
        unnested = _unnest_schema(schema=schema)
        root = _to_class_name(root_class_name)
        schemas_by_root = {root: unnested}

    blueprint = _schemas_to_module(
        schemas_by_root=schemas_by_root,
        include_dtypes=include_dtypes,
        allow_nulls=allow_nulls,
        multi_roots=multi_roots,
    )
    root_names = list(schemas_by_root.keys())

    if path is None:
        return blueprint
    else:
        return _write_model_blueprint_to_python_file(
            blueprint=blueprint,
            path=path,
            model_class_names=root_names,
            print_usage_suggestion=print_usage_suggestion,
        )


def _schemas_to_module(
        *,
        schemas_by_root: Mapping[str, Mapping[str, Any]],
        include_dtypes: bool | Literal["as_values"],
        allow_nulls: bool | None,
        multi_roots: bool,
) -> str:
    """
    Generate a Python module for one or many root schemas.

    Naming policy:
      - Root classes use clean PascalCase of their keys.
      - Single-root mode: child classes keep clean names;
        numeric suffix only if conflict.
      - Multi-root mode:
        child classes ALWAYS get <Base><Root> concatenation (PascalCase).
        If conflict remains,
        append increasing numbers: <Base><Root>2, <Base><Root>3, ...
    """
    # Validate nested identifiers for all roots
    for schema in schemas_by_root.values():
        ensure_nested_keys_are_identifiers(schema)

    # (imports)
    header: list[str] = []
    # header.append("from __future__ import annotations")
    header.append("import paguro as pg")
    header.append("from paguro.models import vfm")
    if include_dtypes == "as_values":
        header.append("import polars as pl")
    header.append("")

    emitted: list[str] = []
    used_class_names: set[str] = set()
    actual_root_names: list[str] = []

    def _reserve_class_name(
            seen: set[str],
            base: str,
    ) -> str:
        """Minimal modification: try base; then base2, base3, ... (no underscores)."""
        if base not in seen:
            seen.add(base)
            return base
        i = 2
        while True:
            cand = f"{base}{i}"
            if cand not in seen:
                seen.add(cand)
                return cand
            i += 1

    def _reserve_attr_name(
            seen: set[str],
            original_key: str,
    ) -> str:
        """Attributes: sanitize, then minimal numeric de-dupe with underscore."""
        base = _sanitize_ident(original_key)
        if base not in seen:
            seen.add(base)
            return base
        i = 2
        while True:
            cand = f"{base}_{i}"
            if cand not in seen:
                seen.add(cand)
                return cand
            i += 1

    def build_class(
            tree: Mapping[str, Any],
            *,
            suggested_name: str,
            root_name: str,
    ) -> str:
        """
        Emit classes in post-order (children first). Returns the *actual* class name.
        """
        base = _to_class_name(suggested_name)
        cls_name = _reserve_class_name(used_class_names, base)

        # Recurse into nested structs first
        child_class_for_key: dict[str, str] = {}
        for k, v in tree.items():
            if isinstance(v, Mapping):
                child_base = _to_class_name(k)
                # Single root: prefer clean child_base;
                # Multi roots: ALWAYS suffix with root;
                if multi_roots:
                    child_suggested = f"{child_base}{root_name}"  # PascalCase concat
                else:
                    child_suggested = child_base

                # Build child and record actual name
                actual_child = build_class(
                    v,
                    suggested_name=child_suggested,
                    root_name=root_name,
                )
                child_class_for_key[k] = actual_child

        # Emit current class body
        lines: list[str] = []
        lines.append(f"class {cls_name}(vfm.VFrameModel):")
        if not tree:
            lines.append("    pass")
            lines.append("")
            emitted.append("\n".join(lines))
            return cls_name

        used_attrs: set[str] = set()

        for original_key, value in tree.items():
            attr = _reserve_attr_name(used_attrs, original_key)

            if isinstance(value, Mapping):
                child_cls = child_class_for_key[original_key]
                lines.append(f"    {attr}: {child_cls}")
            else:
                # Build kwargs for vcol
                name_kw = (
                    []
                    if attr == original_key
                    else [f'name="{original_key}"']
                )

                allow_kw = (
                    []
                    if allow_nulls is None
                    else [f"allow_nulls={allow_nulls}"]
                )

                if include_dtypes is True:
                    head, pos = _render_vcol_ctor_and_posargs(value)
                    kwargs = name_kw + allow_kw
                    if pos and kwargs:
                        call = f"{head}({', '.join(pos + kwargs)})"
                    elif pos:
                        call = f"{head}({', '.join(pos)})"
                    else:
                        call = f"{head}({', '.join(kwargs)})"

                    lines.append(f"    {attr} = {call}")

                elif include_dtypes == "as_values":
                    dtype_name = _dtype_name(value)
                    if dtype_name in {"List", "Array"}:
                        head, _ = _render_vcol_ctor_and_posargs(value)
                        dtype_expr = _render_pl_dtype_expr(value)
                        kwargs = [f"dtype={dtype_expr}"] + name_kw + allow_kw
                        call = f"{head}({', '.join(kwargs)})"
                        lines.append(f"    {attr} = {call}")
                    else:
                        dtype_expr = _render_pl_dtype_expr(value)
                        kwargs = [f"dtype={dtype_expr}"] + name_kw + allow_kw
                        lines.append(f"    {attr} = pg.vcol({', '.join(kwargs)})")

                else:
                    dtype_name = _dtype_name(value)
                    if dtype_name in {"List", "Array"}:
                        head, _ = _render_vcol_ctor_and_posargs(value)
                        kwargs = name_kw + allow_kw
                        call = f"{head}({', '.join(kwargs)})" if kwargs else f"{head}()"
                        lines.append(f"    {attr} = {call}")
                    else:
                        kwargs = name_kw + allow_kw

                        call = (
                            f"pg.vcol({', '.join(kwargs)})"
                            if kwargs else "pg.vcol()"
                        )

                        lines.append(f"    {attr} = {call}")

        lines.append("")
        emitted.append("\n".join(lines))
        return cls_name

    # Build roots
    for root_key, schema in schemas_by_root.items():
        root_name = _to_class_name(root_key)
        actual_root = build_class(
            schema,
            suggested_name=root_name,
            root_name=root_name,
        )
        actual_root_names.append(actual_root)

    # __all__ for the GENERATED module (exports only the root classes)
    all_line = f"__all__ = [{', '.join(repr(n) for n in actual_root_names)}]"

    # Join header + __all__ + emitted class bodies
    return "\n".join(header + [all_line, ""] + emitted)


def ensure_nested_keys_are_identifiers(
        schema: Mapping[str, Any],
) -> None:
    """
    Walk the schema and collect all keys that point to nested mappings
    but are NOT valid Python identifiers.

    Raise ValueError listing them all.
    """
    bad: list[str] = []

    def walk(tree: Mapping[str, Any], path: list[str]) -> None:
        for k, v in tree.items():
            if isinstance(v, Mapping):
                if not _is_valid_identifier(k):
                    bad.append(".".join([*path, k]))
                walk(v, [*path, k])

    walk(schema, [])

    if bad:
        uniq: list[str] = list(dict.fromkeys(bad))
        msg = (
                "Invalid identifiers: "
                "Struct column name must be a valid Python identifier. "
                + ", ".join(uniq)
                + "\nPlease ensure the name is a valid python identifier."
        )
        raise ValueError(msg)


def _is_valid_identifier(
        name: str,
) -> bool:
    return bool(name) and name.isidentifier() and not keyword.iskeyword(name)


_ident_re = re.compile(r"[^0-9a-zA-Z_]+")


def _sanitize_ident(
        name: str,
) -> str:
    s = _ident_re.sub("_", name).strip("_")
    if not s:
        s = "x"
    if s[0].isdigit():
        s = "_" + s
    if keyword.iskeyword(s):
        s += "_"
    return s


def _to_class_name(
        key: str,
) -> str:
    base = _sanitize_ident(key)
    parts = re.split(r"_+", base)
    return "".join(p[:1].upper() + p[1:] for p in parts if p)


# Dtype rendering


def _dtype_name(
        dt: Any,
) -> str:
    """
    Return simple Polars dtype name (e.g., 'Int64', 'Utf8', 'List', 'Array', 'Struct').
    """
    return getattr(dt, "__name__", None) or type(dt).__name__


def _render_vcol_ctor_and_posargs(
        dt: Any,
) -> tuple[str, list[str]]:
    """
    Constructor style (include_dtypes=True):
      - Simple types -> ("pg.vcol.Int64", [])
      - List(...)   -> ("pg.vcol.List",  [])   # no inner args in ctor mode
      - Array(...)  -> ("pg.vcol.Array", [])   # no inner/width in ctor mode
    """
    name = _dtype_name(dt)
    if name == "List":
        return "pg.vcol.List", []
    if name == "Array":
        return "pg.vcol.Array", []
    return f"pg.vcol.{name}", []


def _render_pl_dtype_expr(dt: Any) -> str:
    """
    Keyword style (include_dtypes='as_values'):
      - Simple types -> pl.Int64
      - List(inner)  -> pl.List(<inner>)
      - Array(inner, width) -> pl.Array(<inner>, <width>)
      - Struct(fields) -> pl.Struct([pl.Field("name", <dtype>), ...])
    """
    name = _dtype_name(dt)

    if name == "List":
        inner = getattr(dt, "inner", None)
        inner_expr = _render_pl_dtype_expr(inner)
        return f"pl.List({inner_expr})"

    if name == "Array":
        inner = getattr(dt, "inner", None)
        shape = dt.shape

        inner_expr = _render_pl_dtype_expr(inner)
        return f"pl.Array({inner_expr}, {shape})"

    if name == "Struct":
        # we are currently not including Struct as a value pg.vcol.Struct
        # but only as a constructed class. But maybe we should allow both
        fields: list[Field] = dt.fields

        # if not fields:
        #     msg = "Struct dtype is missing required 'fields'."
        #     raise ValueError(msg)

        items: list[str] = []
        for f in fields:
            field_name, field_dtype = f.name, f.dtype

            field_expr = _render_pl_dtype_expr(field_dtype)
            items.append(f"pl.Field({field_name!r}, {field_expr})")

        return f"pl.Struct([{', '.join(items)}])"

    # Default simple dtype
    return f"pl.{name}"


# I/O


def _write_model_blueprint_to_python_file(
        *,
        blueprint: str,
        path: str | Path | None,
        model_class_names: list[str],
        print_usage_suggestion: bool,
) -> str | None:
    if path is None:
        return blueprint

    p = Path(path)

    # Must be a .py file
    if p.suffix != ".py":
        msg = f"Expected a '.py' path, got: {p}"
        raise ValueError(msg)

    # Parent folder must exist (no auto-creation)
    if not p.parent.exists() or not p.parent.is_dir():
        msg = f"Parent folder must already exist: {p.parent}"
        raise FileNotFoundError(msg)

    # File must not already exist (no overwrites)
    if p.exists():
        msg = f"Refusing to overwrite existing file: {p}"
        raise FileExistsError(msg)

    if print_usage_suggestion:
        parts = ".".join(p.with_suffix("").parts)
        if parts:
            import_line = f"from {parts} import {', '.join(model_class_names)}"
        else:
            import_line = f"import {', '.join(model_class_names)}"
        print(
            "\n"
            f"# ------- Suggested usage for models {', '.join(model_class_names)}\n\n"
            "import paguro as pg\n"
            f"{import_line}\n\n"
            "# Assign a model to a dataset instance:\n"
            "# dataset = dataset.with_model(YourModelClass)\n"
        )

    p.write_text(blueprint, encoding="utf-8")
    return None
