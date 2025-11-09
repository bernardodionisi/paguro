from __future__ import annotations

import keyword
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TYPE_CHECKING, Literal

from paguro.shared.extra_utilities import _unnest_schema

if TYPE_CHECKING:
    from polars import DataFrame, LazyFrame
    from paguro import Dataset, LazyDataset
    from paguro.shared.extra_utilities import SchemaTree

__all__ = ["collect_model_blueprint", "schema_to_module"]


def collect_model_blueprint(
        data: DataFrame | LazyFrame | Dataset | LazyDataset,
        *,
        path: str | Path | None = None,
        root_class_name: str = "DatasetModel",
        include_dtypes: bool | Literal["as_values"] = False,
        allow_nulls: bool | None = False,
        print_usage_suggestion: bool = True,
) -> str | None:
    """
    Build a model template from a dataset-like object's schema.
    """
    schema = data.collect_schema()
    unnested_schema: SchemaTree = _unnest_schema(schema=schema)
    blueprint = schema_to_module(
        schema=unnested_schema,
        root_class_name=root_class_name,
        include_dtypes=include_dtypes,
        allow_nulls=allow_nulls,
    )

    if path is None:
        return blueprint
    else:
        return _write_model_blueprint_to_python_file(
            blueprint=blueprint,
            path=path,
            model_class_name=root_class_name,
            print_usage_suggestion=print_usage_suggestion,
        )


def schema_to_module(
        schema: Mapping[str, Any],
        *,
        root_class_name: str = "DatasetModel",
        include_dtypes: bool | Literal["as_values"] = False,
        allow_nulls: bool | None = False,
) -> str:
    """
    Build a Python module string from a recursive schema:
      { key: pl.DataType | { ... } }

    Header:
        import paguro as pg
        from paguro.models import vfm
        import polars as pl    # only when include_dtypes == "as_values"

    Leaves:
      - include_dtypes=False:
            attr = pg.vcol(name=..., allow_nulls=...)
      - include_dtypes=True:            # constructor style
            attr = pg.vcol.<Name>(..., name=..., allow_nulls=...)
      - include_dtypes="as_values":     # keyword dtype=pl.<expr>
            attr = pg.vcol(dtype=pl.<expr>, name=..., allow_nulls=...)
    """
    # Fail fast on invalid nested identifiers (cannot use name= for annotations)
    ensure_nested_keys_are_identifiers(schema)

    header: list[str] = []
    header.append("import paguro as pg")
    header.append("from paguro.models import vfm")
    if include_dtypes == "as_values":
        header.append("import polars as pl")
    header.append("")

    # Naming utilities
    def _dedupe(seen: set[str], name: str) -> str:
        if name not in seen:
            seen.add(name)
            return name
        i = 2
        while f"{name}_{i}" in seen:
            i += 1
        final = f"{name}_{i}"
        seen.add(final)
        return final

    used_class_names: set[str] = set()
    # Ensure the root class name is reserved up-front
    root_class_name = _to_class_name(root_class_name)

    # ------------------------------------------------------------------
    # Post-order DFS: emit children first, then parent.
    # We build class source strings in `emitted` (a list) in the order we visit.
    # Each subtree call returns the *actual* class name chosen (after dedupe),
    # so parents can annotate with the correct, already-defined child name.
    # ------------------------------------------------------------------
    emitted: list[str] = []

    def build_class(tree: Mapping[str, Any], suggested_name: str) -> str:
        cls_name = _dedupe(used_class_names, _to_class_name(suggested_name))

        # First, assign names to children and recursively emit them (post-order)
        # Remember, per original key, the chosen child class name to annotate with.
        child_class_for_key: dict[str, str] = {}

        for original_key, value in tree.items():
            if _is_nested_mapping(value):
                # We want a stable child suggestion per key under this parent.
                suggested_child = _to_class_name(original_key)
                # Dedup happens inside the recursive call (against global set),
                # which returns the *actual* chosen name for the child class.
                actual_child_name = build_class(value, suggested_child)
                child_class_for_key[original_key] = actual_child_name

        # Now that all children are emitted, we can emit the current class body.
        lines: list[str] = []
        lines.append(f"class {cls_name}(vfm.VFrameModel):")
        if not tree:
            lines.append("    pass")
            lines.append("")
            emitted.append("\n".join(lines))
            return cls_name

        used_attrs: set[str] = set()
        used_child_ann: set[str] = set()

        for original_key, value in tree.items():
            attr = _dedupe(used_attrs, _sanitize_ident(original_key))

            if _is_nested_mapping(value):
                # Annotation with already-defined child class.
                child_cls = child_class_for_key[original_key]
                _ = _dedupe(
                    used_child_ann,
                    child_cls
                )  # no-op, keeps naming style parity
                lines.append(f"    {attr}: {child_cls}")
            else:
                # Build kwargs for vcol
                name_kw = [] if attr == original_key else [f'name="{original_key}"']
                allow_kw = [] if allow_nulls is None else [f"allow_nulls={allow_nulls}"]

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
                    dtype_expr = _render_pl_dtype_expr(value)
                    kwargs = [f"dtype={dtype_expr}"] + name_kw + allow_kw
                    lines.append(f"    {attr} = pg.vcol({', '.join(kwargs)})")

                else:
                    kwargs = name_kw + allow_kw
                    lines.append(f"    {attr} = pg.vcol({', '.join(kwargs)})")

        lines.append("")
        emitted.append("\n".join(lines))
        return cls_name

    # Kick off from root (children get emitted first, then root at the end)
    build_class(schema, root_class_name)

    # `emitted` currently holds classes in strict leaf->...->root order already.
    # We just need to join header + emitted classes.
    return "\n".join(header + emitted)


# ----------------------------- Helpers --------------------------------


def _is_valid_identifier(name: str) -> bool:
    return bool(name) and name.isidentifier() and not keyword.iskeyword(name)


def ensure_nested_keys_are_identifiers(schema: Mapping[str, Any]) -> None:
    """
    Walk the schema and collect all keys that point to nested mappings but are
    NOT valid Python identifiers. Raise ValueError listing them all.
    """
    bad: list[str] = []

    def walk(tree: Mapping[str, Any], path: list[str]) -> None:
        for k, v in tree.items():
            if _is_nested_mapping(v):
                if not _is_valid_identifier(k):
                    bad.append(".".join([*path, k]))
                walk(v, [*path, k])

    walk(schema, [])

    if bad:
        # Order-preserving dedupe without side effects in expressions
        uniq: list[str] = list(dict.fromkeys(bad))
        raise ValueError(
            "Invalid identifiers: Struct column name must be a valid Python identifier. "
            + ", ".join(uniq)
            + "\nPlease ensure the name is a valid python identifier."
        )


_ident_re = re.compile(r"[^0-9a-zA-Z_]+")


def _sanitize_ident(name: str) -> str:
    s = _ident_re.sub("_", name).strip("_")
    if not s:
        s = "x"
    if s[0].isdigit():
        s = "_" + s
    if keyword.iskeyword(s):
        s += "_"
    return s


def _to_class_name(key: str) -> str:
    base = _sanitize_ident(key)
    parts = re.split(r"_+", base)
    return "".join(p[:1].upper() + p[1:] for p in parts if p)


def _is_nested_mapping(v: Any) -> bool:
    return isinstance(v, Mapping)


# ------------------------ Dtype rendering -----------------------------

def _dtype_name(dt: Any) -> str:
    """
    Return simple Polars dtype name (e.g., 'Int64', 'Utf8', 'List', 'Array', 'Struct').
    """
    return getattr(dt, "__name__", None) or type(dt).__name__


def _render_vcol_ctor_and_posargs(dt: Any) -> tuple[str, list[str]]:
    """
    Constructor style (include_dtypes=True):

      Int64 / Utf8 / Boolean -> ("pg.vcol.Int64", [])
      List(...) -> ("pg.vcol.List",  ["pl.Int64"])     # inner ignored for now
      Array(...) -> ("pg.vcol.Array", ["pl.Int64", "4"])# width/inner defaulted
    """
    name = _dtype_name(dt)
    if name == "List":
        return "pg.vcol.List", ["pl.Int64"]
    if name == "Array":
        return "pg.vcol.Array", ["pl.Int64", "4"]
    return f"pg.vcol.{name}", []


def _render_pl_dtype_expr(dt: Any) -> str:
    """
    Keyword style (include_dtypes='as_values'):
      - Simple types -> pl.Int64
      - List(inner)  -> pl.List(<inner>)
      - Array(inner, width) -> pl.Array(<inner>, <width>)
      - Struct(fields) -> pl.Struct([pl.Field("name", <dtype>), ...])

    Recursively qualifies nested inners and struct fields with pl.*.
    """
    name = _dtype_name(dt)

    # List
    if name == "List":
        inner = getattr(dt, "inner", None)
        inner_expr = _render_pl_dtype_expr(inner) if inner is not None else "pl.Int64"
        return f"pl.List({inner_expr})"

    # Array
    if name == "Array":
        inner = getattr(dt, "inner", None)
        width = getattr(dt, "width", None)
        if width is None:
            width = getattr(dt, "shape", None)
        if width is None:
            width = getattr(dt, "size", None)
        inner_expr = _render_pl_dtype_expr(inner) if inner is not None else "pl.Int64"
        width_expr = str(width) if width is not None else "4"
        return f"pl.Array({inner_expr}, {width_expr})"

    # Struct
    if name == "Struct":
        fields = getattr(dt, "fields", None)
        items: list[str] = []
        if fields:
            for f in fields:
                fname = getattr(f, "name", None)
                fdtype = getattr(f, "dtype", None)
                if fname is None:
                    continue
                fdexpr = _render_pl_dtype_expr(
                    fdtype) if fdtype is not None else "pl.Int64"
                items.append(f'pl.Field({fname!r}, {fdexpr})')
        return f"pl.Struct([{', '.join(items)}])"

    # Default simple dtype
    return f"pl.{name}"


# ------------------------------ I/O ------------------------------------

def _write_model_blueprint_to_python_file(
        blueprint: str,
        path: str | Path | None,
        *,
        model_class_name: str,
        print_usage_suggestion: bool,
) -> str | None:
    if path is None:
        return blueprint

    p = Path(path)

    # Must be a .py file
    if p.suffix != ".py":
        raise ValueError(f"Expected a '.py' path, got: {p}")

    # Parent folder must exist (no auto-creation)
    if not p.parent.exists() or not p.parent.is_dir():
        raise FileNotFoundError(f"Parent folder must already exist: {p.parent}")

    # File must not already exist (no overwrites)
    if p.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {p}")

    if print_usage_suggestion:
        parts = ".".join(p.with_suffix("").parts)

        if parts:
            import_line = f"from {parts} import {model_class_name}"
        else:
            import_line = f"import {model_class_name}"

        print(
            "\n"
            f"# ------- Suggested usage for model {model_class_name}\n\n"
            "import paguro as pg\n"
            f"{import_line}\n\n"
            "# TO assign a model to the dataset instance call .with_model()\n\n"
            f"dataset = dataset.with_model({model_class_name})\n\n"
            # "# # If you want to get the expression for a column\n"
            # "# # you can use the 'vcol' namespace in your dataset instance\n"
            # "# # just replace * with your column name (it is statically typed!)\n\n"
            # "# ds.select(\n"
            # "#    ds.vcol.*()\n"
            # "# )\n"
            # "# # for structs you can chain the fields names!\n"
            # "# ds.select(\n"
            # "#    ds.vcol.*.*()\n"
            # "# )\n"
        )

    p.write_text(blueprint, encoding="utf-8")
    return None
