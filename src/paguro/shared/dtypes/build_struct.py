from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, Union

import polars as pl
from polars import DataType
from polars.datatypes import DataTypeClass

if TYPE_CHECKING:
    from paguro.validation.valid_column.valid_column import ValidColumn

DTypeLike: TypeAlias = Union[DataType, DataTypeClass]


# ------ snippet of code taken from polars/schema.py ------


def _required_init_args(tp: DataTypeClass) -> bool:
    """
    Required_init_args.

    Indicates override of the default __init__ (i.e., this dtype
    requires specific args). This is evaluated on the *class*.
    """
    return "__init__" in tp.__dict__


# ---------------------------------------------------------


def build_struct_from_dtype_or_fields(node: ValidColumn) -> DTypeLike:
    """
    Build a Struct dtype from a ValidColumn node, or pass through other dtypes.

    Returns
    -------
        DTypeLike: may be a DataType instance or a DataTypeClass placeholder,
        matching current runtime behavior.
    """
    name = getattr(node, "_name", None)
    if not name:
        msg = "Field 'name' must be specified"
        raise TypeError(msg)

    dt: DTypeLike | None = node._get_supertype()
    if dt is None:
        if node._fields is not None:
            dt = pl.Struct  # unspecified Struct to build from fields
        else:
            msg = (
                f"dtype is None for field {name!r}. "
                f"Specify all the dtypes to construct a Struct from fields."
            )
            raise TypeError(msg)

    # Detect placeholder vs instance — DO NOT use `==` here.
    is_struct_placeholder = dt is pl.Struct  # bare placeholder (class)
    is_struct_instance = isinstance(
        dt, pl.Struct
    )  # fully specified instance

    # Needs-init check (placeholder always needs init).
    # Only call _required_init_args on a DataTypeClass; instances don't need init.
    if is_struct_placeholder:
        needs_init = True
    else:
        needs_init = (
            _required_init_args(dt)
            if isinstance(dt, DataTypeClass)
            else False
        )

    # Non-Struct that needs init -> error
    if not (is_struct_placeholder or is_struct_instance) and needs_init:
        msg = (
            f"dtype '{dt}' for field '{name}' requires "
            f"initialization arguments and is not a Struct"
        )
        raise TypeError(msg)

    # Struct handling
    if is_struct_placeholder or is_struct_instance:
        # Fully specified struct instance: return as-is and ignore any provided fields
        if is_struct_instance and not needs_init:
            return dt

        # Build from provided fields
        fields_validation = node._fields
        fields_vcl = getattr(fields_validation, "_valid_column_list", None)

        if not fields_vcl:
            msg = (
                f"Unable to build dtype for struct field {name!r}. "
                f"Either fully specify a dtype or declare fields for {name!r}."
            )
            raise TypeError(msg)

        return pl.Struct(
            [
                pl.Field(f._name, build_struct_from_dtype_or_fields(f))
                for f in fields_vcl
            ]
        )

    # Non-Struct that doesn't need init args:
    # pass through as-is (may be class or instance)
    return dt
