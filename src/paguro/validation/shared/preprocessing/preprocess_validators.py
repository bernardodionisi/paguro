from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Iterable

import polars as pl
from paguro.validation.shared.preprocessing.duplicates import \
    _raise_exception_duplicates_names
from paguro.validation.shared.preprocessing.utils import _parse_validators_as_iterable
from paguro.validation.valid_column.valid_column import ValidColumn
from paguro.validation.valid_column.utils._vdtypes import ValidStruct
from paguro.validation.valid_frame.valid_frame import ValidFrame

if TYPE_CHECKING:
    from paguro.typing import ValidatorOrExpr
    from paguro.validation.validation import Validation

__all__ = [
    "preprocess_vcs_vfs",
]


def preprocess_vcs_vfs(
        *validators: (
                ValidatorOrExpr |
                Iterable[ValidatorOrExpr] |
                Validation
        ),
        **named_validators: ValidatorOrExpr,
) -> tuple[
    list[ValidColumn] | None,
    list[ValidFrame] | None
]:
    _vals: tuple[ValidatorOrExpr | Validation, ...] = tuple(
        _parse_validators_as_iterable(validators)
    )

    vfe: ValidFrame | None = _preprocess_expressions(
        *_vals, **named_validators
    )

    vcl: list[ValidColumn]
    vfl: list[ValidFrame]
    vcl, vfl = _preprocess_validators(validators=_vals)

    nvcl: list[ValidColumn]
    nvfl: list[ValidFrame]
    nvcl, nvfl = _preprocess_named_validators(
        named_validators=named_validators
    )

    if nvcl:
        vcl.extend(nvcl)
    if nvfl:
        vfl.extend(nvfl)

    if vfe:
        vfl.append(vfe)

    for idx, vf in enumerate(vfl):
        if vf._name is None:
            vf._name = f"vframe_{idx}"

    _raise_exception_duplicates_names(vcl=vcl, vfl=vfl)

    vcl_result: list[ValidColumn] | None = None if not vcl else vcl
    vfl_result: list[ValidFrame] | None = None if not vfl else vfl

    return vcl_result, vfl_result


def _preprocess_expressions(
        *validators: ValidatorOrExpr | Iterable[ValidatorOrExpr] | Validation,
        **named_validators: ValidatorOrExpr,
) -> ValidFrame | None:
    exprs = {}

    # Process positional arguments

    counter = 0
    for arg in validators:
        if isinstance(arg, pl.Expr):
            exprs[f"constraint_{counter}"] = arg
            counter += 1  # use counter (instead of enumerate) otherwise we count also non-pl.Expr

    # Process keyword arguments
    for key, value in named_validators.items():
        if isinstance(value, pl.Expr):
            # Handle duplicate keys by adding suffix
            original_key = key
            counter = 1
            while key in exprs:
                key = f"{original_key}_{counter}"
                counter += 1
            exprs[key] = value

    if not exprs:
        return None

    return ValidFrame(**exprs)


def _preprocess_validators(
        validators: tuple[ValidatorOrExpr | Validation, ...],
) -> tuple[list[ValidColumn], list[ValidFrame]]:
    vcl: list[ValidColumn] = []
    vfl: list[ValidFrame] = []

    for i in validators:
        if isinstance(i, ValidColumn):
            vcl.append(i)
        elif isinstance(i, ValidFrame):
            vfl.append(i)
        else:
            if not isinstance(i, pl.Expr):
                from paguro.validation.validation import Validation

                if isinstance(i, Validation):
                    if i._valid_column_list is not None:
                        vcl.extend(i._valid_column_list._valid_list)

                    if i._valid_frame_list is not None:
                        vfl.extend(i._valid_frame_list._valid_list)
                else:
                    msg = f"Unexpected type {type(i)}"
                    raise TypeError(msg)

    return vcl, vfl


def _preprocess_named_validators(
        named_validators: dict[str, ValidColumn | ValidFrame],
) -> tuple[list[ValidColumn], list[ValidFrame]]:
    vcl: list[ValidColumn] = []
    vfl: list[ValidFrame] = []

    for name, validator in named_validators.items():
        if isinstance(validator, ValidColumn):
            _warn_replacing_name(name, validator)
            validator._name = name
            vcl.append(validator)

        elif isinstance(validator, ValidFrame):
            _warn_replacing_name(name, validator)
            validator._name = name
            vfl.append(validator)

    return vcl, vfl


def _warn_replacing_name(
        name: str,
        validator: ValidColumn | ValidFrame,
) -> None:
    if validator._name is not None and validator._name != name:
        warnings.warn(
            f"The name of {type(validator)}: {validator._name} != {name}. "
            f"The name has been replaced.",
            stacklevel=2,
        )


# ----------------------------------------------------------------------


def _preprocess_validator_mapping(
        mapping: Mapping | pl.Schema,
) -> list[ValidColumn | ValidFrame]:
    """
    Converts a key value mapping to a list of validators.

    _preprocess_validator_mapping({
        "a": int,
        "b": pg.vcol(ge=1),
        "c": {
            "c_1": int,

        }
    })
    """
    # note: if one passes a Polars Schema with python types,
    # (int) these are casted to polars types
    # if one passes a dictionary with python types these are
    # casted to frozensets with polars types
    # int passed in a dictionary will be a superset of in passed in a schema
    validators: list[ValidColumn | ValidFrame] = []

    if isinstance(mapping, pl.Schema):
        mapping = mapping.to_python()

    for k, v in mapping.items():
        if isinstance(v, Mapping):

            _val = _preprocess_validator_mapping(v)
            validators.append(
                ValidStruct(*_val, name=k)
            )

        elif isinstance(v, (ValidColumn, ValidFrame)):
            if v._name is not None and v._name != k:
                msg = (
                    f"Dictionary key ({k}) and "
                    f"validator name ({v._name}) do not match"
                )
                raise ValueError(msg)
            validators.append(v)
        else:  # dtype
            validators.append(ValidColumn(name=k, dtype=v))
        # expression?

    return validators
