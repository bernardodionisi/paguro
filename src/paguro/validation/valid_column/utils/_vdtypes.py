from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Literal, cast

import polars as pl

from paguro.shared._typing._typing import IsBetweenTuple

from paguro.validation.valid_column.valid_column import ValidColumn

if TYPE_CHECKING:
    import decimal

    from paguro.typing import FieldsValidators, IntoNameVC
    from paguro.validation.validation import Validation

__all__ = [
    "ValidStruct",
    "ValidEnum",
    "ValidCategorical",
    "ValidString",
    "ValidBinary",
    "ValidBoolean",
    "ValidDate",
    "ValidDateTime",
    "ValidDuration",
    "ValidTime",
    "ValidArray",
    "ValidList",
    "ValidNumeric",
    "ValidInteger",
    "ValidInt8",
    "ValidInt16",
    "ValidInt32",
    "ValidInt64",
    "ValidInt128",
    "ValidUInteger",
    "ValidUInt8",
    "ValidUInt16",
    "ValidUInt32",
    "ValidUInt64",
    "ValidUInt128",
    "ValidFloat",
    "ValidFloat32",
    "ValidFloat64",
    "ValidDecimal",
]


class ValidStruct(ValidColumn):
    def __init__(
            self,
            name: IntoNameVC = None,
            dtype: pl.Struct | type[pl.Struct] | None = pl.Struct,
            *,
            fields: FieldsValidators | None = None,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> None:
        super().__init__(
            name=name,
            dtype=dtype,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **constraints,
        )

        if fields is not None:
            # TODO: allow_rename not supported when set in fields validators:
            #   - we are not renaming the fields or checking
            #       whether the fields have been renamed
            #   - todo: raise error if the allow_rename
            #       flag has been set for any ValidColumn in fields

            from paguro.validation.validation import Validation
            # validators is now a tuple [ValidatorOrExpr, Iterable[ValidatorOrExpr], Validation]
            self._fields = Validation(fields)  # type: ignore[arg-type]

            if dtype is None:
                # todo: distinguish between required and not required
                self._set_struct_dtype_from_fields(replace=True)

    def __repr__(self):
        return self._repr_or_str(string=False)

    def __str__(self):
        return self._repr_or_str(string=True)

    def _repr_or_str(self, *, string: bool) -> str:
        if string:
            base_str = super().__str__()
            fields_str = self._fields.__str__()
        else:
            base_str = super().__repr__()
            fields_str = self._fields.__repr__()

        base_str = base_str.lstrip(r"ValidColumn")
        if self._fields is None:
            return f"ValidColumn:Struct(fields=None, " + base_str[1:]

        fields_str = fields_str.replace("\n", "\n\t")
        base_str = base_str.rstrip(")")
        return f"ValidColumn:Struct(\n\tfields={fields_str},\n\t" + base_str[1:] + "\n)"

    def __getattr__(self, name: str) -> ValidColumn:
        if name.startswith("__"):
            raise AttributeError(name)
        return self._vfield(name=name)

    def _vfield(self, name: str) -> ValidColumn:
        # todo: search vframes

        if not isinstance(self._name, str):
            msg = (
                f"Field can only be accessed if the vcol name is set to a string: "
                f"the name is set to {self._name}"
            )
            raise TypeError(msg)

        if self._fields is None:
            msg = f"No fields have been set for {self._name}"
            raise AttributeError()
        vcol: ValidColumn | None = self._fields._find_vcol(
            name=name,
            return_first=True,
            include_transformed_frames=False,
            include_fields=False,
        )

        if vcol:
            vcol = copy.deepcopy(vcol)

            if self._root_up:
                vcol._root_up = (*self._root_up, cast(str, self._name))
            else:
                vcol._root_up = (cast(str, self._name),)

            return vcol

        else:
            msg = f"No field named {name} found."
            raise AttributeError(msg)


class ValidArray(ValidColumn):
    def __init__(
            self,
            name: IntoNameVC = None,
            dtype: pl.Array | type[pl.Array] = pl.Array,
            *,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            contains: Any | None = None,
            **constraints: Any,
    ) -> None:
        _constraints = _list_constraints_remove_none(
            contains=contains,
        )

        super().__init__(
            name=name,
            dtype=dtype,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **_constraints,
            **constraints,
        )


class ValidList(ValidColumn):
    def __init__(
            self,
            name: IntoNameVC = None,
            dtype: pl.List | type[pl.List] = pl.List,
            *,
            contains: Any | None = None,
            len_ge: int | None = None,
            len_gt: int | None = None,
            len_le: int | None = None,
            len_lt: int | None = None,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> None:
        _constraints = _list_constraints_remove_none(
            contains=contains,
            len_ge=len_ge,
            len_gt=len_gt,
            len_le=len_le,
            len_lt=len_lt
        )

        super().__init__(
            name=name,
            dtype=dtype,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **_constraints,
            **constraints,
        )


class ValidEnum(ValidColumn):
    def __init__(
            self,
            name: IntoNameVC = None,
            dtype: pl.Enum | type[pl.Enum] = pl.Enum,
            *,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> None:
        super().__init__(
            name=name,
            dtype=dtype,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **constraints,
        )


class ValidCategorical(ValidColumn):
    def __init__(
            self,
            name: IntoNameVC = None,
            *,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> None:
        dtype = pl.Categorical
        super().__init__(
            name=name,
            dtype=dtype,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **constraints,
        )


class ValidString(ValidColumn):
    def __init__(
            self,
            name: IntoNameVC = None,
            *,
            contains: str | None = None,
            contains_any: str | None = None,
            starts_with: str | None = None,
            ends_with: str | None = None,
            len_chars_eq: int | None = None,
            len_chars_ge: int | None = None,
            len_chars_gt: int | None = None,
            len_chars_le: int | None = None,
            len_chars_lt: int | None = None,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> None:
        _constraints = _str_constraints_remove_none(
            contains=contains,
            contains_any=contains_any,
            start=starts_with,
            end=ends_with,
            len_chars_eq=len_chars_eq,
            len_chars_ge=len_chars_ge,
            len_chars_gt=len_chars_gt,
            len_chars_le=len_chars_le,
            len_chars_lt=len_chars_lt,
        )
        dtype = pl.String

        super().__init__(
            name=name,
            dtype=dtype,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **_constraints,
            **constraints,
        )


class ValidBinary(ValidColumn):
    def __init__(
            self,
            name: IntoNameVC = None,
            *,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> None:
        dtype = pl.Binary
        super().__init__(
            name=name,
            dtype=dtype,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **constraints,
        )


class ValidBoolean(ValidColumn):
    def __init__(
            self,
            name: IntoNameVC = None,
            *,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> None:
        super().__init__(
            name=name,
            dtype=pl.Boolean,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **constraints,
        )


class ValidDate(ValidColumn):
    def __init__(
            self,
            name: IntoNameVC = None,
            *,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> None:
        dtype = pl.Date
        super().__init__(
            name=name,
            dtype=dtype,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **constraints,
        )


class ValidDateTime(ValidColumn):
    def __init__(
            self,
            name: IntoNameVC = None,
            dtype: pl.Datetime | type[pl.Datetime] = pl.Datetime,
            *,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> None:
        super().__init__(
            name=name,
            dtype=dtype,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **constraints,
        )


class ValidDuration(ValidColumn):
    def __init__(
            self,
            name: IntoNameVC = None,
            dtype: pl.Duration | type[pl.Duration] = pl.Duration,
            *,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> None:
        super().__init__(
            name=name,
            dtype=dtype,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **constraints,
        )


class ValidTime(ValidColumn):
    def __init__(
            self,
            name: IntoNameVC = None,
            *,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> None:
        dtype = pl.Time
        super().__init__(
            name=name,
            dtype=dtype,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **constraints,
        )


class ValidNumeric(ValidColumn):
    def __init__(
            self,
            name: IntoNameVC = None,
            *,
            ge: int | float | decimal.Decimal | None = None,
            gt: int | float | decimal.Decimal | None = None,
            le: int | float | decimal.Decimal | None = None,
            lt: int | float | decimal.Decimal | None = None,
            is_between: IsBetweenTuple = None,
            is_infinite: bool | None = None,
            is_nan: bool | None = None,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> None:
        _constraints = _int_constraints_remove_none(
            ge=ge,
            gt=gt,
            le=le,
            lt=lt,
            is_between=is_between,
            is_infinite=is_infinite,
            is_nan=is_nan,
        )
        dtype = "numeric"
        super().__init__(
            name=name,
            dtype=dtype,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **_constraints,
            **constraints,
        )


class ValidInteger(ValidColumn):
    """
    Integer.
    """

    def __init__(
            self,
            name: IntoNameVC = None,
            *,
            ge: int | float | decimal.Decimal | None = None,
            gt: int | float | decimal.Decimal | None = None,
            le: int | float | decimal.Decimal | None = None,
            lt: int | float | decimal.Decimal | None = None,
            is_between: IsBetweenTuple = None,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> None:
        _constraints = _int_constraints_remove_none(
            ge=ge,
            gt=gt,
            le=le,
            lt=lt,
            is_between=is_between,
        )
        dtype = int
        super().__init__(
            name=name,
            dtype=dtype,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **_constraints,
            **constraints,
        )


class ValidInt8(ValidColumn):
    def __init__(
            self,
            name: IntoNameVC = None,
            *,
            ge: int | float | decimal.Decimal | None = None,
            gt: int | float | decimal.Decimal | None = None,
            le: int | float | decimal.Decimal | None = None,
            lt: int | float | decimal.Decimal | None = None,
            is_between: IsBetweenTuple = None,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> None:
        _constraints = _int_constraints_remove_none(
            ge=ge,
            gt=gt,
            le=le,
            lt=lt,
            is_between=is_between,
        )
        dtype = pl.Int8
        super().__init__(
            name=name,
            dtype=dtype,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **_constraints,
            **constraints,
        )


class ValidInt16(ValidColumn):
    def __init__(
            self,
            name: IntoNameVC = None,
            *,
            ge: int | float | decimal.Decimal | None = None,
            gt: int | float | decimal.Decimal | None = None,
            le: int | float | decimal.Decimal | None = None,
            lt: int | float | decimal.Decimal | None = None,
            is_between: IsBetweenTuple = None,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> None:
        _constraints = _int_constraints_remove_none(
            ge=ge,
            gt=gt,
            le=le,
            lt=lt,
            is_between=is_between,
        )
        dtype = pl.Int16
        super().__init__(
            name=name,
            dtype=dtype,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **_constraints,
            **constraints,
        )


class ValidInt32(ValidColumn):
    def __init__(
            self,
            name: IntoNameVC = None,
            *,
            ge: int | float | decimal.Decimal | None = None,
            gt: int | float | decimal.Decimal | None = None,
            le: int | float | decimal.Decimal | None = None,
            lt: int | float | decimal.Decimal | None = None,
            is_between: IsBetweenTuple = None,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> None:
        _constraints = _int_constraints_remove_none(
            ge=ge,
            gt=gt,
            le=le,
            lt=lt,
            is_between=is_between,
        )
        dtype = pl.Int32
        super().__init__(
            name=name,
            dtype=dtype,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **_constraints,
            **constraints,
        )


class ValidInt64(ValidColumn):
    def __init__(
            self,
            name: IntoNameVC = None,
            *,
            ge: int | float | decimal.Decimal | None = None,
            gt: int | float | decimal.Decimal | None = None,
            le: int | float | decimal.Decimal | None = None,
            lt: int | float | decimal.Decimal | None = None,
            is_between: IsBetweenTuple = None,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> None:
        _constraints = _int_constraints_remove_none(
            ge=ge,
            gt=gt,
            le=le,
            lt=lt,
            is_between=is_between,
        )
        dtype = pl.Int64
        super().__init__(
            name=name,
            dtype=dtype,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **_constraints,
            **constraints,
        )


class ValidInt128(ValidColumn):
    def __init__(
            self,
            name: IntoNameVC = None,
            *,
            ge: int | float | decimal.Decimal | None = None,
            gt: int | float | decimal.Decimal | None = None,
            le: int | float | decimal.Decimal | None = None,
            lt: int | float | decimal.Decimal | None = None,
            is_between: IsBetweenTuple = None,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> None:
        _constraints = _int_constraints_remove_none(
            ge=ge,
            gt=gt,
            le=le,
            lt=lt,
            is_between=is_between,
        )
        dtype = pl.Int128
        super().__init__(
            name=name,
            dtype=dtype,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **_constraints,
            **constraints,
        )


class ValidUInteger(ValidColumn):
    def __init__(
            self,
            name: IntoNameVC = None,
            *,
            ge: int | float | decimal.Decimal | None = None,
            gt: int | float | decimal.Decimal | None = None,
            le: int | float | decimal.Decimal | None = None,
            lt: int | float | decimal.Decimal | None = None,
            is_between: IsBetweenTuple = None,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> None:
        _constraints = _int_constraints_remove_none(
            ge=ge,
            gt=gt,
            le=le,
            lt=lt,
            is_between=is_between,
        )
        dtype = "uint"
        super().__init__(
            name=name,
            dtype=dtype,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **_constraints,
            **constraints,
        )


class ValidUInt8(ValidColumn):
    def __init__(
            self,
            name: IntoNameVC = None,
            *,
            ge: int | float | decimal.Decimal | None = None,
            gt: int | float | decimal.Decimal | None = None,
            le: int | float | decimal.Decimal | None = None,
            lt: int | float | decimal.Decimal | None = None,
            is_between: IsBetweenTuple = None,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> None:
        _constraints = _int_constraints_remove_none(
            ge=ge,
            gt=gt,
            le=le,
            lt=lt,
            is_between=is_between,
        )
        dtype = pl.UInt8
        super().__init__(
            name=name,
            dtype=dtype,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **_constraints,
            **constraints,
        )


class ValidUInt16(ValidColumn):
    def __init__(
            self,
            name: IntoNameVC = None,
            *,
            ge: int | float | decimal.Decimal | None = None,
            gt: int | float | decimal.Decimal | None = None,
            le: int | float | decimal.Decimal | None = None,
            lt: int | float | decimal.Decimal | None = None,
            is_between: IsBetweenTuple = None,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> None:
        _constraints = _int_constraints_remove_none(
            ge=ge,
            gt=gt,
            le=le,
            lt=lt,
            is_between=is_between,
        )
        dtype = pl.UInt16
        super().__init__(
            name=name,
            dtype=dtype,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **_constraints,
            **constraints,
        )


class ValidUInt32(ValidColumn):
    def __init__(
            self,
            name: IntoNameVC = None,
            *,
            ge: int | float | decimal.Decimal | None = None,
            gt: int | float | decimal.Decimal | None = None,
            le: int | float | decimal.Decimal | None = None,
            lt: int | float | decimal.Decimal | None = None,
            is_between: IsBetweenTuple = None,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> None:
        _constraints = _int_constraints_remove_none(
            ge=ge,
            gt=gt,
            le=le,
            lt=lt,
            is_between=is_between,
        )
        dtype = pl.UInt32
        super().__init__(
            name=name,
            dtype=dtype,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **_constraints,
            **constraints,
        )


class ValidUInt64(ValidColumn):
    def __init__(
            self,
            name: IntoNameVC = None,
            *,
            ge: int | float | decimal.Decimal | None = None,
            gt: int | float | decimal.Decimal | None = None,
            le: int | float | decimal.Decimal | None = None,
            lt: int | float | decimal.Decimal | None = None,
            is_between: IsBetweenTuple = None,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> None:
        _constraints = _int_constraints_remove_none(
            ge=ge,
            gt=gt,
            le=le,
            lt=lt,
            is_between=is_between,
        )
        dtype = pl.UInt64
        super().__init__(
            name=name,
            dtype=dtype,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **_constraints,
            **constraints,
        )


class ValidUInt128(ValidColumn):
    def __init__(
            self,
            name: IntoNameVC = None,
            *,
            ge: int | float | decimal.Decimal | None = None,
            gt: int | float | decimal.Decimal | None = None,
            le: int | float | decimal.Decimal | None = None,
            lt: int | float | decimal.Decimal | None = None,
            is_between: IsBetweenTuple = None,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> None:
        _constraints = _int_constraints_remove_none(
            ge=ge,
            gt=gt,
            le=le,
            lt=lt,
            is_between=is_between,
        )
        dtype = pl.UInt128
        super().__init__(
            name=name,
            dtype=dtype,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **_constraints,
            **constraints,
        )


class ValidFloat(ValidColumn):
    def __init__(
            self,
            name: IntoNameVC = None,
            *,
            ge: int | float | decimal.Decimal | None = None,
            gt: int | float | decimal.Decimal | None = None,
            le: int | float | decimal.Decimal | None = None,
            lt: int | float | decimal.Decimal | None = None,
            is_between: IsBetweenTuple = None,
            is_infinite: bool | None = None,
            is_nan: bool | None = None,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> None:
        _constraints = _int_constraints_remove_none(
            ge=ge,
            gt=gt,
            le=le,
            lt=lt,
            is_between=is_between,
            is_infinite=is_infinite,
            is_nan=is_nan,
        )
        dtype = float
        super().__init__(
            name=name,
            dtype=dtype,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **_constraints,
            **constraints,
        )


class ValidFloat32(ValidColumn):
    def __init__(
            self,
            name: IntoNameVC = None,
            *,
            ge: int | float | decimal.Decimal | None = None,
            gt: int | float | decimal.Decimal | None = None,
            le: int | float | decimal.Decimal | None = None,
            lt: int | float | decimal.Decimal | None = None,
            is_between: IsBetweenTuple = None,
            is_infinite: bool | None = None,
            is_nan: bool | None = None,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> None:
        _constraints = _int_constraints_remove_none(
            ge=ge,
            gt=gt,
            le=le,
            lt=lt,
            is_between=is_between,
            is_infinite=is_infinite,
            is_nan=is_nan,
        )
        dtype = pl.Float32
        super().__init__(
            name=name,
            dtype=dtype,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **_constraints,
            **constraints,
        )


class ValidFloat64(ValidColumn):
    def __init__(
            self,
            name: IntoNameVC = None,
            *,
            ge: int | float | decimal.Decimal | None = None,
            gt: int | float | decimal.Decimal | None = None,
            le: int | float | decimal.Decimal | None = None,
            lt: int | float | decimal.Decimal | None = None,
            is_between: IsBetweenTuple = None,
            is_infinite: bool | None = None,
            is_nan: bool | None = None,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> None:
        _constraints = _int_constraints_remove_none(
            ge=ge,
            gt=gt,
            le=le,
            lt=lt,
            is_between=is_between,
            is_infinite=is_infinite,
            is_nan=is_nan,
        )
        dtype = pl.Float64
        super().__init__(
            name=name,
            dtype=dtype,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **_constraints,
            **constraints,
        )


class ValidDecimal(ValidColumn):
    def __init__(
            self,
            name: IntoNameVC = None,
            dtype: pl.Decimal | type[pl.Decimal] = pl.Decimal,
            *,
            ge: int | float | decimal.Decimal | None = None,
            gt: int | float | decimal.Decimal | None = None,
            le: int | float | decimal.Decimal | None = None,
            lt: int | float | decimal.Decimal | None = None,
            is_between: IsBetweenTuple = None,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> None:
        _constraints = _int_constraints_remove_none(
            ge=ge,
            gt=gt,
            le=le,
            lt=lt,
            is_between=is_between,
        )
        super().__init__(
            name=name,
            dtype=dtype,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **_constraints,
            **constraints,
        )


# ----------------------------------------------------------------------

def _constraints_remove_none(**d: Any) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def _list_constraints_remove_none(**d: Any) -> dict[str, Any]:
    return {f"list_{k}": v for k, v in d.items() if v is not None}


def _array_constraints_remove_none(**d: Any) -> dict[str, Any]:
    return {f"arr_{k}": v for k, v in d.items() if v is not None}


def _str_constraints_remove_none(**d: Any) -> dict[str, Any]:
    return {f"str_{k}": v for k, v in d.items() if v is not None}


def _int_constraints_remove_none(**d: Any) -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        if v is None:
            continue
        elif k == "is_between":
            out[k] = pl.all().is_between(*v)
        else:
            out[k] = v
    return out
