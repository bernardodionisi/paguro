from __future__ import annotations

import typing
from typing import TYPE_CHECKING, Any, Literal, Protocol

import polars as pl

from paguro.shared._typing._typing import IsBetweenTuple
from paguro.validation.shared._docs import set_doc_string, \
    VALID_COLUMNS_SHARED_PARAMETERS

from paguro.validation.valid_column.valid_column import ValidColumn

from paguro.validation.valid_column.utils._vdtypes import (
    ValidStruct,
    ValidEnum,
    ValidCategorical,
    ValidString,
    ValidBinary,
    ValidBoolean,
    ValidDate,
    ValidDateTime,
    ValidDuration,
    ValidTime,
    ValidArray,
    ValidList,
    ValidNumeric,
    ValidInteger,
    ValidInt8,
    ValidInt16,
    ValidInt32,
    ValidInt64,
    ValidInt128,
    ValidUInteger,
    ValidUInt8,
    ValidUInt16,
    ValidUInt32,
    ValidUInt64,
    ValidUInt128,
    ValidFloat,
    ValidFloat32,
    ValidFloat64,
    ValidDecimal,
)
from polars.selectors import Selector

if TYPE_CHECKING:
    from paguro.shared.dtypes.into_dtypes import IntoDataType

    from collections.abc import Iterable

    import decimal
    import enum
    import datetime
    from polars._typing import TimeUnit, PolarsDataType, PythonDataType
    from paguro.typing import FieldsValidators

__all__ = [
    "VCol"
]


class ValidColumnFactory(Protocol):
    def __call__(
            self,
            dtype: IntoDataType | None = None,
            *,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> ValidColumn: ...


class VCol:
    """
    ValidColumn constructor
    """

    @set_doc_string(additional_parameters=VALID_COLUMNS_SHARED_PARAMETERS)
    def __call__(
            self,
            name: str | typing.Collection[str] | Selector | None = None,
            dtype: IntoDataType | None = None,
            *,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> ValidColumn:
        """
        Use the `vcol` as a function to construct a ValidColumn.

        Parameters
        ----------
        name
            The column name.
        dtype
            The expected column data type:
                - a polars datatype
                - python types:
                    int: all the polars integers (signed)
                    float: all the polars floats
                    str: polars string
{{ AdditionalParameters }}
        """
        return ValidColumn._(
            name=name,
            dtype=dtype,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            constraints=constraints
        )

    @classmethod
    def _(
            cls,
            name: str | typing.Collection[str] | Selector | None = None,
            dtype: IntoDataType | None = None,
            *,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            constraints: dict[str, Any] | None = None,
    ) -> ValidColumn:
        return ValidColumn._(
            name=name,
            dtype=dtype,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            constraints=constraints,
        )

    def __getattr__(self, attr: str) -> ValidColumnFactory:
        if attr.startswith("__"):
            raise AttributeError(attr)

        def _valid_column_factory(
                dtype: IntoDataType | None = None,
                *,
                required: bool | Literal["dynamic"] = True,
                allow_nulls: bool = False,
                unique: bool = False,
                **constraints: Any,
        ) -> ValidColumn:
            return ValidColumn(
                name=attr,
                dtype=dtype,
                required=required,
                allow_nulls=allow_nulls,
                unique=unique,
                **constraints,
            )

        return _valid_column_factory  # type: ignore

    @set_doc_string(additional_parameters=VALID_COLUMNS_SHARED_PARAMETERS)
    def Struct(  # noqa: N802
            self,
            name: str | typing.Collection[str] | Selector | None = None,
            dtype: pl.Struct | type[pl.Struct] | None = pl.Struct,
            *,
            fields: FieldsValidators | None = None,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> ValidStruct:
        """
        Struct valid column constructor.

        Parameters
        ----------
        name
            The column name.
        dtype
            The expected column data type: you can pass a fully defined Stuct,
            the default is the Struct base with no fields specified.
{{ AdditionalParameters }}

        Examples
        --------

        .. ipython:: python

            print(
                pg.vcol.Struct()
            )

        .. ipython:: python

            print(
                pg.vcol.Struct(pg.vcol("field", ge=1))
            )

        .. ipython:: python

            print(
                pg.vcol.Struct(pg.vcol.Struct(pg.vcol("field", ge=1)))
            )
        """
        return ValidStruct(
            name=name,
            dtype=dtype,
            fields=fields,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **constraints
        )

    @set_doc_string(additional_parameters=VALID_COLUMNS_SHARED_PARAMETERS)
    def Enum(  # noqa: N802
            self,
            name: str | typing.Collection[str] | Selector | None = None,
            dtype: pl.Enum | type[pl.Enum] = pl.Enum,
            *,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> ValidEnum:
        """
        Enum valid column constructor.

        Parameters
        ----------
        name
        categories
{{ AdditionalParameters }}

        Examples
        --------

        .. ipython:: python

            print(
                pg.vcol.Enum()
            )
        """
        return ValidEnum(
            name=name,
            dtype=dtype,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **constraints
        )

    @set_doc_string(additional_parameters=VALID_COLUMNS_SHARED_PARAMETERS)
    def Categorical(  # noqa: N802
            self,
            name: str | typing.Collection[str] | Selector | None = None,
            *,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> ValidCategorical:
        """
        Categorical valid column constructor.

        Parameters
        ----------
        name
{{ AdditionalParameters }}

        Examples
        --------

        .. ipython:: python

            print(
                pg.vcol.Categorical()
            )
        """
        return ValidCategorical(
            name=name,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **constraints
        )

    @set_doc_string(additional_parameters=VALID_COLUMNS_SHARED_PARAMETERS)
    def String(  # noqa: N802
            self,
            name: str | typing.Collection[str] | Selector | None = None,
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
    ) -> ValidString:
        """
        String valid column constructor.

        Parameters
        ----------
        name
        contains
        contains_any
        starts_with
        ends_with
        len_chars_eq
        len_chars_ge
        len_chars_gt
        len_chars_le
        len_chars_lt

{{ AdditionalParameters }}

        Examples
        --------

        .. ipython:: python

            print(
                pg.vcol.String()
            )
        """
        return ValidString(
            name=name,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            contains=contains,
            contains_any=contains_any,
            starts_with=starts_with,
            ends_with=ends_with,
            len_chars_eq=len_chars_eq,
            len_chars_ge=len_chars_ge,
            len_chars_gt=len_chars_gt,
            len_chars_le=len_chars_le,
            len_chars_lt=len_chars_lt,
            **constraints,
        )

    @set_doc_string(additional_parameters=VALID_COLUMNS_SHARED_PARAMETERS)
    def Binary(  # noqa: N802
            self,
            name: str | typing.Collection[str] | Selector | None = None,
            *,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> ValidBinary:
        """
        Binary valid column constructor.

        Parameters
        ----------
        name
{{ AdditionalParameters }}

        Examples
        --------

        .. ipython:: python

            print(
                pg.vcol.String()
            )
        """
        return ValidBinary(
            name=name,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **constraints
        )

    @set_doc_string(additional_parameters=VALID_COLUMNS_SHARED_PARAMETERS)
    def Boolean(  # noqa: N802
            self,
            name: str | typing.Collection[str] | Selector | None = None,
            *,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> ValidBoolean:
        """
        Boolean valid column constructor.

        Parameters
        ----------
        name
            The column name.
{{ AdditionalParameters }}

        Examples
        --------

        .. ipython:: python

            print(
                pg.vcol.String()
            )
        """
        return ValidBoolean(
            name=name,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **constraints
        )

    @set_doc_string(additional_parameters=VALID_COLUMNS_SHARED_PARAMETERS)
    def Date(  # noqa: N802
            self,
            name: str | typing.Collection[str] | Selector | None = None,
            *,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> ValidDate:
        """
        Date valid column constructor.

        Parameters
        ----------
        name
            The column name.
{{ AdditionalParameters }}

        Examples
        --------

        .. ipython:: python

            print(
                pg.vcol.Date()
            )
        """
        return ValidDate(
            name=name,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **constraints
        )

    @set_doc_string(additional_parameters=VALID_COLUMNS_SHARED_PARAMETERS)
    def DateTime(  # noqa: N802
            self,
            name: str | typing.Collection[str] | Selector | None = None,
            dtype: pl.Datetime | type[pl.Datetime] = pl.Datetime,
            *,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> ValidDateTime:
        """
        DateTime valid column constructor.

        Parameters
        ----------
        name
            The column name.

{{ AdditionalParameters }}

        Examples
        --------

        .. ipython:: python

            print(
                pg.vcol.DateTime()
            )
        """
        return ValidDateTime(
            name=name,
            dtype=dtype,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **constraints
        )

    @set_doc_string(additional_parameters=VALID_COLUMNS_SHARED_PARAMETERS)
    def Duration(  # noqa: N802
            self,
            name: str | typing.Collection[str] | Selector | None = None,
            dtype: pl.Duration | type[pl.Duration] = pl.Duration,
            *,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> ValidDuration:
        """
        Duration valid column constructor.

        Parameters
        ----------
        name
        time_unit
{{ AdditionalParameters }}

        Examples
        --------

        .. ipython:: python

            print(
                pg.vcol.Duration()
            )
        """
        return ValidDuration(
            name=name,
            dtype=dtype,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **constraints
        )

    @set_doc_string(additional_parameters=VALID_COLUMNS_SHARED_PARAMETERS)
    def Time(  # noqa: N802
            self,
            name: str | typing.Collection[str] | Selector | None = None,
            *,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> ValidTime:
        """
        Time valid column constructor.

        Parameters
        ----------
        name
{{ AdditionalParameters }}

        Examples
        --------

        .. ipython:: python

            print(
                pg.vcol.Date()
            )
        """
        return ValidTime(
            name=name,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **constraints
        )

    @set_doc_string(additional_parameters=VALID_COLUMNS_SHARED_PARAMETERS)
    def Array(  # noqa: N802
            self,
            name: str | typing.Collection[str] | Selector | None = None,
            dtype: pl.Array | type[pl.Array] = pl.Array,
            *,
            contains: Any | None = None,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> ValidArray:
        """
        Array valid column constructor.

        Parameters
        ----------
        name
        inner
        shape
        contains
{{ AdditionalParameters }}

        Examples
        --------

        .. ipython:: python

            print(
                pg.vcol.Array()
            )
        """
        return ValidArray(
            name=name,
            dtype=dtype,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            contains=contains,
            **constraints
        )

    @set_doc_string(additional_parameters=VALID_COLUMNS_SHARED_PARAMETERS)
    def List(  # noqa: N802
            self,
            name: str | typing.Collection[str] | Selector | None = None,
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
    ) -> ValidList:
        """
        List valid column constructor.

        Parameters
        ----------
        name
        contains
        len_ge
        len_gt
        len_le
        len_lt
{{ AdditionalParameters }}

        Examples
        --------

        .. ipython:: python

            print(
                pg.vcol.List()
            )
        """
        return ValidList(
            name=name,
            dtype=dtype,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            contains=contains,
            len_ge=len_ge,
            len_gt=len_gt,
            len_le=len_le,
            len_lt=len_lt,
            **constraints
        )

    @set_doc_string(additional_parameters=VALID_COLUMNS_SHARED_PARAMETERS)
    def Decimal(  # noqa: N802
            self,
            name: str | typing.Collection[str] | Selector | None = None,
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
    ) -> ValidDecimal:
        """
        Decimal valid column constructor.

        Parameters
        ----------
        name
        ge
        gt
        le
        lt
        is_between
{{ AdditionalParameters }}

        Examples
        --------

        .. ipython:: python

            print(
                pg.vcol.Decimal()
            )
        """
        return ValidDecimal(
            name=name,
            dtype=dtype,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            ge=ge,
            gt=gt,
            le=le,
            lt=lt,
            is_between=is_between,
            **constraints
        )

    @set_doc_string(additional_parameters=VALID_COLUMNS_SHARED_PARAMETERS)
    def Numeric(  # noqa: N802
            self,
            name: str | typing.Collection[str] | Selector | None = None,
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
    ) -> ValidNumeric:
        """
        Numeric valid column constructor.

        Parameters
        ----------
        name
        ge
        gt
        le
        lt
        is_between
        is_infinite
        is_nan
{{ AdditionalParameters }}

        Examples
        --------

        .. ipython:: python

            print(
                pg.vcol.Numeric()
            )
        """
        return ValidNumeric(
            name=name,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            ge=ge,
            gt=gt,
            le=le,
            lt=lt,
            is_between=is_between,
            is_infinite=is_infinite,
            is_nan=is_nan,
            **constraints
        )

    @set_doc_string(additional_parameters=VALID_COLUMNS_SHARED_PARAMETERS)
    def Integer(  # noqa: N802
            self,
            name: str | typing.Collection[str] | Selector | None = None,
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
    ) -> ValidInteger:
        """
        Integer valid column constructor.

        Parameters
        ----------
        name
        ge
        gt
        le
        lt
        is_between
{{ AdditionalParameters }}

        Examples
        --------

        .. ipython:: python

            print(
                pg.vcol.Integer()
            )
        """
        return ValidInteger(
            name=name,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            ge=ge,
            gt=gt,
            le=le,
            lt=lt,
            is_between=is_between,
            **constraints
        )

    @set_doc_string(additional_parameters=VALID_COLUMNS_SHARED_PARAMETERS)
    def Int8(  # noqa: N802
            self,
            name: str | typing.Collection[str] | Selector | None = None,
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
    ) -> ValidInt8:
        """
        Int8 valid column constructor.

        Parameters
        ----------
        name
        ge
        gt
        le
        lt
        is_between
{{ AdditionalParameters }}

        Examples
        --------

        .. ipython:: python

            print(
                pg.vcol.Int8()
            )
        """
        return ValidInt8(
            name=name,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            ge=ge,
            gt=gt,
            le=le,
            lt=lt,
            is_between=is_between,
            **constraints
        )

    @set_doc_string(additional_parameters=VALID_COLUMNS_SHARED_PARAMETERS)
    def Int16(  # noqa: N802
            self,
            name: str | typing.Collection[str] | Selector | None = None,
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
    ) -> ValidInt16:
        """
        Int16 valid column constructor.

        Parameters
        ----------
        name
        ge
        gt
        le
        lt
        is_between
{{ AdditionalParameters }}

        Examples
        --------

        .. ipython:: python

            print(
                pg.vcol.Int16()
            )
        """
        return ValidInt16(
            name=name,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            ge=ge,
            gt=gt,
            le=le,
            lt=lt,
            is_between=is_between,
            **constraints
        )

    @set_doc_string(additional_parameters=VALID_COLUMNS_SHARED_PARAMETERS)
    def Int32(  # noqa: N802
            self,
            name: str | typing.Collection[str] | Selector | None = None,
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
    ) -> ValidInt32:
        """
        Int32 valid column constructor.

        Parameters
        ----------
        name
        ge
        gt
        le
        lt
        is_between
{{ AdditionalParameters }}

        Examples
        --------

        .. ipython:: python

            print(
                pg.vcol.Int32()
            )
        """
        return ValidInt32(
            name=name,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            ge=ge,
            gt=gt,
            le=le,
            lt=lt,
            is_between=is_between,
            **constraints
        )

    @set_doc_string(additional_parameters=VALID_COLUMNS_SHARED_PARAMETERS)
    def Int64(  # noqa: N802
            self,
            name: str | typing.Collection[str] | Selector | None = None,
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
    ) -> ValidInt64:
        """
        Int64 valid column constructor.

        Parameters
        ----------
        name
        ge
        gt
        le
        lt
        is_between
{{ AdditionalParameters }}

        Examples
        --------

        .. ipython:: python

            print(
                pg.vcol.Int64()
            )
        """
        return ValidInt64(
            name=name,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            ge=ge,
            gt=gt,
            le=le,
            lt=lt,
            is_between=is_between,
            **constraints
        )

    @set_doc_string(additional_parameters=VALID_COLUMNS_SHARED_PARAMETERS)
    def Int128(  # noqa: N802
            self,
            name: str | typing.Collection[str] | Selector | None = None,
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
    ) -> ValidInt128:
        """
        Int128 valid column constructor.

        Parameters
        ----------
        name
        ge
        gt
        le
        lt
        is_between
{{ AdditionalParameters }}

        Examples
        --------

        .. ipython:: python

            print(
                pg.vcol.Int64()
            )
        """
        return ValidInt128(
            name=name,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            ge=ge,
            gt=gt,
            le=le,
            lt=lt,
            is_between=is_between,
            **constraints
        )

    @set_doc_string(additional_parameters=VALID_COLUMNS_SHARED_PARAMETERS)
    def UInteger(  # noqa: N802
            self,
            name: str | typing.Collection[str] | Selector | None = None,
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
    ) -> ValidUInteger:
        """
        UInteger valid column constructor.

        Parameters
        ----------
        name
        ge
        gt
        le
        lt
        is_between
{{ AdditionalParameters }}

        Examples
        --------

        .. ipython:: python

            print(
                pg.vcol.UInteger()
            )
        """
        return ValidUInteger(
            name=name,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            ge=ge,
            gt=gt,
            le=le,
            lt=lt,
            is_between=is_between,
            **constraints
        )

    @set_doc_string(additional_parameters=VALID_COLUMNS_SHARED_PARAMETERS)
    def UInt8(  # noqa: N802
            self,
            name: str | typing.Collection[str] | Selector | None = None,
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
    ) -> ValidUInt8:
        """
        UInt8 valid column constructor.

        Parameters
        ----------
        name
        ge
        gt
        le
        lt
        is_between
{{ AdditionalParameters }}

        Examples
        --------

        .. ipython:: python

            print(
                pg.vcol.UInt8()
            )
        """
        return ValidUInt8(
            name=name,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            ge=ge,
            gt=gt,
            le=le,
            lt=lt,
            is_between=is_between,
            **constraints
        )

    @set_doc_string(additional_parameters=VALID_COLUMNS_SHARED_PARAMETERS)
    def UInt16(  # noqa: N802
            self,
            name: str | typing.Collection[str] | Selector | None = None,
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
    ) -> ValidUInt16:
        """
        UInt16 valid column constructor.

        Parameters
        ----------
        name
        ge
        gt
        le
        lt
        is_between
{{ AdditionalParameters }}

        Examples
        --------

        .. ipython:: python

            print(
                pg.vcol.UInt16()
            )
        """
        return ValidUInt16(
            name=name,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            ge=ge,
            gt=gt,
            le=le,
            lt=lt,
            is_between=is_between,
            **constraints
        )

    @set_doc_string(additional_parameters=VALID_COLUMNS_SHARED_PARAMETERS)
    def UInt32(  # noqa: N802
            self,
            name: str | typing.Collection[str] | Selector | None = None,
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
    ) -> ValidUInt32:
        """
        UInt32 valid column constructor.

        Parameters
        ----------
        name
        ge
        gt
        le
        lt
        is_between
{{ AdditionalParameters }}

        Examples
        --------

        .. ipython:: python

            print(
                pg.vcol.UInt32()
            )
        """
        return ValidUInt32(
            name=name,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            ge=ge,
            gt=gt,
            le=le,
            lt=lt,
            is_between=is_between,
            **constraints
        )

    @set_doc_string(additional_parameters=VALID_COLUMNS_SHARED_PARAMETERS)
    def UInt64(  # noqa: N802
            self,
            name: str | typing.Collection[str] | Selector | None = None,
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
    ) -> ValidUInt64:
        """
        UInt64 valid column constructor.

        Parameters
        ----------
        name
        ge
        gt
        le
        lt
        is_between
{{ AdditionalParameters }}

        Examples
        --------

        .. ipython:: python

            print(
                pg.vcol.UInt64()
            )
        """
        return ValidUInt64(
            name=name,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            ge=ge,
            gt=gt,
            le=le,
            lt=lt,
            is_between=is_between,
            **constraints
        )

    @set_doc_string(additional_parameters=VALID_COLUMNS_SHARED_PARAMETERS)
    def UInt128(  # noqa: N802
            self,
            name: str | typing.Collection[str] | Selector | None = None,
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
    ) -> ValidUInt128:
        """
        UInt128 valid column constructor.

        Parameters
        ----------
        name
        ge
        gt
        le
        lt
        is_between
{{ AdditionalParameters }}

        Examples
        --------

        .. ipython:: python

            print(
                pg.vcol.UInt128()
            )
        """
        return ValidUInt128(
            name=name,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            ge=ge,
            gt=gt,
            le=le,
            lt=lt,
            is_between=is_between,
            **constraints
        )

    @set_doc_string(additional_parameters=VALID_COLUMNS_SHARED_PARAMETERS)
    def Float(  # noqa: N802
            self,
            name: str | typing.Collection[str] | Selector | None = None,
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
    ) -> ValidFloat:
        """
        Float valid column constructor.

        Parameters
        ----------
        name
        ge
        gt
        le
        lt
        is_between
        is_infinite
        is_nan
{{ AdditionalParameters }}

        Examples
        --------

        .. ipython:: python

            print(
                pg.vcol.Float()
            )
        """
        return ValidFloat(
            name=name,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            ge=ge,
            gt=gt,
            le=le,
            lt=lt,
            is_between=is_between,
            is_infinite=is_infinite,
            is_nan=is_nan,
            **constraints
        )

    @set_doc_string(additional_parameters=VALID_COLUMNS_SHARED_PARAMETERS)
    def Float32(  # noqa: N802
            self,
            name: str | typing.Collection[str] | Selector | None = None,
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
    ) -> ValidFloat32:
        """
        Float32 valid column constructor.

        Parameters
        ----------
        name
        ge
        gt
        le
        lt
        is_between
        is_infinite
        is_nan
{{ AdditionalParameters }}

        Examples
        --------

        .. ipython:: python

            print(
                pg.vcol.Float32()
            )
        """
        return ValidFloat32(
            name=name,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            ge=ge,
            gt=gt,
            le=le,
            lt=lt,
            is_between=is_between,
            is_infinite=is_infinite,
            is_nan=is_nan,
            **constraints
        )

    @set_doc_string(additional_parameters=VALID_COLUMNS_SHARED_PARAMETERS)
    def Float64(  # noqa: N802
            self,
            name: str | typing.Collection[str] | Selector | None = None,
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
    ) -> ValidFloat64:
        """
        Float64 valid column constructor.

        Parameters
        ----------
        name
        ge
        gt
        le
        lt
        is_between
        is_infinite
        is_nan
{{ AdditionalParameters }}

        Examples
        --------

        .. ipython:: python

            print(
                pg.vcol.Float64()
            )
        """
        return ValidFloat64(
            name=name,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            ge=ge,
            gt=gt,
            le=le,
            lt=lt,
            is_between=is_between,
            is_infinite=is_infinite,
            is_nan=is_nan,
            **constraints
        )
