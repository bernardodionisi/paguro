from __future__ import annotations

import warnings

import typing
from typing import TYPE_CHECKING, Any, Literal

import polars as pl
import polars.selectors as cs
from polars.selectors import Selector

from paguro.ashi.info.info_collection import InfoCollection

from paguro.shared.dtypes.build_struct import (
    build_struct_from_dtype_or_fields,
)
from paguro.shared.dtypes.dtype_serialize import (
    dict_to_dtype_list,
)
from paguro.shared.dtypes.into_dtypes import parse_dtype_into_frozenset
from paguro.shared.dtypes.supertype import find_supertype_multiple
from paguro.utils.dependencies import copy
from paguro.validation.exception.errors.validation_error import ValidationError
from paguro.validation.exception.utils.filter_utils import _gather_predicates

from paguro.validation.exception.validate_dispatch import (
    _validate_dispatch,
)
from paguro.validation.shared._docs import set_doc_string, VALIDATE_PARAMS
from paguro.validation.shared.cast import cast_frame
from paguro.validation.shared.find_v import find_unique_vcol, find_all_vcols_multi, \
    find_unique_vframe, find_all_vframes_multi
from paguro.validation.shared.keep_columns import _select_keep_columns
from paguro.validation.shared.utils import (
    data_to_frame_like,
    data_to_lazyframe,
)
from paguro.validation.valid_base.valid_base import _ValidBase
from paguro.validation.valid_column.utils.dtype_errors import dtype_errors

from paguro.validation.valid_column.utils.exprs.predicates import (
    get_allow_nulls_predicate,
    get_unique_predicate,
    get_struct_expr
)

from paguro.validation.valid_column.utils.exprs.replace_expr import (
    _dispatch_expr_args_for_errors,
    _dispatch_expr_args_for_predicates
)
from paguro.validation.valid_column.utils.required import _suggest_columns

from paguro.validation.valid_column.utils.utils import (
    _get_duplicates,
    _get_nulls,
    _has_additional_columns,
    _null_counts,
    _select_and_filter,
)
from paguro.validation.valid_column.utils.exprs.build_expression import _build_expr
from paguro.shared._typing import typed_dicts
from paguro.typing import CollectConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable
    from polars.datatypes.classes import DataType, DataTypeClass

    from paguro.validation.valid_frame.valid_frame import ValidFrame
    from paguro.shared.dtypes.into_dtypes import IntoDataType
    from paguro.typing import (
        IntoKeepColumns,
        IntoValidation,
        OnSuccess,
        FrameLike,
        OnFailureExtra, ValidationMode,
    )
    from paguro.validation.validation import Validation
    import sys

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

__all__ = [
    "ValidColumn",
]


# TODO: check that required="dynamic" works as expected when name is None or Selector

class ValidColumn(_ValidBase):

    def __init__(
            self,
            name: str | typing.Collection[str] | Selector | None = None,
            dtype: IntoDataType | None = None,
            *,
            required: bool | Literal["dynamic"] = True,
            allow_nulls: bool = False,
            unique: bool = False,
            **constraints: Any,
    ) -> None:
        """
        Column validator.

        Parameters
        ----------
        name
            The name of the column to validate.

            See :ref:`example <tutorials-shorts-vcol-by parameter-name>`

        dtype
            Expected data type (e.g. `pl.Int64`, `pl.Utf8`, or a custom dtype alias).
            If provided, a type‑mismatch will raise a `ValidationError`.

            See :ref:`example <tutorials-shorts-vcol-dtype>`

        required
            - If `True`, the column must be present.
            - If `"dynamic"`, the column will become required as soon the frame as a column
              with the respective name in it
            - If `False`, missing column is allowed and will skip all further checks.

            See :ref:`example <tutorials-shorts-vcol-required>`

        allow_nulls
            Whether null values are allowed. If `False`, any nulls will fail validation.

            See :ref:`example <tutorials-shorts-vcol-allow_nulls>`

        unique

            See :ref:`example <tutorials-shorts-vcol-unique>`

        fields
            For struct‑type columns, you can supply nested validation rules
            (e.g. child columns or frame‑level checks on the struct’s contents).

            See :ref:`example <tutorials-shorts-vcol-fields>`

        **constraints
            Additional named Polars boolean expressions to apply to this column.
            Each expression is evaluated elementwise; all must be true for the
            column to pass.

            See :ref:`example <tutorials-shorts-vcol-constraints>`
        """
        if name is not None and not isinstance(name, (str, Selector)):
            name = cs.by_name(name)

        super().__init__(name=name, constraints=constraints)

        self._dtype = parse_dtype_into_frozenset(dtype)

        self._required: bool | Literal["dynamic"] = required

        self._allow_nulls: bool = allow_nulls
        self._unique: bool = unique

        self._allow_rename: bool = True

        # allow_drop: useful only in ValidColumnList context
        self._allow_drop: bool = True

        self._fields: Validation | None = None
        self._field_parents: tuple[str, ...] | None = None

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
        if constraints is None:
            constraints = {}
        return cls(
            name=name,
            dtype=dtype,
            required=required,
            allow_nulls=allow_nulls,
            unique=unique,
            **constraints,
        )

    # def __deepcopy__(self, memo: dict) -> Self:
    #     cls = self.__class__
    #     new = cls.__new__(cls)
    #     memo[id(self)] = new
    #
    #     # Immutable - shallow copy is fine
    #     new._name = self._name
    #     new._dtype = self._dtype
    #     new._required = self._required
    #     new._allow_nulls = self._allow_nulls
    #     new._unique = self._unique
    #     new._allow_rename = self._allow_rename
    #     new._allow_drop = self._allow_drop
    #
    #     # Mutable - deep copy
    #     new._constraints = copy.deepcopy(self._constraints, memo)
    #     new._fields = copy.deepcopy(self._fields, memo) if self._fields else None
    #     new._info = copy.deepcopy(self._info, memo) if self._info else None
    #
    #     return new

    # def __copy__(self) -> Self:
    #     return self.__deepcopy__()

    def __repr__(self) -> str:
        name = (
            f"{self._name!r}"
            if isinstance(self._name, str)
            else f"{self._name}"
        )

        # We'll switch to {self.__class__.__qualname__} once we have a plan for the dtype API
        return f"ValidColumn({name}, ...) [at {hex(id(self))}]"

    def __str__(self) -> str:
        name = (
            f"{self._name!r}"
            if isinstance(self._name, str)
            else f"{self._name}"
        )

        dtype = self._dtype
        if dtype is not None:
            dtype = list(dtype)

        if self._constraints:
            # should we display them in order
            constraints = set(self._constraints.keys())
        else:
            constraints = {}

        # class_name = {self.__class__.__qualname__}
        class_name = "ValidColumn"
        return (
            f"{class_name}("
            f"name={name}, "
            f"dtype={dtype}, "
            f"required={self._required}, "
            f"allow_nulls={self._allow_nulls}, "
            f"unique={self._unique}, "
            f"constraints={constraints})"
        )

    def __call__(self) -> pl.Expr:
        if isinstance(self._name, str):
            if self._field_parents is None:
                return pl.col(self._name)
            else:
                return get_struct_expr(self._field_parents).struct.field(self._name)
        else:
            if self._field_parents is not None:
                warnings.warn(
                    f"Fields parents ignored for column with non string name: {self._name}"
                )

            if self._name is None:
                return cs.all()
            else:
                return self._name  # selector

    # ------------------------------------------------------------------

    def _set_struct_dtype_from_fields(self, *, replace: bool) -> None:
        dtype = build_struct_from_dtype_or_fields(self)
        if replace:
            self._dtype = frozenset([dtype])

    def _set_allow_rename(self, *, allow_rename: bool) -> Self:
        # TODO: (do not allow allow_rename=False if None or selector)
        # TODO: make non inplace before making public
        self._allow_rename = allow_rename
        return self

    def _set_allow_drop(self, *, allow_drop: bool) -> Self:
        # TODO: make non inplace before making public
        self._allow_drop = allow_drop
        return self

    # ------------------------------------------------------------------

    def _find_vcol(
            self,
            name: str | None | Selector,
            *,
            include_transformed_frames: bool = False,
            include_fields: bool = False,
            return_first: bool = False,
    ) -> ValidColumn | None:
        return find_unique_vcol(
            root=self._fields,
            name=name,
            return_first=return_first,
            include_transformed_frames=include_transformed_frames,
            include_fields=include_fields,
        )

    def _find_vcols(
            self,
            names: Iterable[str | None | Selector],
            *,
            include_transformed_frames: bool = False,
            include_fields: bool = False,
            dedupe: bool = True,
            group_by_name: bool = False,
    ) -> list[ValidColumn] | dict[str | None | Selector, list[ValidColumn]]:
        return find_all_vcols_multi(
            root=self._fields,
            names=names,
            include_transformed_frames=include_transformed_frames,
            include_fields=include_fields,
            dedupe=dedupe,
            group_by_name=group_by_name,
        )

    def _find_vframe(
            self,
            name: str | None,
            *,
            include_transformed_frames: bool = False,
            include_fields: bool = False,
            return_first: bool = False,
    ) -> ValidFrame | None:
        return find_unique_vframe(
            root=self._fields,
            name=name,
            return_first=return_first,
            include_transformed_frames=include_transformed_frames,
            include_fields=include_fields,
        )

    def _find_vframes(
            self,
            names: Iterable[str | None],
            *,
            include_transformed_frames: bool = False,
            include_fields: bool = False,
            dedupe: bool = True,
            group_by_name: bool = False,
    ) -> list[ValidFrame] | dict[str | None, list[ValidFrame]]:
        return find_all_vframes_multi(
            root=self._fields,
            names=names,
            include_transformed_frames=include_transformed_frames,
            include_fields=include_fields,
            dedupe=dedupe,
            group_by_name=group_by_name,
        )

    # ------------------------------------------------------------------

    # @property
    # def name(self) -> str | Selector | None:  # or Selector
    #     return self._name
    #
    # @name.setter
    # def name(self, value: str | Selector | None) -> None:
    #     if not self._allow_rename:
    #         raise ValueError(
    #             f"Renaming is not allowed for vcol={self._name!r}. "
    #             f"Please set allow_rename=True to be able to rename the column."
    #         )
    #     if value is None:
    #         self._name = None
    #     elif isinstance(value, (str, Selector)):
    #         self._name = value
    #     else:
    #         msg = f"name must be a str or Selector, not {type(value)}"
    #         raise TypeError(msg)

    def with_name(self, name: str | Selector | None) -> Self:
        """
        Renames the column validator.

        Group
        -----
            Settings
        """
        if not self._allow_rename:
            raise ValueError(
                f"Renaming is not allowed for vcol={self._name!r}. "
                f"Please set allow_rename=True to be able to rename the column."
            )

        new = copy.deepcopy(self)

        if name is None:
            new._name = None
        elif isinstance(name, (str, Selector)):
            new._name = name
        else:
            msg = f"name must be a str or Selector, not {type(name)}"
            raise TypeError(msg)
        return new

    # ------------------------------------------------------------------

    def to_schema(
            self,
            *,
            build_struct: bool = True,
            fully_specified: bool = True,
    ) -> pl.Schema:
        # convenience method for single column schema

        if self._dtype is None and not build_struct:
            msg = f"{self._name}'s dtype is not set. Unable to construct schema."
            raise TypeError(msg)

        schema = {
            self._name: self._get_supertype(
                build_struct=build_struct,
            )
        }
        return pl.Schema(schema, check_dtypes=fully_specified)

    def _get_supertype(
            self,
            *,
            build_struct: bool = False,
    ) -> DataTypeClass | pl.DataType:
        if self._dtype is None and self._fields is None:
            # this may be a bit confusing
            # because determining the supertype is internal
            # one instance in which we call it
            # is when we are trying to build a struct from fields
            msg = f"dtype is None. Unable to determine supertype of {self._name!r}"
            raise TypeError(msg)

        if build_struct:
            return build_struct_from_dtype_or_fields(self)
        else:
            if self._dtype is None and self._fields is not None:
                return pl.Struct
            else:
                return find_supertype_multiple(self._dtype)

    # ------------------------------------------------------------------

    def _to_dict(
            self, *, _fingerprint: bool, include_info: bool
    ) -> dict[str, Any]:

        fields = None
        if self._fields is not None:
            fields = self._fields._to_dict(
                _fingerprint=_fingerprint,
                include_info=include_info
            )

        out = {
            "name": self._name,
            "dtype": self._dtype,
            "required": self._required,
            "allow_nulls": self._allow_nulls,
            "unique": self._unique,
            "fields": fields,
            "constraints": self._constraints,
            # -------
            "allow_rename": self._allow_rename,
            "allow_drop": self._allow_drop,
            # TODO: add self._set_membership
            # -------
        }

        if include_info:
            out["info"] = (
                None if self._info is None else self._info.to_dict()
            )

        return out

    @classmethod
    def _from_dict(cls, source: dict) -> ValidColumn:
        vcol_dict = copy.deepcopy(source)
        # vcol_dict: dict = deserialize(source, cls=CustomJSONDecoder)

        dtype = vcol_dict.get("dtype")

        if dtype is not None:
            vcol_dict["dtype"] = dict_to_dtype_list(dtype)
        else:
            vcol_dict["dtype"] = None

        allow_drop = vcol_dict.pop("allow_drop", True)
        allow_rename = vcol_dict.pop("allow_rename", False)

        # TODO: check that fields deserializes correctly
        fields = vcol_dict.pop("fields")
        if fields is not None:
            from paguro.validation.validation import Validation
            fields = Validation._from_dict(fields)

        _info: dict | None = vcol_dict.pop("info", None)

        # --------

        instance = cls._(**vcol_dict, )

        instance._fields = fields
        instance = instance._set_allow_drop(allow_drop=allow_drop)
        instance = instance._set_allow_rename(allow_rename=allow_rename)

        if _info is not None:
            instance._info = InfoCollection._from_dict_with_attributes(
                info_list_dict=_info)
        # else:
        #     instance._info = None

        return instance

    # ------------------------------------------------------------------

    # docs building tool is sharing base methods in the docs,
    # keeping here so ValidFrame gets its own.
    # Otherwise we could just delete and have ValidBase.validate
    @set_doc_string(parameters=VALIDATE_PARAMS)
    def validate(
            self,
            data: IntoValidation,
            *,
            mode: ValidationMode = "all",
            keep_columns: IntoKeepColumns = False,
            collect: bool | CollectConfig = True,
            on_success: OnSuccess = "return_none",
            on_failure: OnFailureExtra = "raise",
            cast: bool = False,
    ) -> FrameLike | ValidationError | None:
        """
        Validate the target data using the validator.

{{ Parameters }}

        Group
        -----
            Validation
        """
        return self._validate(
            data=data,
            mode=mode,
            keep_columns=keep_columns,
            with_row_index=False,
            collect=collect,
            on_success=on_success,
            on_failure=on_failure,
            get_expr=_build_expr,
            cast=cast,
        )

    def _validate(
            self,
            data: IntoValidation,
            *,
            mode: ValidationMode,
            keep_columns: IntoKeepColumns,
            with_row_index: bool | str,
            collect: bool | CollectConfig,
            on_success: OnSuccess,
            on_failure: OnFailureExtra,
            get_expr: Callable[[str, Any, str | pl.Expr | None], pl.Expr],
            cast: bool,
    ) -> FrameLike | ValidationError | None:
        # expand validation if _name is None or Selector
        if not isinstance(self._name, str):
            from paguro.validation.validation import Validation

            validation = Validation(
                self
            )  # passing vcol to Validation to delegate expansion
            return validation._validate(
                data=data,
                mode=mode,
                keep_columns=keep_columns,
                with_row_index=with_row_index,
                collect=collect,
                on_success=on_success,
                on_failure=on_failure,
                get_expr=get_expr,
                cast=cast,
            )

        data = data_to_frame_like(data=data)

        if cast:
            data = cast_frame(
                frame=data, schema=self.to_schema(fully_specified=True),
            )

        errors: typed_dicts.ValidColumnErrors = self._gather_errors(
            frame=data_to_lazyframe(data=data),
            mode=mode,
            keep_columns=keep_columns,
            with_row_index=with_row_index,
            get_expr=get_expr,
            cast=cast,
            _struct_fields=None,
        )

        out: typed_dicts.ValidationErrors = {
            "valid_column_list": {self._name: errors}
        }

        return _validate_dispatch(
            errors=out,
            data=data,
            with_row_index=with_row_index,
            collect=collect,
            on_success=on_success,
            on_failure=on_failure,
        )

    # ------------------------------------------------------------------

    def _gather_schema_errors(
            self,
            frame: pl.LazyFrame,
            schema: pl.Schema,
    ) -> typed_dicts.ValidColumnSchemaErrors:
        out: typed_dicts.ValidColumnSchemaErrors = {}

        name_in_columns = False

        if self._name in schema:
            name_in_columns = True
            # dynamically make column required if _required = "dynamic"
            self._required = bool(self._required)

        # --------------------------------------------------------------

        if name_in_columns and self._dtype is not None:
            dt_error = dtype_errors(
                column_dtype=schema.get(self._name),
                valid_dtypes=self._dtype,
            )
            if dt_error is not None:
                out["dtype"] = dt_error

        # --------------------------------------------------------------

        if not name_in_columns and isinstance(self._required, bool):
            if self._required:
                message = f"Column {self._name!r} is missing."
                suggested_columns = _suggest_columns(
                    column_name=self._name,
                    columns=schema.names(),
                    n=5
                )
                if suggested_columns:
                    message += f"\nSuggested names: {suggested_columns}."
                out["required"] = typed_dicts.BaseErrors(errors=message)

        return out

    def _gather_data_errors(
            self,
            frame: pl.LazyFrame,
            schema: pl.Schema,
            *,
            keep_columns: IntoKeepColumns,
            with_row_index: bool | str,
            get_expr: Callable[
                [str, Any, str | pl.Expr | None], pl.Expr
            ],  # for ValidColumn
            _struct_fields: tuple[str, ...] | None,
    ) -> typed_dicts.ValidColumnDataErrors:
        if self._name not in schema:
            return {}

        out: typed_dicts.ValidColumnDataErrors = {}

        if _struct_fields:
            _struct_fields = (*_struct_fields, self._name)

        if not self._allow_nulls:
            if "allow_nulls" not in out:
                out["allow_nulls"] = typed_dicts.Errors()

            out["allow_nulls"]["predicate"] = get_allow_nulls_predicate(
                column_name=self._name,
                struct_fields=_struct_fields,
            )

            if not _has_additional_columns(
                    keep_columns=keep_columns, with_row_index=with_row_index
            ):
                # if no additional columns requested, then just count the nulls
                # we would not have other columns to use for filtering anyway
                out["allow_nulls"]["maybe_errors"] = _null_counts(
                    data=frame, column_name=self._name
                )
            else:
                # if additional columns requested return the same level of observation
                out["allow_nulls"]["maybe_errors"] = _get_nulls(
                    data=frame,
                    column_name=self._name,
                    keep_columns=keep_columns,
                    with_row_index=with_row_index,
                )

        if self._unique:
            # expr = pl.col(name).is_first_distinct()
            # expr = pl.col(name).is_last_distinct()

            if "unique" not in out:
                out["unique"] = typed_dicts.Errors()

            out["unique"]["predicate"] = get_unique_predicate(
                column_name=self._name,
                struct_fields=_struct_fields,
            )

            out["unique"]["maybe_errors"] = _get_duplicates(
                data=frame,
                column_name=self._name,
                keep_columns=keep_columns,
                with_row_index=with_row_index,
            )

        info = None
        if self._constraints:
            out["constraints"] = {}

            if (
                    self._info is not None
            ):  # lets collect the info only if we may have constraints
                # make dict instead of getting directly from InfoList
                info = self._info[0].to_dict(
                    include_name=False
                )  # Info.to_dict()
                # InfoList only has 1 element here. which is named "info"
                # instead of getting "info" we get
                # the first element in case we may rename the info

        for attr, value in self._constraints.items():
            keep_columns, with_row_index, expr, predicate = _dispatch_expr_args_for_errors(
                column_name=self._name,
                value=value,
                attr=attr,
                keep_columns=keep_columns,
                with_row_index=with_row_index,
                _struct_fields=_struct_fields,
                get_expr=get_expr,
            )
            # if isinstance(value, pl.Expr):
            #     keep_columns, with_row_index, expr = replace_expr(
            #         expr=value,
            #         column_name=self._name,
            #         keep_columns=keep_columns,
            #         with_row_index=with_row_index,
            #     )
            #     if _struct_fields:
            #         predicate: pl.Expr | None = replace_predicate(
            #             expr=value,
            #             struct_fields=_struct_fields,
            #         )
            #     else:
            #         predicate = expr
            # else:
            #     expr = get_expr(
            #         attr, value, self._name
            #     )
            #     if _struct_fields:
            #         predicate = get_expr(
            #             attr,
            #             value,
            #             get_struct_expr(struct_fields=_struct_fields)
            #         )
            #     else:
            #         predicate = expr

            if attr not in out["constraints"]:
                out["constraints"][attr] = typed_dicts.ConstraintsErrors()

            # - should we also show components
            # instead of constructed expression, i.e. ge=1?
            if not isinstance(value, pl.Expr):
                out["constraints"][attr]["value"] = value

            # --- add info to the errors dict. (should we make this configurable?)
            if info is not None:
                attr_info = info.get("constraints", {}).get(attr)
                if attr_info is not None:
                    out["constraints"][attr]["info"] = attr_info
            # ----

            out["constraints"][attr]["predicate"] = predicate

            # used to be: _lazy_search_constraints
            out["constraints"][attr]["maybe_errors"] = _select_and_filter(
                data=frame,
                column_name=self._name,
                keep_columns=keep_columns,
                with_row_index=with_row_index,
                expr=expr,
            )

        return out

    def _gather_fields_errors(
            self,
            frame: pl.LazyFrame,
            schema: pl.Schema,
            *,
            mode: ValidationMode,
            keep_columns: IntoKeepColumns,
            with_row_index: bool | str,
            get_expr: Callable[[str, Any, str | pl.Expr | None], pl.Expr],
            cast: bool,
            _struct_fields: tuple[str, ...] | None,
    ) -> typed_dicts.ValidColumnFieldsErrors:
        if self._fields is None:
            return {}

        if self._name not in schema:
            return {}

        # keep columns that are not the Struct column
        frame = _select_keep_columns(
            frame=frame,
            column_names=self._name,  # the Struct column
            keep_columns=keep_columns,  # other columns that are not the Struct column
            with_row_index=False,  # row index is created after unnest
        )

        struct_schema = (
            frame
            .select(self._name)
            .collect_schema()
            [self._name]
            .to_schema()  # type: ignore[attr-defined]
            # .select(self._name).unnest(self._name).collect_schema()
        )

        frame = frame.unnest(self._name)

        if cast:
            frame = cast_frame(
                frame=frame,
                schema=self._fields.to_schema(check_dtypes=True),
            )

        if not _struct_fields:
            _struct_fields = (self._name,)
        else:
            _struct_fields = (*_struct_fields, self._name)

        errors: typed_dicts.ValidationErrors = self._fields._gather_errors(
            frame=frame,
            schema=struct_schema,
            mode=mode,
            keep_columns=keep_columns,
            # each struct col would have different cols/fields itself
            with_row_index=with_row_index,
            get_expr=get_expr,
            cast=cast,
            _struct_fields=_struct_fields,
        )

        if errors:
            return {"fields": errors}
        return {}

    def _gather_errors(
            self,
            frame: pl.LazyFrame,
            *,
            mode: ValidationMode,
            keep_columns: IntoKeepColumns,
            with_row_index: bool | str,
            get_expr: Callable[[str, Any, str | pl.Expr | None], pl.Expr],
            cast: bool,
            _struct_fields: tuple[str, ...] | None,
    ) -> typed_dicts.ValidColumnErrors:
        out: typed_dicts.ValidColumnErrors = {}

        schema = frame.collect_schema()

        if mode != "data":
            lazy_errors = self._gather_schema_errors(
                frame=frame, schema=schema
            )
            out.update(lazy_errors)

        if mode in ("data", "all") or (mode == "all-conditional" and not out):
            eager_errors: typed_dicts.ValidColumnDataErrors = self._gather_data_errors(
                frame=frame,
                schema=schema,
                keep_columns=keep_columns,
                with_row_index=with_row_index,
                get_expr=get_expr,
                _struct_fields=_struct_fields,
            )

            if eager_errors:
                out.update(eager_errors)

        if self._fields is not None:
            fields_errors = self._gather_fields_errors(
                frame=frame,
                schema=schema,
                mode=mode,
                keep_columns=keep_columns,
                with_row_index=with_row_index,
                get_expr=get_expr,
                cast=cast,
                _struct_fields=_struct_fields,
            )
            if fields_errors:
                out.update(fields_errors)

        return out

    # ------------------------------------------------------------------

    def gather_predicates(self, target: pl.Schema | None = None) -> list[pl.Expr]:
        """
        Gather all the set rules as predicate expressions.

        A target schema is needed if valid columns are specified using selectors.
        """
        return self._predicates(
            schema=target,
            get_expr=_build_expr,
        )

    def _predicates(
            self,
            schema: pl.Schema | None,
            *,
            get_expr: Callable[[str, Any, str | pl.Expr | None], pl.Expr],
    ) -> list[pl.Expr]:
        # expand validation if _name is None or Selector
        if not isinstance(self._name, str):
            from paguro.validation.validation import Validation

            validation = Validation(
                self
            )  # passing vcol to Validation to delegate expansion
            return validation._predicates(
                schema=schema,
                get_expr=get_expr,
            )

        predicates = self._gather_predicates(
            schema=schema,
            get_expr=get_expr,
            _struct_fields=None,
        )

        out = {
            "valid_column_list": {self._name: predicates}
        }

        return _gather_predicates(out, leaf_key="predicate")

    def _gather_predicates(
            self,
            schema: pl.Schema | None,
            *,
            get_expr: Callable[[str, Any, str | pl.Expr | None], pl.Expr],
            _struct_fields: tuple[str, ...] | None,
    ) -> dict[str, Any]:

        out = super()._gather_predicates(
            schema=schema,
            get_expr=get_expr,
            _struct_fields=_struct_fields,
        )

        if self._fields is not None:
            validators_errors = self._gather_fields_predicates(
                schema=schema,
                get_expr=get_expr,
                _struct_fields=_struct_fields,
            )
            if validators_errors:
                out.update(validators_errors)

        return out

    def _gather_fields_predicates(
            self,
            schema: pl.Schema | None,
            *,
            get_expr: Callable[[str, Any, str | pl.Expr | None], pl.Expr],
            _struct_fields: tuple[str, ...] | None,
    ) -> dict[str, Any]:
        if self._fields is None:
            return {}

        if schema is not None:
            if self._name not in schema:
                return {}

        if not _struct_fields:
            _struct_fields = (self._name,)
        else:
            _struct_fields = (*_struct_fields, self._name)

        if schema is not None:
            struct_: pl.Struct = schema[self._name]
            if not isinstance(struct_, pl.Struct):
                msg = (f"{self._name} must be a Struct "
                       f"in order to extract fields schema, got {struct_}."
                       )
                raise ValueError(msg)
            schema = struct_.to_schema()  # todo: check!

        predicates = self._fields._gather_predicates(
            schema=schema,
            get_expr=get_expr,
            _struct_fields=_struct_fields
        )

        if predicates:
            return {"fields": predicates}
        return {}

    def _gather_focal_predicates(
            self,
            schema: pl.Schema | None,
            *,
            get_expr: Callable[[str, Any, str | pl.Expr | None], pl.Expr],
            _struct_fields: tuple[str, ...] | None,
    ) -> dict[str, Any]:

        if schema is not None:
            if self._name not in schema:
                return {}

        out: dict[str, Any] = {}

        if _struct_fields:
            _struct_fields = (*_struct_fields, self._name)

        if not self._allow_nulls:
            if "allow_nulls" not in out:
                out["allow_nulls"] = {}

            out["allow_nulls"]["predicate"] = get_allow_nulls_predicate(
                column_name=self._name,
                struct_fields=_struct_fields,
            )

        if self._unique:
            # expr = pl.col(name).is_first_distinct()
            # expr = pl.col(name).is_last_distinct()

            if "unique" not in out:
                out["unique"] = {}

            out["unique"]["predicate"] = get_unique_predicate(
                column_name=self._name,
                struct_fields=_struct_fields,
            )

        if self._constraints:
            out["constraints"] = {}

        for attr, value in self._constraints.items():
            predicate = _dispatch_expr_args_for_predicates(
                column_name=self._name,
                value=value,
                attr=attr,
                _struct_fields=_struct_fields,
                get_expr=get_expr,
            )

            if attr not in out["constraints"]:
                out["constraints"][attr] = {}

            out["constraints"][attr]["predicate"] = predicate

        return out

    # ----------------------------------------------------------------------


# ----------------------------------------------------------------------


def report_dropped_or_required(
        frame: pl.LazyFrame,
        valid_columns: list[ValidColumn],
) -> list[str]:
    """
    Identify columns that are dropped or required but not present in the dataframe.

    This method checks each column registered in `_valid_columns` to
    determine if it has been dropped or if it is a required column that
    is not present in the provided `frame`. The check is performed
    only for columns that are not allowed to be dropped or are
    explicitly marked as required.

    Parameters
    ----------
    frame : pl.LazyFrame
        A LazyFrame object from the Polars library
        which represents a lazily evaluated data frame
        where transformations will be applied but not executed until needed.

    Returns
    -------
    list[str]
        A list of column names that have been
        identified as either dropped or required but are
        missing from the dataframe.
    """
    latest_columns = frame.columns
    not_allowed_missing = []  # list the columns that are not allowed to be missing from the data
    for vc in valid_columns:
        if not vc._allow_drop or vc._required:
            if vc._name not in latest_columns:
                not_allowed_missing.append(vc._name)
    return not_allowed_missing


def _validate_vc_lazyframe(
        frame: pl.LazyFrame,
        valid_columns: list[ValidColumn],
) -> dict:
    out = {}

    drop_or_required = report_dropped_or_required(frame, valid_columns)
    if drop_or_required:
        out["required/allow_drop"] = drop_or_required

    return out
