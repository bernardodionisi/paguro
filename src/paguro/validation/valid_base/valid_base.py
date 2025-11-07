from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from paguro.ashi.info.info import Info
from paguro.ashi.info.info_collection import InfoCollection

from paguro.shared.serialize.encoder import CustomJSONEncoder
from paguro.utils.dependencies import copy, hashlib, json
from paguro.validation.exception.errors.validation_error import ValidationError
from paguro.validation.shared._docs import set_doc_string, VALIDATE_PARAMS
from paguro.validation.shared.utils import (
    _fixed_info_mapping,
    custom_serializer_for_repr,
)
from paguro.validation.valid_column.utils.exprs.build_expression import _build_expr
from paguro.shared._typing import typed_dicts

if TYPE_CHECKING:
    import sys
    from collections.abc import Callable
    import polars as pl

    from paguro.typing import (
        IntoKeepColumns,
        IntoValidation,
        OnSuccess, FrameLike, OnFailureExtra, ValidationMode, CollectConfig,
    )

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self


class _ValidBase(ABC):
    def __init__(
            self,
            name,  # noqa: ANN001
            constraints: dict[str, Any],
    ) -> None:
        # TODO: if not expr.meta.root_names() if != [column_name] or []
        #  raise error saying expressions should just refer to the column,
        #  use valid_frame for frame level expressions,
        #  dont do it here since we actually may
        #  just change the column name at runtime so to allow renaming vcol

        self._name = name
        self._constraints = constraints

        self._info: InfoCollection | None = None

    def __repr__(self) -> str:
        # return self.__str__()
        return f"{self.__class__.__qualname__}(...)"

    def _display(self) -> str:
        content = json.dumps(
            self._to_dict(_fingerprint=False, include_info=False),
            indent=3,
            default=custom_serializer_for_repr,
        )
        return f"{self.__class__.__qualname__}({content})"

    # ----------------------- info -------------------------------------

    def with_info(
            self,
            *,
            title: str | None = None,
            description: str | None = None,
            constraints: dict[str, Any] | None = None,
            **mapping: Any,
    ) -> Self:
        """
        Add info to the validator.

        Group
        -----
            Information
        """
        fixed_mapping, free_mapping = _fixed_info_mapping(
            title=title,
            description=description,
            constraints=constraints,
            mapping=mapping,
            valid_constraints=set(self._constraints.keys()),
        )

        new = copy.deepcopy(self)
        if new._info is None:
            new._info = InfoCollection()

        info_name = "info"  # single named info

        # todo: add warning if new._info is non empty and name changed
        if info_name in new._info:
            # Immutable update on existing Info
            new._info = new._info.update(
                info_name,
                **fixed_mapping,
                **free_mapping,
            )
        else:
            # Create new Info, keep it schema-free
            info = (
                Info(info_name)
                .set_schema(mode="off")
                .update(**fixed_mapping, **free_mapping)
            )
            new._info = new._info.append(info)

        return new

    # ------------------------------------------------------------------
    @abstractmethod
    def _to_dict(
            self, *, _fingerprint: bool, include_info: bool
    ) -> dict[str, Any]:
        """
        Convert object state to a dictionary.

        Implementation must handle serialization of
        expressions if serialize_expr is True.
        """
        raise NotImplementedError

    @classmethod
    def _from_dict(cls, source: dict) -> Self:
        raise NotImplementedError

    def serialize(
            self,
            **kwargs: Any,
    ) -> str:
        """
        Serialize the object to JSON string.

        Group
        -----
            Export
        """
        return json.dumps(
            self._to_dict(_fingerprint=False, include_info=True),
            cls=CustomJSONEncoder,
            **kwargs,
        )

    def _fingerprint(
            self, *, as_bytes: bool, include_info: bool
    ) -> str | bytes:
        data: str = json.dumps(
            self._to_dict(_fingerprint=True, include_info=include_info),
            cls=CustomJSONEncoder,
            sort_keys=True,
            separators=(",", ":"),
        )
        hash_ = hashlib.sha256(data.encode())
        if as_bytes:
            return hash_.digest()
        return hash_.hexdigest()

    # ------------------------------------------------------------------
    @set_doc_string(parameters=VALIDATE_PARAMS)
    def validate(
            self,
            data: IntoValidation,
            *,
            mode: ValidationMode = "all",
            keep_columns: IntoKeepColumns = False,
            collect: bool | dict = True,
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
            on_success: OnSuccess,
            on_failure: OnFailureExtra,
            collect: bool | CollectConfig,
            get_expr: Callable[[str, Any, str | pl.Expr | None], pl.Expr],
            cast: bool,
    ) -> FrameLike | ValidationError | None:
        raise NotImplementedError
        # errors = self._gather_errors(
        #     frame=data_to_lazyframe(data=data),
        #     mode=mode,
        #     keep_columns=keep_columns,
        #     with_row_index=with_row_index,
        #     get_expr=get_expr,
        #     cast=cast,
        #     _struct_fields=None,
        # )
        #
        # return _validate_dispatch(
        #     errors=errors,
        #     data=data,
        #     with_row_index=with_row_index,
        #     on_success=on_success,
        #     on_failure=on_failure,
        #     collect=collect,
        # )

    def _gather_schema_errors(
            self,
            frame: pl.LazyFrame,
            schema: pl.Schema,
    ) -> typed_dicts.ValidColumnSchemaErrors | typed_dicts.ValidFrameSchemaErrors:
        raise NotImplementedError

    def _gather_data_errors(
            self,
            frame: pl.LazyFrame,
            schema: pl.Schema,
            *,
            keep_columns: IntoKeepColumns,
            with_row_index: bool | str,
            get_expr: Callable[[str, Any, str | pl.Expr | None], pl.Expr],
            _struct_fields: tuple[str, ...] | None,
    ) -> typed_dicts.ValidColumnDataErrors | typed_dicts.ValidFrameDataErrors:
        raise NotImplementedError

    def _gather_errors(
            self,
            frame: pl.LazyFrame,
            *,
            mode: ValidationMode,
            keep_columns: IntoKeepColumns,
            with_row_index: bool | str,
            get_expr: Callable[[str, Any, str | pl.Expr | None], pl.Expr],
            cast: bool,  # for fields errors in ValidColumn, validators errors
            _struct_fields: tuple[str, ...] | None,
    ) -> typed_dicts.ValidColumnErrors | typed_dicts.ValidFrameErrors:
        out: typed_dicts.ValidColumnErrors | typed_dicts.ValidFrameErrors = {}  # type: ignore

        schema = frame.collect_schema()

        if mode != "data":
            try:
                schema_errors = self._gather_schema_errors(
                    frame=frame, schema=schema
                )
                if schema_errors:
                    out.update(schema_errors)

            except NotImplementedError:
                pass

        if mode in ("data", "all") or (mode == "all-conditional" and not out):
            data_errors = self._gather_data_errors(
                frame=frame,
                schema=schema,
                keep_columns=keep_columns,
                with_row_index=with_row_index,
                get_expr=get_expr,  # needs to be inserted in ValidColumn
                _struct_fields=_struct_fields,
            )

            if data_errors:
                out.update(data_errors)

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
        raise NotImplementedError

    def _gather_focal_predicates(
            self,
            schema: pl.Schema | None,
            *,
            get_expr: Callable[[str, Any, str | pl.Expr | None], pl.Expr],
            _struct_fields: tuple[str, ...] | None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def _gather_predicates(
            self,
            schema: pl.Schema | None,
            *,
            get_expr: Callable[[str, Any, str | pl.Expr | None], pl.Expr],
            _struct_fields: tuple[str, ...] | None,
    ) -> dict[str, Any]:
        out = {}

        predicates = self._gather_focal_predicates(
            schema=schema,
            get_expr=get_expr,
            _struct_fields=_struct_fields,
        )

        if predicates:
            out.update(predicates)
        return out
