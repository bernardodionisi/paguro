from __future__ import annotations

from typing import TYPE_CHECKING, Any

from polars.selectors import Selector

from paguro.ashi.info.info import Info
from paguro.ashi.info.info_collection import InfoCollection

from paguro.shared.serialize import CustomJSONEncoder, CustomJSONDecoder
from paguro.utils.dependencies import copy, hashlib, json
from paguro.validation.exception.errors.validation_error import ValidationError
from paguro.validation.exception.utils.filter_utils import _gather_predicates
from paguro.validation.exception.validate_dispatch import (
    _validate_dispatch,
)
from paguro.validation.shared.cast import cast_frame
from paguro.validation.shared.expand_names import expand_valid_column_list
from paguro.validation.shared.preprocessing.preprocess_validators import (
    preprocess_vcs_vfs,
)
from paguro.validation.shared.utils import (
    custom_serializer_for_repr,
    data_to_frame_like,
    data_to_lazyframe,
)

from paguro.validation.shared.find_v import (
    find_unique_vcol,
    find_all_vcols_multi,
    find_unique_vframe,
    find_all_vframes_multi
)
from paguro.validation.shared.rename_valid_columns import rename_valid_columns, \
    MappingOrFunc
from paguro.validation.valid_column.utils.exprs.build_expression import _build_expr
from paguro.validation.valid_column.valid_column_list import (
    ValidColumnList,
)
from paguro.validation.valid_frame.valid_frame_list import ValidFrameList
from paguro.shared._typing import typed_dicts
from paguro.typing import CollectConfig

if TYPE_CHECKING:
    import sys
    from collections.abc import Callable, Iterable, Mapping
    import polars as pl
    from polars._typing import JoinStrategy
    from paguro.validation.valid_frame.valid_frame import ValidFrame
    from paguro.validation.valid_column.valid_column import ValidColumn

    from paguro.typing import (
        IntoKeepColumns,
        IntoValidation,
        OnSuccess,
        ValidatorOrExpr, FrameLike, OnFailureExtra, ValidationMode,
    )

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self


class Validation:
    """Validation."""

    def __init__(
            self,
            *validators: ValidatorOrExpr
                         | Iterable[ValidatorOrExpr]
                         | Validation,
            **named_validators: ValidatorOrExpr,
    ) -> None:
        vcl, vfl = preprocess_vcs_vfs(*validators, **named_validators)

        self._valid_frame_list: ValidFrameList | None = None
        if vfl is not None:
            self._valid_frame_list = ValidFrameList(vfl)

        self._valid_column_list: ValidColumnList | None = None
        if vcl is not None:
            self._valid_column_list = ValidColumnList(vcl)

        self._info: InfoCollection | None = (
            None  # TODO: INFO ---> improve management of Validation Level info
        )
        self._name: str | None = None

    # --------------

    def __repr__(self) -> str:
        return self._repr_or_str(string=False)

    def __str__(self) -> str:
        return self._repr_or_str(string=True)

    def _repr_or_str(self, *, string: bool) -> str:
        vcl: str | None = None
        vfl: str | None = None

        if self._valid_column_list is not None:
            if string:
                vcl = self._valid_column_list.__str__()
            else:
                vcl = self._valid_column_list.__repr__()

            vcl = vcl.replace("\n", "\n\t")

        if self._valid_frame_list is not None:

            if string:
                vfl = self._valid_frame_list.__str__()
            else:
                vfl = self._valid_frame_list.__repr__()

            vfl = vfl.replace("\n", "\n\t")

        out = (
            f"{self.__class__.__qualname__}(\n"
            f"\tvalid_columns={vcl},\n"
            f"\tvalid_frames={vfl},\n"
            f")"
        )

        return out

    def _display(self) -> str:
        content = json.dumps(
            self._to_dict(_fingerprint=False, include_info=False),
            indent=3,
            default=custom_serializer_for_repr,
        )
        # content = self.to_dict()
        return f"{self.__class__.__qualname__}({content})"

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
            root=self,
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
            root=self,
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
            root=self,
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
            root=self,
            names=names,
            include_transformed_frames=include_transformed_frames,
            include_fields=include_fields,
            dedupe=dedupe,
            group_by_name=group_by_name,
        )

    def _rename_valid_columns(
            self,
            mapping: MappingOrFunc,
            *,
            include_transformed_frames: bool = False,
            include_fields: bool = False,
    ) -> Validation:
        validation = copy.deepcopy(self)
        return rename_valid_columns(
            validation=validation,
            mapping=mapping,
            include_transformed_frames=include_transformed_frames,
            include_fields=include_fields,
        )

    # ------------------------------------------------------------------

    # def _append(
    #         self,
    #         *validators,
    #         **named_validators,
    # ) -> None:
    #     vcl, vfl = preprocess_vcs_vfs(*validators, **named_validators)
    #
    #     if vcl is not None:
    #         if self._valid_column_list is not None:
    #             self._valid_column_list._extend(vcl)
    #             # warns/replace if name is existing
    #         else:
    #             self._valid_column_list = vcl
    #
    #     if vfl is not None:
    #         if self._valid_frame_list is not None:
    #             self._valid_frame_list._extend(vfl)
    #             # warns/replace if name is existing
    #         else:
    #             self._valid_frame_list = vfl

    # ------------------------------------------------------------------

    def _join_validation(
            self,
            other: Validation,
            how: JoinStrategy,
            *,
            join_constraints: bool = False,
            suffix: str | None = None,
    ) -> Self:
        """Create a new Validation instance by joining the valid lists."""
        new = copy.deepcopy(self)

        # todo: just use Validation(Validation(), Validation())
        # --------------------------- frame ----------------------------

        if other._valid_frame_list is not None:
            if new._valid_frame_list is not None:
                new._valid_frame_list = new._valid_frame_list.join(
                    other=other._valid_frame_list,
                    how=how,
                    join_constraints=join_constraints,
                    suffix=suffix,
                )
            else:
                # assign other valid frames to self
                new._valid_frame_list = copy.deepcopy(
                    other._valid_frame_list
                )

            # if new._valid_frame_list is not None, its already set

        # --------------------------- column ---------------------------

        if other._valid_column_list is not None:
            if new._valid_column_list is not None:
                new._valid_column_list = new._valid_column_list.join(
                    other=other._valid_column_list,
                    how=how,
                    join_constraints=join_constraints,
                    suffix=suffix,
                )
            else:
                # assign other valid columns to self
                new._valid_column_list = copy.deepcopy(
                    other._valid_column_list
                )

            # if new._valid_column_list is not None, its already set

        return new

    # ------------------------------------------------------------------

    def with_name(self, name: str) -> Self:
        """Assigns name."""
        new = copy.deepcopy(self)
        new._name = name
        return new

    @property
    def name(self) -> str | None:
        """Returns the assigned name."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if not isinstance(value, str):
            msg = "name must be of type str"
            raise TypeError(msg)
        self._name = value

    # def with_info(
    #         self,
    #         name: str | None = None,
    #         **mapping: Any,
    # ) -> Self:
    #     """Assigns info."""
    #     new = copy.deepcopy(self)
    #     if new._info is None:
    #         new._info = InfoList()  # legacy
    #     name = self._name if name is None else name
    #     if name is None:
    #         name = "<unnamed-info>"
    #     new._info = new._info.update(
    #         info=name,
    #         **mapping,
    #     )
    #     return new

    def with_info(self, name: str | None = None, **mapping: Any) -> Self:
        """
        Return a copy with Info(name) updated by `mapping`.
        """
        new = copy.deepcopy(self)

        if name is None:
            name = new.name
            if name is None:
                name = "<unnamed-info>"

        if new._info is None:
            new._info = InfoCollection()

        if name in new._info:
            new._info = new._info.update(name, **mapping)
        else:
            info = Info(name).update(**mapping)
            # explicitly disable schema policy
            info.set_schema(mode="off")
            new._info = new._info.append(info)

        return new

    @property
    def info(self) -> InfoCollection | None:
        """User defined information."""
        return self._info

    # ------------------------------------------------------------------

    def to_schema(
            self,
            *,
            check_dtypes: bool = True,
            build_struct: bool = False,
    ) -> pl.Schema:
        """Extracts a schema from the ValidColumns."""
        # note: this is the schema defined in valid columns,
        # schema can also be defined in vframes
        if self._valid_column_list is None:
            msg = "Unable to generate expected schema without vcol(s)"
            raise ValueError(msg)
        return self._valid_column_list.to_schema(
            check_dtypes=check_dtypes, build_struct=build_struct
        )

    # -----------

    def serialize(self) -> str:
        return json.dumps(
            self._to_dict(_fingerprint=False, include_info=True),
            cls=CustomJSONEncoder,
        )

    @classmethod
    def deserialize(cls, source: str) -> Validation:
        return cls._from_dict(json.loads(source, cls=CustomJSONDecoder))

    def _to_dict(
            self, *, _fingerprint: bool, include_info: bool
    ) -> dict[str, list[dict[str, Any]]]:
        out: dict = {}

        if _fingerprint:
            if self._valid_frame_list is not None:
                out["valid_frame_list"] = (
                    self._valid_frame_list._fingerprint(
                        as_bytes=False, include_info=include_info
                    )
                )

            if self._valid_column_list is not None:
                out["valid_column_list"] = (
                    self._valid_column_list._fingerprint(
                        as_bytes=False, include_info=include_info
                    )
                )

        else:
            if self._valid_frame_list is not None:
                # info is always included here
                out["valid_frame_list"] = (
                    self._valid_frame_list._to_list_of_dicts()
                )

            if self._valid_column_list is not None:
                # info is always included here
                out["valid_column_list"] = (
                    self._valid_column_list._to_list_of_dicts()
                )

        if include_info:
            out["name"] = self._name
            out["info"] = (
                None if self._info is None else self._info.to_dict()
            )

        return out

    @classmethod
    def _from_dict(
            cls,
            source: Mapping[str, Any],
    ) -> Self:
        instance = cls()

        # ------

        vfl: list[dict] | None = source.get("valid_frame_list")
        if vfl is not None:
            instance._valid_frame_list = ValidFrameList._from_list_dict(
                vfl
            )
        else:
            instance._valid_frame_list = None

        # ------

        vcl: list[dict] | None = source.get("valid_column_list")
        if vcl is not None:
            instance._valid_column_list = ValidColumnList._from_list_dict(
                vcl
            )
        else:
            instance._valid_column_list = None

        # ------

        instance._name = source.get("name", None)

        info_dict: dict | None = source.get("info")

        if info_dict is not None:
            instance._info = InfoCollection.from_dict(info_dict)
        else:
            instance._info = None

        return instance

    def _fingerprint(self, *, include_info: bool) -> str:
        other_json = json.dumps(
            self._to_dict(_fingerprint=True, include_info=include_info),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(other_json.encode()).hexdigest()

    # ------------------------------------------------------------------

    def _expand_valid_column_list(
            self,
            schema: pl.Schema | None,
    ) -> ValidColumnList | None:
        # expand only at validation, not before!
        return expand_valid_column_list(
            vcl=self._valid_column_list,
            schema=schema,
            required_only=False,
        )

    # ------------------------------------------------------------------

    def validate(  # noqa: D102
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

    def _validate(  # noqa: ANN202
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
        frame = data_to_lazyframe(data=data)
        errors = self._gather_errors(
            frame=frame,
            schema=None,
            mode=mode,
            keep_columns=keep_columns,
            with_row_index=with_row_index,
            get_expr=get_expr,
            cast=cast,
            _struct_fields=None,
        )

        # store data as frame like:
        # LazyFrame, DataFrame, Dataset, LazyDataset
        data = data_to_frame_like(data=data)
        if cast:
            # we are repeating the work here,
            # we also need to cast when gathering validation
            data = cast_frame(
                data, schema=self.to_schema(check_dtypes=True)
            )

        return _validate_dispatch(
            errors=errors,
            data=data,
            with_row_index=with_row_index,
            collect=collect,
            on_success=on_success,
            on_failure=on_failure,
        )

    def _gather_errors(
            self,
            frame: pl.LazyFrame,
            schema: pl.Schema | None,
            *,
            mode: ValidationMode,
            keep_columns: IntoKeepColumns,
            with_row_index: bool | str,
            get_expr: Callable[[str, Any, str | pl.Expr | None], pl.Expr],
            cast: bool,
            _struct_fields: tuple[str, ...] | None,
    ) -> typed_dicts.ValidationErrors:
        if schema is None:
            schema = frame.collect_schema()

        vcl: ValidColumnList | None = self._expand_valid_column_list(schema=schema)

        if cast:
            # we need to cast here so we can propagate in fields and validators
            frame = cast_frame(
                frame=frame,
                schema=self.to_schema(check_dtypes=True),
            )

        out: typed_dicts.ValidationErrors = {}

        if vcl is not None:
            column_errors: dict[
                str, typed_dicts.ValidColumnErrors] = vcl._gather_errors(
                frame=frame,
                mode=mode,
                keep_columns=keep_columns,
                with_row_index=with_row_index,
                get_expr=get_expr,
                cast=cast,
                _struct_fields=_struct_fields,
            )
            if column_errors:
                out["valid_column_list"] = column_errors

        if self._valid_frame_list is not None:
            frame_errors: dict[
                str, typed_dicts.ValidFrameErrors] = self._valid_frame_list._gather_errors(
                frame=frame,
                mode=mode,
                keep_columns=keep_columns,
                with_row_index=with_row_index,
                get_expr=get_expr,
                cast=cast,
                _struct_fields=_struct_fields,
            )
            if frame_errors:
                out["valid_frame_list"] = frame_errors
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
        predicates = self._gather_predicates(
            schema=schema,
            get_expr=get_expr,
            _struct_fields=None,
        )
        return _gather_predicates(predicates, leaf_key="predicate")

    def _gather_predicates(
            self,
            schema: pl.Schema | None,
            *,
            get_expr: Callable[[str, Any, str | pl.Expr | None], pl.Expr],
            _struct_fields: tuple[str, ...] | None,
    ) -> dict[str, Any]:

        vcl: ValidColumnList | None = self._expand_valid_column_list(schema=schema)

        out = {}

        if vcl is not None:
            column_predicates = vcl._gather_predicates(
                schema=schema,
                get_expr=get_expr,
                _struct_fields=_struct_fields,
            )
            if column_predicates:
                out["valid_column_list"] = column_predicates

        if self._valid_frame_list is not None:
            frame_predicates = self._valid_frame_list._gather_predicates(
                schema=schema,
                get_expr=get_expr,
                _struct_fields=_struct_fields,
            )
            if frame_predicates:
                out["valid_frame_list"] = frame_predicates
        return out

    # ------------------------------------------------------------------

    def _gather_transforms(
            self,
            frame: pl.LazyFrame,
    ) -> dict[str, Any]:
        # extra utility - not for validation

        if self._valid_frame_list is None:
            return {}

        out: dict[str, Any] = {}
        transforms = self._valid_frame_list._gather_transforms(
            frame=frame,
        )
        if transforms:
            # out["Frames"] = transforms
            return transforms

        return out
