from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import polars as pl
from polars.selectors import Selector

from paguro.ashi.info.info_collection import InfoCollection
from paguro.defer import LazyFrameExpr
from paguro.utils.dependencies import copy
from paguro.validation.exception.errors.validation_error import ValidationError
from paguro.validation.exception.utils.filter_utils import _gather_predicates
from paguro.validation.shared.find_v import find_unique_vcol, find_unique_vframe, \
    find_all_vcols_multi, find_all_vframes_multi

from paguro.validation.exception.validate_dispatch import (
    _validate_dispatch,
)
from paguro.validation.shared.cast import cast_frame
from paguro.validation.shared.preprocessing.preprocess_validators_vframe import \
    split_validators_into_validation_and_constraints
from paguro.validation.shared.utils import (
    data_to_frame_like,
    data_to_lazyframe,
)

from paguro.validation.valid_base.valid_base import _ValidBase
from paguro.validation.valid_frame.utils.columns_policy import ColumnsPolicy
from paguro.validation.valid_frame.utils.transform import (
    _to_transform_frame_tree,
)
from paguro.validation.valid_frame.utils.utils import (
    _negate_filter_from_expr,
    _negate_filter_from_expr_unique_by,
)

from paguro.shared._typing import typed_dicts

if TYPE_CHECKING:
    import sys
    import typing

    from collections.abc import Callable, Iterable
    from paguro.validation.validation import Validation
    from paguro.validation.valid_column.valid_column import ValidColumn

    from polars import Expr

    from paguro.typing import (
        IntoKeepColumns,
        IntoValidation,
        IntoValidators,
        OnSuccess,
        FrameLike, OnFailureExtra, ValidationMode, CollectConfig,
    )
    from paguro.validation.valid_frame.utils.transform import (
        TransformFrameTree,
    )
    from paguro.validation.validation import Validation

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

__all__ = [
    "ValidFrame",
]


class ValidFrame(_ValidBase):
    def __init__(
            self,
            *validators: IntoValidators | typing.Collection[IntoValidators],
            transform: LazyFrameExpr | pl.Expr | None = None,
            name: str | None = None,
            unique: str | typing.Collection[str] | None = None,
            **constraints: pl.Expr,
    ) -> None:
        self._validators: Validation | None = None
        if validators:
            validation, constraints = split_validators_into_validation_and_constraints(
                validators=validators,
                constraints=constraints,
            )
            self._validators = validation

        super().__init__(name=name, constraints=constraints)

        self._unique = unique

        self._columns_policy: ColumnsPolicy | None = None

        if isinstance(transform, pl.Expr):
            self._transform: LazyFrameExpr | None = (
                LazyFrameExpr().select(transform)
            )
        else:
            self._transform = transform

    @classmethod
    def _(
            cls,
            *validators: IntoValidators | typing.Collection[IntoValidators],
            transform: LazyFrameExpr | pl.Expr | None = None,
            name: str | None = None,
            unique: str | typing.Collection[str] | None = None,
            constraints: dict[str, pl.Expr] | None = None,
    ) -> ValidFrame:
        if constraints is None:
            constraints = {}
        return cls(
            *validators,
            transform=transform,
            name=name,
            unique=unique,
            **constraints,
        )

    def __repr__(self) -> str:
        out = f"{self.__class__.__qualname__}("
        out += f"\nvalidators={self._validators.__repr__()}\n"

        name = (
            f"{self._name!r}"
            if isinstance(self._name, str)
            else f"{self._name}"
        )

        out += f"name={name}, ..."  # \n)
        out = out.replace("\n", "\n\t")
        out += "\n)"
        return out

    def __str__(self) -> str:
        out = f"{self.__class__.__qualname__}("

        name = (
            f"{self._name!r}"
            if isinstance(self._name, str)
            else f"{self._name}"
        )
        if self._constraints:
            # should we display them in order
            constraints = set(self._constraints.keys())
        else:
            constraints = set()

        if self._validators is not None:
            out += f"\nvalidators={self._validators.__str__()},\n"
        else:
            out += "validators=None, "

        out += (
            f"transform={self._transform.__repr__()}, "
            f"name={name}, "
            f"unique={self._unique.__repr__()}, "
            f"constraints={constraints}"
        )

        out = out.replace("\n", "\n\t") + ")"
        return out

    # def __repr__(self) -> str:
    #     name = (
    #         f"{self._name!r}"
    #         if isinstance(self.name, str)
    #         else f"{self._name}"
    #     )
    #     return f"{self.__class__.__qualname__}(name={name}, ...) at {hex(id(self))}"

    # ------------------------------------------------------------------

    def with_name(self, name: str | None) -> Self:
        """
        Add a name to the valid frame object.

        Group
        -----
            Settings
        """
        new = copy.deepcopy(self)
        if isinstance(name, str):
            new._name = name
        elif name is None:
            new._name = None
        else:
            msg = f"name must be a string or None, got {type(name)}"
            raise TypeError(msg)
        return new

    def with_columns_policy(
            self,
            *,
            ordered: bool = False,
            allow_extra: bool = True,
            allow_missing: bool = True,
            expected_column_names: typing.Collection[str] | None = None,
    ) -> Self:
        """
        Frame wide rules for columns.
        
        Group
        -----
            Settings
        """
        new = copy.deepcopy(self)
        new._columns_policy = ColumnsPolicy(
            ordered=ordered,
            allow_extra=allow_extra,
            allow_missing=allow_missing,
            expected_column_names=expected_column_names,
        )
        return new

    def _find_vcol(
            self,
            name: str | None | Selector,
            *,
            include_transformed_frames: bool = False,
            include_fields: bool = False,
            return_first: bool = False,
    ) -> ValidColumn | None:
        return find_unique_vcol(
            root=self._validators,
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
            root=self._validators,
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
            root=self._validators,
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
            root=self._validators,
            names=names,
            include_transformed_frames=include_transformed_frames,
            include_fields=include_fields,
            dedupe=dedupe,
            group_by_name=group_by_name,
        )

    # ------------------------------------------------------------------

    def with_info(
            self,
            *,
            title: str | None = None,
            description: str | None = None,
            constraints: dict[str, Any] | None = None,
            transform_description: Any | None = None,
            **mapping: Any,
    ) -> Self:
        """
        Add info to the valid frame object.

        Group
        -----
            Information
        """
        if transform_description is not None:
            mapping.update({"transform_description": transform_description})
        return super().with_info(
            title=title,
            description=description,
            constraints=constraints,
            **mapping,
        )

    # ------------------------------------------------------------------

    def _to_dict(
            self, *, _fingerprint: bool, include_info: bool
    ) -> dict[str, Any]:
        validators: dict | None = None
        if self._validators is not None:
            validators = self._validators._to_dict(
                _fingerprint=_fingerprint, include_info=include_info
            )

        transform: dict | None = None
        if self._transform is not None:
            transform = self._transform._to_dict()

        out = {
            "name": self._name,
            "transform": transform,
            # "fixed_columns": None #
            # if self._schema is None else self._schema._to_dict(),
            "unique": self._unique,
            "validators": validators,
            "constraints": self._constraints,
        }

        if include_info:
            out["info"] = (
                None if self._info is None else self._info.to_dict()
            )
        return out

    @classmethod
    def _from_dict(
            cls,
            source: dict[str, Any],
    ) -> ValidFrame:
        # vf_dict = deserialize(source, cls=CustomJSONDecoder)
        vf_dict = copy.deepcopy(source)

        # vf_dict = source.pop("fixed_columns")  # TODO

        validators: dict[str, Any] = vf_dict.pop("validators")
        transform: dict[str, Any] = vf_dict.pop("transform")
        constraints: dict[str, Any] = vf_dict.pop("constraints")

        info_dict: dict = vf_dict.pop("info", None)

        validation: Validation | None = None
        if validators is not None:
            from paguro.validation.validation import Validation

            validation = Validation._from_dict(validators)

        if transform is not None:
            vf_dict["transform"] = LazyFrameExpr._from_dict(transform)

        # -----
        if validation is not None:
            instance: ValidFrame = cls(validation, **vf_dict, **constraints)
        else:
            instance = cls(**vf_dict, **constraints)
        # -----

        if info_dict is not None:
            instance._info = InfoCollection.from_dict(info_dict)
        else:
            instance._info = None

        return instance

    # ------------------------------------------------------------------

    def _gather_schema_errors(
            self,
            frame: pl.LazyFrame,
            schema: pl.Schema,
    ) -> typed_dicts.ValidFrameSchemaErrors:

        out = typed_dicts.ValidFrameSchemaErrors()

        # here frame should be the transformed one

        if self._columns_policy is not None:
            if (
                    self._validators is not None
                    and self._validators._valid_column_list is not None
                    and self._columns_policy._expected_column_names is None
            ):
                expected_column_names = [
                    vc._name
                    for vc in self._validators._valid_column_list
                    if isinstance(vc._name, str)  # selectors or None ignored
                ]
                errors = self._columns_policy._gather_errors(
                    actual=schema.names(),
                    expected_column_names_override=expected_column_names,
                )

            else:
                errors = self._columns_policy._gather_errors(
                    actual=schema.names()
                )

            if errors:
                out["columns_policy"] = errors

        return out

    def _gather_data_errors(
            self,
            frame: pl.LazyFrame,
            schema: pl.Schema,
            *,
            keep_columns: IntoKeepColumns,
            with_row_index: bool | str,
            get_expr: Callable[
                [str, Any, str | Expr | None], Expr
            ],  # only used for ValidColumn
            _root_down: tuple[str, ...] | None,
    ) -> typed_dicts.ValidFrameDataErrors:
        out = typed_dicts.ValidFrameDataErrors()

        if self._unique is not None or self._constraints:
            # TODO: add transform info?
            if self._transform is not None:
                out["transform"] = {
                    "pipeline": self._transform,
                    "frame": frame,
                }

        if self._unique is not None:
            # # TODO: check feasibility of unique: link in constraints
            #  if the columns specified in unique are in the collected_schema

            if "unique" not in out:
                out["unique"] = typed_dicts.Errors()

            # out["unique"]["predicate"] = pl.struct(self._unique).is_duplicated()
            out["unique"]["predicate"] = pl.struct(
                self._unique
            ).is_unique()

            out["unique"]["maybe_errors"] = (
                _negate_filter_from_expr_unique_by(
                    frame=frame,
                    unique_by=self._unique,
                    keep_columns=keep_columns,
                    with_row_index=with_row_index,
                    sort=True,
                )
            )

        # --- just for info
        info = None
        if self._constraints:
            if self._info is not None:
                # lets collect the info only if we may have constraints
                # make dict instead of getting directly from InfoList
                info = self._info[0].to_dict(
                    include_name=False
                )  # Info.to_dict()
                # InfoList only has 1 element here. which is named "info"
                # instead of getting "info" we
                # get the first element in case we may rename the info
        # ---

        for attr, expr in self._constraints.items():
            columns = schema.names()

            # TODO: try, except in case root_names fails
            missing = set(expr.meta.root_names()) - set(columns)
            if missing:
                msg = (
                    f"\nvframe: {self._name!r}"
                    f"\n\tskipped constraint: {attr!r} ({expr!s}).\n"
                    f"\t\tColumns: {missing} are not in the schema."
                )
                warnings.warn(msg, stacklevel=2, )
                continue
            else:
                # we encountered a valid constraint
                if "constraints" not in out:
                    out["constraints"] = {}
                if attr not in out["constraints"]:
                    out["constraints"][attr] = {}

                # --- add info to the errors dict. (should we make this configurable?)
                if info is not None:
                    attr_info = info.get("constraints", {}).get(attr)
                    if attr_info is not None:
                        out["constraints"][attr]["info"] = attr_info
                # ----

                out["constraints"][attr]["predicate"] = expr
                out["constraints"][attr]["maybe_errors"] = (
                    _negate_filter_from_expr(
                        frame=frame,
                        expr=expr,
                        keep_columns=keep_columns,
                        with_row_index=with_row_index,
                    )
                )

        if not set(out.keys()) - {
            "transform"
        }:  # check if "constraints" or "unique" in out
            return {}  # if there were no valid constraints or unique, do not populate the dict

        return out

    def _gather_validators_errors(
            self,
            frame: pl.LazyFrame,
            *,
            mode: ValidationMode,
            keep_columns: IntoKeepColumns,
            with_row_index: bool | str,
            get_expr: Callable[[str, Any, str | pl.Expr | None], pl.Expr],
            cast: bool,
            _root_down: tuple[str, ...] | None,
    ) -> typed_dicts.ValidFrameValidatorsErrors:
        if self._validators is None:
            return {}

        errors: typed_dicts.ValidationErrors = self._validators._gather_errors(
            frame=frame,
            schema=None,
            mode=mode,
            keep_columns=keep_columns,
            with_row_index=with_row_index,
            get_expr=get_expr,
            cast=cast,
            _root_down=_root_down,
        )
        if errors:
            return {"validators": errors}
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
            _root_down: tuple[str, ...] | None,
    ) -> typed_dicts.ValidFrameErrors:
        if self._transform is not None:
            frame = self._transform(frame)
            # schema = frame.collect_schema()

        if cast:
            if self._validators is not None:
                frame = cast_frame(
                    frame=frame,
                    schema=self._validators.to_schema(
                        check_dtypes=True
                    ),
                )
            else:
                msg = "cast only supported when vcol(s) are specified within validators"
                raise ValueError(msg)

        out: typed_dicts.ValidFrameErrors = super()._gather_errors(
            # type: ignore[assignment]
            frame=frame,
            mode=mode,
            keep_columns=keep_columns,
            with_row_index=with_row_index,
            get_expr=get_expr,  # only used for ValidColumn now
            cast=cast,  # inconsequential here, only passed because ValidColumn
            _root_down=_root_down,
        )

        if self._validators is not None:
            validators_errors = self._gather_validators_errors(
                frame=frame,
                mode=mode,
                keep_columns=keep_columns,
                with_row_index=with_row_index,
                get_expr=get_expr,
                cast=cast,
                _root_down=_root_down,
            )
            if validators_errors:
                out.update(validators_errors)

        return out

    # ------------------------------------------------------------------

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
        data = data_to_frame_like(data=data)

        if cast:
            if self._validators is not None:
                # cast here to store the data already casted
                data = cast_frame(
                    frame=data,
                    schema=self._validators.to_schema(
                        check_dtypes=True
                    ),
                )

        errors: typed_dicts.ValidFrameErrors = self._gather_errors(
            frame=data_to_lazyframe(data=data),
            mode=mode,
            keep_columns=keep_columns,
            with_row_index=with_row_index,
            get_expr=get_expr,
            cast=cast,
            _root_down=None,
        )

        # self._validate is redefined here just to modify the errors dict
        # TODO: check inconsistency on how we store name in validators frames
        name = (
            "" if self._name is None else self._name
        )  # watch it str(None)

        out: typed_dicts.ValidationErrors = {"valid_frame_list": {name: errors}}

        return _validate_dispatch(
            errors=out,
            data=data,
            with_row_index=with_row_index,
            collect=collect,
            on_success=on_success,
            on_failure=on_failure,
        )

    # ------------------------------------------------------------------

    def _predicates(
            self,
            schema: pl.Schema | None,
            *,
            get_expr: Callable[[str, Any, str | pl.Expr | None], pl.Expr],
    ) -> list[pl.Expr]:

        predicates = self._gather_predicates(
            schema=schema,
            get_expr=get_expr,
            _root_down=None,
        )

        # TODO: check inconsistency on how we store name in validators frames
        name = (
            "" if self._name is None else self._name
        )  # watch it str(None)

        out = {"valid_frame_list": {name: predicates}}

        return _gather_predicates(out, leaf_key="predicate")

    def _gather_predicates(
            self,
            schema: pl.Schema | None,
            *,
            get_expr: Callable[[str, Any, str | pl.Expr | None], pl.Expr],
            _root_down: tuple[str, ...] | None,
    ) -> dict[str, Any]:
        if self._transform is not None:
            msg = (
                f"Skipping predicates of vframe {self._name!r}. "
                f"Predicates of vframes with "
                f"transformations can be gathered from ValidationError"
            )
            warnings.warn(msg, stacklevel=2, )
            return {}

        out = super()._gather_predicates(
            schema=schema,
            get_expr=get_expr,
            _root_down=_root_down,
        )

        if self._validators is not None:
            validators_errors = self._gather_validators_predicates(
                schema=schema,
                get_expr=get_expr,
                _root_down=_root_down,
            )
            if validators_errors:
                out.update(validators_errors)

        return out

    def _gather_focal_predicates(
            self,
            schema: pl.Schema | None,
            *,
            get_expr: Callable[[str, Any, str | pl.Expr | None], pl.Expr],
            _root_down: tuple[str, ...] | None,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}

        if self._transform is not None:
            return out  # no predicates if transform is set

        if _root_down is not None:
            msg = (
                f"Unable to gather predicates for vframe with name: {self._name}\n"
                f"The valid frame is placed inside a nested path: {_root_down}\n"
                f"If you want predicate constraints that reflect the nested path "
                f"you need to write them on root "
                f"using the relevant expression namespaces"
            )
            raise ValueError(msg)  # todo: example

        if self._unique is not None:
            # # TODO: check feasibility of unique: link in constraints
            #  if the columns specified in unique are in the collected_schema

            if "unique" not in out:
                out["unique"] = {}

            # out["unique"]["predicate"] = pl.struct(self._unique).is_duplicated()
            out["unique"]["predicate"] = pl.struct(self._unique).is_unique()

        for attr, expr in self._constraints.items():

            if schema is not None:
                columns = schema.names()

                try:
                    missing = set(expr.meta.root_names()) - set(columns)
                except Exception as e:  # in case root_names fails for some reason
                    msg = f"{e}\nUnable to determine the root names of {attr}"
                    warnings.warn(msg, stacklevel=2, )
                    missing = set()

                if missing:
                    msg = (
                        f"\nvframe: {self._name!r}"
                        f"\n\tskipped constraint: {attr!r} ({expr!s}).\n"
                        f"\t\tColumns: {missing} are not in the schema."
                    )
                    warnings.warn(msg, stacklevel=2, )
                    continue
            # we encountered a valid constraint
            if "constraints" not in out:
                out["constraints"] = {}
            if attr not in out["constraints"]:
                out["constraints"][attr] = {}

            out["constraints"][attr]["predicate"] = expr

        return out

    def _gather_validators_predicates(
            self,
            schema: pl.Schema | None,
            *,
            get_expr: Callable[[str, Any, str | pl.Expr | None], pl.Expr],
            _root_down: tuple[str, ...] | None,
    ) -> dict[str, Any]:
        if self._validators is None:
            return {}

        predicates = self._validators._gather_predicates(
            schema=schema,
            get_expr=get_expr,
            _root_down=_root_down,
        )
        if predicates:
            return {"validators": predicates}
        return {}

    # ------------------------------------------------------------------

    def _gather_validators_transforms(
            self,
            frame: pl.LazyFrame,
    ) -> dict:
        # extra utility, not for validation

        if self._validators is None:
            return {}

        transforms = self._validators._gather_transforms(
            frame=frame,
        )
        if transforms:
            return {"validators": transforms}
        return {}

    def _gather_transforms(
            self,
            frame: pl.LazyFrame,
    ) -> dict[str, Any]:
        # extra utility, not for validation

        out = {}
        if self._transform is not None:
            frame = self._transform(frame)
            out["transform"] = {
                "pipeline": self._transform,
                "frame": frame,
            }

        if self._validators is not None:
            transforms = self._gather_validators_transforms(
                frame=frame,
            )
            if transforms:
                out.update(transforms)

        return out

    def transform(
            self,
            data: IntoValidation,
            *,
            collect: bool = False
    ) -> TransformFrameTree:
        """
        Applies the specified transformation and returns the transformed target data tree.

        Group
        -----
            Transformations
        """
        transforms = self._gather_transforms(
            frame=data_to_lazyframe(data=data),
        )

        # transforms = {"Frames": {self._name: transforms}}
        transforms = {self._name: transforms}
        return _to_transform_frame_tree(
            mapping=transforms,
            collect=collect,
            collect_kwargs={}
        )
