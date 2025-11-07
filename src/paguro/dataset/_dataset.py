from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast, Literal

import polars as pl

from paguro.ashi.info.info_collection import InfoCollection
from paguro.ashi.info.info import Info

from paguro.ashi.repr.html.html_dict import DictHTML, transform_nested_dict
from paguro.ashi.repr.html.utility import html_repr_as_str
from paguro.ashi.repr.string.box.box import Box

from paguro.dataset.utils.utils import _unnest
from paguro.eda.skim.skim import skim
from paguro.shared._typing._typing import PolarsGroupByTypesTuple

from paguro.shared.various import insert_columns_where_ellipsis, pl_column_names
from paguro.utils.dependencies import copy
from paguro.validation.validation import Validation
from paguro.models.vfm import VFrameModel
from paguro.eda.print_schema import print_schema
from paguro.dataset.io.metadata.metadata import (
    to_paguro_dataset_metadata_serialized_key_value,
    _from_paguro_dataset_metadata_to_attrs
)

from paguro.models.vfm._blueprint import collect_model_blueprint
from paguro.typing import _CollectConfig

if TYPE_CHECKING:
    import sys
    from collections.abc import Callable, Iterable, Sequence, Mapping
    from types import EllipsisType

    from polars import DataFrame, LazyFrame
    from polars._typing import (
        ColumnNameOrSelector,
        FrameInitTypes,
        IntoExpr,
        JoinStrategy,
    )
    from paguro.dataset.utils._group_by import _GroupBy

    from paguro.ashi import StStr
    from paguro.collection.collection import Collection
    from paguro.typing import (
        OnFailure, ValidatorOrExpr, ValidationMode, IntoKeepColumns
    )
    from paguro.dataset.dataset import Dataset
    from paguro.dataset.lazydataset import LazyDataset

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

_FrameT = TypeVar("_FrameT", pl.DataFrame, pl.LazyFrame)


class _Dataset(Generic[_FrameT]):
    _data: _FrameT
    _name: str | None

    def __init__(
            self,
            data: DataFrame | LazyFrame | FrameInitTypes | _Dataset,
            name: str | None = None,
            **kwargs: Any,
    ) -> None:
        if isinstance(data, _Dataset):
            _name = data._name
            data = data._data
            if name is None:
                name = _name

        elif not isinstance(data, (pl.DataFrame, pl.LazyFrame)):
            data = pl.LazyFrame(data, **kwargs)

        if not isinstance(data, (pl.DataFrame, pl.LazyFrame)):
            msg = f"Unexpected data type: {type(data)}"
            raise TypeError(msg)

        self._data = cast("_FrameT", data)
        self._name = name

        self._info: InfoCollection | None = None
        self._validation: Validation | None = None
        self._model: type[VFrameModel] | None = None

        self._is_modified: bool = False

    # ------------------------------------------------------------------

    def _print_schema(self) -> None:
        print_schema(self._data)

    def __repr__(self) -> str:
        active = os.environ.pop("PAGURO_DATASET_ASHI_REPR", None)
        if active or active is None:
            return self.__str__()
        return self._non_ashi_string_repr()

    def __str__(self) -> str:
        return str(self._ashi_string_repr())

    def _repr_html_(self) -> str:
        return self._ashi_html_repr()

    def _ashi_html_repr(self, *, as_str: bool = False) -> str:
        if as_str:
            return html_repr_as_str(self.__str__())

        else:
            name = f"{type(self).__qualname__}"

            if self._name is not None:
                name += f'<p style="font-size:13px;color:gray;display:inline""> {self._name}</p>'

            out: dict = {"data": self._data}
            if self._info:
                info = self._info.to_dict(keep_empty=False)

                out["info"] = {
                    title: transform_nested_dict(inner)
                    for title, inner in info.items()
                }

            return DictHTML({name: out})._repr_html_()

    def _non_ashi_string_repr(self) -> str:
        if self._model is not None:
            top_name = f"{type(self).__qualname__}[{self._model.__name__}]"
        else:
            top_name = f"{type(self).__qualname__}"

        if self._validation is not None:
            _v = _get_auto_mode_repr()
            top_name += f"<validation[{_v}]>"

        out = f"{top_name}\n{self._data.__repr__()}"
        if self._info is not None:
            out += "\nprint for more info"
        return out

    def _ashi_string_repr(self, box: Box | None = None) -> str | StStr:

        if box is None:
            box = Box()

        bottom_name = None

        top_name = f"{type(self).__qualname__}"
        if self._model is not None:
            top_name += f"[{self._model.__name__}]"

        if self._name is not None:
            _name = (
                self._name
                if len(self._name) < 25
                else f"...{self._name[-25:]}"
            )
            top_name += f"\n{_name}"

        # if self._is_modified:
        #     top_name += "\n(modified)"

        if self._validation is not None:
            _v = _get_auto_mode_repr()

            bottom_name = f"validation[{_v}]"

        (
            box
            .set_top_name(top_name=f"{top_name}")
            .set_bottom_name(bottom_name=bottom_name)
            .set_align_content("center-ind")
            .set_dict_nested_levels(1)
            # .set_dict_num_columns(2)
            .set_inner_boxes(Box("horizontal_top_ascii"))
            .set_width_chars()
        )

        width_chars = None
        if os.environ.get("ASHI_WIDTH_CHARS") is None:
            if isinstance(self._data, pl.DataFrame):
                n_columns = self._data.shape[1]
                width_chars = 35 + n_columns * 15

                if width_chars > 100:
                    if box._inner_box is not None:
                        box._inner_box.set_dict_positioning("left")

        if self._info is not None:

            # info_boxes = [i._get_box() for i in self.info if i.info]
            # # add information
            # out: str = box.join_boxes(
            #     other=info_boxes,
            #     content=self._data,
            #     width_chars=width_char
            # )

            try:
                return box.to_string(
                    [self._data, self._info.to_dict(keep_empty=False)],
                    width_chars=width_chars,
                )
            except ValueError as e:
                warnings.warn(
                    f"\nUnable to display dataset with the set width, "
                    f"defaulting to 80 chars"
                    f"\n{e!s}",
                    stacklevel=2,
                )
                return box.to_string(
                    [self._data, self._info.to_dict(keep_empty=False)],
                    width_chars=80,
                )

        else:
            return box._to_string(
                content=self._data,
                width_chars=width_chars,
            )

    @classmethod
    def _from_object(
            cls,
            base_object: _Dataset | Self,  # Self not working because of Generic
            frame: _FrameT
    ) -> Self:
        """
        Create a new Dataset instance from an existing instance with new data.

        Copies all internal state from obj but uses the provided data.
        """
        new = cls.__new__(cls)
        new.__dict__ = copy.deepcopy(base_object.__dict__)
        new._data = frame
        return new

    def _from_instance(self, frame: _FrameT) -> Self:
        return self._from_object(base_object=self, frame=frame)

    def __dir__(self) -> list[str]:
        return sorted(set(dir(self._data) + list(super().__dir__())))

    def _getattr(
            self,
            attr: str,
            *,
            validation: bool | Validation | None = None,
    ) -> Any:
        """Delegate to Polars DataFrame or LazyFrame."""
        if attr.startswith("_repr_"):  # or attr.startswith("_ipython_")
            msg = f"{type(self).__name__} object has no attribute {attr}"
            raise AttributeError(msg)

        if hasattr(self._data.__class__, attr):
            attr_value = getattr(self._data, attr)

            if callable(attr_value):

                def wrapper(
                        *args,
                        **kwargs: Any,
                ) -> Any:

                    result = attr_value(*args, **kwargs)

                    if isinstance(
                            result, (pl.DataFrame, pl.LazyFrame)
                    ):  # no self._data.__class__, would break lazy/collect

                        _mode = _get_auto_validation_mode()

                        if _mode is not None and _mode != "none":

                            if validation is None:  # most cases
                                this_validation = self._validation

                            elif isinstance(validation, bool):
                                if validation:
                                    this_validation = self._validation
                                else:
                                    this_validation = None

                            else:  # Validation
                                this_validation = validation

                            if this_validation is not None:
                                # data returned from validation
                                # should always be "consistent" with input
                                limit_ = os.environ.get("PAGURO_AUTO_VALIDATION_LIMIT")
                                collect_ = True
                                if limit_ is not None:
                                    collect_ = _CollectConfig(
                                        limit=int(limit_),
                                        row_counts=False,
                                    )

                                result = (
                                    this_validation
                                    .validate(
                                        data=result,
                                        mode=_mode,
                                        keep_columns=False,
                                        collect=collect_,
                                        on_success="return_data",
                                        on_failure="raise",
                                        cast=False,
                                    )
                                )

                        # ----------------------------------------------

                        # new = copy.deepcopy(self)
                        # new._data = result

                        # ----------------------------------------------

                        new = _new_dataset_or_lazydataset(
                            base_object=self,  # type: ignore[arg-type]
                            frame=result
                        )
                        # from paguro.dataset.dataset import Dataset
                        # from paguro.dataset.lazydataset import LazyDataset
                        #
                        # target_cls = (
                        #     LazyDataset
                        #     if isinstance(result, pl.LazyFrame)
                        #     else Dataset
                        # )
                        # new = target_cls._from_object(  # type: ignore
                        #     base_object=self,
                        #     frame=result
                        # )  # copying all the attrs from self

                        # ----------------------------------------------

                        if not new._is_modified and attr not in [
                            "lazy",
                            "collect",
                        ]:
                            new._is_modified = True

                        # if new._info is not None:
                        #     # careful: collect_schema may panic with
                        #     # an exception that is different than .collect()
                        #
                        #     # TODO: make syncing schema optional for lazydataset
                        #     new._info.sync_schema(
                        #         pl_column_names(result),
                        #         drop_empty="never"
                        #     )

                        return new

                    elif isinstance(result, PolarsGroupByTypesTuple):
                        from paguro.dataset.utils._group_by import _GroupBy
                        return _GroupBy(group_by_object=result, dataset=self)

                    else:
                        return result  # non dataframe output

                return wrapper
            else:
                return attr_value  # return attribute (not callable) from Polars
        else:
            msg = (
                f"{type(self).__name__} [{self._data.__class__}] "
                f"has no attribute {attr}"
            )
            raise AttributeError(msg)

    # --------------------------- name and info ------------------------

    def with_name(self, name: str | None) -> Self:
        if isinstance(name, str) or name is None:
            new = copy.deepcopy(self)

            new._name = name
            new._is_modified = False
        else:
            msg = "name must be of type string or None"
            raise TypeError(msg)

        return new

    def with_info(self, k: str, /, **mapping: Any) -> Self:
        new = copy.deepcopy(self)

        if new._info is None:
            new._info = InfoCollection()

        cols_set: set[str] = set(pl_column_names(new._data))
        keys = set(mapping.keys())

        if k in new._info:

            # Existing info: if schema-level,
            # first expand its allowed keys to the *current* schema
            target = new._info[k]
            if target._is_schema_level:
                target.set_schema(
                    mode="enforced",
                    keys=cols_set,
                    normalize_existing=False
                )

            new._info = new._info.update(k, **mapping)
        else:
            info = Info(k).update(**mapping)
            # auto-detect schema-level ONCE, at creation
            if keys and keys.issubset(cols_set):
                info.set_schema(
                    mode="enforced", keys=cols_set, normalize_existing=True, )
                info._is_schema_level = True
            else:
                info.set_schema(mode="off")  # non-schema info stays free-form
                info._is_schema_level = False

            new._info = new._info.append(info)

        for info in new._info:
            if info._is_schema_level:
                info.set_schema(
                    mode="enforced",
                    keys=cols_set,
                    normalize_existing=True
                )

        return new

    # ------------------------- model ----------------------------------

    def _with_model(
            self,
            model: type[VFrameModel],
            *,
            overwrite: bool = False,
    ) -> LazyDataset | Dataset:
        # technically this should be defined wither in
        if self._model is not None:
            raise ValueError(
                "Model has already been set. Set overwrite=True to replace it.")
        if self._validation is not None:
            raise ValueError(
                "Validation has already been set. Set overwrite=True to replace it."
            )

        model._valid_frame.validate(
            data=self._data,
            mode="all",
            collect=True,
            on_success="return_none",
            on_failure="raise",
            cast=False,
        )

        new = copy.deepcopy(self)
        new._model = model
        new._validation = Validation(copy.deepcopy(model._valid_frame))

        return new  # type: ignore

    def _without_model(
            self,
    ) -> Self:
        new = copy.deepcopy(self)
        new._model = None
        return new

    def collect_model_blueprint(
            self,
            path: str | Path | None = None,
            *,
            name: str | None = "DatasetModel",
            dtypes: bool | Literal["as_values"] = False,
            allow_nulls: bool | None = None,
    ) -> str:
        """
        Generate a blueptrint for a model of the datased based on VFrameModel.

        Note that in order to generate the blueprint .collect_schema() is called.

        Group
        -----
            Model
        """
        if name is None:
            # we should make sure the dataset name is a valid string for a class name
            if self._name is not None:
                name = self._name.capitalize()

            if name is None:
                raise ValueError(
                    "Please specify a name for the root class model, either by:\n"
                    "- .to_model_blueprint(name='DatasetModel')\n"
                    "- assigning a name to the dataset Dataset.with_name('name')\n"
                )
        return collect_model_blueprint(
            data=self._data,
            path=path,
            root_class_name=name,
            include_dtypes=dtypes,
            allow_nulls=allow_nulls,
            print_usage_suggestion=True,
        )

    # ---------------------- validation --------------------------------

    @property
    def validation(self) -> Validation | None:
        """
        Retrives the validation that has been added to the dataset.

        Group
        -----
            Validation
        """
        # do not define a @validation.setter, use with_validation
        return self._validation

    def with_validation(
            self,
            *validators: (
                    ValidatorOrExpr
                    | Iterable[ValidatorOrExpr]
                    | Validation
            ),
            overwrite: bool = False,
            **named_validators: ValidatorOrExpr,
    ) -> Self:
        """
        Add validation to the dataset.

        Group
        -----
            Validation
        """
        # TODO! IMPORTANT MODEL?

        if not validators and not named_validators:
            msg = (
                "No validators specified, "
                "please specify at least one validator to set validation."
            )
            raise TypeError(msg)

        # TODO: _with_validation_update

        validation = Validation(*validators, **named_validators)

        if self._validation is not None:
            if not overwrite:
                raise ValueError(
                    "Validation has already been set; set overwrite=True to replace it."
                )
        return self._with_validation_replace(validation)

    def _with_validation_replace(
            self,
            validator: Validation,
    ) -> Self:
        if self._model is not None:
            msg = (
                f"{self.__class__.__qualname__} "
                f"has been set to be of model: {self._model}. "
                f"In order to replace the validation attribute you "
                f"must remove the model from the {self.__class__.__qualname__} using "
                f"{self.__class__.__qualname__}.without_model()."
                f"Alternativly to validate the data without "
                f"setting the attribute validation "
                f"you can call {self.__class__.__qualname__}.validate()."
            )
            raise ValueError(msg)

        mode = _get_auto_validation_mode()

        if mode is not None:
            result = self._validate(
                validator=validator,
                mode=mode,  # type: ignore[arg-type]
                keep_columns=False,
                on_failure="raise",
                cast=False,
            )

            new = copy.deepcopy(result)

            # result is Self
            # new = copy.deepcopy(self)  When it used to be Frame
            # new._data = data._data
            new._validation = validator

        else:
            new = copy.deepcopy(self)
            new._validation = validator

        return new

    def validate(
            self,
            *validators: (
                    ValidatorOrExpr
                    | Iterable[ValidatorOrExpr]
                    | Validation
            ),
            mode: ValidationMode = "all",
            keep_columns: IntoKeepColumns = False,
            on_failure: OnFailure = "raise",
            cast: bool = False,
    ) -> Self:
        """
        Validate the dataset.

        If no validators are provided, the validation added with the method
        :meth:`.with_validation()` is used.


        Group
        -----
            Validation
        """
        # this is inplace: should we make it not inplace?

        validation = None
        if validators:
            validation = Validation(*validators)

        return self._validate(
            validator=validation,
            mode=mode,
            keep_columns=keep_columns,
            on_failure=on_failure,
            cast=cast
        )

    def _validate(
            self,
            *,
            validator: Validation | None,
            mode: ValidationMode,
            keep_columns: IntoKeepColumns,
            on_failure: OnFailure,
            cast: bool,
    ) -> Self:

        if self._validation is None and validator is None:
            msg = (
                "A 'validator' must be provided either as an argument "
                "or set within .with_validation()"
            )
            raise ValueError(msg)

        if validator is None:
            return self._validate(
                validator=self._validation,
                mode=mode,
                keep_columns=keep_columns,
                on_failure=on_failure,
                cast=cast,
            )

        out = validator.validate(
            data=self,
            mode=mode,
            keep_columns=keep_columns,
            on_success="return_data",
            on_failure=on_failure,
            cast=cast,
        )
        return out

    # ------------------------- metadata -------------------------------

    def _attrs_to_paguro_metadata(
            self,
            *,
            use_pyarrow_format: bool,
    ) -> dict[str, str] | dict[bytes, bytes]:

        validation: str | None = None
        if self._validation is not None:
            validation = self._validation.serialize()

        info: str | None = None
        if self._info is not None:
            info = self._info.serialize()  # todo with info attrs

        attrs = dict(
            name=self._name,
            validation=validation,
            info=info,
        )
        return to_paguro_dataset_metadata_serialized_key_value(
            class_name=self.__class__.__name__,
            attrs=attrs,
            use_pyarrow_format=use_pyarrow_format,
            json_encoder=None,
        )

    @classmethod
    def _from_paguro_metadata(
            cls,
            frame: _FrameT,
            paguro_metadata: dict[str, str],  # | dict[bytes, bytes]
    ) -> Self:
        # validation and info still need deserialization
        attrs: dict[str, Any] = _from_paguro_dataset_metadata_to_attrs(
            source=paguro_metadata,
            json_decoder=None,
        )
        name = attrs.pop("name")
        validation = attrs.pop("validation", None)
        if validation is not None:
            validation = Validation.deserialize(validation)

        info = attrs.pop("info")
        if info is not None:
            info = InfoCollection.deserialize(info)  # todo: with attributes

        out = cls(
            data=frame,
            name=name,
        )
        out._validation = validation
        out._info = info
        return out

    # ------------------------- descriptives ---------------------------

    def skim(
            self,
            config: list[tuple] | None = None,
            *,
            by: str | list[str] | None = None,
            hist: bool = False,
            unnest_structs: bool | ColumnNameOrSelector = False,
    ) -> Collection:
        """
        Generate a summary of the dataset based on specified configurations.

        This method provides a comprehensive summary for a dataset, allowing
        customization of grouped statistics, histogram generation, and deeper
        dataset analysis by unnesting nested structures.

        Parameters
        ----------
        config
            A list of tuples defining the configuration for the summary statistics.
            Each tuple typically represents a specific computation or analysis to
            perform on the dataset.
        by
            Defines the columns by which to group data before generating the summary.
            If multiple strings are provided, data is grouped hierarchically by the
            specified columns.
        hist
            If True, includes histograms for numeric fields in the summary output.
            Useful for visualizing the distribution of numeric data.
        unnest_structs
            If True, unnests any nested structures in the dataset to enable deeper
            analysis. A ColumnNameOrSelector can also be provided to target specific
            fields to unnest.

        Group
        -----
            EDA
        """
        return skim(
            data=self,  # type: ignore[arg-type]
            config=config,
            by=by,
            hist=hist,
            unnest_structs=unnest_structs,
        )

    # ------------------------------------------------------------------

    def _unnest(
            self,
            columns: ColumnNameOrSelector | None = None,
            separator: str = "_",
    ):
        data: pl.DataFrame | pl.LazyFrame = _unnest(
            data=self._data,
            columns=columns,
            separator=separator,
        )
        new = copy.deepcopy(self)

        new._data = cast("_FrameT", data)
        new._is_modified = True

        return new

    def _select(
            self,
            *exprs: (
                    IntoExpr | EllipsisType
                    | Iterable[IntoExpr | EllipsisType]
            ),
            **named_exprs: IntoExpr,
    ) -> Self:
        # allows to reorder column by using ...
        exprs_ = insert_columns_where_ellipsis(exprs=exprs)
        return self._getattr("select")(*exprs_, **named_exprs)

    # ------------------- polars methods -------------------------------

    def _rename(
            self,
            mapping: Mapping[str, str] | Callable[[str], str],
            *,
            strict: bool,
    ) -> Self:
        # rename is defined for renaming metadata

        validation: Validation | None = None

        if self._validation is not None:
            validation = copy.deepcopy(self._validation)
            validation = validation._rename_valid_columns(mapping)  # inplace operation

        new = copy.deepcopy(self)

        new._data = new._data.rename(mapping, strict=strict)
        new._is_modified = True
        new._validation = validation

        if new._info:
            if isinstance(mapping, dict):
                new._info.rename_schema_keys(mapping, apply_to_members="enforced")
            else:
                warnings.warn(
                    "Info elements have not been renamed. "
                    "To rename info elements please provide a dict[str,str]",
                    stacklevel=2,
                )

        return new

    # ------------ methods that take other Frame

    def _join_validation_inplace(
            self,
            other_validation: Validation | None,
            how: JoinStrategy,
    ) -> Self:
        if other_validation is None:
            return self

        other_validation = copy.deepcopy(other_validation)

        # if there is no validation is self either assign or join, otherwise keep as is
        if self._validation is None:
            self._validation = other_validation
        else:
            self._validation = self._validation._join_validation(
                other_validation, how=how
            )

        if self._validation is not None:
            mode = _get_auto_validation_mode()
            return self._validate(
                validator=self._validation,
                mode=mode,  # type: ignore[arg-type]
                on_failure="raise",
                keep_columns=False,
                cast=False,
            )
        return self

    # -------------------------- groupby -------------------------------

    def _group_by(
            self,
            *args,
            **kwargs: Any,
    ) -> _GroupBy:
        return self._getattr("group_by")(*args, **kwargs)

    def _group_by_dynamic(
            self,
            *args,
            **kwargs: Any,
    ) -> _GroupBy:
        return self._getattr("group_by_dynamic")(*args, **kwargs)

    def _rolling(
            self,
            *args,
            **kwargs: Any,
    ) -> _GroupBy:
        return self._getattr("rolling")(*args, **kwargs)

    # -------------------------- other ---------------------------------

    def _join(
            self,
            other: Self | pl.DataFrame | pl.LazyFrame,
            on: str | pl.Expr | Sequence[str | pl.Expr] | None = None,
            how: JoinStrategy = 'inner',
            **kwargs: Any,
    ) -> Self:
        # warning: currently info is not joined, only left info is kept

        other_has_no_validation = True
        if not isinstance(other, (pl.DataFrame, pl.LazyFrame)):
            # we check that other is either Dataset | LazyDataset in .join_asof
            other_has_no_validation = other._validation is None
            if other._info is not None:
                warnings.warn(
                    "Other Info elements are not preserved through the join. "
                    "This behaviour is going to change."
                )
            other = other._data

        if other_has_no_validation or (
                how in ("semi", "anti") or kwargs.get("coalesce")
        ):
            return self._getattr("join")(
                other=other,
                on=on,
                how=how,
                **kwargs,
            )
        else:
            warnings.warn(
                "other validation has been dropped, "
                "currently joining other validation is not supported."
            )
            return self._getattr("join")(
                other=other,
                on=on,
                how=how,
                **kwargs,
            )  # remove this return an finish below

            # # not validating in getattr, we need to join validation first
            # result = self._data.join(other, *args, **kwargs)
            #  # we need to collect the schemas here to resolve renaming
            # other_new_columns_mapping = get_other_rename_mapping(
            #     *args,
            #     **kwargs,
            #     left_data=self._data,
            #     right_data=other,
            #     new_data=result
            # )
            #
            # ov = copy.deepcopy(other._validation)
            # ov = ov._rename_valid_columns(other_new_columns_mapping)
            # # todo: finish, join the two validations and validate
            # new = _new_dataset_or_lazydataset(
            #     base_object=self,
            #     frame=result
            # )

    def _join_asof(
            self,
            other: Self | DataFrame | LazyFrame,
            **kwargs,
    ) -> Self:
        other_has_validation = False
        if not isinstance(other, (pl.DataFrame, pl.LazyFrame)):
            # we check that other is either Dataset | LazyDataset in .join_asof
            if other._info is not None:
                warnings.warn(
                    "Other Info elements are not preserved through the join_asof. "
                )
            other_has_validation = other._validation is not None
            other = other._data

        if not other_has_validation or kwargs.get("coalesce"):
            return self._getattr("join_asof")(
                other=other,
                **kwargs,
            )
        else:
            warnings.warn(
                "other validation has been dropped, "
                "currently joining other validation is not supported."
            )
            return self._getattr("join_asof")(
                other=other,
                **kwargs,
            )

    def _join_where(
            self,
            other: Self | DataFrame | LazyFrame,
            *args,
            **kwargs,
    ) -> Self:
        other_has_validation = False
        if not isinstance(other, (pl.DataFrame, pl.LazyFrame)):
            # we check that other is either Dataset | LazyDataset in .join_where
            if other._info is not None:
                warnings.warn(
                    "Other Info elements are not preserved through the join. "
                    "This behaviour is going to change."
                )
            other_has_validation = other._validation is not None
            other = other._data

        if not other_has_validation:
            return self._getattr("join_where")(
                other,
                *args,
                **kwargs,
            )
        else:
            warnings.warn(
                "other validation has been dropped, "
                "currently joining other validation is not supported."
            )
            return self._getattr("join_where")(
                other=other,
                *args,
                **kwargs,
            )

    def _merge_sorted(
            self,
            other: Self | DataFrame | LazyFrame,
            **kwargs,
    ) -> Self:
        if not isinstance(other, (pl.DataFrame, pl.LazyFrame)):
            if other._info is not None:
                warnings.warn(
                    "Other: Info is not preserved through the merge_sorted."
                )
            if other._validation is not None:
                warnings.warn(
                    "Other: validation is not preserved through the merge_sorted."
                )

            other = other._data
        return self._getattr("merge_sorted")(
            other=other,
            **kwargs,
        )

    def _update(
            self,
            other: Self | DataFrame | LazyFrame,
            **kwargs,
    ) -> Self:
        if not isinstance(other, (pl.DataFrame, pl.LazyFrame)):
            if other._info is not None:
                warnings.warn(
                    "Other: Info is not preserved through the update."
                )
            if other._validation is not None:
                warnings.warn(
                    "Other: validation is not preserved through the update."
                )

            other = other._data
        return self._getattr("update")(other=other, **kwargs)

    def _vstack(
            self,
            other: Self | DataFrame | LazyFrame,
            **kwargs,
    ) -> Self:
        if not isinstance(other, (pl.DataFrame, pl.LazyFrame)):
            if other._info is not None:
                warnings.warn(
                    "Other: Info is not preserved through the vstack."
                )
            if other._validation is not None:
                warnings.warn(
                    "Other: validation is not preserved through the vstack."
                )

            other = other._data
        return self._getattr("vstack")(other=other, **kwargs)

    # todo: hstack

    # ------------------------- exports --------------------------------

    def to_dataframe(self) -> pl.DataFrame:
        """
        To Polars DataFrame. Collects if the underlying dataset is a LazyFrame.

        Group
        -----
            Export
        """
        if isinstance(self._data, pl.LazyFrame):
            return self._data.collect()
        return self._data

    def to_lazyframe(self) -> pl.LazyFrame:
        """
        To Polars LazyFrame.

        Group
        -----
            Export
        """
        return self._data.lazy()


def _new_dataset_or_lazydataset(
        base_object: Dataset | LazyDataset,
        frame: pl.LazyFrame | pl.DataFrame
) -> Dataset | LazyDataset:
    from paguro.dataset.dataset import Dataset
    from paguro.dataset.lazydataset import LazyDataset

    target_cls = (
        LazyDataset
        if isinstance(frame, pl.LazyFrame)
        else Dataset
    )
    # should define in each class
    new = target_cls._from_object(  # type: ignore[var-annotated]
        base_object=base_object,
        frame=frame  # type: ignore[arg-type]
    )  # copying all the attrs from self

    return new


def _get_auto_mode_repr() -> str:
    _mode = _get_auto_validation_mode()
    _on = True if _mode is not None and _mode != "none" else False
    if _on:
        return f"{_mode}:raise"
    return "off"


def _get_auto_validation_mode():
    return os.environ.get("PAGURO_AUTO_VALIDATION_MODE")
