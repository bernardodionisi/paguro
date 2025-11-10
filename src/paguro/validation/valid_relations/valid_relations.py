from __future__ import annotations

import warnings
from collections.abc import Callable
import typing
from typing import TYPE_CHECKING, Iterable, Any

import polars as pl

from paguro.validation.shared._docs import set_doc_string, VALIDATE_PARAMS
from paguro.validation.shared.preprocessing.utils import _parse_validators_as_iterable
from paguro.validation.valid_relations.utils.relation_strings import (
    parse_relationship_strings,
)

from paguro.validation.exception.errors.validation_error_relation import (
    _to_relation_validation_error,
)
from paguro.validation.exception.validate_dispatch import (
    _validate_dispatch_base,
)
from paguro.validation.valid_relations.utils.topological_order import (
    topological_order,
)
from paguro.validation.valid_relations.utils.utils import (
    required_params,
)
from paguro.validation.shared.data_to import _data_to_dict_lazyframes

from paguro.validation.valid_relations.valid_pair import (
    ValidPairRelation,
)

from paguro.validation.valid_frame.valid_frame_list import ValidFrameList

if TYPE_CHECKING:
    import sys

    from paguro.collection.collection import Collection
    from paguro.collection.lazycollection import LazyCollection
    from paguro.typing import (
        IntoKeepColumns, OnSuccess, OnFailureExtra,
        ValidationMode, CollectConfig)
    from paguro.validation.valid_frame.valid_frame import ValidFrame

    if sys.version_info >= (3, 11):
        from typing import Unpack
    else:
        from typing_extensions import Unpack

    CustomConstraint = Callable[
        [Unpack[tuple[pl.LazyFrame, ...]]], pl.LazyFrame
    ]

IntoRelations: typing.TypeAlias = typing.Union[
    str, ValidPairRelation, Iterable[ValidPairRelation], Iterable[str]
]

__all__ = [
    "ValidRelations",
]


class ValidRelations:
    def __init__(
            self,
            *validators: ValidFrame | Iterable[ValidFrame],
            relations: IntoRelations | None = None,
            **constraints: CustomConstraint,
    ) -> None:
        # todo: check that all the valid frames have a name (not None)

        self._valid_frame_list: ValidFrameList = ValidFrameList(
            _parse_validators_as_iterable(validators)
            # validators
        )

        if relations is None:
            self._relations = None
        else:
            self._relations = preprocess_valid_relations(relations)

        self._constraints: dict[str, CustomConstraint] = (
            constraints
        )

    @classmethod
    def _(
            cls,
            *validators: ValidFrame | typing.Iterable[ValidFrame],
            relations: IntoRelations | None = None,
            constraints: dict[str, CustomConstraint] | None = None,
    ) -> ValidRelations:
        if constraints is None:
            constraints = {}
        return cls(
            *validators,
            relations=relations,
            **constraints,
        )

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"validators={self._valid_frame_list.__repr__()}, "
            f"relations={self._relations}, "
            f"constraints={self._constraints})"
            f")"
        )

    def __repr__(self) -> str:
        return self.__str__()

    # ------------------------------------------------------------------

    @property
    def _nodes(self) -> set[str]:
        out: set[str] = set()
        if self._relations is not None:
            for vr in self._relations:
                out |= vr._nodes
        return out

    @property
    def _edges(self) -> set[tuple[str, str]]:
        out: set[tuple[str, str]] = set()
        if self._relations is not None:
            for vr in self._relations:
                out |= vr._edges
        return out

    def _topological_order(self) -> list[str]:
        # topological order only useful if there are no custom_constraints
        return topological_order(self._edges)

    # ------------------------------------------------------------------

    def _gather_custom_constraints_errors(
            self,
            data: dict[str, pl.LazyFrame],
            *,
            with_row_index: bool | str,
            with_frame_index: bool | str,
    ) -> dict[str, Any]:
        errors: dict[str, Any] = {}

        if self._constraints:
            if "constraints" not in errors:
                errors["constraints"] = {}

        for name, constraint_func in self._constraints.items():
            try:
                # constraint_func should only require one or more LazyFrame
                # where the parameters names should correspond to the keys
                # of the dictionary

                params = required_params(constraint_func)

                # --------
                extra_params = []
                for i in params:
                    if i not in data:
                        extra_params.append(i)

                if extra_params:
                    msg = f"Constraint parameters [{''.join(extra_params)}] not in data"
                    raise ValueError(msg)
                # ----------

                fill_params = {}
                for n, d in data.items():
                    if n in params:
                        # Adding row_index and frame_index,
                        # this is a convenience but
                        # note that the user may modify both row_index
                        # and frame_index columns
                        # within the function.
                        # In that case, row_index and/or frame_index
                        # may introduce errors when
                        # filtering; this should be clear from the documentation.
                        if with_row_index:
                            if isinstance(with_row_index, str):
                                d = d.with_row_index(name=with_row_index)
                            else:
                                d = d.with_row_index()
                        if with_frame_index:
                            if isinstance(with_row_index, bool):
                                with_frame_index = "__frame_index__"

                            d = d.with_columns(
                                pl.lit(n).alias(str(with_frame_index))
                            )
                        fill_params[n] = d
                # ----------

                # if not fill_params, constraint_func may require no params
                custom_errors = constraint_func(**fill_params)

                if not isinstance(custom_errors, pl.LazyFrame):
                    if custom_errors is None:
                        # if None is returned assume no errors?
                        custom_errors = pl.LazyFrame()
                    else:
                        warnings.warn(
                            f"Skipping constraint {name!r}: Return type of {name!r} is not a LazyFrame",
                            stacklevel=2,
                        )
                        continue

                errors["constraints"][name] = {
                    "maybe_errors": custom_errors
                }
                # note that even if we pass here there
                # could be an Exception when collecting
            except Exception as e:
                errors["constraints"][name] = {
                    "exception": e  # pl.LazyFrame({"__exception__": e})
                }

        if not errors.get("constraints"):
            # could be None if there are no constraints
            # or empty if they are misspecified (no LazyFrame return)
            return {}

        return errors

    # ------------------------------------------------------------------

    def _gather_relation_errors(
            self,
            data: dict[str, pl.LazyFrame],
            *,
            keep_columns: IntoKeepColumns,
            with_row_index: bool | str,
    ) -> dict:
        if self._relations is None:
            return {}

        errors = {}

        for r in self._relations:
            # name is  the combination of the two tables
            errors[r._name] = r._gather_errors(
                data=data,
                keep_columns=keep_columns,
                with_row_index=with_row_index,
            )

        return errors

    def _gather_errors(
            self,
            data: dict[str, pl.LazyFrame],
            *,
            mode: ValidationMode,
            keep_columns: IntoKeepColumns,
            with_row_index: bool | str,
            with_frame_index: bool | str,
            cast: bool,
    ) -> dict:
        errors = {}

        frame_errors = self._valid_frame_list._gather_errors_from_dict(
            data=data,
            keep_columns=keep_columns,
            with_row_index=with_row_index,
            mode=mode,
            cast=cast,
        )

        if frame_errors:
            errors["frame_errors"] = frame_errors

        relation_errors = self._gather_relation_errors(
            data=data,
            keep_columns=keep_columns,
            with_row_index=with_row_index,
        )
        if relation_errors:
            errors["relation_errors"] = relation_errors

        custom_errors = self._gather_custom_constraints_errors(
            data=data,
            with_row_index=with_row_index,
            with_frame_index=with_frame_index,
        )
        if custom_errors:
            errors["constraints_errors"] = custom_errors

        return errors

    # ------------------------------------------------------------------

    @set_doc_string(parameters=VALIDATE_PARAMS)
    def validate(  # noqa: ANN201
            self,
            data: dict | Collection | LazyCollection,
            *,
            mode: ValidationMode = "all",  # todo: add "relation"
            keep_columns: IntoKeepColumns = False,
            with_frame_index: bool | str = False,
            collect: bool | dict[str, Any] = True,
            on_success: OnSuccess = "return_none",
            on_failure: OnFailureExtra = "raise",
            cast: bool = False,
    ):
        """
        Validates relations against the provided data.

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
            with_frame_index=with_frame_index,
            collect=collect,
            on_success=on_success,
            on_failure=on_failure,
            cast=cast,
        )

    def _validate(  # noqa: ANN202
            self,
            data: dict | Collection | LazyCollection,
            *,
            mode: ValidationMode,
            keep_columns: IntoKeepColumns,
            with_row_index: bool | str,
            with_frame_index: bool | str,
            collect: bool | CollectConfig,
            on_success: OnSuccess,
            on_failure: OnFailureExtra,
            cast: bool,
    ):
        errors = self._gather_errors(
            data=_data_to_dict_lazyframes(data),
            mode=mode,
            keep_columns=keep_columns,
            with_row_index=with_row_index,
            with_frame_index=with_frame_index,
            cast=cast,
        )

        try:
            top_order = self._topological_order()
        except ValueError:  # cycle detected
            top_order = None

        # instance of RelationValidationError
        # TODO: add frame flag name to error as we do for row index
        validation_error = _to_relation_validation_error(
            errors=errors,
            data=data,
            with_row_index=with_row_index,
            topological_order=top_order,
        )

        return _validate_dispatch_base(
            errors=errors,
            data=data,
            with_row_index=with_row_index,
            collect=collect,
            on_success=on_success,
            on_failure=on_failure,
            validation_error_type=validation_error,
        )


def preprocess_valid_relations(
        valid_relations: IntoRelations,
) -> tuple[ValidPairRelation, ...]:
    if isinstance(valid_relations, str):
        # mypy: tell it this branch is tuple[str, ...]
        return to_valid_pair_relation_list(valid_relations)

    elif isinstance(valid_relations, ValidPairRelation):
        return (valid_relations,)

    elif all(isinstance(i, str) for i in valid_relations):
        # mypy: tell it this branch is tuple[str, ...]
        return to_valid_pair_relation_list(
            typing.cast(tuple[str, ...], valid_relations)
        )

    elif any(
            not isinstance(i, ValidPairRelation) for i in valid_relations
    ):
        raise TypeError(
            "valid_relations must be all a valid pair relation or strings"
        )
    else:
        # TODO: check that they have unique names
        return typing.cast(tuple[ValidPairRelation, ...], valid_relations)


def to_valid_pair_relation_list(
        strings: str | Iterable[str],
) -> tuple[ValidPairRelation, ...]:
    if isinstance(strings, str):
        strings = [strings]
    pairs = parse_relationship_strings(strings)

    vpt_list: list[ValidPairRelation] = []
    for tables, rel_list in pairs.items():
        left, right = tables
        vpr = ValidPairRelation(left, right)

        for rel in rel_list:
            vpr = vpr.with_subset_relation(**rel)

        vpt_list.append(vpr)
    return tuple(vpt_list)
