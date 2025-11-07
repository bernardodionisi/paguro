from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from paguro.validation.exception.errors.validation_error_relation import (
    _to_relation_validation_error,
)
from paguro.validation.exception.validate_dispatch import (
    _validate_dispatch_base,
)
from paguro.validation.shared.keep_columns import _select_keep_columns

from paguro.validation.valid_relations.utils.topological_order import (
    topological_order,
)
from paguro.validation.valid_relations.utils.utils import (
    _fk,
)
from paguro.validation.shared.data_to import _data_to_dict_lazyframes

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl

    from paguro.collection.collection import Collection
    from paguro.collection.lazycollection import LazyCollection
    from paguro.typing import (IntoKeepColumns, OnSuccess, OnFailureExtra,
    CollectConfig)

__all__ = [
    "ValidPairRelation",
]


class ValidPairRelation:
    def __init__(
            self,
            left: str,
            right: str,
            *,
            name: str | None = None,
            _relations: tuple[dict, ...] | None = None,
    ) -> None:
        self._left = left
        self._right = right

        if _relations is None:
            _relations = ()
        self._relations = _relations

        if name is None:
            name = f"{left}, {right}"

        self._name = name

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self._left!r}, {self._right!r}, relations={self._relations})"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self._left!r}, {self._right!r})"
        )

    def with_subset_relation(
            self,
            relation: Literal["<>", ">", "<"] = "<>",
            on: str | Sequence[str] | None = None,
            left_on: str | Sequence[str] | None = None,
            right_on: str | Sequence[str] | None = None,
    ) -> ValidPairRelation:
        left_relation, right_relation = None, None
        if "<" in relation:
            left_relation = "<"
        if ">" in relation:
            right_relation = ">"

        if on is not None:
            if left_on is not None or right_on is not None:
                msg = (
                    "on and (left_on and right_on) are mutually exclusive"
                )
                raise ValueError(msg)
            left_on, right_on = on, on
        else:
            if left_on is None or right_on is None:
                msg = "Please specify on or (left_on and right_on)"
                raise ValueError(msg)

        rel = {
            "subset": {
                "left": {"relation": left_relation, "on": left_on},
                "right": {"relation": right_relation, "on": right_on},
            }
        }
        return self.__class__(
            left=self._left,
            right=self._right,
            name=self._name,
            _relations=self._relations + (rel,),
        )

    # ------------------------------------------------------------------

    @property
    def _nodes(self) -> set[str]:
        # convenience, the two frame names
        return {self._left, self._right}

    @property
    def _edges(self) -> set[tuple[str, str]]:
        # (child, parent)
        out = set()
        for d in self._relations:
            if d.get("subset", {}).get("left").get("relation") == "<":
                out |= {(self._left, self._right)}

            if d.get("subset", {}).get("right").get("relation") == ">":
                out |= {(self._right, self._left)}

        return out

    def _topological_order(self) -> list[str]:
        return topological_order(self._edges)

    # ------------------------------------------------------------------

    def _gather_subset_relation_errors(
            self,
            data: dict[str, pl.LazyFrame],
            *,
            left: str,
            right: str,
            keep_columns: IntoKeepColumns,
            with_row_index: bool | str,
    ) -> dict[str, pl.LazyFrame]:
        left_frame = data.get(left)

        if left_frame is None:
            msg = f"Missing left frame {left!r} from data"
            raise ValueError(msg)

        right_frame = data.get(right)

        if right_frame is None:
            msg = f"Missing right frame {right!r} from data"
            raise ValueError(msg)

        out: dict = {}

        for r in self._relations:
            subset_relation = r.get("subset", {})

            left_on = subset_relation.get("left", {}).get("on")

            left_frame = _select_keep_columns(
                frame=left_frame,
                column_names=left_on,
                keep_columns=keep_columns,
                with_row_index=with_row_index,
            )

            right_on = subset_relation.get("right", {}).get("on")

            right_frame = _select_keep_columns(
                frame=right_frame,
                column_names=right_on,
                keep_columns=keep_columns,
                with_row_index=with_row_index,
            )

            if subset_relation.get("left", {}).get("relation"):
                maybe_errors = _fk(
                    left=left_frame,
                    right=right_frame,
                    left_on=left_on,
                    right_on=right_on,
                    flag=None,
                )

                out[left] = {}
                rel_name = f"{left}{left_on!r} ⊆ {right}{right_on!r}"

                if rel_name not in out[left]:
                    out[left][rel_name] = {}
                    # TODO: raise error if already in

                out[left][rel_name]["maybe_errors"] = maybe_errors

            if subset_relation.get("right", {}).get("relation"):
                maybe_errors = _fk(
                    left=right_frame,
                    right=left_frame,
                    left_on=right_on,
                    right_on=left_on,
                    flag=None,
                )

                out[right] = {}
                rel_name = f"{right}{right_on!r} ⊆ {left}{left_on!r}"

                if rel_name not in out[right]:
                    out[right][rel_name] = {}

                out[right][rel_name]["maybe_errors"] = maybe_errors

        return out

    def _gather_errors(
            self,
            data: dict,
            *,
            keep_columns: IntoKeepColumns,
            with_row_index: bool | str,
    ) -> dict:
        errors = self._gather_subset_relation_errors(
            data=data,
            left=self._left,
            right=self._right,
            keep_columns=keep_columns,
            with_row_index=with_row_index,
        )
        return errors

    # ------------------------------------------------------------------

    def validate(  # noqa: ANN201
            self,
            data: dict | Collection | LazyCollection,
            keep_columns: IntoKeepColumns = False,
            *,
            collect: bool | dict = True,
            on_success: OnSuccess = "return_none",
            on_failure: OnFailureExtra = "raise",
    ):
        return self._validate(
            data=data,
            keep_columns=keep_columns,
            with_row_index=False,
            collect=collect,
            on_success=on_success,
            on_failure=on_failure,
        )

    def _validate(  # noqa: ANN202
            self,
            data: dict | Collection | LazyCollection,
            *,
            keep_columns: IntoKeepColumns,
            with_row_index: bool | str,
            collect: bool | CollectConfig,
            on_success: OnSuccess,
            on_failure: OnFailureExtra,
    ):
        errors = self._gather_errors(
            data=_data_to_dict_lazyframes(data),
            keep_columns=keep_columns,
            with_row_index=with_row_index,
        )

        if errors:
            errors = {"relation_errors": {self._name: errors}}

        try:
            topological_order = self._topological_order()
        except ValueError:  # cycle detected
            topological_order = None

        # instance of RelationValidationError
        validation_error = _to_relation_validation_error(
            errors=errors,
            data=data,
            with_row_index=with_row_index,
            topological_order=topological_order,
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
