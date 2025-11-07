from __future__ import annotations

from paguro.validation.exception.errors.base_validation_error import (
    _BaseValidationError,
)

# this one contains ONLY relations error

# return this:
#   - if no single frame validation is specified
#   - if single validation is specified,
#       we are pre-filtering by that first and it is a DAG
#       - if it is not a DAG and there are single frame
#           validation we need them all together for fixed point


class RelationValidationError(_BaseValidationError):
    def __init__(self, mapping: dict) -> None:
        super().__init__(mapping=mapping)

        # defined in BaseValidationError
        # self._data: dict | None = None
        # self._row_index: str | None = None

        # if topological order is None, assume it is not a DAG
        self._topological_order: list | None = None

    def _set_topological_order(self, order: list | None) -> None:
        self._topological_order = order


def _to_relation_validation_error(
    errors: dict,
    *,
    data,
    with_row_index: bool | str,
    topological_order: list[str] | None,
) -> RelationValidationError:
    ve = RelationValidationError(errors)

    ve._set_data_and_row_index(
        data=data,  # should this be (Lazy)Collection?
        with_row_index=with_row_index,
    )
    if topological_order is not None:
        ve._set_topological_order(topological_order)
    return ve
