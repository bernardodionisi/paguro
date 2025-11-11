from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence

from paguro.shared.extra_utilities import collect_data_len
from paguro.shared.frame_tree.utils.counts import count_keys_per_ancestor
from paguro.validation.exception.errors.base_validation_error import (
    _BaseValidationError,
)
from paguro.validation.exception.errors.utils.counts import (
    clean_and_group_error_counts_by_type,
    drop_zeros_errors,
)
from paguro.validation.exception.utils.filter_utils import (
    _filter,
    _gather_predicates,
)
from paguro.shared._typing import typed_dicts

if TYPE_CHECKING:
    import polars as pl
    from paguro.typing import FrameLike, CollectConfig


class ValidationError(_BaseValidationError):
    _data: FrameLike
    _mapping: typed_dicts.ValidationErrors  # type: ignore[assignment]

    def __init__(
            self,
            mapping: typed_dicts.ValidationErrors
    ) -> None:
        super().__init__(dict(mapping))

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        error_counts = self._get_error_counts_by_group(groups="baseline")
        return f"{self.__class__.__qualname__} {error_counts}\n[print for more details]"

    # def __str__(self) -> str:
    #     return f"\n{self._ashi_string_repr(errors_only=True)}"

    def _get_error_counts_by_group(
            self, groups: Literal["schema-data", "baseline"] = "baseline"
    ) -> dict[str, int] | dict[str, dict[str, int]]:
        error_keys: tuple[str, ...] = (
            "errors",
            "maybe_errors",
            "no_errors",
            "exception",
        )

        if groups:
            keys: tuple[str, ...] = (
                "dtype",
                "required",
                "allow_nulls",
                "unique",
                "constraints",
            )

            out: dict[str, dict[str, int]] = count_keys_per_ancestor(
                tree=self._mapping,
                keys=error_keys,
                ancestor_keys=keys,
            )

            if groups == "schema-data":
                return clean_and_group_error_counts_by_type(out)
            else:
                return drop_zeros_errors(out)

        return self._count_leaf_keys(keys=error_keys)

    # ------------------------------------------------------------------

    def _gather_predicates(self) -> list[pl.Expr]:
        return _gather_predicates(self._mapping, leaf_key="predicate")

    def _filter(
            self,
            how: (
                    Literal["using_predicates", "using_index"]
                    | None
            ) = "using_predicates",
            *,
            return_valid: bool,
            collect: bool | CollectConfig,
            row_index: str | Sequence[str] | None,
    ) -> FrameLike:
        lazy = _filter(
            root=self._mapping,
            data=self._data,
            how=how,
            return_valid=return_valid,
            row_index=self._row_index if row_index is None else row_index,
        )
        if collect:
            if isinstance(collect, dict):
                return lazy.collect(**collect)  # type: ignore[return-value]
            return lazy.collect()
        return lazy

    def _collect_row_count(
            self,
            *,
            valid_data: bool | None,
    ) -> int:
        if isinstance(valid_data, bool):
            data = self._filter(
                return_valid=valid_data,
                collect=False,
                row_index=None,
            )
        else:
            data = self._data

        return collect_data_len(data)
