from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import polars as pl

from paguro.ashi.info.info_collection import callable_to_column_mapping
from paguro.shared.extra_utilities import _expand_selectors
from paguro.typing import IntoKeepColumns, ValidationMode
from paguro.shared._typing import typed_dicts
from paguro.utils.dependencies import copy
from paguro.validation.valid_base.valid_list_base import (
    _ValidListBase,
)
from paguro.validation.valid_column.valid_column import ValidColumn

if TYPE_CHECKING:
    import sys
    from collections.abc import Callable, Iterable
    from polars._typing import JoinStrategy
    from polars.datatypes import DataTypeClass

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self


class ValidColumnList(_ValidListBase[ValidColumn]):
    def __init__(
            self,
            valid_cols: ValidColumn | Iterable[ValidColumn],
    ) -> None:
        if isinstance(valid_cols, ValidColumn):
            valid_cols = [valid_cols]

        super().__init__(valid_list=valid_cols)

    @classmethod
    def _from_list_dict(
            cls,
            source: list[dict],
    ) -> Self:
        valid_list = [ValidColumn._from_dict(source=i) for i in source]
        return cls(valid_cols=valid_list)

    def names(
            self,
            *,
            required: bool = False,
    ) -> list[str]:
        if required:
            return [c._name for c in self._valid_list if c._required]
        return [c._name for c in self._valid_list]

    def to_schema(
            self,
            *,
            build_struct: bool = False,
            check_dtypes: bool = True,
    ) -> pl.Schema:
        schema: dict = self._get_supertypes(
            build_struct=build_struct,
            include_missing=False,
        )
        return pl.Schema(schema, check_dtypes=check_dtypes)

    def _get_supertypes(
            self,
            *,
            build_struct: bool,
            include_missing: bool,
    ) -> dict[str, DataTypeClass | pl.DataType | None]:
        out: dict[str, DataTypeClass | pl.DataType | None] = {}
        for vc in self._valid_list:
            # if dtype is None but fields is specified we can build struct
            if (
                    vc._dtype is None
                    and vc._fields is None
                    and not include_missing
            ):
                warnings.warn(
                    f"\nskipped vcol -- {vc._name!r} -- dtype missing",
                    stacklevel=2,
                )
                continue

            dtypes = vc._get_supertype(build_struct=build_struct)

            # dtypes can now be None if we were unable to determine supertype
            # or if not specified and 'include_missing' = True
            out[vc._name] = dtypes

        return out

    # ------------------------------------------------------------------

    def _gather_errors(
            self,
            frame: pl.LazyFrame,
            mode: ValidationMode,
            *,
            keep_columns: IntoKeepColumns,
            with_row_index: bool | str,
            get_expr: Callable[[str, Any, str | pl.Expr | None], pl.Expr],
            cast: bool,
            _struct_fields: tuple[str, ...] | None,
    ) -> dict[str, typed_dicts.ValidColumnErrors]:
        return super()._gather_errors(
            frame=frame,
            mode=mode,
            keep_columns=keep_columns,
            with_row_index=with_row_index,
            get_expr=get_expr,
            cast=cast,
            _struct_fields=_struct_fields,
        )

    # ------------------------------------------------------------------

    def join(
            self,
            other: ValidColumnList,
            how: JoinStrategy,
            *,
            suffix: str | None = None,
            join_constraints: bool,
    ) -> ValidColumnList:
        # right (left)
        # self (other) is primary:
        #   - keep all self (other) items and
        #   append any other (self) items whose _name is not
        # in self (other_).
        joined_list: list[ValidColumn] = self._join(
            other=other,
            how=how,
            join_constraints=join_constraints,
            suffix=suffix,
        )

        return self.__class__(joined_list)

    def _drop_column(
            self, frame: pl.LazyFrame, *col_names: str | Iterable[str]
    ) -> Self:
        """
        Drop specified columns from the list of valid_columns.

        Useful to sync columns between Dataset and ValidColumnList,
        in case the column is not in the Dataset anymore
            - if a column is required the validation will raise an error before sync
            - if ValidColumn._allow_drop is True, then we can drop from ValidColumnList
                else we keep it
        """
        # warning: inplace

        cols = _expand_selectors(frame, *col_names)

        # TODO: vc.name can be None or Selector. Must implement that, before exposing

        # list all the columns that are not allowed to be dropped
        not_allowed_drop = []
        for vc in self._valid_list:
            if not vc._allow_drop:
                if vc._name in cols:
                    not_allowed_drop.append(vc._name)

        if not_allowed_drop:
            msg = f"{not_allowed_drop} are not allowed to be dropped"
            raise ValueError(msg)

        new_valid_columns = []
        for vc in self._valid_list:
            if vc._allow_drop:
                if vc._name not in cols:
                    new_valid_columns.append(copy.deepcopy(vc))

        self._valid_list = new_valid_columns

        return self

    def _rename(
            self,
            mapping: dict[str, str] | Callable[[str], str],
            *,
            inplace: bool = False,
    ) -> ValidColumnList:
        # rename columns

        if callable(mapping):
            mapping = callable_to_column_mapping(self.names(), mapping)

        validate_renaming(
            strings_list=self.names(),
            renaming_map=mapping,
        )

        not_allowed = []
        for f in self._valid_list:
            if f._name in mapping and not f._allow_rename:
                not_allowed.append(f._name)

        if not_allowed:
            msg = f"{not_allowed} can't be renamed. [allow_rename=False in validation]"
            raise ValueError(msg)

        # --------------------------------------------------------------

        if inplace:
            target_obj = self
        else:
            target_obj = copy.deepcopy(self)

        return _rename_columns(target_obj, mapping)


# ----------------------------------------------------------------------


def _rename_columns(
        obj: ValidColumnList,
        mapping: dict[str, str],
) -> ValidColumnList:
    # inplace
    for vc in obj._valid_list:
        if vc._name in mapping:
            vc._name = mapping[
                vc._name
            ]  # replacing the valid_column column name
    return obj


def validate_renaming(
        strings_list: list[str],
        renaming_map: dict[str, str],
) -> bool:
    """Validates that renaming strings with the provided mapping won't create duplicates."""
    # Map each string to what it would become after renaming
    new_strings = [renaming_map.get(s, s) for s in strings_list]

    duplicates = set()
    seen = set()

    for new_string in new_strings:
        if new_string in seen:
            duplicates.add(new_string)
        else:
            seen.add(new_string)

    if duplicates:
        conflict_details = []

        for duplicate in duplicates:
            # Find all sources for this duplicate
            # sources = []
            existing_unchanged = False
            rename_conflicts = []

            for i, (old, new) in enumerate(
                    zip(strings_list, new_strings, strict=False)
            ):
                if new == duplicate:
                    if (
                            old == new
                    ):  # This string wasn't renamed but causes conflict
                        existing_unchanged = True
                    else:  # This string was renamed and causes conflict
                        rename_conflicts.append(f"'{old}' → '{new}'")

            # Create appropriate error message based on conflict type
            if existing_unchanged and rename_conflicts:
                conflict_details.append(
                    f"'{duplicate}' already exists and would also be created from: {', '.join(rename_conflicts)}"
                )
            elif len(rename_conflicts) > 1:
                conflict_details.append(
                    f"'{duplicate}' would be created from multiple sources: {', '.join(rename_conflicts)}"
                )

        error_message = "Renaming would create duplicates:\n" + "\n".join(
            conflict_details
        )
        raise ValueError(error_message)

    return True
