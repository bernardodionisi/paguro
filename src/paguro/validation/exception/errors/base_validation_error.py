from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from paguro.shared.frame_tree.frame_tree import FrameTree
from paguro.shared.frame_tree.utils.modify_dict import (
    prune_on_leaf_pair_by_type, _transform_pairs_with_errors_renaming,
)
from paguro.shared.various import write_text_to_html, write_text_to_svg
from paguro.utils.dependencies import pathlib
from paguro.validation.exception.errors._repr.validation_error_repr import (
    validation_exception_box,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path
    from paguro.ashi import StStr
    from paguro.typing import FrameLike, CollectionLike, CollectConfig


class _BaseValidationError(FrameTree, Exception):
    def __init__(
            self,
            mapping: dict[str, Any],
    ) -> None:
        # super().__init__(mapping=errors)  # dont call super here
        FrameTree.__init__(self, mapping=mapping)

        self._data: FrameLike | CollectionLike | None = (
            None
            # could be a single frame like object or a dictionary of frame like objects
        )
        self._row_index: str | None = None

    def _set_data_and_row_index(
            self,
            data: FrameLike | CollectionLike,
            *,
            with_row_index: str | bool,
    ) -> None:
        self._data = data

        if with_row_index:
            # the name "index" is polars' default. We use this is with_row_index is True
            self._row_index = (
                with_row_index
                if isinstance(with_row_index, str)
                else "index"
            )

    # ------------------------------------------------------------------

    def __str__(self) -> str:
        return f"\n{self._ashi_string_repr(errors_only=True)}"

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__} (print for more details)"

    def _non_ashi_string_repr(self) -> str:
        return f"{self.__class__.__qualname__}({self._mapping})"

    def print(self, *, errors_only: bool = False) -> None:
        print(self._ashi_string_repr(errors_only=errors_only))

    def _ashi_string_repr(
            self, *, errors_only: bool = False
    ) -> str | StStr:
        if not errors_only:
            return validation_exception_box(content=self._mapping)

        mapping = self._prune_dict_by_type(
            allowed_keys={
                "errors",
                "errors_count",
                "errors_limited",
                "maybe_errors",
                "exception",
            },
            # allowed_value_types=(pl.DataFrame, pl.LazyFrame, Exception)
            allowed_value_types=None,
        )
        return validation_exception_box(content=mapping)

    def _prune_dict_by_type(
            self,
            allowed_keys: str | Iterable[str],
            allowed_value_types: type[object] | Iterable[type[object]] | None,
    ) -> dict:
        return prune_on_leaf_pair_by_type(
            tree=self._mapping,
            allowed_keys=allowed_keys,
            allowed_value_types=allowed_value_types,
        )

    # ------------------------------------------------------------------

    def _write_string_to_svg(
            self,
            path: str | Path,
            width: int = 100,
            title: str = "Validation Error",
            **kwargs: Any,
    ) -> None:
        text = self.__str__()
        write_text_to_svg(
            path=path, text=text, width=width, title=title, **kwargs
        )

    def _write_string_to_html(
            self,
            path: str | Path,
            width: int = 100,
            **kwargs: Any,
    ) -> None:
        text = self.__str__()
        write_text_to_html(path=path, text=text, width=width, **kwargs)

    # ------------------------------------------------------------------

    def _count_errors(
            self,
    ) -> int:
        # always raise ValidationError even if there are maybe_errors
        # we are in the clear only when no error realized/potential is there
        return sum(i for i in self._get_error_counts().values())

    def _get_error_counts(
            self,
    ) -> dict[str, int]:
        error_keys: tuple[str, ...] = (
            "errors",
            "maybe_errors",
            "errors_counts",
            "errors_limited",
            "exception",
        )

        return self._count_leaf_keys(keys=error_keys)

    # ------------------------------------------------------------------

    def _collect_and_replace(
            self,
            key: str,
            *,
            collect: bool | CollectConfig
    ) -> None:
        if isinstance(collect, bool):
            collect = {}

        iterator_dfs = self._collect(
            key=key,
            sequentially=collect.get("sequentially", False),
            limit=collect.get("limit", False),
            stop_on_non_empty=bool(collect.get("stop_on_non_empty", False)),
            row_counts=collect.get("row_counts", False),
            collect_kwargs=collect.get("collect_kwargs", {}),
        )

        # TODO: allow not inplace?
        _transform_pairs_with_errors_renaming(
            root=self._mapping,
            items=iterator_dfs,
            deepcopy=False,
            on_conflict="raise",
            is_errors_mapping=True,
            is_errors_limit=bool(collect.get("limit", False)),
            is_errors_row_counts=collect.get("row_counts", False),
        )

    # ------------------------------------------------------------------

    def _warn(self, stacklevel: int = 2) -> None:
        warnings.warn(
            str(self), category=UserWarning, stacklevel=stacklevel
        )

    # ------------------------------------------------------------------

    def write_html(
            self,
            path: str | pathlib.Path | None = None,
            *,
            errors_only: bool = True,
            width: int = 80,
            color: str = "#000000",
            background: str = "#ffffff",
            **kwargs: Any,
    ) -> str | None:
        _repr = self._ashi_string_repr(errors_only=errors_only)
        return write_text_to_html(
            path=path,
            width=width,
            text=str(_repr),
            color=color,
            background=background,
            **kwargs,
        )

    def write_svg(
            self,
            path: str | pathlib.Path | None = None,
            *,
            errors_only: bool = True,
            width: int = 80,
            title: str | None = None,
            **kwargs: Any,
    ) -> str | None:
        _repr = str(self._ashi_string_repr(errors_only=errors_only))

        return write_text_to_svg(
            path=path,
            text=_repr,
            width=width,
            title=title,
        )

