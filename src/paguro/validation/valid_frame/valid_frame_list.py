from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Callable, Any

from paguro.typing import IntoKeepColumns, ValidationMode
from paguro.shared._typing import typed_dicts
from paguro.validation.valid_base.valid_list_base import (
    _ValidListBase,
)
from paguro.validation.valid_column.utils.exprs.build_expression import _build_expr
from paguro.validation.valid_frame.valid_frame import ValidFrame

if TYPE_CHECKING:
    import sys
    import polars as pl

    from polars import LazyFrame
    from polars._typing import JoinStrategy

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self


class ValidFrameList(_ValidListBase[ValidFrame]):
    def __init__(
            self, valid_frames: ValidFrame | Iterable[ValidFrame]
    ) -> None:
        if isinstance(valid_frames, ValidFrame):
            valid_frames = [valid_frames]

        super().__init__(valid_list=valid_frames)

    @classmethod
    def _from_list_dict(cls, source: list[dict[str, Any]]) -> Self:
        valid_list = [
            ValidFrame._from_dict(source=i)
            for i in source
        ]
        return cls(valid_frames=valid_list)

    # ------------------------------------------------------------------

    def join(
            self,
            other: ValidFrameList,
            how: JoinStrategy,
            *,
            suffix: str | None,
            join_constraints: bool,
    ) -> ValidFrameList:
        # right (left)
        # self (other) is primary:
        #   - keep all self (other) items and append
        #   any other (self) items whose _name is not
        # in self (other_).

        joined_list: list[ValidFrame] = self._join(
            other=other,
            how=how,
            join_constraints=join_constraints,
            suffix=suffix,
        )
        return self.__class__(joined_list)

    # ------------------------------------------------------------------

    def _gather_transforms(
            self,
            frame: LazyFrame,
    ) -> dict:
        # extra utility - not for validation

        out = {}

        for v in self._valid_list:
            transforms = v._gather_transforms(frame=frame)

            if transforms:
                out[v._name] = transforms

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
            _root_down: tuple[str, ...] | None,
    ) -> dict[str, typed_dicts.ValidFrameErrors]:
        return super()._gather_errors(
            frame=frame,
            mode=mode,
            keep_columns=keep_columns,
            with_row_index=with_row_index,
            get_expr=get_expr,
            cast=cast,
            _root_down=_root_down,
        )

    # ------------------------------------------------------------------

    def _gather_errors_from_dict(
            self,
            data: dict[str, LazyFrame],
            *,
            mode: ValidationMode,
            keep_columns: IntoKeepColumns,
            with_row_index: bool | str,
            cast: bool,
            # _root_down: tuple[str, ...] | None,
    ) -> dict[str, typed_dicts.ValidFrameErrors]:
        out: dict[str, typed_dicts.ValidFrameErrors] = {}

        for v in self._valid_list:
            name = v._name
            if name is None:
                raise TypeError(
                    f"Name must be set for {v.__class__.__name__} "
                    f"when using it within relations."
                )
            frame = data.get(v._name)
            if frame is None:
                continue  # no validator for this dict key

            errors: typed_dicts.ValidFrameErrors = v._gather_errors(
                frame=frame,
                mode=mode,
                keep_columns=keep_columns,
                with_row_index=with_row_index,
                get_expr=_build_expr,
                cast=cast,
                _root_down=None  # todo!
            )

            if errors:
                out[v._name] = errors

        return out
