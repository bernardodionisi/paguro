from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, TYPE_CHECKING

import polars as pl

from paguro.defer import LazyFrameExpr
from paguro.utils.dependencies import functools
from paguro.shared._typing import typed_dicts

if TYPE_CHECKING:
    from paguro.typing import FrameLike
    from paguro.dataset.lazydataset import LazyDataset


def _filter(
        root: typed_dicts.ValidationErrors,
        how: Literal["using_index", "using_predicates"] | None,
        *,
        data: FrameLike,
        row_index: str | Sequence[str] | None,
        return_valid: bool,
) -> pl.LazyFrame | LazyDataset:
    # right now we only rely on predicates. But for some reason if predicates
    # constructor fails we should fall back to joining on a user provided column

    data = data.lazy()

    if how is None:
        return data

    elif how == "using_index":
        if row_index is None:
            msg = (
                "\nWe are trying to filter the data using an index column "
                "(filtering with predicates "
                "may not work due to nested use of vframes),\n"
                "however no index was provided. \n"
                "In order to provide a column index to filter with please set "
                "keep_columns to either True or to the column(s) that "
                "you want to keep during validation. "
                "Then specify the index column as "
                "on_failure='return_valid_data[index column]'"
            )
            raise FilterWithIndexException(msg)

        # only whe row_index was bool if it a string we are passing a column
        # data = data.with_row_index(name=row_index)

        return _filter_using_index(
            root=root,
            frame=data,
            row_index=row_index,
            return_valid=return_valid,
        )
    else:  # using_predicates
        try:
            return _filter_using_predicates(
                root=root,
                frame=data,
                return_valid=return_valid,
            )
        except FilterWithPredicatesException:
            return _filter(
                root=root,
                data=data,
                how="using_index",
                row_index=row_index,
                return_valid=return_valid,
            )


def _filter_using_predicates(
        root: Mapping[str, Any],
        *,
        frame: pl.LazyFrame | LazyDataset,
        return_valid: bool,
) -> pl.LazyFrame | LazyDataset:
    # raises FilterWithPredicatesException, if predicates can't be used
    _predicates: list[pl.Expr] = _gather_predicates(
        root,
        leaf_key="predicate",
    )

    if (
            not _predicates
    ):  # this could be non-empty even if validation passed if we collected
        if return_valid:
            return frame
        else:
            return pl.LazyFrame()

    predicate: pl.Expr = _reduce_expr_list(_predicates, logic="and")

    if return_valid:
        return frame.filter(predicate)
    else:  # invalid
        return frame.filter(~predicate)


def _filter_using_index(
        root: Mapping[str, Any],
        *,
        frame: pl.LazyFrame | LazyDataset,
        row_index: str | Sequence[str] | None,
        return_valid: bool,
) -> pl.LazyFrame | LazyDataset:
    if row_index is None:
        msg = "row_index_name cannot be None when filtering"
        raise ValueError(msg)

    # all the rows that are in the 'errors' frames
    _index_frames: list[pl.LazyFrame] = [
        i.select(row_index)
        for i in _gather_non_transformed_errors_frames(
            root,
            leaf_key=None,
            # errors & maybe_errors (other frames are only transform which we skip)
        )
    ]
    if not _index_frames:
        if return_valid:
            return frame
        else:
            return pl.LazyFrame([])

    index_frame: pl.LazyFrame = pl.concat(
        _index_frames, how="vertical"
    ).unique()

    if return_valid:
        # all the rows that are on the left frame that are not in the 'errors' frames
        return frame.join(index_frame, on=row_index, how="anti")
    else:
        return frame.join(index_frame, on=row_index, how="inner")


# ----------------------------------------------------------------------


def _gather_predicates(
        root: Mapping[str, Any],
        leaf_key: Literal["predicate"],
) -> list[pl.Expr]:
    out = []
    stack: list[Mapping[str, Any]] = [root]

    while stack:
        d = stack.pop()

        # prune: check the presence of "transform" first (O(1))
        t = d.get("transform", None)
        if (
                t is not None
                and "pipeline" in t
                and isinstance(t.get("pipeline"), LazyFrameExpr)
        ):
            # skip this entire dict (including siblings)
            continue

        t = d.get("fields", None)  # do not collect predicates for fields
        if (
                t is not None
                and "valid_frame_list" in t
        ):
            vfl = set(t.get("valid_frame_list", {}).keys())
            msg = (
                f"\nUnable to filter the data using "
                f"predicates then vframes are set within nested fields: {vfl}\n"
                f"Please filter using an index."
            )
            raise FilterWithPredicatesException(msg)

        # Process values: only push nested dicts; ignore sequences entirely
        for k, v in d.items():
            if k == leaf_key and isinstance(v, pl.Expr):
                out.append(v)
            elif isinstance(v, Mapping):
                stack.append(v)
            # else: ignore (sequences, scalars, etc.)

    return out


def _gather_non_transformed_errors_frames(
        root: Mapping[str, Any],
        leaf_key: Literal["errors", "maybe_errors"] | None,
) -> list[pl.LazyFrame]:
    out = []
    stack: list[Mapping[str, Any]] = [root]

    while stack:
        d = stack.pop()

        # --------
        # prune: check the presence of "transform"
        #
        t = d.get("transform", None)
        if (
                t is not None
                and "pipeline" in t
                and isinstance(t.get("pipeline"), LazyFrameExpr)
        ):
            # skip this entire dict (including siblings).
            # Siblings: the error frames derived from transformations
            continue
        # --------

        for k, v in d.items():
            if (leaf_key is None or k == leaf_key) and isinstance(
                    v, (pl.DataFrame, pl.LazyFrame)
            ):
                out.append(
                    v.lazy()  # all lazyframes here
                )

            elif isinstance(v, Mapping):
                stack.append(v)
            # else: ignore (sequences, scalars, etc.)

    return out


# ----------------------------------------------------------------------


def _reduce_expr_list(
        expressions: list[pl.Expr], logic: Literal["and", "or"]
) -> pl.Expr:
    if logic == "and":
        return functools.reduce(lambda x, y: x & y, expressions)
    else:  # logic == "or":
        return functools.reduce(lambda x, y: x | y, expressions)


class FilterWithPredicatesException(Exception):
    ...


class FilterWithIndexException(Exception):
    ...
