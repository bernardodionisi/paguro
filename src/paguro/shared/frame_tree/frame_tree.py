from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from paguro.shared.frame_tree._repr.frame_tree_box import frame_tree_box
from paguro.shared.frame_tree.utils.counts import count_leaf_keys
from paguro.shared.frame_tree.utils.find_by_type import (
    _iter_keyed_typed_matches_fast,
    find_keyed_typed_lists,
)
from paguro.shared.frame_tree.utils.modify_dict import _transform_pairs
from paguro.typing import CollectConfig
from paguro.utils.dependencies import pathlib
from paguro.shared.various import write_text_to_html, write_text_to_svg

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from paguro.ashi import StStr
    from paguro.collection.collection import Collection
    from paguro.collection.lazycollection import LazyCollection


def to_frame_tree(
        mapping: dict[str, Any],
        collect: bool = True,
) -> FrameTree:
    ft = FrameTree(mapping)
    if collect:
        ft._collect_and_replace(
            key="frame",
            collect=collect,
        )  # TODO: allow None
    return ft


class FrameTree:
    def __init__(self, mapping: dict[str, Any],
                 ) -> None:
        self._mapping = mapping

    # -------------------------

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}\n{self._ashi_string_repr()}"

    def __repr__(self) -> str:
        return self.__str__()

    def _non_ashi_string_repr(self) -> str:
        return f"{self.__class__.__qualname__}({self._mapping})"

    def _ashi_string_repr(self) -> str | StStr:
        return frame_tree_box(content=self._mapping)

    # -------------------------

    def _count_leaf_keys(
            self, keys: str | Iterable[str]
    ) -> dict[str, int]:
        return count_leaf_keys(self._mapping, keys)

    # -------------------------

    # TODO: allow to pass key None and not have to specify the key
    #  at which the frame is stored when gathering frames. gather all
    def _gather_keys_and_value_types(
            self, type_: type | tuple[type, ...], key: str
    ) -> tuple[list[tuple[str, ...]], list[Any]]:
        return find_keyed_typed_lists(
            root=self._mapping, want=type_, key=key
        )

    def _gather_keys_and_lazyframes(
            self,
            key: str,  # "frame"
    ) -> tuple[list[tuple], list[pl.LazyFrame]]:
        return self._gather_keys_and_value_types(
            type_=pl.LazyFrame, key=key
        )

    def _collect(
            self,
            key: str,
            *,
            sequentially: bool,
            limit: int | bool,
            row_counts: bool,
            stop_on_non_empty: bool,
            collect_kwargs: dict[str, Any],
    ) -> Iterator[tuple[tuple, pl.DataFrame | Exception]]:
        keys: list[tuple]
        frames: list[pl.LazyFrame]

        keys, frames = self._gather_keys_and_lazyframes(key=key)

        if limit or row_counts:
            if isinstance(limit, bool):
                limit = 1
            frames = _limit_or_len_frames(
                frames=frames,
                limit=limit,
                row_counts=row_counts,
            )

        if sequentially:
            # todo: add stop

            return zip(
                keys,
                _collect_sequentially(
                    frames=frames,
                    stop_on_non_empty=stop_on_non_empty,
                    collect_kwargs=collect_kwargs,
                ),
                strict=False,
            )

        try:
            return zip(
                keys, pl.collect_all(
                    frames,
                    **collect_kwargs,
                ), strict=False
            )
        except Exception:
            return zip(
                keys,
                _collect_sequentially(
                    frames=frames,
                    stop_on_non_empty=stop_on_non_empty,
                    collect_kwargs=collect_kwargs,
                ),
                strict=False,
            )

    def _collect_and_replace(
            self,
            key: str,
            *,
            collect: bool | CollectConfig,
    ) -> None:
        if isinstance(collect, bool):
            collect = {}
        iterator_dfs = self._collect(
            key=key,
            sequentially=collect.get("sequentially", False),
            limit=collect.get("limit", False),
            row_counts=collect.get("row_counts", False),
            stop_on_non_empty=False,
            collect_kwargs=collect.get("collect_kwargs", {}),
        )

        # TODO: allow not inplace?
        _transform_pairs(
            root=self._mapping,
            items=iterator_dfs,
            deepcopy=False,
        )

    def to_lazycollection(
            self,
            key: str,
            *,
            include_dataframes: bool = True,
    ) -> LazyCollection:
        from paguro.collection.lazycollection import LazyCollection

        if include_dataframes:
            data = self._to_type_dict(
                type_=(pl.LazyFrame, pl.DataFrame),
                key=key, to_lazyframe=True)
        else:
            data = self._to_type_dict(
                type_=pl.LazyFrame,
                key=key, to_lazyframe=True)

        return LazyCollection(data)

    def to_collection(self, key: str) -> Collection:
        from paguro.collection.collection import Collection

        type_ = pl.DataFrame
        data = self._to_type_dict(type_=type_, key=key, to_lazyframe=False)
        return Collection(data)

    def _to_type_dict(
            self,
            *,
            type_: type | tuple[type, ...],
            key: str,
            to_lazyframe: bool,
    ) -> dict[str, Any]:
        out = {}
        for k, v in _iter_keyed_typed_matches_fast(
                root=self._mapping, want=type_, key=key
        ):
            key = str(pathlib.Path(*map(str, k)))
            if to_lazyframe:
                out[key] = v.lazy()
            else:
                out[key] = v
        return out

    # ------------------------------------------------------------------

    def write_html(
            self,
            path: str | pathlib.Path | None = None,
            *,
            width: int = 80,
            color: str = "#000000",
            background: str = "#ffffff",
            **kwargs: Any,
    ) -> str | None:
        return write_text_to_html(
            path=path,
            width=width,
            text=self.__str__(),
            color=color,
            background=background,
            **kwargs,
        )

    def write_svg(
            self,
            path: str | pathlib.Path | None = None,
            *,
            width: int = 80,
            title: str | None = None,
            **kwargs: Any,
    ) -> str | None:
        return write_text_to_svg(
            path=path,
            text=self.__str__(),
            width=width,
            title=title,
        )


def _limit_or_len_frames(
        frames: list[pl.LazyFrame], *, limit: int | None, row_counts: bool
) -> list[pl.LazyFrame]:
    out: list[pl.LazyFrame] = []
    for f in frames:
        if limit is not None:
            out.append(f.limit(limit))
        if row_counts:
            out.append(f.select(pl.len()))
    return out


def _collect_sequentially(
        frames: list[pl.LazyFrame],
        stop_on_non_empty: bool,
        collect_kwargs: dict[str, Any],
) -> list[pl.DataFrame | Exception]:
    out: list[pl.DataFrame | Exception] = []

    for f in frames:
        try:
            df = f.collect(**collect_kwargs)
            if stop_on_non_empty and df.shape[0] > 0:
                return [df]

            out.append(df)
        except Exception as e:
            out.append(e)
            # pl.DataFrame({"__exception__": e})

    return out
