from __future__ import annotations

from typing import Any, Literal

from paguro.shared.frame_tree.frame_tree import FrameTree
from paguro.shared.frame_tree.utils.find_by_type import (
    _iter_keyed_typed_matches_fast,
)
from paguro.utils.dependencies import pathlib


def _to_transform_frame_tree(
        mapping: dict,
        collect: bool,
        collect_kwargs: dict[str, Any],
) -> TransformFrameTree:
    trft = TransformFrameTree(mapping)
    if collect:
        _collect = {"collect_kwargs": collect_kwargs}
        trft._collect_and_replace(
            key="frame",
            collect=_collect,
        )
    return trft


class TransformFrameTree(FrameTree):
    def __init__(self, mapping: dict) -> None:
        super().__init__(mapping)

    def _to_type_dict(
            self,
            *,
            type_: type | tuple[type, ...],
            key: str,
            to_lazyframe: bool,
    ) -> dict[str, Any]:
        out = {}
        for k, v in _iter_keyed_typed_matches_fast(
                root=self._mapping, want=type_,
                key=key,
        ):
            k = _clean_tuple_key(
                data=k,
                target="validators",
                last_two=("transform", "frame"),
            )
            key = str(pathlib.Path(*k))
            if to_lazyframe:
                out[key] = v.lazy()
            else:
                out[key] = v
        return out


def _clean_tuple_key(
        *,
        data: tuple,
        target: str,
        last_two: tuple | None = None
) -> tuple:
    # Step 1: Remove target at even positions
    filtered = tuple(
        str(val)
        for i, val in enumerate(data)
        if not (i % 2 == 1 and val == target)
    )

    # Step 2: Remove last two if specified and matched
    if last_two and len(filtered) >= 2:
        if filtered[-2:] == last_two:
            filtered = filtered[:-2]

    return filtered
