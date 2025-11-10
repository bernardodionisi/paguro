from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar

from paguro.dataset.dataset import Dataset
from paguro.dataset.lazydataset import LazyDataset

from paguro.collection.collection import Collection
from paguro.collection.lazycollection import LazyCollection
from paguro.utils.dependencies import copy

if TYPE_CHECKING:
    import sys
    from collections.abc import Callable, Iterable

    import polars as pl
    from polars._typing import IntoExpr, SchemaDict

    from paguro.collection.collection import _Collection
    from paguro.dataset.utils._group_by import _GroupBy

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

__all__ = ["_CollectionGroupBy"]

_C = TypeVar("_C", Collection, LazyCollection)


class _CollectionGroupBy(Generic[_C]):

    def __init__(
            self,
            group_by_or_data_objects: dict[str, _GroupBy | LazyDataset | Dataset],
            collection: _C,
    ) -> None:
        self._group_by_or_data_objects = group_by_or_data_objects
        self._collection: _C = collection

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self._group_by_or_data_objects})"

    def __getattr__(self, attr: str) -> Callable[..., Any]:
        if attr.startswith("_repr_"):
            msg = f"{type(self).__name__} has no attribute {attr!r}"
            raise AttributeError(msg)

        new = copy.deepcopy(self)

        def dispatcher(
                *args,
                **kwargs: Any,
        ) -> Any:
            if new._collection._scope is None:
                scope: tuple[str, ...] = tuple(new._collection._data.keys())
            else:
                scope = new._collection._scope

            for s in scope:
                obj = new._group_by_or_data_objects[s]
                try:
                    member = getattr(obj, attr)
                except AttributeError:
                    msg = (
                        f"Object under key {s!r} ({type(obj).__name__}) "
                        f"has no attribute {attr!r}"
                    )
                    raise AttributeError(msg) from None

                out = (
                    member(*args, **kwargs)
                    if callable(member) else member
                )
                new._group_by_or_data_objects[s] = out

            if all(
                    isinstance(v, (LazyDataset, Dataset))
                    for v in new._group_by_or_data_objects.values()
            ):
                return new._from_instance(new._group_by_or_data_objects)
            else:
                return new._group_by_or_data_objects

        return dispatcher

    def agg(
            self,
            *aggs: IntoExpr | Iterable[IntoExpr],
            **named_aggs: IntoExpr,
    ) -> _C:
        return self.__getattr__("agg")(*aggs, **named_aggs)

    def map_groups(
            self,
            function: Callable[[pl.DataFrame], pl.DataFrame],
            schema: SchemaDict | None,
    ) -> _C:
        return self.__getattr__("map_groups")(function, schema)
