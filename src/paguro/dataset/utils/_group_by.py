from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar

import polars as pl

from paguro.dataset.dataset import Dataset
from paguro.dataset.lazydataset import LazyDataset

if TYPE_CHECKING:
    import sys
    from collections.abc import Callable, Iterable

    from polars._typing import IntoExpr, SchemaDict

    from paguro.shared._typing._typing import PolarsGroupByTypes

    if sys.version_info >= (3, 11):
        pass
    else:
        pass

__all__ = ["_GroupBy"]

_D = TypeVar("_D", Dataset, LazyDataset)


# based groupby definitions shared between Lazy and Eager
class _GroupBy(Generic[_D]):
    def __init__(
            self,
            group_by_object: PolarsGroupByTypes,
            dataset: _D,
    ) -> None:
        self._group_by_object = group_by_object
        self._dataset: _D = dataset

    def __iter__(self):
        if "__iter__" in dir(self._group_by_object):
            return self._group_by_object.__iter__()

        else:
            msg = f"iter() returned non-iterator of type 'NoneType' for {type(self._group_by_object)}"
            raise TypeError(msg)

    def __getattr__(self, attr) -> Any:
        """Delegate to Polars GroupBy or LazyGroupBy."""
        if attr.startswith("_repr_"):  # or attr.startswith("_ipython_")
            msg = f"{type(self).__name__} object has no attribute {attr}"
            raise AttributeError(msg)

        if hasattr(self._group_by_object.__class__, attr):
            attr_value = getattr(self._group_by_object, attr)

            if callable(attr_value):

                def wrapper(
                        *args,
                        **kwargs: Any,
                ) -> Any:
                    result = attr_value(*args, **kwargs)

                    if isinstance(result, (pl.DataFrame, pl.LazyFrame)):

                        # todo: auto sync schema
                        # if self._dataset._info is not None:
                        #     self._dataset = (
                        #         self._dataset._info
                        #         ._sync_schema(columns=result.columns)
                        #     )
                        return self._dataset._from_instance(
                            frame=result  # type: ignore[arg-type]
                        )
                    else:
                        return result  # non dataframe output

                return wrapper
            else:
                return attr_value  # return attribute (not callable) from Polars
        else:
            msg = (
                f"{type(self).__name__} "
                f"[{self._group_by_object.__class__}] has no attribute {attr}"
            )
            raise AttributeError(msg)

    def agg(
            self,
            *aggs: IntoExpr | Iterable[IntoExpr],
            **named_aggs: IntoExpr,
    ) -> _D:
        """
        Aggregate.
        """
        return self.__getattr__("agg")(*aggs, **named_aggs)

    def map_groups(
            self,
            function: Callable[[pl.DataFrame], pl.DataFrame],
            schema: SchemaDict | None,
    ) -> _D:
        """
        Map groups.
        """
        return self.__getattr__("map_groups")(function, schema)
