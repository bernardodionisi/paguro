from __future__ import annotations

import os
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import polars as pl

from paguro.ashi.info.info_collection import InfoCollection
from paguro.ashi.repr.html.html_dict import DictHTML
from paguro.ashi.repr.html.utility import html_repr_as_str
from paguro.ashi.repr.string.box.box import Box
from paguro.dataset.dataset import Dataset
from paguro.dataset.utils._group_by import _GroupBy
from paguro.dataset.lazydataset import LazyDataset
from paguro.utils.dependencies import copy
from paguro.shared.functions import concat

if TYPE_CHECKING:
    import sys
    from collections.abc import Iterable
    from collections.abc import ItemsView, KeysView, ValuesView
    from paguro.collection.utils._group_by import _CollectionGroupBy

    from polars._typing import ConcatMethod, IntoExpr

    from paguro.ashi import StStr

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

_DST = TypeVar("_DST", Dataset, LazyDataset)


class _Collection(Generic[_DST]):
    def __init__(
            self,
            data: dict[str, _DST],
            name: str | None = None,
    ) -> None:
        self._data: dict[str, _DST] = data
        self._scope: tuple[str, ...] | None = None

        self._name: str | None = name
        self._info: InfoCollection | None = None

        self._box: Box = Box()

    def set_scope(self, key: str | Iterable[str] | None = None) -> Self:
        if key is not None:
            if isinstance(key, str):
                key = [key]

            missing_in_data = set(key) - set(self._data.keys())
            if missing_in_data:
                msg = f"{missing_in_data} not in keys: {self._data.keys()}"
                raise ValueError(msg)

        new = copy.deepcopy(self)
        new._scope = tuple(key) if key is not None else None
        return new

    def set_scope_by_index(
            self, idx: int | Iterable[int] | None = None
    ) -> Self:
        if idx is None:
            return self.set_scope(key=None)

        elif isinstance(idx, int):
            return self.set_scope(key=list(self._data.keys())[idx])

        else:
            indices = tuple(idx)
            keys = [
                k for i, k in enumerate(self._data.keys()) if i in indices
            ]
            return self.set_scope(key=keys)

    def _expand_scope(self) -> tuple[str, ...]:
        return (
            tuple(self._data.keys())
            if self._scope is None
            else self._scope
        )

    # dict methods: these may be overriding some dataframe methods

    def items(self) -> ItemsView[str, _DST]:
        return self._data.items()

    def keys(self) -> KeysView[str]:
        return self._data.keys()

    def values(self) -> ValuesView[_DST]:
        return self._data.values()

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __iter__(self) -> Iterator:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __delitem__(self, key: str):
        del self._data[key]

    def __setitem__(self, key: str, value: _DST):
        self._data[key] = value

    # ------------------------------------------------------------------

    @classmethod
    def _from_object(
            cls,
            base_object: _Collection | Self,
            data: dict[str, _DST],
    ) -> Self:
        """Create a new _Collection instance from an existing instance and new data."""
        new = cls.__new__(cls)
        new.__dict__ = copy.deepcopy({
            k: v for k, v in base_object.__dict__.items() if k != "_data"
        })
        new._data = copy.deepcopy(data)
        return new

    def _from_instance(self, data: dict[str, _DST], ) -> Self:
        return self._from_object(base_object=self, data=data)

    # ------------------------------------------------------------------

    def _getattr(self, attr) -> Any:
        if attr.startswith("_repr_"):
            msg = f"{type(self).__name__} object has no attribute {attr}"
            raise AttributeError(msg)

        # values should always be of the same type, within a _collection
        _ds_type: type[_DST] = type(next(iter(self._data.values())))

        if hasattr(_ds_type, attr):
            new = copy.deepcopy(self)
            results = {}

            scope = new._expand_scope()

            def wrapper(
                    *args,
                    **kwargs: Any,
            ) -> Self | Any:
                non_frame_results = {}

                for s in scope:
                    data = new._data[s]

                    attr_value = getattr(data, attr)

                    if callable(attr_value):

                        try:
                            result = attr_value(*args, **kwargs)
                        except Exception as e:
                            msg = (
                                f"{e}\nTrying setting the scope "
                                f"to the specific {_ds_type.__name__} you "
                                f"are trying to target: "
                                f"{self.__class__.__qualname__}.set_scope(*), "
                                f" * from {list(self._data.keys())}."
                            )
                            raise type(e)(msg) from e
                        # this could raise,
                        # for example ColumnNotFoundError.
                        # User should be mindful of the scope

                        if isinstance(result, _ds_type):
                            new._data[s] = result
                            # no validation until we have all the
                            # final dict in self._data
                            # then we call validation over all the dict

                            # here we could validate each frame separately?
                        else:
                            non_frame_results[s] = result
                    else:
                        non_frame_results[s] = attr_value

                if non_frame_results:
                    if any(
                            # isinstance(i, PolarsGroupByTypesTuple)
                            isinstance(i, _GroupBy)
                            for i in non_frame_results.values()
                    ):
                        gb_object = {}
                        for i in new._data:
                            if i in scope:
                                gb_object[i] = non_frame_results[i]
                            else:
                                gb_object[i] = new._data[i]

                        from paguro.collection.utils._group_by import _CollectionGroupBy
                        return _CollectionGroupBy(
                            group_by_or_data_objects=gb_object,
                            collection=self,
                        )

                    # if len(scope) == 1:
                    #     return non_frame_results[scope[0]]
                    # else:
                    return non_frame_results
                else:
                    return new

            # If the attribute is a method that will be called later,
            # return the wrapper function.
            # If the attribute is a property, the wrapper is not needed
            # and non-callable results are returned directly.
            if callable(getattr(_ds_type, attr)):
                return wrapper
            else:
                # Directly return the property values from the DataFrames
                for s in scope:
                    data = new._data[s]
                    results[s] = getattr(data, attr)

                # if len(scope) == 1:
                #     return results[scope[0]]
                # else:
                return results
        else:
            msg = f"{type(self).__name__} object has no attribute {attr}"
            raise AttributeError(msg)

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"{self._ashi_string_repr()}"

    def _repr_html_(self):
        return self._ashi_html_repr(as_str=False)

    def _ashi_html_repr(self, *, as_str: bool = True) -> str:
        if as_str:
            return html_repr_as_str(self.__str__())

        name = f"{type(self).__qualname__}"

        if self._name is not None:
            name += (
                f'<p style="font-size:13px;color:gray;display:inline""> '
                f"{self._name}</p>"
            )

        return DictHTML({name: self._data})._repr_html_()

    def _ashi_string_repr(
            self,
    ) -> str | StStr:

        top_name = f"{type(self).__qualname__}"

        if self._name is not None:
            top_name += f"\n{self._name}"

        if not self._data:  # empty collection
            return self._box.set_top_name(top_name=top_name).to_string(
                content=self._data
            )

        width: str | None = os.environ.get("ASHI_WIDTH_CHARS")
        width_chars: int = 80
        if width is None:
            # then auto determine width
            if all(isinstance(i, Dataset) for i in self._data.values()):
                max_width_chars = max(
                    len(i.collect_schema().names())
                    for i in self._data.values()
                )
                width_chars = 45 + max_width_chars * 10
        else:
            width_chars = int(width)  # never None

        scope = self._scope

        bottom_name: str | None = None
        if scope is not None:
            if len(scope) > 1:
                bottom_name = f"scope: {len(scope)} frames"
            else:
                bottom_name = f"scope: {scope[0]!r}"

            if width_chars is None:
                bottom_name = bottom_name[:70]
            else:
                # max between key len and width_chars. because keys don't get adjusted
                # don't do this, if an error is raised for the string repr then
                # prompt the user to just adjust width_chars
                # width_chars = max(
                #     width_chars,
                #     max(len(str(k)) for k in self._data.keys()) + 10
                # )
                bottom_name = bottom_name[: width_chars - 10]

        out: str | StStr = (
            self._box
            .set_top_name(top_name=top_name)
            .set_bottom_name(bottom_name=bottom_name)
            # .set_width_chars(width_chars=width_chars)
            .to_string(
                # use polars repr for simplicity
                content={k: v.to_polars() for k, v in self._data.items()},
                # content=self._data,
                width_chars=width_chars,
            )
        )

        return out

    # ----------------- paguro _dataset methods ------------------------

    # def with_info(self, name: str, /, **mapping: Any) -> MyObject:
    #     """
    #     Return a copy with Info(name) updated by `mapping`.
    #     No schema detection or enforcement—free-form metadata.
    #
    #     Behavior
    #     --------
    #     - If `name` exists, it’s updated (immutably) with `mapping`.
    #     - Otherwise it’s created and appended.
    #     - Non-serializable dict values are auto-coerced by Info (per your Info class).
    #     """
    #     new = copy.deepcopy(self)
    #
    #     if new._info is None:
    #         new._info = InfoList()
    #
    #     if name in new._info:
    #         new._info = new._info.update(name, **mapping)
    #     else:
    #         info = Info(name).update(**mapping)
    #         # Ensure it stays free-form: explicitly disable schema policy.
    #         info.set_schema(mode="off")
    #         new._info = new._info.append(info)
    #
    #     return new
    # ----------------------

    def concat(
            self,
            *,
            with_key_column: bool | str = False,
            how: ConcatMethod = "vertical",
            **kwargs: Any,
    ) -> _DST:
        """Concatenate all items into a single Dataset/LazyDataset."""
        if not self._data:
            msg = "Cannot concatenate an empty collection."
            raise ValueError(msg)

        if with_key_column:
            if not isinstance(with_key_column, str):
                with_key_column = "__key_column__"
            items = [
                d.with_columns(pl.lit(k).alias(with_key_column))
                for k, d in self._data.items()
            ]
        else:
            items = list(self._data.values())
        return concat(  # type: ignore[return-value]
            items,
            how=how,
            **kwargs,
        )

    # ---------------------- polars methods ----------------------------

    # define .group_by for type hinting
    def group_by(
            self,
            *by: IntoExpr | Iterable[IntoExpr],
            maintain_order: bool = False,
            **named_by: IntoExpr,
    ) -> _CollectionGroupBy:
        return self._getattr("group_by")(
            *by, maintain_order=maintain_order, **named_by,

        )

    def join(
            self,
            other: str | pl.LazyFrame | pl.DataFrame | Dataset | LazyDataset,
            **kwargs: Any,
    ) -> Self:
        if isinstance(other, str):
            data: Dataset | LazyDataset | None = self._data.get(other)
            if data is None:
                msg = f"Missing {other}"
                raise ValueError(msg)
            return self._getattr("join")(other=data, **kwargs)

        else:
            return self._getattr("join")(other=other, **kwargs)

    # ------------------------- export ---------------------------------

    def to_dict(self, *, to_polars: bool = False) -> dict:
        if to_polars:
            return {k: v.to_polars() for k, v in self._data.items()}
        return {k: v for k, v in self._data.items()}
