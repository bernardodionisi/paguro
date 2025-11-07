from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, TypeAlias, Union, Any

import polars as pl
from polars.selectors import Selector

if TYPE_CHECKING:
    from paguro.typing import FrameLike
    from polars._typing import PolarsDataType


def expand_selector(
        target: pl.DataFrame | pl.LazyFrame | Mapping[str, PolarsDataType],
        selector: Selector,
) -> tuple[str, ...]:
    """
    https://github.com/pola-rs/polars/blob/py-0.20.25/py-polars/polars/selectors.py#L127.

    Expand a selector to column names with respect to a specific frame or schema target.

    Parameters
    ----------
    target
        A polars DataFrame, LazyFrame or schema.
    selector
        An arbitrary polars selector (or compound selector).
    """
    if isinstance(target, Mapping):
        from polars.dataframe import DataFrame
        target = DataFrame(schema=target)

    return tuple(target.select(selector).columns)


def _expand_selectors(
        frame: pl.DataFrame | pl.LazyFrame, *items: Any
) -> list[Any]:
    """
    Expand selectors.

    https://github.com/pola-rs/polars/blob/py-0.20.25/py-polars/polars/selectors.py#L127

    Internal function that expands any selectors to column names in the given input.

    Non-selector values are left as-is.
    """
    items_iter = _parse_inputs_as_iterable(items)

    expanded: list[Any] = []
    for item in items_iter:
        if is_selector(item):
            selector_cols = expand_selector(frame, item)
            expanded.extend(selector_cols)
        else:
            expanded.append(item)
    return expanded


def is_selector(obj: Any) -> bool:
    """Indicate whether the given object/expression is a selector."""
    # note: don't want to expose the "Selector" object
    return isinstance(obj, Selector)


def _parse_inputs_as_iterable(
        inputs: tuple[Any, ...] | tuple[Iterable[Any]],
) -> Iterable[Any]:
    if not inputs:
        return []

    # Treat elements of a single iterable as separate inputs
    if len(inputs) == 1 and _is_iterable(inputs[0]):
        return inputs[0]

    return inputs


def _is_iterable(input: Any | Iterable[Any]) -> bool:
    return isinstance(input, Iterable) and not isinstance(
        input, (str, bytes)
    )


# ----------------------------------------------------------------------


def collect_data_len(data: FrameLike) -> int:
    from paguro.dataset.lazydataset import LazyDataset

    if isinstance(data, (pl.LazyFrame, LazyDataset)):
        return data.select(pl.len()).collect().item(0, 0)
    else:
        return data.shape[0]


# ----------------------------------------------------------------------


SchemaTree: TypeAlias = Mapping[str, Union["PolarsDataType", "SchemaTree"]]


def _unnest_schema(
        schema: pl.Schema | dict,
) -> SchemaTree:
    """
    Recursively expand a Polars schema into a nested dictionary of datatypes.

    Structs are expanded into nested dicts.
    """
    result: dict = {}
    for name, dtype in schema.items():
        if isinstance(dtype, pl.datatypes.Struct):
            # Recursively unnest the fields of the struct
            nested_schema = dict(
                dtype
            )  # Struct is dict-like: {field: dtype}
            result[name] = _unnest_schema(nested_schema)
        else:
            result[name] = dtype
    return result
