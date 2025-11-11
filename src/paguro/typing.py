from __future__ import annotations

import typing
from collections import abc
from collections.abc import Sequence, Mapping
from typing import TYPE_CHECKING, Literal, TypeAlias, Union, TypedDict, Any

import polars as pl
import polars.selectors as cs
from polars._typing import FrameInitTypes

if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing import TypeAlias

    from paguro.dataset.dataset import Dataset
    from paguro.dataset.lazydataset import LazyDataset
    from paguro.validation.valid_column.valid_column import ValidColumn
    from paguro.validation.valid_frame.valid_frame import ValidFrame
    from paguro.validation.validation import Validation

    from paguro.collection.collection import Collection
    from paguro.collection.lazycollection import LazyCollection

__all__ = [
    "FrameLike",
    "CollectionLike",
    "HasValidate",
    "ValidatorOrExpr",
    "IntoValidators",
    "IntoValidation",
    "OnSuccess",
    "OnFailure",
    "OnFailureExtra",
    "IntoKeepColumns",
    "ValidationMode",
    "FieldsValidators",

]
# _CollectConfig
# CollectConfig

FrameLike: TypeAlias = Union[pl.DataFrame, pl.LazyFrame, "Dataset", "LazyDataset"]
# FrameLike = TypeVar("DataFrame", "LazyFrame", "Dataset", "LazyDataset")

CollectionLike: TypeAlias = Union[
    "Collection", "LazyCollection", Mapping[str, Union["Dataset", "LazyDataset"]]
]

# validation

HasValidate: TypeAlias = Union["ValidColumn", "ValidFrame", "Validation"]
"""
HasValidate
"""

ValidatorOrExpr: TypeAlias = Union["ValidColumn", "ValidFrame", pl.Expr]
"""
Validators
"""

IntoValidators: TypeAlias = Union["ValidatorOrExpr", "Validation"]
"""
IntoValidators
"""

FieldsValidators: TypeAlias = Union[
    ValidatorOrExpr, abc.Collection[ValidatorOrExpr], "Validation"
    , str  # to allow to pass a name as first argument
    , cs.Selector
    , None
]
"""
FieldsValidators
"""

# ------------------

IntoNameVC: TypeAlias = Union[
    str,
    typing.Collection[str],
    cs.Selector,
    int,
    None
]
"""
IntoNameVC
"""

# ------------------

IntoValidation: TypeAlias = Union[
    pl.LazyFrame,
    pl.DataFrame,
    pl.Series,
    "Dataset",
    "LazyDataset",
    FrameInitTypes,
]
"""
IntoValidation
"""

OnSuccess: TypeAlias = Literal["return_data", "return_none"]
"""
OnSuccess

- `"return_data"`
- `"return_none"`

"""

OnFailure: TypeAlias = Literal[
    "raise",
    "warn-return_data",
    "warn-return_valid_data",
    "warn-return_invalid_data",
    "return_data",
    "return_valid_data",
    "return_invalid_data",
]
"""
OnFailure

It can take the following literal values:

**Error**

- `"raise"`
    raises exception
    
**Data**

- `"return_data"`
- `"return_valid_data"`

- `"return_invalid_data"`
- `"warn-return_data"`
- `"warn-return_valid_data"`
- `"warn-return_invalid_data"`
"""

OnFailureExtra: TypeAlias = Union[OnFailure, Literal["return_error"]]
"""
Same as OnFailure, plus:

- `"return_error"`
"""

IntoKeepColumns: TypeAlias = Union[
    bool,
    str,
    cs.Selector,
    pl.Expr,
    Sequence[Union[pl.Expr, cs.Selector, str]]
]
"""
IntoKeepColumns
"""

ValidationMode: TypeAlias = Literal[
    "schema",
    "data",
    "none",
    "all",
    # "all-conditional"
]
"""
ValidationMode

- `"schema"` : validate the schema
- `"data"` : validate the data content
- `"all"`  : validate both schema and data content
- `"none"`
"""


class _CollectConfig(TypedDict, total=False):
    limit: int
    row_counts: bool
    sequentially: bool
    collect_kwargs: dict[str, Any]


CollectConfig: TypeAlias = Union[_CollectConfig, dict[str, Any]]
