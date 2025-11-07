from __future__ import annotations

from typing import get_args

from polars.dataframe.group_by import (
    DynamicGroupBy,
    GroupBy,
    RollingGroupBy,
)
from polars.lazyframe.group_by import LazyGroupBy
from typing import TypeAlias, Union
from paguro.utils.dependencies import decimal

# ----------------------------------------------------------------------

PolarsGroupByTypes = Union[
    GroupBy, LazyGroupBy, RollingGroupBy, DynamicGroupBy
]
PolarsGroupByTypesTuple = get_args(PolarsGroupByTypes)

# ----------------------------------------------------------------------

IsBetweenTuple: TypeAlias = Union[
    tuple[int | float | decimal.Decimal, int | float | decimal.Decimal],
    tuple[int | float | decimal.Decimal, int | float | decimal.Decimal, str],
    None
]
