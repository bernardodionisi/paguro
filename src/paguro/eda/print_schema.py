from __future__ import annotations

import polars as pl

from paguro.ashi import Box
from paguro.shared.extra_utilities import _unnest_schema


def print_schema(
    data: pl.DataFrame | pl.LazyFrame, *, unnest_structs: bool = True
) -> None:
    schema = data.lazy().collect_schema()
    box = Box().set_key_equal_symbol("")
    if unnest_structs:
        print(box(_unnest_schema(schema)))
    else:
        print(box(dict(schema)))


