from __future__ import annotations

import datetime
from typing import Any

import polars as pl
from polars import DataType
from polars.datatypes import DataTypeClass

from paguro.defer import LazyFrameExpr, export_lfe_to_dict
from paguro.shared.dtypes.dtype_serialize import pl_dtype_to_dict
from paguro.utils.dependencies import decimal, json


class CustomJSONEncoder(json.JSONEncoder):
    """
    JSON encoder for serializing types we use in Paguro.

    Preserves type information using configurable
     metadata key for accurate deserialization.
    """

    # Configurable type key - users can override this
    TYPE_KEY = "__T__"

    @classmethod
    def set_type_key(cls, key: str) -> None:
        """Set the metadata key used for type information."""
        cls.TYPE_KEY = key

    def encode(self, obj: Any) -> str:
        """Encode object to JSON string, preprocessing nested tuples."""
        return super().encode(
            _process_nested_types(
                obj, self.TYPE_KEY, sort_sets_as_str=True
            )
        )

    def default(self, obj: Any) -> Any:
        """Convert non-JSON-serializable objects to serializable format."""
        match obj:
            case _ if isinstance(obj, LazyFrameExpr):
                raw = export_lfe_to_dict(obj)
                # still run the tuple-prepass so you don’t lose tuples nested inside
                processed = _process_nested_types(
                    raw, self.TYPE_KEY, sort_sets_as_str=True
                )
                return {self.TYPE_KEY: "deferred", "__V__": processed}

            case dt if isinstance(dt, (DataType, DataTypeClass)):
                return {
                    self.TYPE_KEY: "dtype",
                    "__V__": pl_dtype_to_dict(dt),
                }

            case pl.Expr():
                return {
                    self.TYPE_KEY: "expr",
                    "__V__": obj.meta.serialize(format="json"),
                }
            case decimal.ValidDecimal():
                return {self.TYPE_KEY: "dec", "__V__": str(obj)}
            case datetime.datetime():  # must come before dt.date()
                return {self.TYPE_KEY: "dt", "__V__": obj.isoformat()}
            case datetime.date():
                return {self.TYPE_KEY: "date", "__V__": obj.isoformat()}
            case datetime.time():
                return {self.TYPE_KEY: "time", "__V__": obj.isoformat()}
            case datetime.timedelta():
                return {self.TYPE_KEY: "td", "__V__": obj.total_seconds()}
            case datetime.tzinfo():
                try:
                    offset = obj.utcoffset(datetime.datetime.now())
                    return {
                        self.TYPE_KEY: "tzinfo",
                        "offset": offset.total_seconds()
                        if offset is not None
                        else None,
                        "name": obj.tzname(None),
                    }
                except (TypeError, ValueError):
                    return {
                        self.TYPE_KEY: "tzinfo",
                        "offset": None,
                        "name": None,
                    }
            case _:
                return super().default(obj)


def _process_nested_types(
        item: Any,
        type_key: str,
        *,
        sort_sets_as_str: bool,
) -> Any:
    """
    Recursively process nested structures to add type hints for special types.

    Special types include: tuples...
    """
    if isinstance(item, (set, tuple, frozenset, list)):
        if (
                isinstance(item, (frozenset, set))
                and sort_sets_as_str
                and len(item) > 1
        ):
            item = sorted(item, key=lambda x: str(x))

        value = [
            _process_nested_types(
                i, type_key, sort_sets_as_str=sort_sets_as_str
            )
            for i in item
        ]

        if isinstance(item, list):
            return value
        elif isinstance(item, set):
            return {type_key: "set", "__V__": value}
        elif isinstance(item, frozenset):
            return {type_key: "fset", "__V__": value}
        elif isinstance(item, tuple):
            return {type_key: "tuple", "__V__": value}

    elif isinstance(item, dict):
        return {
            k: _process_nested_types(
                v, type_key, sort_sets_as_str=sort_sets_as_str
            )
            for k, v in item.items()
        }
    return item
