from __future__ import annotations

import datetime
from io import StringIO
from typing import Any

import polars as pl

from paguro.defer import LazyFrameExpr, import_lfe_dict
from paguro.shared.dtypes.dtype_serialize import dict_to_pl_dtype
from paguro.utils.dependencies import decimal, json


class CustomJSONDecoder(json.JSONDecoder):
    """JSON decoder that mirrors CustomJSONEncoder."""

    # configurable type key - should match encoder
    TYPE_KEY = "__T__"

    @classmethod
    def set_type_key(cls, key: str) -> None:
        """Set the metadata key used for type information."""
        cls.TYPE_KEY = key

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(object_hook=self._decode_object, *args, **kwargs)

    def _decode_object(self, data: dict[str, Any]) -> Any:
        """Convert dictionary back to original objects."""
        if self.TYPE_KEY not in data:
            return data

        type_name = data[self.TYPE_KEY]
        v = data["__V__"]

        match type_name:

            case "tuple":
                return tuple(v)

            case "set":
                return set(v)

            case "fset":
                return frozenset(v)

            case "dtype":
                return dict_to_pl_dtype(v)

            case "expr":
                if not isinstance(v, str):
                    msg = f"Expected string for expression value, got {type(v)}"
                    raise TypeError(msg)
                # data = BytesIO(value.encode("utf-8"))
                # data = StringIO(v)
                return pl.Expr.deserialize(StringIO(v), format="json")

            case "deferred":
                return import_lfe_dict(cls=LazyFrameExpr, d=v)

            case "dec":
                return decimal.ValidDecimal(v)

            case "dt":
                return datetime.datetime.fromisoformat(v)

            case "date":
                return datetime.date.fromisoformat(v)

            case "time":
                return datetime.time.fromisoformat(v)

            case "td":
                return datetime.timedelta(seconds=float(v))

            case "tzinfo":
                offset = data.get("offset")
                name = data.get("name")

                if offset is not None:
                    tz = datetime.timezone(
                        datetime.timedelta(seconds=float(offset))
                    )
                    return tz

                if name == "UTC":
                    return datetime.timezone.utc
                return datetime.timezone(datetime.timedelta(0))

            case _:
                return data
