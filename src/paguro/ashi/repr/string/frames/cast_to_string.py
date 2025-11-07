from __future__ import annotations

import logging
from collections.abc import Set
from typing import Any, TypeVar, cast

import polars as pl
from polars import selectors as cs
from polars.exceptions import InvalidOperationError

logger = logging.getLogger(__name__)

T = TypeVar("T")


def frame_cast_to_utf8(data: pl.DataFrame, round: int = 2) -> pl.DataFrame:
    """Replace numeric values with strings"""

    def _expr_string_float_to_int(columns: list) -> list[pl.Expr]:
        return [
            pl.when(pl.col(i).str.contains(r"^-?\d+\.0+$"))
            .then(pl.col(i).str.split(".").list.first())
            .otherwise(pl.col(i))
            for i in columns
        ]

    data = data.with_columns(
        cs.float().round(round),
        cs.duration().dt.to_string(format="polars"),
        cs.datetime().dt.to_string(format="%d/%m/%Y (%H.%M.%S)"),
        cs.date().dt.to_string(format="iso"),
    )

    try:
        data = data.cast(pl.String)
    except InvalidOperationError:
        data = data.with_columns(
            pl.all().map_elements(str, return_dtype=pl.String)
        )

    data = data.fill_null("").with_columns(
        _expr_string_float_to_int(data.columns)
    )
    return data


def map_dataframes_to_string(
    obj: T,
    *,
    round: int = 2,
    max_depth: int = 100,
    preserve_polars_series: bool = True,
) -> T:
    """
    Recursively traverse `obj` and apply `frame_cast_to_utf8` to every
    polars.DataFrame encountered. Returns a new structure with the same type.
    """
    if round < 0:
        raise ValueError("round parameter must be non-negative")

    seen: dict[int, Any] = {}
    current_depth = 0

    def walk(x: Any, depth: int = 0) -> Any:
        nonlocal current_depth
        current_depth = max(current_depth, depth)

        if depth > max_depth:
            logging.warning(
                f"Max depth {max_depth} exceeded, returning object as-is"
            )
            raise RecursionError(
                f"Maximum recursion depth {max_depth} exceeded"
            )

        # Fast path for primitives and None
        if x is None or isinstance(
            x, (str, bytes, bytearray, int, float, bool)
        ):
            return x

        # Check for cycles
        oid = id(x)
        if oid in seen:
            return seen[oid]

        # Handle Polars objects
        if isinstance(x, pl.DataFrame):
            try:
                result = frame_cast_to_utf8(x, round=round)
                seen[oid] = result
                return result
            except Exception as e:
                logger.error(f"Failed to convert DataFrame to string: {e}")
                seen[oid] = x
                return x

        if isinstance(x, pl.Series):
            if preserve_polars_series:
                seen[oid] = x
                return x
            else:
                try:
                    # Convert Series to DataFrame, transform, then back to Series
                    df = x.to_frame()
                    transformed_df = frame_cast_to_utf8(df, round=round)

                    series: pl.Series = transformed_df.to_series()
                    seen[oid] = series
                    return series

                except Exception as e:
                    logger.error(
                        f"Failed to convert Series to string: {e}"
                    )
                    seen[oid] = x
                    return x

        # Handle dictionaries (most common mapping)
        if isinstance(x, dict):
            result_dict: dict[Any, Any] = {}
            seen[oid] = result_dict

            for key, value in x.items():
                try:
                    new_value = walk(value, depth + 1)
                    result_dict[key] = new_value
                except Exception as e:
                    logger.error(
                        f"Failed to process dict value for key {key}: {e}"
                    )
                    result_dict[key] = value

            return result_dict

        # Handle lists
        if isinstance(x, list):
            result_list: list[Any] = []
            seen[oid] = result_list

            for i, item in enumerate(x):
                try:
                    new_item = walk(item, depth + 1)
                    result_list.append(new_item)
                except Exception as e:
                    logger.error(
                        f"Failed to process list item at index {i}: {e}"
                    )
                    result_list.append(item)

            return result_list

        # ---------

        # # Handle dataclasses
        # if dc.is_dataclass(x) and not isinstance(x, type):
        #     try:
        #         # Pre-register to handle self-references
        #         temp_result = x  # placeholder
        #         seen[oid] = temp_result
        #
        #         field_updates = {}
        #         for field in dc.fields(x):
        #             try:
        #                 old_value = getattr(x, field.name)
        #                 new_value = walk(old_value, depth + 1)
        #                 if new_value is not old_value:  # Only update if changed
        #                     field_updates[field.name] = new_value
        #             except AttributeError:
        #                 # Field might not be accessible
        #                 continue
        #
        #         if field_updates:
        #             result = dc.replace(x, **field_updates)
        #         else:
        #             result = x
        #
        #         seen[oid] = result
        #         return result
        #     except Exception as e:
        #         logger.error(f"Failed to process dataclass {type(x).__name__}: {e}")
        #         seen[oid] = x
        #         return x

        # ---------

        # # Handle other mappings
        # if isinstance(x, Mapping):
        #     temp_dict: dict = {}
        #     seen[oid] = temp_dict
        #
        #     items: list = []
        #     for key, value in x.items():
        #         try:
        #             new_value = walk(value, depth + 1)
        #             items.append((key, new_value))
        #         except Exception as e:
        #             logger.error(f"Failed to process mapping value for key {key}: {e}")
        #             items.append((key, value))
        #
        #     try:
        #         # Try to preserve original type
        #         result = x.__class__(items)
        #         seen[oid] = result
        #         return result
        #     except Exception:
        #         # Fall back to dict if constructor fails
        #         result = dict(items)
        #         seen[oid] = result
        #         return result

        # ---------

        # # Handle tuples and namedtuples
        # if isinstance(x, tuple):
        #     try:
        #         new_items = []
        #         for i, item in enumerate(x):
        #             try:
        #                 new_item = walk(item, depth + 1)
        #                 new_items.append(new_item)
        #             except Exception as e:
        #                 logger.error(f"Failed to process tuple item at index {i}: {e}")
        #                 new_items.append(item)
        #
        #         new_tuple = tuple(new_items)
        #
        #         # Handle namedtuples
        #         if hasattr(x, '_fields'):
        #             try:
        #                 result = x.__class__(*new_tuple)
        #                 seen[oid] = result
        #                 return result
        #             except Exception as e:
        #                 logger.error(f"Failed to create namedtuple {type(x).__name__}: {e}")
        #                 seen[oid] = new_tuple
        #                 return new_tuple
        #         else:
        #             seen[oid] = new_tuple
        #             return new_tuple
        #
        #     except Exception as e:
        #         logger.error(f"Failed to process tuple: {e}")
        #         seen[oid] = x
        #         return x

        # ---------

        # Handle sets
        if isinstance(x, (set, frozenset, Set)):
            try:
                new_items = set()
                for item in x:
                    try:
                        new_item = walk(item, depth + 1)
                        new_items.add(new_item)
                    except (TypeError, Exception) as e:
                        # Item might not be hashable after transformation
                        logger.warning(
                            f"Failed to process set item, keeping original: {e}"
                        )
                        try:
                            new_items.add(item)
                        except TypeError:
                            # If even the original isn't hashable, skip it
                            logger.warning(
                                f"Skipping non-hashable set item: {item}"
                            )
                            continue

                # Try to preserve set type (set vs frozenset)
                if isinstance(x, frozenset):
                    fs = frozenset(new_items)
                    seen[oid] = fs
                    return fs
                else:
                    s = new_items
                    seen[oid] = s
                    return s

            except Exception as e:
                logger.error(f"Failed to process set: {e}")
                seen[oid] = x
                return x

        # ---------

        # For any other object types, return as-is
        seen[oid] = x
        return x

    try:
        result = walk(obj)
        logger.debug(
            f"Transformation completed. Max depth reached: {current_depth}"
        )
        return cast("T", result)
    except RecursionError:
        logger.error("Recursion limit exceeded during transformation")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during transformation: {e}")
        raise
