from __future__ import annotations

import warnings
from collections import defaultdict
from typing import TYPE_CHECKING

import polars.selectors as cs

from paguro.validation.valid_column.valid_column_list import (
    ValidColumnList,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    import polars as pl

    from paguro.validation.valid_column.valid_column import ValidColumn


def expand_valid_column_list(
        vcl: ValidColumnList | None,
        schema: pl.Schema | None,
        *, required_only: bool,
) -> ValidColumnList | None:
    if vcl is None:
        return None

    names = vcl.names(
        required=required_only
    )

    if all(isinstance(n, str) for n in names):
        return vcl
    else:
        if not schema:
            raise TypeError(
                "Empty schema: unable to expand column names, "
                "either provide a non empty schema or specify vcol names."
            )

    # if not all columns are specified, we need the schema to expand vcl
    if schema is None:
        msg = "Missing schema when expanding valid columns"
        raise TypeError(msg)

    # We need to create a list of ValidColumn that are named
    string_columns: list[str] = []
    selector_columns: dict[int, cs.Selector] = {}

    for idx, n in enumerate(names):
        if isinstance(n, str):
            string_columns.append(n)

        elif n is None:
            selector_columns[idx] = cs.all()
        elif isinstance(n, cs.Selector):
            selector_columns[idx] = n

    exclude = cs.by_name(string_columns)

    # -------

    expanded_columns: dict[int, Iterable[str]] = {}
    for k, s in selector_columns.items():
        expanded_columns[k] = cs.expand_selector(schema, s - exclude)

    deduped_expanded_columns: dict[int, list[str]] = (
        _deduplicate_dict_of_lists(expanded_columns)
    )

    new_vcl: list[ValidColumn] = []
    for i, v in enumerate(vcl._valid_list):
        cols = deduped_expanded_columns.get(i)
        if cols is None:
            new_vcl.append(
                v
            )  # should we copy here? not necessary, only used at validate
        else:
            for c in cols:
                temp: ValidColumn = v.with_name(c)
                new_vcl.append(temp)

    return ValidColumnList(new_vcl)


def _deduplicate_dict_of_lists(
        input_dict: dict[int, Iterable[str]],
) -> dict[int, list[str]]:
    seen = set()
    deletions = defaultdict(
        list
    )  # key: removed from, value: list of (string, kept_by)
    ownership = {}  # string -> owner int
    emptied = []

    new_dict = {}
    for key in sorted(input_dict):  # sort to prioritize lower keys
        new_values = []
        for s in input_dict[key]:
            if s not in seen:
                seen.add(s)
                ownership[s] = key
                new_values.append(s)
            else:
                deletions[key].append((s, ownership[s]))
        if not new_values:
            emptied.append(key)
        new_dict[key] = new_values

    if deletions or emptied:
        warnings.warn(
            _warning_message(deletions, emptied),
            OverlappingSelectorsWarning,
            stacklevel=2,
        )

    return new_dict


def _warning_message(deletions: dict, emptied: list):
    # Build warning message
    message: list[str] = ["\n⚠ Overlapping selectors: "]

    for key in sorted(deletions):
        removed = deletions[key]
        line = (
                f"\t- Column selector at index {key} does not include columns: "
                + ", ".join(
            f'"{s}" (claimed by {owner})' for s, owner in removed
        )
        )
        message.append(line)

    if emptied:
        message.append(
            f"The column validators at index  {', '.join(map(str, emptied))} "
            f"are fully skipped eithe\n"
            f"- because the columns were claimed by earlier selectors."
            f"- no columns selected."
        )

    return "\n".join(message)


class OverlappingSelectorsWarning(UserWarning):
    pass
