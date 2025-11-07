from __future__ import annotations

from collections import Counter

import polars as pl

from paguro.validation.valid_column.valid_column import ValidColumn
from paguro.validation.valid_frame.valid_frame import ValidFrame


class DuplicateNameError(Exception):
    pass


def _raise_exception_duplicates_names(
        vcl: list[ValidColumn],
        vfl: list[ValidFrame],
) -> None:
    names_columns: dict = _find_duplicates_names(vcl)
    names_frames: dict = _find_duplicates_names_str(vfl)

    if names_columns or names_frames:
        error_message = "Duplicate Names Found:\n"

        if names_columns:
            error_message += "\nValidColumns(s):\n"
            for name, count in names_columns.items():
                error_message += f"  - {name}: {count} times\n"
            error_message += "Column names must be unique.\n"

        if names_frames:
            error_message += "\nValidFrame(s):\n"
            for name, count in names_frames.items():
                if name is None:
                    error_message += f"  - name is None: {count}\n"
                else:
                    error_message += f"  - {name}: {count}\n"
            # error_message += (
            #     "Frame names are determined by the"
            #     " name of the function that modifies the data. "
            #     "If no function has been specified, "
            #     "the data is considered the original data, "
            #     "and the name will be None. "
            #     "Only one ValidFrame is allowed for each data modification.\n"
            # )

        raise DuplicateNameError(error_message)


def _find_duplicates_names_str(
        valid_list: list[ValidFrame],
) -> dict[str | None, int]:
    counts: Counter[str | None] = Counter([i._name for i in valid_list])
    return {i: c for i, c in counts.items() if c > 1}


# def _find_duplicates_names(
#         valid_list: list[ValidColumn],
# ) -> dict[str, int]:
#     name_counts: dict[str, int] = {}
#     other_counts: dict[str, int] = {}
#     # For pl.Expr we must use meta.eq, so keep small list of (rep_expr, count)
#     expr_reps: list[tuple[pl.Expr, int]] = []
#
#     for item in valid_list:
#         if isinstance(
#                 item, pl.Expr
#         ):  # TODO: we should not have an expression here
#             matched = False
#             for i, (rep, cnt) in enumerate(expr_reps):
#                 if rep.meta.eq(item):
#                     expr_reps[i] = (rep, cnt + 1)
#                     matched = True
#                     break
#             if not matched:
#                 expr_reps.append((item, 1))
#             continue
#
#         elif hasattr(item, "_name"):
#             key = str(item._name)
#             name_counts[key] = name_counts.get(key, 0) + 1
#             continue
#
#         key = str(item)
#         other_counts[key] = other_counts.get(key, 0) + 1
#
#     # Assemble only duplicates
#     result: dict[str, int] = {}
#
#     for k, c in name_counts.items():
#         if c > 1:
#             result[k] = c
#
#     for rep, c in expr_reps:
#         if c > 1:
#             result[str(rep)] = c
#
#     for k, c in other_counts.items():
#         if c > 1:
#             result[k] = c
#
#     return result


def _find_duplicates_names(
        valid_list: list[ValidColumn],
) -> dict[str, int]:
    name_counts: dict[str, int] = {}

    for item in valid_list:
        if not isinstance(item, ValidColumn):
            raise TypeError(
                f"Encountered a wrong type when assessing duplicate names, "
                f"it should be of type ValidColumn, got {type(item)}"
            )
        key = str(
            item._name)  # todo: fix here selector.meta.eq (get doesnt work otherwise)
        name_counts[key] = name_counts.get(key, 0) + 1

    # Assemble only duplicates
    result: dict[str, int] = {}

    for k, c in name_counts.items():
        if c > 1:
            result[k] = c

    return result
