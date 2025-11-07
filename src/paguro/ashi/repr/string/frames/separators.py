from __future__ import annotations

import polars as pl


def insert_table_separators(
    table: str | list[str],
    col_positions: int | list[int] | None,
    row_positions: int | list[int] | None,
    header_position: int | None,
) -> str:
    if col_positions is None and row_positions is None:
        # return table
        if isinstance(table, list):
            return "\n".join(table)
        else:
            return table  # str

    else:
        if col_positions is not None:
            table = _table_vertical_column_separator(
                table=table,
                positions=col_positions,
                header_position=header_position,
            )

        if row_positions is not None:
            table = _table_horizontal_row_separator(
                table=table,
                positions=row_positions,
                col_positions=col_positions,
            )

        return "\n".join(table)


def get_separators_indices(
    data: pl.DataFrame,
    separators: tuple[
        int | list[int] | None, int | list[int] | str | pl.Expr | None
    ],
    **polars_tbl_config,
) -> tuple[list, list, int]:
    col_indices, row_indices, header_index = (
        _find_frame_repr_separator_indices(data=data, **polars_tbl_config)
    )

    col_separator_idx, row_separator_idx = separators

    if isinstance(row_separator_idx, (str, pl.Expr)):
        # based on column
        row_separator_idx = _get_row_change_indices(
            data=data, by=row_separator_idx
        )

    col_indices = _to_chars_indices(
        separator_idx=col_separator_idx, positions=col_indices
    )

    row_indices = _to_chars_indices(
        separator_idx=row_separator_idx, positions=row_indices
    )

    return col_indices, row_indices, header_index


# -------------------------- positions ---------------------------------


def _get_row_change_indices(
    data: pl.DataFrame, by: str | pl.Expr
) -> list[int]:
    """Finds where the column (by) changes value"""
    if isinstance(by, str):
        by = pl.col(by)
    s = data.select(by.ne(by.shift(1)).arg_true()).to_series()

    return (s - 1).to_list()  # series to list


def _to_chars_indices(
    separator_idx: int | list[int] | None,
    positions: list[int],
) -> list[int]:
    """Convert integers indicating rows/cols to string chars indices"""
    if separator_idx is None or not positions:
        return []

    if isinstance(separator_idx, int):
        separator_idx = [separator_idx]

    return [positions[i] for i in separator_idx]


def _find_frame_repr_separator_indices(
    data: pl.DataFrame, **polars_tbl_config
):
    with pl.Config(**polars_tbl_config):
        with pl.Config(tbl_formatting="UTF8_FULL"):
            table = data.__repr__()

        col_indices, row_indices, header_index = (
            _find_table_separator_indices(table=table)
        )

    return col_indices, row_indices, header_index


def _find_table_separator_indices(
    table: str | list[str], chars_separators: tuple = ("┬", "├", "╞")
) -> tuple[list[int], list[int], int | None]:
    # Characters used for column and row separation

    columns_sep, rows_sep, header_sep = chars_separators

    if isinstance(table, str):
        table = table.split("\n")

    top_row = table[0]
    has_shape = False
    if top_row.startswith("shape"):
        has_shape = True
        top_row = table[1]

    # ------------------------------------------------------------------

    col_indices: list[int] = [  # find horizontal indices (columns)
        i
        for i, char in enumerate(
            top_row
        )  # first row must be the top line (no shape)
        if char == columns_sep
    ]

    # ------------------------------------------------------------------

    row_indices: list[int] = []  # find vertical indices (rows)

    for i, row in enumerate(table):
        if rows_sep in row:
            if has_shape:
                i -= 1

            row_indices.append(i)

    row_indices = [idx - i for i, idx in enumerate(row_indices)]

    # ------------------------------------------------------------------

    header_index = next(
        (  # find the index of the header row separator
            i for i, row in enumerate(table) if header_sep in row
        ),
        None,
    )

    return col_indices, row_indices, header_index


# -------------------- vertical/horizontal lines -----------------------


def _table_vertical_column_separator(
    table: str | list[str],
    positions: int | list[int],
    header_position: int | None = None,
    column_separator: tuple = ("┬", "│", "┴", "╪"),
) -> list[str]:
    if isinstance(table, str):
        table = table.split("\n")
    else:
        table = list(table)

    if not table or len(table) < 3:
        return table

    if isinstance(positions, int):
        positions = [positions]

    # Calculate the width of the table
    width_chars = max(len(row) for row in table)

    # Normalize positions: positive from the start, negative from the end
    normalized_positions = [
        p if p >= 0 else width_chars + p for p in positions
    ]

    # Sort and remove duplicates
    normalized_positions = sorted(set(normalized_positions))

    table_indices = {
        i for i, row in enumerate(table) if len(row) == width_chars
    }
    min_idx, max_idx = min(table_indices), max(table_indices)

    for i, row in enumerate(table):
        if i > max_idx or i < min_idx:
            continue  # shape row
        elif i == min_idx:
            sep = column_separator[0]
        elif i == max_idx:
            sep = column_separator[2]
        else:
            sep = column_separator[1]

        for pos in normalized_positions:
            if 0 <= pos < len(row):
                # Use ╪ for header row
                if i == header_position:
                    sep_char = column_separator[3]
                else:
                    sep_char = sep

                row = row[:pos] + sep_char + row[pos + 1 :]

        table[i] = row

    return table


def _table_horizontal_row_separator(
    table: str | list[str],
    positions: int | list[int],
    col_positions: int | list[int] | None = None,
    row_separator: tuple = ("├", "─", "┼", "┤"),
) -> list[str]:
    if isinstance(table, str):
        table_list: list[str] = table.split("\n")
    else:
        table_list = list(table)

    if not table_list or len(table_list) < 2:
        return table_list

    table_list, width_chars, shape_row, shape_row_index = (
        _handle_shape_row(table_list)
    )

    if isinstance(positions, int):
        positions = [positions]

    if col_positions is None:
        col_positions = []
    elif isinstance(col_positions, int):
        col_positions = [col_positions]

    col_positions = sorted(set(col_positions))

    sep_line_parts = [
        row_separator[1] if i not in col_positions else row_separator[2]
        for i in range(1, width_chars - 1)
    ]

    sep_line = (
        row_separator[0] + "".join(sep_line_parts) + row_separator[3]
    )

    normalized_positions = sorted(
        {p if p >= 0 else len(table_list) + p for p in positions}
    )

    for i, pos in enumerate(normalized_positions):
        adjusted_pos = pos + i
        if 0 <= adjusted_pos < len(table_list) + i:
            table_list.insert(adjusted_pos, sep_line)

    return _reinsert_shape_row(table_list, shape_row, shape_row_index)


def _handle_shape_row(
    table: list[str],
) -> tuple[list[str], int, str | None, int | None]:
    width_chars = max(len(row) for row in table)
    shape_row = None
    shape_row_index = None

    if len(table[0]) != width_chars:
        shape_row = table.pop(0)
        shape_row_index = 0

    elif len(table[-1]) != width_chars:
        shape_row = table.pop()
        shape_row_index = -1

    return table, width_chars, shape_row, shape_row_index


def _reinsert_shape_row(
    table: list[str],
    shape_row: str | None,
    shape_row_index: int | None,
) -> list[str]:
    if shape_row is not None:
        if shape_row_index == 0:
            table.insert(shape_row_index, shape_row)
        else:
            table.append(shape_row)

    return table


# def _table_vertical_column_separator(
#         table: str | list[str],
#         positions: int | list[int],
#         column_separator: tuple = ("┬", "│", "┴")
# ) -> list[str]:
#     if isinstance(table, str):
#         table = table.split("\n")
#     else:
#         table = list(table)
#
#     if not table or len(table) < 3:
#         return table
#
#     if isinstance(positions, int):
#         positions = [positions]
#
#     # Calculate the width of the table
#     width_chars = max(len(row) for row in table)
#
#     # Normalize positions: positive from the start, negative from the end
#     normalized_positions = [p if p >= 0 else width_chars + p for p in positions]
#
#     # Sort and remove duplicates
#     normalized_positions = sorted(set(normalized_positions))
#
#     table_indices = {i for i, row in enumerate(table) if len(row) == width_chars}
#     min_idx, max_idx = min(table_indices), max(table_indices)
#
#     for i, row in enumerate(table):
#         # if len(row) != width_chars:
#         #     continue  # Skip non-standard rows
#         if i > max_idx or i < min_idx:
#             continue  # shape row
#         elif i == min_idx:
#             sep = column_separator[0]
#         elif i == max_idx:
#             sep = column_separator[2]
#         else:
#             sep = column_separator[1]
#
#         for pos in normalized_positions:
#             if 0 <= pos < len(row):
#                 row = row[:pos] + sep + row[pos + 1:]
#
#         table[i] = row
#
#     return table


# def _table_horizontal_row_separator(
#         table: str | list[str],
#         positions: int | list[int],
#         row_separator: tuple = ("├", "─", "┤")
# ):
#     if isinstance(table, str):
#         table = table.split("\n")
#     else:
#         table = list(table)
#
#     if not table or len(table) < 2:
#         return table
#
#     table, width_chars, shape_row, shape_row_index = _handle_shape_row(table)
#
#     if isinstance(positions, int):
#         positions = [positions]
#
#     sep_line = row_separator[0] + row_separator[1] * (width_chars - 2) + row_separator[2]
#     normalized_positions = sorted({p if p >= 0 else len(table) + p for p in positions})
#
#     for i, pos in enumerate(normalized_positions):
#         adjusted_pos = pos + i
#         if 0 <= adjusted_pos < len(table) + i:
#             table.insert(adjusted_pos, sep_line)
#
#     return _reinsert_shape_row(table, shape_row, shape_row_index)
#
#
# def _handle_shape_row(table):
#     width_chars = max(len(row) for row in table)
#     shape_row = None
#     shape_row_index = None
#
#     if len(table[0]) != width_chars:
#         shape_row = table.pop(0)
#         shape_row_index = 0
#     elif len(table[-1]) != width_chars:
#         shape_row = table.pop()
#         shape_row_index = len(table)
#
#     return table, width_chars, shape_row, shape_row_index
#
#
# def _reinsert_shape_row(table, shape_row, shape_row_index):
#     if shape_row is not None:
#         table.insert(shape_row_index, shape_row)
#     return table
