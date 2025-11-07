from __future__ import annotations

from typing import Literal, cast

from paguro.ashi.repr.string.box.utils import _concatenate_strings


def add_title_to_table(
        table: list[str] | str,
        *,
        title: str,
        on_side: bool = False,
        position: Literal["top", "center", "bottom"] = "top",
        alignment: Literal["center", "left", "right"] = "center",
        max_width: int | None = None,
) -> str:
    if isinstance(table, str):
        # table = table.split("\n")
        table = cast(list[str], table.split("\n"))

    table_width_chars = max(len(line) for line in table)

    if max_width is not None:
        if on_side:
            # if the width
            if table_width_chars + len(title) > max_width:
                return "\n".join(table)
        else:
            if len(title) > max_width:
                return "\n".join(table)

    if not on_side:
        if position == "center":
            raise ValueError("position cannot be 'center' if not on_side")
        table = _add_title_to_table(
            table=table,
            title=title,
            table_width_chars=table_width_chars,
            position=position,  # center not a valid insertion if not on_side
            alignment=alignment,
        )
        return table

    else:  # if on_side:
        # add some padding
        title = f"{title} "

        title_list = _create_strings_with_insertion(
            insert_str=title,
            num_strings=len(table)
                        + 3,  # should fix this and exclude the headers
            position=position,
            alignment=alignment,
        )

        return _concatenate_strings(
            ["\n".join(title_list), "\n".join(table)],
            num_columns=2
        )

    # if isinstance(table, list):
    #     table = "\n".join(table)
    #
    # return table


def _add_title_to_table(
        table: list[str] | str,
        title: str,
        table_width_chars: int | None = None,
        position: Literal["top", "bottom"] = "top",
        alignment: Literal["center", "left", "right"] = "center",
) -> str:
    """Add title top or bottom"""
    if isinstance(table, str):
        table = table.split("\n")

    if table_width_chars is None:
        table_width_chars = max(len(line) for line in table)

    # format the title based on the specified alignment
    if alignment == "left":
        title = title.ljust(table_width_chars)
    elif alignment == "right":
        title = title.rjust(table_width_chars)
    else:  # alignment == 'center'
        title = title.center(table_width_chars)

    # TODO: do not split table into list

    # add the title to the specified position
    if position == "top":
        return title + "\n" + "\n".join(table)
    else:  # position == 'bottom'
        return title + "\n" + "\n".join(table)


def _create_strings_with_insertion(
        insert_str: str,
        num_strings: int,
        position: Literal["top", "bottom", "center"] = "center",
        alignment: Literal["center", "left", "right"] = "center",
) -> list[str]:
    """Crete a list of empty strings with some of them filled in with insert_str"""
    # Split the string to be inserted into lines and find the max length
    insert_lines = insert_str.split("\n")

    if num_strings < len(insert_lines):
        num_strings = len(insert_lines)

    max_line_length = max(len(line) for line in insert_lines)

    # Create a list of strings filled with spaces, each having a length of max_line_length
    space_filled_strings = [
        " " * max_line_length for _ in range(num_strings)
    ]

    # Calculate the start position for insertion
    if position == "top":
        start_pos = 0
    elif position == "bottom":
        start_pos = num_strings - len(insert_lines)
    else:  # 'center'
        start_pos = (num_strings - len(insert_lines)) // 2

    # Insert the lines with specified alignment
    for i, line in enumerate(insert_lines):
        if 0 <= start_pos + i < num_strings:
            space_filled_strings[start_pos + i] = _insert_with_alignment(
                original_string=space_filled_strings[start_pos + i],
                string_to_insert=line,
                alignment=alignment,
            )

    return space_filled_strings


def _insert_with_alignment(
        original_string: str,
        string_to_insert: str,
        alignment: Literal["center", "left", "right"] = "center",
) -> str:
    """
    Helper
    Inserts string_to_insert into original_string with the specified alignment.
    """
    original_length = len(original_string)
    insert_length = min(len(string_to_insert), original_length)

    if alignment == "left":
        return (
                string_to_insert[:insert_length]
                + original_string[insert_length:]
        )

    elif alignment == "right":
        return (
                original_string[:-insert_length]
                + string_to_insert[-insert_length:]
        )

    else:  # 'center'
        start = (original_length - insert_length) // 2
        end = start + insert_length
        return (
                original_string[:start]
                + string_to_insert[:insert_length]
                + original_string[end:]
        )
