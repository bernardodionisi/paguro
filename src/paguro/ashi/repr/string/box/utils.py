from __future__ import annotations

from typing import Literal, cast

from paguro.ashi.repr.string.styled.styled_str import StStr
from paguro.ashi.repr.string.utils import join_ststr
from paguro.utils.config import should_style


def boxed_str(
        text: str | StStr | list[str | StStr],
        top_name: str | None,
        top_name_align: Literal["left", "center", "right"],
        bottom_name: str | None,
        bottom_name_align: Literal["left", "center", "right"],
        width_chars: int,
        align: Literal["left", "center", "center-ind", "right"] = "left",
        box: str | None = None,
        style: dict[str, str] | None = None,
) -> list[str | StStr]:
    if box is None:
        box = "   \n   \n   \n"

    if style is not None and should_style():  # defaulting to style
        box = StStr(box).set_style(**style)  # type: ignore[arg-type]

    line_1, line_2, line_3 = box.splitlines()

    top_left_corner, top_line, top_right_corner = iter(line_1)
    left_border, _, right_border = iter(line_2)
    bottom_left_corner, bottom_line, bottom_right_corner = iter(line_3)

    out: list[str | StStr] = []

    out.extend(
        _box_top(
            top_name=top_name,
            top_name_align=top_name_align,
            width_chars=width_chars,
            top_line=top_line,
            top_left_corner=top_left_corner,
            top_right_corner=top_right_corner,
            left_border=left_border,
            right_border=right_border,
        )
    )

    out.extend(
        _box_center(
            text=text,
            width_chars=width_chars,
            left_border=left_border,
            right_border=right_border,
            align=align,
        )
    )

    out.extend(
        _box_bottom(
            bottom_name=bottom_name,
            width_chars=width_chars,
            bottom_name_align=bottom_name_align,
            bottom_line=bottom_line,
            bottom_left_corner=bottom_left_corner,
            bottom_right_corner=bottom_right_corner,
            left_border=left_border,
            right_border=right_border,
        )
    )

    # warning: the next two line would mess up the width!
    # if any(isinstance(i, StStr) for i in out):
    #     out = [str(i) for i in out]

    return out


def _box_top(
        top_name: str | None,
        top_name_align: Literal["left", "center", "right"],
        width_chars: int,
        *,
        top_line: str,
        top_left_corner: str,
        top_right_corner: str,
        left_border: str,
        right_border: str,
) -> list[str]:
    width_chars_short = width_chars - 2  # remove the border
    # (assuming here its 1 char each side, could be " ")

    top_line *= width_chars_short

    top_name_list: list[str | StStr] = center_strings(
        text=top_name,
        left_margin=1,
        right_margin=1,
    )

    out = []

    top = top_left_corner + top_line + top_right_corner

    if top_name_list:
        top = center_replace(
            base_string=top,
            insert_string=top_name_list[0],
            insert_string_align=top_name_align,
        )

    out.append(top)

    other_lines = left_border + " " * width_chars_short + right_border
    for n in top_name_list[1:]:
        out.append(
            center_replace(
                base_string=other_lines,
                insert_string=n,
                insert_string_align=top_name_align,
            )
        )

    # if len(out[0]) != width_chars:
    #     raise ValueError("Something went wrong in formatting the top of the box")

    return out


def _box_center(
        text: str | StStr | list[str | StStr],
        width_chars: int,
        align: Literal["left", "center", "center-ind", "right"] = "left",
        *,
        left_border: str,
        right_border: str,
) -> list[str]:
    width_chars_short = width_chars - 2  # remove the border

    if isinstance(text, (str, StStr)):
        # text: list[str | StStr] = text.split("\n")
        text = cast("list[str | StStr]", text.split("\n"))

    if align == "center" or align == "right":
        # TODO: FIX: this is not useful if we ar joining, and a single block needs to be centered
        # maybe make the argument centered-independent (each row is centered?
        text = ljust_strings(
            text=text
        )  # make sure all the strings are the same length

    out = []
    for t in text:
        if isinstance(
                t, str
        ):  # TODO: remove this check, only useful for unicode 🔥
            if len(t) > width_chars_short:
                # pass
                _max_len_ = max(
                    len(t) for t in text
                )  # so that we prompt only once
                raise ValueError(
                    f"Line: '{t}' is too long. It is {_max_len_} characters, while it should be {width_chars_short}\n"
                    # if still fails also after adjusting width_chars it's a problem in the string
                    f"Try adjusting width_chars by:\n\nimport paguro as pg\npg.Config.set_width_chars({_max_len_ + 2})"
                )

        if align.startswith("center"):
            t = t.center(width_chars_short)
        elif align == "right":
            t = t.rjust(width_chars_short)
        else:
            t = t.ljust(width_chars_short)

        if isinstance(t, str):
            line = f"{left_border}{t}{right_border}"

        else:  # styled str (keep different just to remember)
            line = left_border + t + right_border

        out.append(line)

    return out


def _box_bottom(
        bottom_name: str | None,
        width_chars: int,
        bottom_name_align: Literal["left", "center", "right"],
        *,
        bottom_line: str,
        bottom_left_corner: str,
        bottom_right_corner: str,
        left_border: str,
        right_border: str,
) -> list[str]:
    width_chars_short = width_chars - 2  # remove the border
    # (assuming here its 1 char each side, could be " ")

    bottom_line *= width_chars_short

    bottom_name_list: list[str | StStr] = center_strings(
        text=bottom_name,
        right_margin=1,
        left_margin=1,
    )

    out = []

    other_lines = left_border + " " * width_chars_short + right_border
    for n in bottom_name_list[:-1]:
        out.append(
            center_replace(
                base_string=other_lines,
                insert_string=n,
                insert_string_align=bottom_name_align,
            )
        )

    bottom = bottom_left_corner + bottom_line + bottom_right_corner

    if bottom_name_list:
        bottom = center_replace(
            base_string=bottom_left_corner
                        + bottom_line
                        + bottom_right_corner,
            insert_string=bottom_name_list[-1],
            insert_string_align=bottom_name_align,
        )

    out.append(bottom)

    return out


def center_strings(
        text: str | list[str] | None,
        left_margin: str | int = 0,
        right_margin: str | int = 0,
) -> list[str | StStr]:
    if not text:
        return []

    if isinstance(left_margin, int):
        left_margin = " " * left_margin

    if isinstance(right_margin, int):
        right_margin = " " * right_margin

    if isinstance(text, (str, StStr)):
        text = text.split("\n")

    max_len = max(len(i) for i in text)

    return [left_margin + i.center(max_len) + right_margin for i in text]


def ljust_strings(
        text: str | StStr | list[str | StStr],
        left_margin: str | int = 0,
        right_margin: str | int = 0,
) -> list[str | StStr]:
    if not text:
        return []

    if isinstance(text, (str, StStr)):
        text = cast("list[str | StStr]", text.split("\n"))

    if isinstance(left_margin, int):
        left_margin = " " * left_margin

    if isinstance(right_margin, int):
        right_margin = " " * right_margin

    max_len = max(len(i) for i in text)

    text = [left_margin + i.ljust(max_len) + right_margin for i in text]

    return text


def center_replace(
        base_string: str | StStr,
        insert_string: str | StStr,
        insert_string_align: Literal["center", "right", "left"],
) -> str:
    # check if the insert_string is longer than the base_string
    if len(insert_string) > len(base_string):
        insert_string = insert_string.strip()
        if len(insert_string) > len(base_string):
            raise ValueError(
                f"insert_string: '{insert_string}' should be shorter "
                f"than base_string: {base_string}"
            )

    # # calculate the start position
    # start_pos = (len(base_string) - len(insert_string)) // 2

    # if isinstance(insert_string, StStr):
    #     len_insert_string = insert_string.length(unicode=True)
    #     # display_len_insert_string = insert_string.length(unicode=False)
    # else:
    #     len_insert_string = len(insert_string)
    #     # display_len_insert_string = len(insert_string)

    len_insert_string = len(insert_string)

    # Calculate the start position based on the alignment within the effective length
    if insert_string_align == "center":
        start_pos = (len(base_string) - len_insert_string) // 2
    elif insert_string_align == "left":
        start_pos = 2
    elif insert_string_align == "right":
        start_pos = len(base_string) - 2 - len_insert_string
    else:
        raise ValueError("Alignment must be 'center', 'right', or 'left'")

    new_string = (
            base_string[:start_pos]
            + insert_string
            + base_string[start_pos + len_insert_string:]
    )

    return new_string


# ----------------------------------------------------------------------


def _concatenate_strings(
        strings: list[str | StStr],
        num_columns: int = 1,
) -> str | StStr:
    return concatenate_strings(
        strings=strings,
        num_columns=num_columns,
    )


def concatenate_strings(
        strings: list[str | StStr],
        num_columns: int = 1,
) -> str | StStr:
    # TODO: make sure it works with StyledStr
    if num_columns == 1:
        out = join_ststr(str_list=strings, separator="\n")

        return out

    strings = process_strings(strings=strings, n=num_columns)
    # Convert each string representation into lists of lines
    lists_of_lines = [s.split("\n") for s in strings]

    # Get the maximum width for each string representation, considering the minimum width
    max_widths = [
        max(len(line) for line in lines) for lines in lists_of_lines
    ]

    # Ensure each line in each string representation is of the correct width (by padding)
    for i, lines in enumerate(lists_of_lines):
        for j, line in enumerate(lines):
            lists_of_lines[i][j] = line.ljust(max_widths[i])

    # Get the maximum number of lines among all string representations
    max_lines = max(len(lines) for lines in lists_of_lines)

    # Ensure all lists of lines are of equal length (fill with empty lines if necessary)
    for lines in lists_of_lines:
        while len(lines) < max_lines:
            lines.append(" " * max_widths[lists_of_lines.index(lines)])

    # Join lines side by side
    joined_lines = []
    for i in range(max_lines):
        # line = "".join(lines[i] for lines in lists_of_lines)
        line = join_ststr(
            [lines[i] for lines in lists_of_lines], separator=""
        )

        joined_lines.append(line)

    # return "\n".join(joined_lines)
    return join_ststr(joined_lines, separator="\n")


def process_strings(
        strings: list[str | StStr], n: int
) -> list[str | StStr]:
    """
    Processes a list of strings based on the provided number 'n'.
    - If n is 1, returns the list as it is.
    - If n is 2, joins strings at even indexes and odd indexes separately.
    - If n is 3, joins strings at indexes 0, 3, 6, ...; 1, 4, 7, ...; and so on.
    - If n is 4 or more, groups and joins strings based on their index modulo n.
    """
    if n == 1:
        return strings

    else:
        result: list[list[str | StStr]] = [[] for _ in range(n)]

        for i, string in enumerate(strings):
            result[i % n].append(string)

        # return ['\n'.join(group) for group in result]
        return [join_ststr(group, separator="\n") for group in result]

    # if n == 1:
    #     return strings
    # else:
    #     result = ['' for _ in range(n)]
    #     for i, string in enumerate(strings):
    #         result[i % n] += str(string) + '\n'
    #     # careful with the strip here. if we want to add newlines
    #     return [s.lstrip("\n") for s in result]
