from __future__ import annotations

import textwrap
from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

from paguro.ashi.repr.string.styled.styled_str import StStr

if TYPE_CHECKING:
    from collections.abc import Iterable


def join_ststr(
    str_list: Iterable[str | StStr], separator: str = "\n"
) -> str | StStr:
    # if isinstance(str_list, (str, StStr)):
    #     return str_list

    if any(isinstance(i, StStr) for i in str_list):
        return StStr(separator).join(str_list)
    else:
        return separator.join(cast("Iterable[str]", str_list))


def format_list_of_strings(
    data: str | list[str],
    *,
    width_chars: int,
    left_padding: int,
    right_padding: int = 0,
    to_string: bool = False,
) -> str | StStr | list:
    """
    Formats a string or list of strings with specified padding on each line.

    The function processes each line in the input string or list of strings,
    adjusts its width according to the specified character count, and applies
    the desired left and right padding. The result can either be returned as
    a list of strings or a single concatenated string.

    Parameters
    ----------
    data : str or list[str]
        The string or list of strings to be formatted.

    width_chars : int
        The total width of each resulting line, in characters.

    left_padding : int
        The number of space characters to be added to the left side of each line.

    right_padding : int, optional
        The number of space characters to be added to the right side of each line.
        Default is 0.

    to_string : bool, optional
        If True, the result will be a single concatenated string.
        If False, the result will be a list of strings. Default is False.

    Returns
    -------
    str or list[str]
        A formatted string or list of strings with the desired width and padding.

    Notes
    -----
    The function uses the `split_text_with_padding` utility internally to handle
    the splitting and padding of each line. If the input `data` is a list of strings,
    it first concatenates the list with newline characters to process the data uniformly.
    """
    if isinstance(data, list):
        # joining because the strings in the list may also have new lines
        # data: str = "\n".join(data) # TODO: check
        data_str: str | StStr = join_ststr(str_list=data)
    else:
        data_str = data

    out: list = []
    for val_line in data_str.split("\n"):
        value_lines = text_split_with_padding(
            text=val_line,
            width_chars=width_chars,
            left_padding=left_padding,
            right_padding=right_padding,
        )
        out.extend(value_lines)

    if to_string:
        return join_ststr(str_list=out, separator="\n")

    return out


def text_split_with_padding(
    text: str,
    width_chars: int,
    left_padding: int,
    right_padding: int = 0,
) -> list[str | StStr]:
    """
    Splits a text based on width_chars and appends hyphens for word breaks.
    """
    width_chars_with_padding = width_chars - left_padding - right_padding

    if width_chars_with_padding <= 1:
        raise ValueError("width_chars_with_padding must be > 1")

    # Use textwrap to wrap the text
    elif isinstance(text, StStr):
        wrapped_lines = text.wrap(
            width=width_chars_with_padding,
            break_long_words=False,
            replace_whitespace=False,
        )
    else:
        wrapped_lines = textwrap.wrap(
            str(text),
            width=width_chars_with_padding,
            break_long_words=False,
            replace_whitespace=False,
        )

    # Manually handle long words that couldn't be wrapped by textwrap
    split_lines = []
    for line in wrapped_lines:
        while len(line) > width_chars_with_padding:
            break_point = (
                width_chars_with_padding - 1
            )  # Account for the hyphen
            split_lines.append(line[:break_point] + "-")
            line = line[break_point:]

        split_lines.append(line)

    # Apply padding
    padded_lines = [
        left_padding * " " + line + right_padding * " "
        for line in split_lines
    ]

    # Ensure each line is of width `width_chars`

    # TODO: if we pad here then we can't center
    # padded_lines = [line.ljust(width_chars) for line in padded_lines]

    return padded_lines


def line_with_centered_caption(
    width_chars: int,
    left_padding: int = 0,
    right_padding: int = 0,
    title: str | None = None,
    symbol: str = "-",
) -> str:
    """
    Creates a line of specified width with an optional centered title.

    The function generates a string of a given width, primarily composed of a specified
    symbol. Optionally, a title can be centered within this line. Additional padding can
    be added to the left or right side of the line separately.

    Parameters
    ----------
    width_chars : int
        The total width of the resulting line, in characters.

    left_padding : int, optional
        The number of space characters to be added to the left side of the line.
        Default is 0.

    right_padding : int, optional
        The number of space characters to be added to the right side of the line.
        Default is 0.

    title : str, optional
        The title to be centered within the line. If not provided, the line will
        consist only of the specified symbol. Default is None.

    symbol : str, optional
        The character used to create the line. Default is '-'.

    Returns
    -------
    str
        A string of specified width, with the given symbol, and an optional centered
        title.

    Notes
    -----
    If the total width of the line (including the title, symbols, and padding) exceeds
    `width_chars`, the line will be truncated to `width_chars`.

    If `title` is provided and its length combined with symbols exceeds `width_chars`,
    the title may be surrounded by fewer symbols than expected or may be partially
    truncated.
    """
    # Calculate the total width for the dashes considering the padding on both sides
    dash_length = width_chars - left_padding - right_padding

    if title:
        title = " " + title + " "

    # If title is provided, adjust the dash length to fit the title in the center
    if title:
        title_length = len(title)
        dash_side_length = (dash_length - title_length) // 2

        line = (
            symbol * dash_side_length
            + title
            + symbol * (dash_length - dash_side_length - title_length)
        )
    else:
        line = symbol * dash_length

    # Add padding
    line = " " * left_padding + line + " " * right_padding

    # Ensure the line doesn't exceed the max_length
    return line[:width_chars]


def join_strings(
    *strings, separator: str = "|", min_width: int = 0
) -> list[str | StStr]:
    """
    Join multiple strings vertically, side by side, separated by a given separator.

    The function takes care to pad strings with spaces such that the output is neat
    and the strings are vertically aligned. Strings can have multiple lines.

    Parameters
    ----------
    *strings : str
        Any number of input strings that need to be joined. Multiple lines in a string
        should be separated by newline characters.

    separator : str, optional
        The separator string used to join the input strings side by side.
        Default is '|'.

    min_width : int, optional
        Minimum width for each string. If the actual string width is smaller than this,
        it will be padded with spaces on the right. Default is 0.

    Returns
    -------
    list
        A list of strings with the input strings joined side by side, separated by the specified
        separator. Each line of the input strings is joined with the corresponding line
        of other strings. If a string has fewer lines than others, it will be padded
        with empty lines.

    Notes
    -----
    If the input strings have different number of lines, the output will be aligned
    based on the string with the maximum number of lines.

    The function ensures that each string block in the output has a width equal to the
    maximum width of its lines or the specified `min_width`, whichever is larger.
    """
    # Convert each string representation into lists of lines
    lists_of_lines = [s.split("\n") for s in strings]

    # Get the maximum width for each string representation, considering the minimum width
    max_widths = [
        max(len(line) for line in lines) for lines in lists_of_lines
    ]
    max_widths = [max(w, min_width) for w in max_widths]

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
        # line = separator.join(lines[i] for lines in lists_of_lines)
        line = join_ststr(
            [lines[i] for lines in lists_of_lines], separator=separator
        )

        joined_lines.append(line)

    return joined_lines


def _determine_indent(
    indent: int | tuple,
) -> tuple[int, int, int, int]:
    base_indent, key_indent, value_indent, right_padding = 0, 0, 0, 0

    if isinstance(indent, int):
        base_indent = indent

    elif isinstance(indent, tuple):
        if len(indent) == 1:
            base_indent = indent[0]

        elif len(indent) == 2:
            base_indent, key_indent = indent

        elif len(indent) == 3:
            base_indent, key_indent, value_indent = indent

        elif len(indent) == 4:
            base_indent, key_indent, value_indent, right_padding = indent

        else:
            raise ValueError("len of indent should max 4")

    return base_indent, key_indent, value_indent, right_padding
