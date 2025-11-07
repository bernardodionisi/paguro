from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from paguro.ashi.repr.string.frames.frames import format_dicts_with_frames
from paguro.ashi.repr.string.styled.styled_str import StStr
from paguro.ashi.repr.string.utils import (
    _determine_indent,
    format_list_of_strings,
    join_ststr,
)


def determine_fmt_func(
    fmt_type: Literal[1, 2] | Callable = 1, *, to_string: bool = True
) -> Callable:
    if fmt_type == 1:
        func: Callable = (
            format_dict_v1_to_str if to_string else _format_dict_v1_to_list
        )
    elif fmt_type == 2:
        func = (
            format_dict_v2_to_str if to_string else _format_dict_v2_to_list
        )
    elif callable(fmt_type):
        func = fmt_type
    else:
        raise ValueError("no such version")

    return func


# ---------------------------- version 1 -------------------------------


def format_dict_v1_to_str(
    data: dict,
    width_chars: int = 80,
    *,
    indent: int | tuple = (0, 2, 4),
    equal_symbol: str = "",
    separators: tuple[int | list[int] | None, int | list[int] | None]
    | None = None,  # pl frame
    titles: tuple[str, str] | str | None = None,  # pl frame
    style: dict[str, str] | None = None,
    **set_polars_config,
) -> str | StStr:
    out = _format_dict_v1_to_list(
        data=data,
        width_chars=width_chars,
        indent=indent,
        equal_symbol=equal_symbol,
        separators=separators,
        titles=titles,
        style=style,
        **set_polars_config,
    )

    return join_ststr(out, separator="\n")


def _format_dict_v1_to_list(
    data: dict,
    width_chars: int,
    *,
    indent: int | tuple,
    equal_symbol: str,
    separators: tuple[int | list[int] | None, int | list[int] | None]
    | None = None,  # pl frame
    titles: tuple[str, str] | str | None = None,  # pl frame
    style: dict[str, str] | None = None,
    _current_level: int = 0,
    **set_polars_config,
) -> list[str]:
    (base_indent, key_indent, value_indent, right_padding) = (
        _determine_indent(indent=indent)
    )

    # ------------------------------------------------------------------

    data = format_dicts_with_frames(
        data=data,
        width_chars=width_chars,
        indent=(
            base_indent,
            key_indent,
            value_indent,
            right_padding,  # no right padding for frames?
        ),
        separators=separators,
        titles=titles,
        style=style,
        **set_polars_config,
    )

    # ------------------------------------------------------------------

    left_padding_str = " " * base_indent + " " * (
        key_indent * _current_level
    )

    lines = []
    for key, value in data.items():
        # fstring would return a string not styled
        # key_str = f"{left_padding_str}{key}{equal_symbol}"
        # key_str = left_padding_str + key + equal_symbol

        # TODO: be careful if it is styled string!!! dont str(key)
        if isinstance(key, StStr):
            pass
        else:
            key = str(key)  # TODO: watch it str(None)

        key_str = left_padding_str + key + equal_symbol

        if isinstance(value, dict):
            lines.append(key_str)

            sub_lines = _format_dict_v1_to_list(
                data=value,
                width_chars=width_chars,
                indent=indent,
                _current_level=_current_level + 1,
                equal_symbol=equal_symbol,
                separators=separators,
                titles=titles,
                style=style,
                **set_polars_config,  # should it be here?
            )

            lines.extend(sub_lines)

        else:
            lines.append(key_str)

            value_str = value

            if not isinstance(value, StStr):
                value_str = str(value_str)

            value_lines = format_list_of_strings(
                data=value_str,
                width_chars=width_chars,
                left_padding=len(left_padding_str) + value_indent,
                right_padding=right_padding,
            )

            lines.extend(value_lines)

    return lines


# --------------------------- version 2 --------------------------------


def format_dict_v2_to_str(
    data: dict,
    width_chars: int = 80,
    *,
    indent: int | tuple = (0, 2, 4),
    equal_symbol: str = "",
    separators: tuple[int | list[int] | None, int | list[int] | None]
    | None = None,  # pl frame
    titles: tuple[str, str] | str | None = None,  # pl frame
    style: dict[str, str] | None = None,
    **set_polars_config,
) -> str | StStr:
    out = _format_dict_v2_to_list(
        data=data,
        width_chars=width_chars,
        indent=indent,
        equal_symbol=equal_symbol,
        separators=separators,
        titles=titles,
        style=style,
        **set_polars_config,
    )

    return join_ststr(out, separator="\n")


def _format_dict_v2_to_list(
    data: dict,
    width_chars: int,
    *,
    indent: int | tuple,
    equal_symbol: str,
    separators: tuple[
        int | list[int] | None, int | list[int] | None
    ] | None,  # pl frame
    titles: tuple[str, str] | str | None = None,  # pl frame
    style: dict[str, str] | None = None,
    _current_level: int = 0,
    **set_polars_config,
) -> list[str]:
    (base_indent, key_indent, value_indent, right_padding) = (
        _determine_indent(indent=indent)
    )

    # ------------------------------------------------------------------

    data = format_dicts_with_frames(
        data=data,
        width_chars=width_chars,
        indent=(
            base_indent * 2,
            key_indent,
            value_indent,
            right_padding,  # no right padding for frames?
        ),
        separators=separators,
        titles=titles,
        style=style,
        **set_polars_config,
    )

    # ------------------------------------------------------------------

    left_padding_str = " " * base_indent + " " * (
        key_indent * _current_level
    )

    lines = []
    for key, value in data.items():
        # key_str = f"{key}{equal_symbol}"

        # key_str = key + equal_symbol

        key_str = key
        if not isinstance(key, StStr):
            key_str = str(key)

        key_str = key_str + equal_symbol

        if isinstance(value, dict):
            lines.append(left_padding_str + key_str)

            sub_lines = _format_dict_v2_to_list(
                data=value,
                width_chars=width_chars,
                indent=indent,
                _current_level=_current_level + 1,
                equal_symbol=equal_symbol,
                separators=separators,
                titles=titles,
                style=style,
                **set_polars_config,  # should it be here?
            )

            lines.extend(sub_lines)
        else:
            value_str = value
            if not isinstance(value, StStr):
                value_str = str(value_str)

            total_padding = (
                width_chars
                - len(left_padding_str)
                - len(key_str)
                - len(value_str)
                - right_padding
            )

            # line = (
            #         f"{left_padding_str}{key_str}"
            #         f"{' ' * total_padding}{value_str}"
            #         + " " * right_padding
            # )

            line = (
                left_padding_str
                + key_str
                + " " * total_padding
                + value_str
                + " " * right_padding
            )

            if len(line) <= width_chars:
                lines.append(line)

            else:
                # If it doesn't fit, then fall back to putting value on new line

                lines.append(left_padding_str + key_str)

                value_lines = format_list_of_strings(
                    data=value_str,
                    width_chars=width_chars,
                    left_padding=len(left_padding_str) + value_indent,
                    right_padding=right_padding,
                )

                lines.extend(value_lines)

    return lines
