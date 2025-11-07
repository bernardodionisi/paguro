from __future__ import annotations

import os
from typing import Any

import polars as pl

from paguro.ashi.repr.string.frames.separators import (
    get_separators_indices,
    insert_table_separators,
)
from paguro.ashi.repr.string.frames.titles import add_title_to_table
from paguro.ashi.repr.string.styled.styled_str import StStr
from paguro.ashi.repr.string.utils import (
    _determine_indent,
    format_list_of_strings,
    join_ststr,
)
from paguro.utils.config import should_style

# KEEP AN EYE ON (RFC: reworked Config): https://github.com/pola-rs/polars/issues/12252


def format_dicts_with_frames(
    data: dict[str, pl.DataFrame | pl.LazyFrame | Any],
    width_chars: int,
    *,
    indent: int | tuple,
    separators: tuple[int | list[int] | None, int | list[int] | None]
    | None = None,
    titles: tuple[str, str] | str | None = None,
    style: dict[str, str] | None = None,
    **set_polars_config,
) -> dict:
    base_indent, _, value_indent, right_padding = _determine_indent(
        indent=indent
    )

    # data should just be at first level of nestedness.

    width_dataframe = width_chars - (
        base_indent + value_indent + right_padding
    )

    out = {}
    for key, value in data.items():
        if isinstance(value, pl.DataFrame):
            out[key] = format_dataframe_repr(
                data=value,
                width_chars=width_dataframe,
                separators=separators,
                titles=titles,
                style=style,
                **set_polars_config,
            )
        elif isinstance(value, pl.LazyFrame):
            out[key] = value.__repr__()
        else:
            out[key] = value

    return out


def format_frames(  # not dict of frames
    data: list[pl.DataFrame | pl.LazyFrame] | pl.DataFrame | pl.LazyFrame,
    width_chars: int,
    *,
    indent: int | tuple,
    center: bool = True,
    separators: tuple[int | list[int] | None, int | list[int] | None]
    | None = None,
    titles: tuple[str, str] | str | None = None,
    style: dict[str, str] | None = None,
    **set_polars_config,
) -> str | StStr:
    # TODO: indent here is used only to find the width, it does not actually indent
    base_indent, _, value_indent, right_padding = _determine_indent(
        indent=indent
    )

    # data should just be at first level of nestedness.

    width_dataframe = width_chars - (
        base_indent + value_indent + right_padding
    )

    if not isinstance(data, list):
        data = [data]  # careful here !!

    out = []
    for d in data:
        if isinstance(d, pl.DataFrame):
            data_repr_ = format_dataframe_repr(
                data=d,
                width_chars=width_dataframe,
                separators=separators,
                titles=titles,
                **set_polars_config,
            )

        elif isinstance(d, pl.LazyFrame):
            data_repr_ = d.__repr__()

            _temp = format_list_of_strings(
                data=data_repr_,
                width_chars=width_chars,
                left_padding=base_indent,
                right_padding=right_padding,
                to_string=True,
            )

            data_repr_ = f"\n{_temp}"

        else:
            data_repr_ = d.__repr__()  # careful??

        data_repr_ = f"{data_repr_}\n"

        if center:
            data_repr_ = "\n".join(
                [
                    i.strip().center(
                        # TODO: this is likely the cause of some wrong formatting if no borders
                        width_chars
                    )
                    for i in data_repr_.split("\n")
                ]
            )

        if style is not None:
            data_repr_ = StStr(data_repr_).set_style(**style)  # type: ignore[arg-type]

        out.append(data_repr_)

    # -------------------------- pad frame -----------------------------

    lines: list[str | StStr] = []

    for i in out:
        line = format_list_of_strings(
            data=i,
            width_chars=width_chars,
            left_padding=base_indent,
            right_padding=right_padding,
        )
        lines.extend(line)

    return join_ststr(lines, separator="\n")


def format_dataframe_repr(
    data: pl.DataFrame,
    width_chars: int | None = None,
    separators: tuple[int | list[int] | None, int | list[int] | None]
    | None = None,
    titles: tuple[str, str] | str | None = None,
    style: dict[str, str] | None = None,  # TODO: improve typing
    **polars_tbl_config,
) -> str | StStr:
    polars_tbl_config = adjust_width_polars_tbl_config(
        data=data, width_chars=width_chars, **polars_tbl_config
    )

    # if isinstance(titles, str):
    #     titles = [titles]
    #
    # if titles is not None:
    #     column_titles = [i for i in titles if i in data.columns]
    #
    #     if column_titles:
    #         data = data.rename({column_titles[0]: ""})  # only one column

    with pl.Config(**polars_tbl_config):
        table: str = data.__repr__()

    if separators is not None:
        col_positions, row_positions, header_position = (
            get_separators_indices(
                data=data, separators=separators, **polars_tbl_config
            )
        )

        # add vertical/horizontal line to the representation
        table = insert_table_separators(
            table=table,
            col_positions=col_positions,
            row_positions=row_positions,
            header_position=header_position,
        )

    if titles is not None:
        # TODO: process better to allow just left title
        if isinstance(titles, str):
            top_title, left_title = titles, None
        else:
            top_title, left_title = titles

        table = add_title_to_table(
            table=table,
            title=top_title,
            on_side=False,
            position="top",
            alignment="right",
            max_width=width_chars,
        )
        if left_title is not None:
            table = add_title_to_table(
                table=table,
                title=left_title,
                on_side=True,
                position="center",
                alignment="center",
                max_width=width_chars,
            )

    if style is not None and should_style():
        table = StStr(table).set_style(**style)  # type: ignore[arg-type]

    return table


def adjust_width_polars_tbl_config(
    data: pl.DataFrame, width_chars: int | None = None, **polars_tbl_config
) -> dict:
    """Reduce the width of the displayed dataframe to width_chars"""
    if width_chars is None:  # if no pre-specified width
        return polars_tbl_config

    with pl.Config(**polars_tbl_config):
        # if the width has been specified, but there are many displayed columns (from tbl_cols):
        # polars will still display all the columns (maybe this will change)

        with pl.Config(
            tbl_width_chars=width_chars,
            ascii_tables=True,  # using ascii_tables to find out how many columns are displayed
            tbl_hide_dataframe_shape=True,
        ):
            data_repr_ = data.__repr__()

            # this is very hacky, but the user defined format may not include a top line
            # this approach may break, oh well, if it breaks, the repr may not look as desired

            # when 3 columns the top bar should be a string as +---+---+---+

            rows = [
                line
                for line in data_repr_.split("\n")
                if line.startswith("+")
            ]

            if rows:
                top_line = rows[0]  # 0 because here we only kept the +
                # n_displayed_columns = top_line.count("+") - 1
                n_displayed_columns = top_line.count("+") - 2

        if not rows:
            return polars_tbl_config

        polars_tbl_config["tbl_width_chars"] = width_chars

        # --------------------------------------------------------------

        POLARS_FMT_MAX_COLS = os.environ.get("POLARS_FMT_MAX_COLS", None)

        if len(top_line) <= width_chars and POLARS_FMT_MAX_COLS is None:
            new_displayed_cols = n_displayed_columns

            while len(
                top_line
            ) <= width_chars and new_displayed_cols < len(data.columns):
                new_displayed_cols += (
                    1  # increase the number of columns to fit the width
                )

                with pl.Config(
                    tbl_width_chars=width_chars,
                    tbl_cols=new_displayed_cols,
                ):
                    data_repr_ = data.__repr__()

                top_line = data_repr_.split("\n")[1]

                if len(top_line) <= width_chars:
                    n_displayed_columns = new_displayed_cols

        else:
            while len(top_line) > width_chars:
                n_displayed_columns -= (
                    1  # reduce the number of column to fit the width
                )

                with pl.Config(
                    tbl_width_chars=width_chars,
                    tbl_cols=n_displayed_columns,
                ):
                    data_repr_ = data.__repr__()

                # actual_frame_repr_width = max(len(i) for i in data_repr_.split("\n"))
                top_line = data_repr_.split("\n")[1]

        polars_tbl_config.update({"tbl_cols": n_displayed_columns})

        return polars_tbl_config
