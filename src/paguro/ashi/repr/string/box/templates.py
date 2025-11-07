from __future__ import annotations

from collections.abc import Callable

from paguro.ashi.repr.string.box.box import Box

__all__ = [
    "COLLECTION_BOXES",
]


def table_box(group_columns: list[str]) -> Box:
    separators: list[int] | int = 0
    n_groups_columns = len(group_columns)

    if n_groups_columns:
        # first columns are groups then the id_column (easier readibility)
        separators = [n_groups_columns - 1, n_groups_columns]

    box = (
        Box()
        .set_pl_tbl_config(
            # tbl_rows=100,
            tbl_formatting="UTF8_BORDERS_ONLY",  # check UTF8_NO_BORDERS
            tbl_hide_column_data_types=True,
            tbl_hide_dataframe_shape=True,
        )
        .set_align_content("center-ind")
        .set_inner_boxes(
            Box("double").set_dict_num_columns(2), Box("heavy")
        )
        .set_dict_nested_levels(2)
        .set_dict_num_columns(1)
        .set_pl_col_separator(separators)
    )
    return box


def collection_box() -> Box:
    box = (
        Box()
        .set_dict_nested_levels(0)
        .set_dict_positioning("left")
        .set_align_content("center")
        .set_pl_tbl_config(
            tbl_formatting="UTF8_BORDERS_ONLY",
            tbl_hide_dataframe_shape=True,
            tbl_hide_column_data_types=True,
        )
    )
    return box


def collection_skim_box(
    display_rounding: int = 2, by: list[str] | None = None
) -> Box:
    col_separators = [0]

    if by is not None:
        col_separators = [
            len(by) - 1,
            len(by),
        ]  # by columns are before the column
        # col_separators.append(len(by)) # by columns are after the column

    box = (
        collection_box()
        .set_pl_col_separator(col_separators)
        ._set_rounding_for_frame_string_casting(
            rounding_for_frame_string_casting=display_rounding
        )
    )

    return box


def collection_tabulate_box(
    *,
    margin_sum: bool,
    output_names: list[str],
    group_by: bool | str,
) -> Box:
    box = collection_box()

    box = box.set_pl_col_separator([0])

    if len(output_names) >= 2:
        if margin_sum:  # add vertical and horizontal lines
            box = box.set_pl_col_separator([0, -1]).set_pl_row_separator(
                -1
            )

        if not isinstance(group_by, bool):
            box = (
                box
                # left_title, top_title
                .set_pl_titles(titles=(output_names[1], output_names[0]))
            )
    return box


def tabulate_box():
    box = (
        Box(box="horizontals_double")
        .set_align_content("center")
        .set_indent_content(0, 0, 0, 0)
    )
    return box


COLLECTION_BOXES: dict[str | None, Callable[..., Box] | None] = {
    None: collection_box,
    "tabulate": collection_tabulate_box,
    "skim": collection_skim_box,
}
