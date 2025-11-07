from __future__ import annotations

import os
from typing import TYPE_CHECKING

from paguro.ashi import Box
from paguro.shared.frame_tree._repr.utils import calculate_width_with_depth

if TYPE_CHECKING:
    from paguro.ashi import StStr


def frame_tree_box(
    content: dict,
) -> str | StStr:
    width_chars = os.environ.get("ASHI_WIDTH_CHARS", None)
    if width_chars is None:
        _wc = calculate_width_with_depth(content)
    else:
        _wc = int(width_chars)
    return _frame_tree_box().to_string(content, width_chars=_wc)


def _frame_tree_box(
    # title: str
) -> Box:
    # TODO: condition if styling is possible
    #
    # if should_style():
    #     title = StStr(title).set_style(color=(173, 93, 93), bold=True, underline=True)
    #
    box = (
        Box("horizontals_double_top")
        # .set_top_name(title)
        .set_dict_positioning("left")
        .set_dict_nested_levels(2)
        .set_top_name_align("center")
        # .set_dict_style(style=_get_validation_style_config())
        # .set_box_style({"color": (92, 131, 116)})  # green
    )

    box_1 = (
        Box("rounded").set_top_name_align("left")
        # .set_box_style({"color": (103, 103, 132)})
    )

    box_2 = (
        Box("horizontal_top_ascii")
        # .set_indent_content(0, 0, 4, 0)
        .set_top_name_align("right")
        .set_pl_tbl_config(tbl_dataframe_shape_below=True)
        .set_dict_positioning("left")
        .set_key_equal_symbol("")
        # .set_pl_style({"color": (173, 93, 93)})  # red
        # .set_box_style({"color": (58, 90, 103)})  # blue
    )

    return box.set_inner_boxes(box_1, box_2)
