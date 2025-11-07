from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from paguro.ashi.repr.string.box.box import Box
from paguro.ashi.repr.string.styled.styled_str import StStr
from paguro.shared.frame_tree._repr.utils import calculate_width_with_depth
from paguro.utils.config import should_style
from paguro.validation.exception.utils.utils import (
    find_nested_keys_paths,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from paguro.ashi.typing import (
        AffixMapsLike,
        SimpleStyle,
        StyleMapsLike,
    )

COLOR_RED: tuple[int, int, int] = (173, 93, 93)
COLOR_GREEN: tuple[int, int, int] = (33, 145, 73)
COLOR_YELLOW: tuple[int, int, int] = (255, 206, 27)


def validation_exception_box(
        content: dict[str, Any],
) -> str | StStr:
    # out = []

    # --------
    width: str | None = os.environ.get("ASHI_WIDTH_CHARS", None)

    if width is None:
        width_chars: int = calculate_width_with_depth(content)
    else:
        width_chars = int(width)
    # --------

    # TODO:
    # content = replace_dict_keys(
    #     content,
    #     {
    #         "valid_column_list": "Column Errors",
    #         "valid_frame_list": "Frame Errors",
    #     }
    # )

    vcl_name = "valid_column_list"
    vfl_name = "valid_frame_list"
    styles = get_validation_styles(vcl_name=vcl_name, vfl_name=vfl_name)
    affixes = get_validation_affixes(
        content=content, vcl_name=vcl_name, vfl_name=vfl_name
    )
    # for key, value in content.items():
    #     # TODO: add number of errors under key (title)
    #
    #     out.append(
    #         _validation_exception_box(
    #             # title=key,
    #             styles=styles,
    #             affixes=affixes
    #         )
    #         # .set_width_chars(width_chars)
    #         .to_string(content={key: value}, width_chars=width_chars)
    #     )
    #
    # return join_ststr(out, separator="\n")
    return (
        _validation_exception_box(
            styles=styles, affixes=affixes, title="ValidationError"
        )
        # .set_width_chars(width_chars)
        .to_string(content=content, width_chars=width_chars)
    )


# -------------------------------- styled ------------------------------


def _validation_exception_box(
        styles: dict | None, affixes: dict | None, title: str | None = None
) -> Box:
    # TODO: condition if styling is possible

    if title is not None and should_style():
        title = StStr(title).set_style(
            color=COLOR_RED, bold=True, underline=False
        )
    box = (
        Box("horizontals_double_tb")
        .set_top_name(title)
        .set_dict_positioning("left")
        .set_dict_nested_levels(3)
        .set_top_name_align("left")
    )
    if affixes is not None:
        box = box.set_dict_affixes(affixes=affixes)
    if styles is not None:
        box = box.set_dict_style(style=styles).set_box_style(
            {"color": COLOR_RED}
        )

    box_1 = (
        Box("horizontals_heavy_top")
        # .set_top_name(title)
        .set_dict_positioning("left")
        .set_dict_nested_levels(2)
        .set_top_name_align("center")
    )
    if styles is not None:
        box_1 = (
            box_1
            .set_box_style({"color": (58, 90, 103)})
        )

    box_2 = (
        Box("rounded")
        .set_dict_nested_levels(1)
        .set_top_name_align("left")
    )

    if styles is not None:
        box_2 = box_2.set_box_style({"color": (103, 103, 132)})

    box_3 = (
        Box("horizontal_top_ascii")
        # .set_indent_content(0, 0, 4, 0)
        .set_top_name_align("right")
        .set_pl_tbl_config(tbl_dataframe_shape_below=True)
        .set_dict_positioning("left")
        .set_key_equal_symbol("")
    )
    if styles is not None:
        box_3 = (
            (
                box_3
                .set_pl_style({"color": COLOR_RED})
                .set_box_style({"color": (58, 90, 103)}
                               ))
        )

    return box.set_inner_boxes(box_1, box_2, box_3)


def _set_key_styling_dict(
        keys: Iterable[str | int | tuple[str, ...]], style: SimpleStyle | dict
) -> dict:
    out: dict = {}
    for k in keys:
        out[k] = style
    return out


def _validation_errors_key_styling(
        vcl_name: str,
        vfl_name: str,
) -> dict[str, dict]:
    out: dict[str, dict] = {"key": {}}

    # valid_column_names = find_nested_keys_dict(content, "valid_column_list")
    # valid_frame_names = find_nested_keys_dict(content, "valid_frame_list")
    # constraint_names = find_nested_keys_dict(content, "constraints")

    out["key"].update(
        _set_key_styling_dict(
            keys={
                "dtype",
                "required",
                "allow_nulls",
                "unique",
                "fields",
                "constraints",
                "validators",
                "columns_policy",
            },
            style={
                "color": (209, 139, 71),  # orange
                "italic": True,
            },
        )
    )

    out["key"].update(
        {
            vcl_name: {
                "bold": True,
                "inverted": False,
                "underline": True,
            },

            vfl_name: {
                "bold": True,
                "inverted": False,
                "underline": True,
            },

            # columns policy

            "missing": {
                "color": (148, 143, 142),
                "italic": True,
            },

            "extra": {
                "color": (148, 143, 142),
                "italic": True,
            },

            "order": {
                "color": (148, 143, 142),
                "italic": True,
            },

            # --------------------------

            "transform": {
                "bold": True,
                "inverted": True,
            },

            # ------- status keys ------
            "maybe_errors": {
                "color": COLOR_YELLOW,
                "italic": False,
            },

            "no_errors": {
                "color": COLOR_GREEN,
                "italic": False,
            },

            "errors": {
                "color": COLOR_RED,
                "italic": False,
            },

            "errors_limited": {
                "color": COLOR_RED,
                "italic": False,
            },

            "errors_count": {
                "color": COLOR_RED,
                "italic": False,
            },

            "exception": {
                "color": COLOR_RED,
                "italic": False,
            },

            # depth selectors (root=0, children=1, etc.)
            # 0: {
            #     "color": (255, 255, 255),
            #     "bold": True,
            #     "inverted": False,
            #     "background": (103, 103, 132),
            # },
            # 1: {
            #     "color": (58, 90, 103),
            #     "bold": False,
            # },
        }
    )
    return out


def _validation_errors_value_styling():
    out = {"value": {}}

    out["value"].update(
        {
            # status keys
            "maybe_errors": {"color": COLOR_YELLOW},
            "no_errors": {"color": COLOR_GREEN},
            "errors": {"color": COLOR_RED},
            "errors_count": {"color": COLOR_RED},
            "errors_limited": {"color": COLOR_RED},
            "exception": {"color": COLOR_RED},
        }
    )
    return out


def get_validation_styles(
        vcl_name: str,
        vfl_name: str,
) -> StyleMapsLike:
    """
    get_validation_styles.

    New API:
      styles = {
        "key": {...depth/int, key/str, path/tuple -> style dict...},
        "value": {...},
        "default_key": {...},      # optional
        "default_value": {...},    # optional
      }
    """
    return {
        **_validation_errors_key_styling(
            vcl_name=vcl_name, vfl_name=vfl_name
        ),
        **_validation_errors_value_styling(),
        # optional top-level defaults (kept empty here)
        "default_key": {},
        "default_value": {},
    }


# ----------------------------------------------------------------------


def _validation_errors_key_affixes(
        content: dict,
        vcl_name: str,
        vfl_name: str,
):
    # using get just because we are rendering separately for now.
    vcl_sub_keys = find_nested_keys_paths(content, vcl_name)
    vfl_sub_keys = find_nested_keys_paths(content, vfl_name)
    constraint_sub_keys = find_nested_keys_paths(content, "constraints")

    out: dict = {"key": {}}

    # out["key"].update(
    #     {
    #         0: {
    #             "prefix": " >> ",
    #             "suffix": " << ",
    #             "start_level": 0,
    #             "apply_to_deeper_levels": False,
    #         },
    #     }
    # )

    if vcl_sub_keys:
        for i in vcl_sub_keys:
            if isinstance(i, tuple):
                start_level = len(i) - 1
            else:
                start_level = 0

            out["key"][i] = {
                "prefix": "* '",
                "suffix": "'",
                "start_level": start_level,
                "apply_to_deeper_levels": False,
            }

    if vfl_sub_keys:
        for i in vfl_sub_keys:
            if isinstance(i, tuple):
                start_level = len(i) - 1
            else:
                start_level = 0

            out["key"][i] = {
                "prefix": '> "',
                "suffix": '"',
                "start_level": start_level,
                "apply_to_deeper_levels": False,
            }

    if constraint_sub_keys:
        for i in constraint_sub_keys:
            if isinstance(i, tuple):
                start_level = len(i) - 1
            else:
                start_level = 0

            out["key"][i] = {
                "prefix": "‣ ",
                # "suffix": '',
                "start_level": start_level,
                "apply_to_deeper_levels": False,
            }

    return out


def get_validation_affixes(
        content: dict,
        vcl_name: str,
        vfl_name: str,
) -> AffixMapsLike:
    """
    Get affixes.

    New API:
      affixes = {
        "key": {...depth/int, key/str, path/tuple -> affix config...}
      }
    """
    return {
        **_validation_errors_key_affixes(
            content=content, vcl_name=vcl_name, vfl_name=vfl_name
        ),
    }
