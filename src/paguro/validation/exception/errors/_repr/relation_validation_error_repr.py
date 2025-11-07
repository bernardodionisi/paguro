from __future__ import annotations

from paguro.ashi import Box, StStr
from paguro.utils.config import should_style


def _get_validation_style_config():
    config = {
        "key_styles": {
            0: {
                # "color": (255, 255, 255),
                # "bold": True,
                # "inverted": False,
                "background": (103, 103, 132),
            },
            1: {
                # "color": (58, 90, 103),
                # "bold": False
            },
            # 2: {
            #     "color": (209, 139, 71),  # orange
            #     "italic": True,
            # },
        },
        # "value_styles": {
        #     # "errors": {"color": "green"},
        #     "expected": {
        #         "color": (173, 93, 93),  # red
        #         "italic": True,
        #     }
        # },
        # "key_affixes": {
        #     0: {
        #         "prefix": " >> ",
        #         "suffix": " << ",
        #         "start_level": 0,
        #         "apply_to_deeper_levels": False,
        #     }
        # },
    }
    return config


def _validation_error_box_collection(title: str) -> Box:
    # TODO: condition if styling is possible

    if should_style():
        title = StStr(title).set_style(color=(173, 93, 93))

    box = (
        Box("horizontals_double_top")
        .set_top_name(title)
        .set_dict_positioning("left")
        .set_dict_nested_levels(2)
        .set_top_name_align("center")
        .set_dict_style(style=_get_validation_style_config())
        .set_box_style({"color": (92, 131, 116)})  # green
    )
    box_1 = (
        Box("rounded")
        .set_top_name_align("left")
        .set_box_style({"color": (103, 103, 132)})
    )

    box_2 = (
        Box("horizontal_top_ascii")
        # .set_indent_content(0, 0, 4, 0)
        .set_top_name_align("right")
        .set_pl_tbl_config(tbl_dataframe_shape_below=True)
        .set_dict_positioning("left")
        .set_key_equal_symbol("")
        .set_pl_style({"color": (173, 93, 93)})  # red
        .set_box_style({"color": (58, 90, 103)})  # blue
    )

    return box.set_inner_boxes(box_1, box_2)


def _validation_error_box_multiple_frames(title: str) -> Box:
    # TODO: condition if styling is possible

    if should_style():
        title = StStr(title).set_style(color=(173, 93, 93))

    box = (
        Box("heavy")
        .set_top_name(title)
        .set_dict_positioning("left")
        .set_dict_nested_levels(3)
        .set_top_name_align("left")
        .set_dict_style(style=_get_validation_style_config())
        # .set_box_style({"color": (92, 131, 116)})  # green
    )

    box_1 = (
        Box("horizontals_double_top")
        .set_top_name_align("center")
        .set_box_style({"color": (92, 131, 116)})  # green
    )

    box_2 = (
        Box("rounded")
        .set_top_name_align("left")
        .set_box_style({"color": (103, 103, 132)})
    )

    box_3 = (
        Box("horizontal_top_ascii")
        # .set_indent_content(0, 0, 4, 0)
        .set_top_name_align("right")
        .set_pl_tbl_config(tbl_dataframe_shape_below=True)
        .set_dict_positioning("left")
        .set_key_equal_symbol("")
        .set_pl_style({"color": (173, 93, 93)})  # red
        .set_box_style({"color": (58, 90, 103)})  # blue
    )

    return box.set_inner_boxes(box_1, box_2, box_3)


#
#
# import paguro as pg
#
# pg.Config.set_styled(True)
#
# print(_validation_exception_box("CHECK").to_string({
#     "a": {
#         "b": {
#             "c": "this is a check",
#             "d": "sd"
#         }
#     }
# }))
#
# box = (
#     Box("horizontals_double_top")
#     .set_top_name("TOP TITLE")
#     .set_dict_positioning("left")
#     .set_dict_nested_levels(1)
#     .set_top_name_align("center")
#     .set_dict_style(style=_get_validation_style_config())
#     .set_box_style({"color": (92, 131, 116)})  # green
# )
#
# box_1 = (
#     Box("rounded")
#     # .set_top_name_align("left")
#     # .set_dict_nested_levels(1)
#     .set_box_style({"color": (103, 103, 132)})
# )
#
# box.set_inner_boxes(box_1)
#
# print(box.to_string({"a": {"b": {"c": "this is a check"}}}))
#
# box_2 = (
#     Box("horizontal_top_ascii")
#     # .set_indent_content(0, 0, 4, 0)
#     .set_top_name_align("right")
#     .set_pl_tbl_config(tbl_dataframe_shape_below=True)
#     .set_dict_positioning("left")
#     .set_key_equal_symbol("")
#     .set_pl_style({"color": (173, 93, 93)})  # red
#     .set_box_style({"color": (58, 90, 103)})  # blue
# )
