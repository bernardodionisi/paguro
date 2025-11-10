from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

from paguro.ashi.repr.html.utility import html_repr_as_str
from paguro.ashi.repr.string.box.utils import (
    boxed_str,
    concatenate_strings,
)
from paguro.ashi.repr.string.dicts import (
    format_dict_v1_to_str,
    format_dict_v2_to_str,
)
from paguro.ashi.repr.string.frames.cast_to_string import (
    map_dataframes_to_string,
)
from paguro.ashi.repr.string.frames.frames import format_dataframe_repr
from paguro.ashi.repr.string.styled.styled_dict import (
    render_nested_structure,
)
from paguro.ashi.repr.string.styled.styled_str import StStr
from paguro.ashi.repr.string.utils import (
    _determine_indent,
    join_ststr,
    text_split_with_padding,
)
from paguro.utils.config import should_style

if TYPE_CHECKING:
    import sys
    from collections.abc import Iterable

    from paguro.ashi.typing import (
        AffixMapsLike,
        SimpleStyle,
        StyleMapsLike,
    )

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

    from paguro.ashi.info.info import Info
    from paguro.ashi.info.info_collection import InfoCollection


class Box:
    """
    Box representation

    Group
    -----
        representation_ns
    """

    def __init__(self, box: BoxesShapes = "rounded"):
        self.set_box_shape(box=box)
        self._box_style: SimpleStyle | None = None
        # --------------------------------------------------------------

        self._content = None

        self._dict_style: StyleMapsLike | None = None
        self._dict_affixes: AffixMapsLike | None = None  # styling

        self._dict_skip_keys: set | None = None

        self._width_chars: int = int(
            os.environ.get("ASHI_WIDTH_CHARS", 80)
        )
        self._dict_positioning: Literal["right", "left"] = "right"

        self._indent_content: tuple = (2, 2, 2, 2)
        self._align_content: Literal[
            "left", "center", "center-ind", "right"
        ] = "left"

        self._nested_levels: int = 0
        self._num_columns: int = 1

        self._polars_tbl_config: dict = {}

        self._borders: bool | None = (
            True  # if None top_name and bottom_name will be ignored
        )

        self._top_name: str | None = None
        self._top_name_align: Literal["center", "left", "right"] = "center"

        self._bottom_name: str | None = None
        self._bottom_name_align: Literal["center", "left", "right"] = (
            "center"
        )

        self._key_equal_symbol: str = ": "

        self._rounding_for_frame_string_casting: int | None = None

        self._pl_col_separator: int | list[int] | None = None
        self._pl_row_separator: int | list[int] | None = None
        self._pl_titles: tuple[str, str] | str | None = None
        self._pl_style: SimpleStyle | None = None

        self._inner_box: Box | None = None

    def __str__(self):
        return self._to_string()

    def __repr__(self):
        return "Box(...)"

    def _repr_html_(self) -> str:
        return html_repr_as_str(self.__repr__())

    # -------------------- config settings -----------------------------

    def set_content(self, content) -> Self:
        self._content = content
        return self

    def set_box_shape(self, box: BoxesShapes) -> Self:
        # TODO: or pass your own string
        self._box = BOXES.get(box)

        if self._box is None:
            raise ValueError(
                f"Invalid box: please select one of {BoxesShapes}"
            )

        return self

    def set_width_chars(self, width_chars: int | None = None) -> Self:
        if width_chars is None:
            self._width_chars = int(
                os.environ.get("ASHI_WIDTH_CHARS", 80)
            )
        else:
            self._width_chars = width_chars
        return self

    def set_top_name(self, top_name: str | None) -> Self:
        self._top_name = top_name
        return self

    def set_top_name_align(
            self, top_name_align: Literal["center", "left", "right"]
    ) -> Self:
        self._top_name_align = top_name_align
        return self

    def set_bottom_name(self, bottom_name: str | None) -> Self:
        self._bottom_name = bottom_name
        return self

    def set_bottom_name_align(
            self, bottom_name_align: Literal["center", "left", "right"]
    ) -> Self:
        self._bottom_name_align = bottom_name_align
        return self

    def set_borders(self, borders: bool | None) -> Self:
        self._borders = borders
        return self

    def set_indent_content(
            self, *indent: tuple[int, int, int, int] | int
    ) -> Self:
        if isinstance(indent[0], tuple):
            indent = indent[0]

        self._indent_content = indent
        return self

    def set_align_content(
            self, align: Literal["left", "center", "center-ind", "right"]
    ) -> Self:
        self._align_content = align
        return self

    def set_key_equal_symbol(self, key_equal_symbol: str) -> Self:
        self._key_equal_symbol = key_equal_symbol
        return self

    # --------------------------- dict ---------------------------------

    def set_dict_positioning(
            self, positioning: Literal["right", "left"]
    ) -> Self:
        self._dict_positioning = positioning
        return self

    def set_dict_nested_levels(self, nested_levels: int) -> Self:
        self._nested_levels = nested_levels
        return self

    def set_dict_num_columns(self, num_columns: int) -> Self:
        # only for nested levels
        self._num_columns = num_columns
        return self

    def _set_dict_skip_keys(self, skip_keys: Iterable[str]) -> Self:
        if isinstance(skip_keys, str):
            skip_keys = {skip_keys}

        self._dict_skip_keys = set(skip_keys)
        return self

    def set_inner_boxes(self, *box: Box) -> Self:
        if box:
            self._inner_box: Box = box[0]  # type: ignore[no-redef]

        if self._inner_box is not None:
            self._inner_box.set_inner_boxes(*box[1:])

        return self

    # ---------------------- polars tbl config -------------------------

    def _set_rounding_for_frame_string_casting(
            self, rounding_for_frame_string_casting: int
    ) -> Self:
        # careful, once this is set the other polars configurations may not work because all columns will be strings
        self._rounding_for_frame_string_casting = (
            rounding_for_frame_string_casting
        )
        return self

    def set_pl_tbl_config(self, **polars_tbl_config) -> Self:
        self._polars_tbl_config = polars_tbl_config
        return self

    def set_pl_col_separator(
            self, col_separator_idx: int | list[int]
    ) -> Self:
        self._pl_col_separator = col_separator_idx
        return self

    def set_pl_row_separator(
            self, row_separator_idx: int | list[int]
    ) -> Self:
        self._pl_row_separator = row_separator_idx
        return self

    def set_pl_titles(self, titles: tuple[str, str] | str | None) -> Self:
        self._pl_titles = titles
        return self

    # def set_dict_style(
    #         self, style: StyleMapping | None
    # ) -> Self:
    #     # config = {
    #     #
    #     #     "key_styles": {
    #     #         1: {"color": "blue", "bold": True},
    #     #         2: {"color": "yellow", "italic": True},
    #     #
    #     #     },
    #     #
    #     #     "value_styles": {
    #     #         "errors": {"color": "green"},
    #     #         "a": {"color": "yellow", "italic": True},
    #     #
    #     #     },
    #     #
    #     #     # "key_affixes": {
    #     #     #
    #     #     #     "name": {"prefix": ">> ", "suffix": " <<", "start_level": -1,
    #     #     #              "apply_to_deeper_levels": True}
    #     #     # },
    #     # }
    #     #
    #     self._dict_style = style
    #     return self

    def set_dict_style(self, style: StyleMapsLike | None) -> Self:
        self._dict_style = style
        return self

    def set_dict_affixes(self, affixes: AffixMapsLike | None) -> Self:
        self._dict_affixes = affixes
        return self

    def set_box_style(self, style: SimpleStyle | None) -> Self:
        self._box_style = style
        return self

    def set_pl_style(self, style: SimpleStyle | None = None) -> Self:
        self._pl_style = style
        return self

    # ------------------------------------------------------------------

    def __call__(
            self,
            content: Any = None,
            *,
            width_chars: int | None = None,
            boxed: bool = True,
    ) -> str | StStr:
        return self.to_string(
            content, width_chars=width_chars, boxed=boxed
        )

    def to_string(
            self,
            content: Any = None,
            *,
            width_chars: int | None = None,
            boxed: bool = True,
    ) -> str | StStr:
        if width_chars is None:
            width_chars = self._width_chars

        if content is None:
            content = self._content

        if content is None and boxed:
            return self._get_boxed_str(text="", width_chars=width_chars)

        if isinstance(content, str):
            content = content.split("\n")  # to list

        if isinstance(content, list):
            out = []
            for e in content:
                text = self._to_string(
                    content=e,
                    # TODO: check this, but we are subtracting 2 because now we are crating a frameless box
                    width_chars=width_chars - 2,
                    boxed=False,
                )
                out.append(text)

            if boxed:
                return self._get_boxed_str(
                    text=join_ststr(out, "\n"), width_chars=width_chars
                )
            else:
                # return None  # ?
                return self._to_string(
                    content=join_ststr(out, "\n"),
                    width_chars=width_chars,
                    boxed=False,
                )

        else:
            return self._to_string(
                content=content, width_chars=width_chars, boxed=True
            )

    def _to_string(
            self,
            content: Any = None,
            *,
            width_chars: int | None = None,
            boxed: bool = True,
            # we need this argument to insert content in join without a box and
            # then a box around everything
    ) -> str | StStr:
        if width_chars is None:
            width_chars = self._width_chars

        if content is None:
            content = self._content

        if content is None and boxed:
            return self._get_boxed_str(text="", width_chars=width_chars)

        if self._rounding_for_frame_string_casting is not None:
            # it modifies all the dataframes by casting their columns to strings
            content = map_dataframes_to_string(
                obj=content,
                round=self._rounding_for_frame_string_casting,
            )

        if isinstance(content, dict):
            # if self._dict_style and should_style():
            #     content = style_nested_structure(
            #         data=content,
            #         config=self._dict_style,
            #     )
            if self._dict_style or self._dict_affixes:
                content = render_nested_structure(
                    data=content,
                    supports_styling=should_style(),
                    styles=self._dict_style,  # type: ignore[arg-type]
                    affixes=self._dict_affixes,  # type: ignore[arg-type]
                    custom_formatters=None,
                    path_based_styling=True,
                )

            # is threshold is lower than the actual nestendess the threshold is returned
            # nested_levels = find_min_depth_iterative(content, threshold=self._nested_levels)

            text: str | StStr = self._from_nested_dict(
                content=content,
                width_chars=width_chars,
                boxed=boxed,
                nested_levels=self._nested_levels,
            )

        elif isinstance(content, (pl.DataFrame, pl.LazyFrame)):
            text = self._from_frame(
                content=content, width_chars=width_chars, boxed=boxed
            )

        elif is_info_collection(content=content):
            text = self._from_info_collection(
                content=content, width_chars=width_chars
            )

        elif is_info(content=content):
            text = self._from_info(
                content=content, width_chars=width_chars
            )

        else:
            text = self._from_other(
                content=content, width_chars=width_chars, boxed=boxed
            )

        return text

    def _get_boxed_str(
            self,
            text: str | StStr | list[str | StStr],
            width_chars: int,
    ) -> str | StStr:
        if isinstance(self._borders, bool):  # if
            text = boxed_str(
                text=text,
                top_name=self._top_name,
                top_name_align=self._top_name_align,
                bottom_name=self._bottom_name,
                bottom_name_align=self._bottom_name_align,
                width_chars=width_chars,
                align=self._align_content,  # here content gets centered
                box=self._box if self._borders else None,
                style=self._box_style,  # type: ignore[arg-type]
            )

        if isinstance(text, (str, StStr)):
            return text

        return join_ststr(text, separator="\n")

    def _from_dict(
            self, content: dict, *, width_chars: int, boxed: bool
    ) -> str | StStr:
        """Renders a dictionary into a nested string of key and associated values in a list"""
        if self._dict_positioning == "left":
            func = format_dict_v1_to_str
        else:
            func = format_dict_v2_to_str

        text: str | StStr = func(
            data=content,
            width_chars=width_chars if not boxed else width_chars - 2,
            indent=self._indent_content,
            equal_symbol=self._key_equal_symbol,  # ": "
            separators=(self._pl_col_separator, self._pl_row_separator),
            titles=self._pl_titles,
            style=self._pl_style,  # type: ignore[arg-type]
            **self._polars_tbl_config,
        )

        if boxed:
            return self._get_boxed_str(text=text, width_chars=width_chars)

        return text

    def _from_nested_dict(
            self,
            content: dict,
            *,
            width_chars: int,
            boxed: bool,
            nested_levels: int,
    ) -> str | StStr:
        if self._dict_skip_keys is not None:
            content = prune_keys(content, skip_keys=self._dict_skip_keys)

        if nested_levels == 0:
            return self._from_dict(
                content=content, width_chars=width_chars, boxed=boxed
            )

        elif nested_levels >= 1:
            return self._from_nested_dict_levels(
                content=content,
                width_chars=width_chars,
                boxed=boxed,
                nested_levels=nested_levels,
            )
        else:
            raise ValueError(
                f"nested levels not supported: {nested_levels}"
            )

    def _from_nested_dict_levels(
            self,
            content: dict,
            *,
            width_chars: int,
            boxed: bool,
            nested_levels: int,
    ):
        box: Box | None = self._inner_box

        if box is None:
            box = Box("horizontal_top_ascii")

        width_chars_inner = width_chars // self._num_columns
        width_chars_inner -= 2

        box = box.set_dict_nested_levels(
            nested_levels - 1
        ).set_width_chars(
            width_chars_inner
        )  # go down one level from where we are

        out: list[str | StStr] = []

        for title, value in content.items():
            box.set_top_name(title)

            text = box._to_string(
                content=value, width_chars=width_chars_inner, boxed=True
            )

            out.append(text)

        # -------

        out_str: str | StStr = concatenate_strings(
            strings=out, num_columns=self._num_columns
        )

        if not boxed:
            return out_str
        else:
            return self._get_boxed_str(
                text=out_str, width_chars=width_chars
            )

    def _from_frame(
            self,
            content: pl.DataFrame | pl.LazyFrame,
            *,
            width_chars: int,
            boxed: bool,
    ) -> str | StStr:
        """Renders a data/lazy frame into a string"""
        # width_chars_text = width_chars if not boxed else width_chars - 2
        width_chars_text = width_chars - 2

        if isinstance(content, pl.LazyFrame):
            return self._from_other(
                content=content.__repr__(),
                width_chars=width_chars_text,
                boxed=boxed,
            )

        text: str | StStr = format_dataframe_repr(
            data=content,
            width_chars=width_chars_text,
            separators=(self._pl_col_separator, self._pl_row_separator),
            titles=self._pl_titles,
            style=self._pl_style,  # type: ignore[arg-type]
            **self._polars_tbl_config,
        )

        if boxed:
            return self._get_boxed_str(text=text, width_chars=width_chars)

        return text

    def _from_other(
            self, content: Any, *, width_chars: int, boxed: bool
    ) -> str | StStr:
        """Renders other content into a string: int/float/string..."""
        # TODO: manage lists

        base_indent, _, _, right_padding = _determine_indent(
            indent=self._indent_content
        )

        text: list[str | StStr] = text_split_with_padding(
            text=content,
            width_chars=width_chars if not boxed else width_chars - 2,
            # - (base_indent + right_padding)
            left_padding=base_indent,
            right_padding=right_padding,
        )

        if boxed:
            return self._get_boxed_str(text=text, width_chars=width_chars)

        return join_ststr(text, separator="\n")

    def _from_info(
            self,
            content: Info,
            width_chars: int,
    ) -> str | StStr:
        text = (
            content
            ._get_box(set_content=True)
            ._to_string(width_chars=width_chars - 2)
        )
        return text

    def _from_info_collection(
            self,
            content: InfoCollection,
            width_chars: int,
    ) -> str | StStr:
        out = []

        for info in content:
            text = (
                info
                ._get_box(set_content=True)
                ._to_string(width_chars=width_chars - 2)
            )
            out.append(text)

        return join_ststr(out, separator="\n")

    # ------------------------------------------------------------------

    def join_boxes(
            self,
            other: Box | Iterable[Box],
            content: Any = None,
            width_chars: int | None = None,
    ) -> str | StStr:
        # TODO: allow to pass some boxes to go on top

        if width_chars is None:
            width_chars = self._width_chars

        if content is None:
            content = self._content

        if isinstance(other, Box):
            other = [other]

        text = self._to_string(
            content=content, width_chars=width_chars, boxed=False
        )

        # get the strings from all the boxes (to be placed inside the current box)
        width_chars_boxes = width_chars - 2

        # other_text = "\n".join([b._to_string(width_chars=width_chars_boxes) for b in other])

        other_text: str | StStr = join_ststr(
            [b._to_string(width_chars=width_chars_boxes) for b in other],
            separator="\n",
        )

        # for i in text.split("\n"):
        #     print(len(i))
        #
        # print(f"{text}\n{other_text}")

        return self._get_boxed_str(
            text=text + "\n" + other_text,  # f"{text}\n{other_text}"
            width_chars=width_chars,
        )


# ----------------------------------------------------------------------


def prune_keys(obj: Any, skip_keys: Iterable[str]) -> Any:
    """
    Return a deep-copied version of `obj` (dict/list/other) with any keys in `skip_keys` removed.
    If a key is skipped, its entire subtree is dropped and we do not search deeper on that branch.
    Works inside lists as well.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in skip_keys:
                # Drop this key and its subtree; do not recurse into it.
                continue
            out[k] = prune_keys(v, skip_keys)
        return out

    if isinstance(obj, list):
        return [prune_keys(item, skip_keys) for item in obj]

    # Primitives / anything else: return as-is
    return obj


def is_info_collection(content: Any) -> bool:
    from paguro.ashi.info.info_collection import InfoCollection

    return isinstance(content, InfoCollection)


def is_info(content: Any) -> bool:
    from paguro.ashi.info.info import Info
    return isinstance(content, Info)


def is_dict_of_dicts(d, level=1):
    """
    Check if a given dictionary is a dictionary of dictionaries,
     up to a specified level of nestedness.

    Parameters
    ----------
    d : dict
        The dictionary to check.

    level : int, optional
        The level of nestedness to check for. Default is 1.

    Returns
    -------
    bool
        True if the dictionary is a dictionary of dictionaries
        up to the specified level, False otherwise.
    """
    if not isinstance(d, dict) or level < 1:
        return False

    if level == 1:
        return all(isinstance(value, dict) for value in d.values())

    # For deeper levels, recursively check each nested dictionary
    return all(
        is_dict_of_dicts(value, level - 1)
        for value in d.values()
        if isinstance(value, dict)
    )


BoxesShapes = Literal[
    "no_boarder",
    "rounded",
    "horizontal_top",
    "ascii",
    "ascii_double",
    "square",
    "horizontal_top_ascii",
    "horizontals",
    "horizontals_double",
    "horizontals_double_top",
    "horizontals_double_tb",
    "double",
    "heavy",
    "horizontals_heavy_top",
]

BOXES = {
    "no_boarder": "   \n   \n   \n",
    "rounded": "╭─╮\n│ │\n╰─╯\n",
    "horizontal_top": " ─ \n   \n   \n",
    "ascii": "+-+\n| |\n+-+\n",
    "ascii_double": "+=+\n| |\n+=+\n",
    "square": "┌─┐\n│ │\n└─┘\n",
    "horizontal_top_ascii": " - \n   \n   \n",
    "horizontals": " ─ \n   \n ─ \n",
    "horizontals_double": " ═ \n   \n ═ \n",
    "horizontals_double_top": "═══\n   \n   \n",
    "horizontals_double_tb": "═══\n   \n═══\n",
    "double": "╔═╗\n║ ║\n╚═╝\n",
    "heavy": "┏━┓\n┃ ┃\n┗━┛\n",
    "horizontals_heavy_top": "━━━\n   \n   \n",
}
