from __future__ import annotations

import textwrap
from collections.abc import Iterable
from itertools import groupby
from typing import Literal

from paguro.ashi.terminal import terminal_detector
from paguro.utils.config import _forced_color_system
from paguro.utils.dependencies import dataclasses, re

# Extended list of 256-color ANSI escape codes
COLORS_256 = {
    # System colors (0-15)
    "black": "0",
    "maroon": "1",
    "green": "2",
    "olive": "3",
    "navy": "4",
    "purple": "5",
    "teal": "6",
    "silver": "7",
    "grey": "8",
    "red": "9",
    "lime": "10",
    "yellow": "11",
    "blue": "12",
    "fuchsia": "13",
    "aqua": "14",
    "white": "15",
    # Standard colors (16-231)
    "grey0": "16",
    "navy_blue": "17",
    "dark_blue": "18",
    "blue3": "19",
    "blue3_1": "20",
    "blue1": "21",
    "dark_green": "22",
    "deep_sky_blue4": "23",
    "deep_sky_blue4_1": "24",
    "deep_sky_blue4_2": "25",
    "dodger_blue3": "26",
    "dodger_blue2": "27",
    "green4": "28",
    "spring_green4": "29",
    "turquoise4": "30",
    "deep_sky_blue3": "31",
    "deep_sky_blue3_1": "32",
    "dodger_blue1": "33",
    "green3": "34",
    "spring_green3": "35",
    "dark_cyan": "36",
    "light_sea_green": "37",
    "deep_sky_blue2": "38",
    "deep_sky_blue1": "39",
    "green3_1": "40",
    "spring_green3_1": "41",
    "spring_green2": "42",
    "cyan3": "43",
    "dark_turquoise": "44",
    "turquoise2": "45",
    "green1": "46",
    "spring_green2_1": "47",
    "spring_green1": "48",
    "medium_spring_green": "49",
    "cyan2": "50",
    "cyan1": "51",
    "dark_red": "52",
    "deep_pink4": "53",
    "purple4": "54",
    "purple4_1": "55",
    "purple3": "56",
    "blue_violet": "57",
    "orange4": "58",
    "grey37": "59",
    "medium_purple4": "60",
    "slate_blue3": "61",
    "slate_blue3_1": "62",
    "royal_blue1": "63",
    "chartreuse4": "64",
    "dark_sea_green4": "65",
    "pale_turquoise4": "66",
    "steel_blue": "67",
    "steel_blue3": "68",
    "cornflower_blue": "69",
    "chartreuse3": "70",
    "dark_sea_green4_1": "71",
    "cadet_blue": "72",
    "cadet_blue_1": "73",
    "sky_blue3": "74",
    "steel_blue1": "75",
    "chartreuse3_1": "76",
    "pale_green3": "77",
    "sea_green3": "78",
    "aquamarine3": "79",
    "medium_turquoise": "80",
    "steel_blue1_1": "81",
    "chartreuse2": "82",
    "sea_green2": "83",
    "sea_green1": "84",
    "sea_green1_1": "85",
    "aquamarine1": "86",
    "dark_slate_gray2": "87",
    "dark_red_1": "88",
    "deep_pink4_1": "89",
    "dark_magenta": "90",
    "dark_magenta_1": "91",
    "dark_violet": "92",
    "purple_1": "93",
    "orange4_1": "94",
    "light_pink4": "95",
    "plum4": "96",
    "medium_purple3": "97",
    "medium_purple3_1": "98",
    "slate_blue1": "99",
    "yellow4": "100",
    "wheat4": "101",
    "grey53": "102",
    "light_slate_grey": "103",
    "medium_purple": "104",
    "light_slate_blue": "105",
    "yellow4_1": "106",
    "dark_olive_green3": "107",
    "dark_sea_green": "108",
    "light_sky_blue3": "109",
    "light_sky_blue3_1": "110",
    "sky_blue2": "111",
    "chartreuse2_1": "112",
    "dark_olive_green3_1": "113",
    "pale_green3_1": "114",
    "dark_sea_green3": "115",
    "dark_slate_gray3": "116",
    "sky_blue1": "117",
    "chartreuse1": "118",
    "light_green_1": "119",
    "light_green_2": "120",
    "pale_green1": "121",
    "aquamarine1_1": "122",
    "dark_slate_gray1": "123",
    "red3": "124",
    "deep_pink4_2": "125",
    "medium_violet_red": "126",
    "magenta3": "127",
    "dark_violet_1": "128",
    "purple_2": "129",
    "dark_orange3": "130",
    "indian_red": "131",
    "hot_pink3": "132",
    "medium_orchid3": "133",
    "medium_orchid": "134",
    "medium_purple2": "135",
    "dark_goldenrod": "136",
    "light_salmon3": "137",
    "rosy_brown": "138",
    "grey63": "139",
    "medium_purple2_1": "140",
    "medium_purple1": "141",
    "gold3": "142",
    "dark_khaki": "143",
    "navajo_white3": "144",
    "grey69": "145",
    "light_steel_blue3": "146",
    "light_steel_blue": "147",
    "yellow3": "148",
    "dark_olive_green3_2": "149",
    "dark_sea_green3_1": "150",
    "dark_sea_green2": "151",
    "light_cyan3": "152",
    "light_sky_blue1": "153",
    "green_yellow": "154",
    "dark_olive_green2": "155",
    "pale_green1_1": "156",
    "dark_sea_green5": "157",
    "dark_sea_green5_1": "158",
    "pale_turquoise1": "159",
    "red3_1": "160",
    "deep_pink3": "161",
    "deep_pink3_1": "162",
    "magenta3_1": "163",
    "magenta3_2": "164",
    "magenta2": "165",
    "dark_orange3_1": "166",
    "indian_red_1": "167",
    "hot_pink3_1": "168",
    "hot_pink2": "169",
    "orchid": "170",
    "medium_orchid1": "171",
    "orange3": "172",
    "light_salmon3_1": "173",
    "light_pink3": "174",
    "pink3": "175",
    "plum3": "176",
    "violet": "177",
    "gold3_1": "178",
    "light_goldenrod3": "179",
    "tan": "180",
    "misty_rose3": "181",
    "thistle3": "182",
    "plum2": "183",
    "yellow3_1": "184",
    "khaki3": "185",
    "light_goldenrod2": "186",
    "light_yellow3": "187",
    "grey84": "188",
    "light_steel_blue1": "189",
    "yellow2": "190",
    "dark_olive_green1": "191",
    "dark_olive_green1_1": "192",
    "dark_sea_green1": "193",
    "honeydew2": "194",
    "light_cyan1": "195",
    "red1": "196",
    "deep_pink2": "197",
    "deep_pink1": "198",
    "deep_pink1_1": "199",
    "magenta2_1": "200",
    "magenta1": "201",
    "orange_red1": "202",
    "indian_red1": "203",
    "indian_red1_1": "204",
    "hot_pink": "205",
    "hot_pink_1": "206",
    "medium_orchid1_1": "207",
    "dark_orange": "208",
    "salmon1": "209",
    "light_coral": "210",
    "pale_violet_red1": "211",
    "orchid2": "212",
    "orchid1": "213",
    "orange1": "214",
    "sandy_brown": "215",
    "light_salmon1": "216",
    "light_pink1": "217",
    "pink1": "218",
    "plum1": "219",
    "gold1": "220",
    "light_goldenrod2_1": "221",
    "light_goldenrod2_2": "222",
    "navajo_white1": "223",
    "misty_rose1": "224",
    "thistle1": "225",
    "yellow1": "226",
    "light_goldenrod1": "227",
    "khaki1": "228",
    "wheat1": "229",
    "cornsilk1": "230",
    "grey100": "231",
    # Grayscale (232-255)
    "grey3": "232",
    "grey7": "233",
    "grey11": "234",
    "grey15": "235",
    "grey19": "236",
    "grey23": "237",
    "grey27": "238",
    "grey30": "239",
    "grey35": "240",
    "grey39": "241",
    "grey42": "242",
    "grey46": "243",
    "grey50": "244",
    "grey54": "245",
    "grey58": "246",
    "grey62": "247",
    "grey66": "248",
    "grey70": "249",
    "grey74": "250",
    "grey78": "251",
    "grey82": "252",
    "grey85": "253",
    "grey89": "254",
    "grey93": "255",
}

# Updated ANSI escape codes and color dictionaries
STYLE_CODES = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "italic": "\033[3m",
    "underline": "\033[4m",
    "blink": "\033[5m",
    "inverted": "\033[7m",
    "hidden": "\033[8m",
    "strikethrough": "\033[9m",
}

COLORS_4BIT = {
    "black": "30",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "white": "37",
    "bright_black": "90",
    "bright_red": "91",
    "bright_green": "92",
    "bright_yellow": "93",
    "bright_blue": "94",
    "bright_magenta": "95",
    "bright_cyan": "96",
    "bright_white": "97",
}

BACKGROUND_COLORS_4BIT = {
    "on_black": "40",
    "on_red": "41",
    "on_green": "42",
    "on_yellow": "43",
    "on_blue": "44",
    "on_magenta": "45",
    "on_cyan": "46",
    "on_white": "47",
    "on_bright_black": "100",
    "on_bright_red": "101",
    "on_bright_green": "102",
    "on_bright_yellow": "103",
    "on_bright_blue": "104",
    "on_bright_magenta": "105",
    "on_bright_cyan": "106",
    "on_bright_white": "107",
}

# Keep COLORS_256 as is

ANSI_ESCAPE_RE = re.compile(r"\033\[(?:[0-9]{1,3};?)*m")


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert a hex color code to an RGB tuple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(c * 2 for c in hex_color)
    return tuple(int(hex_color[i: i + 2], 16) for i in (0, 2, 4))


def rgb_to_ansi_256(r: int, g: int, b: int) -> int:
    """Convert RGB to the closest ANSI 256 (8-bit) color code."""
    if r == g == b:
        if r < 8:
            return 16
        if r > 248:
            return 231
        return round(((r - 8) / 247) * 24) + 232

    ansi = (
            16
            + (36 * round(r / 255 * 5))
            + (6 * round(g / 255 * 5))
            + round(b / 255 * 5)
    )
    return ansi


def rgb_to_ansi_16(r: int, g: int, b: int) -> str:
    """Convert RGB to the closest ANSI 16 (4-bit) color name."""
    ansi_16 = {
        (0, 0, 0): "black",
        (128, 0, 0): "red",
        (0, 128, 0): "green",
        (128, 128, 0): "yellow",
        (0, 0, 128): "blue",
        (128, 0, 128): "magenta",
        (0, 128, 128): "cyan",
        (192, 192, 192): "white",
        (128, 128, 128): "bright_black",
        (255, 0, 0): "bright_red",
        (0, 255, 0): "bright_green",
        (255, 255, 0): "bright_yellow",
        (0, 0, 255): "bright_blue",
        (255, 0, 255): "bright_magenta",
        (0, 255, 255): "bright_cyan",
        (255, 255, 255): "bright_white",
    }

    def color_distance(color1, color2):
        return sum(
            (a - b) ** 2 for a, b in zip(color1, color2, strict=False)
        )

    closest_color = min(
        ansi_16, key=lambda x: color_distance((r, g, b), x)
    )
    return ansi_16[closest_color]


@dataclasses.dataclass
class Style:
    color: str | tuple | None = None
    background: str | tuple | None = None
    bold: bool | None = False
    dim: bool | None = False
    italic: bool | None = False
    underline: bool | None = False
    blink: bool | None = False
    inverted: bool | None = False
    hidden: bool | None = False
    strikethrough: bool | None = False
    color_mode: Literal["auto", "truecolor", "256", "16"] = (
        "auto"  # Can be "auto", "truecolor", "256", or "16"
    )

    def __post_init__(self) -> None:
        self.color = self._process_color(self.color)
        self.background = self._process_color(self.background)

    def _process_color(self, color):
        if isinstance(color, str) and color.startswith("#"):
            return hex_to_rgb(color)
        return color

    def to_tuple(self) -> tuple:
        ascii_styles = []

        if self.color:
            ascii_styles.extend(
                self._color_to_ansi(self.color, foreground=True)
            )

        if self.background:
            ascii_styles.extend(
                self._color_to_ansi(self.background, foreground=False)
            )

        if self.bold:
            ascii_styles.append(STYLE_CODES["bold"])
        if self.dim:
            ascii_styles.append(STYLE_CODES["dim"])
        if self.italic:
            ascii_styles.append(STYLE_CODES["italic"])
        if self.underline:
            ascii_styles.append(STYLE_CODES["underline"])
        if self.blink:
            ascii_styles.append(STYLE_CODES["blink"])
        if self.inverted:
            ascii_styles.append(STYLE_CODES["inverted"])
        if self.hidden:
            ascii_styles.append(STYLE_CODES["hidden"])
        if self.strikethrough:
            ascii_styles.append(STYLE_CODES["strikethrough"])

        return tuple(ascii_styles)

    def _color_to_ansi(self, color, foreground=True):
        if isinstance(color, str):
            if color in (
                    COLORS_4BIT if foreground else BACKGROUND_COLORS_4BIT
            ):
                return [
                    f"\033[{COLORS_4BIT[color]}m"
                    if foreground
                    else f"\033[{BACKGROUND_COLORS_4BIT[color]}m"
                ]

            elif color in COLORS_256:
                return [
                    f"\033[{'38' if foreground else '48'};5;{COLORS_256[color]}m"
                ]

        elif isinstance(color, tuple):
            r, g, b = color

            # we are going with the terminal_detector color system,
            # we should allow to force first

            if (
                    self.color_mode == "truecolor" or (
                    self.color_mode == "auto"
                    and (
                            terminal_detector().color_system == "truecolor" or
                            _forced_color_system() == "truecolor"
                    )
            )
            ):
                return [
                    f"\033[{'38' if foreground else '48'};2;{r};{g};{b}m"
                ]

            elif (
                    self.color_mode == "256" or (
                    self.color_mode == "auto"
                    and (
                            terminal_detector().color_system == "256" or
                            _forced_color_system() == "256"
                    )
            )
            ):
                ansi_256 = rgb_to_ansi_256(r, g, b)

                return [
                    f"\033[{'38' if foreground else '48'};5;{ansi_256}m"
                ]

            else:
                ansi_16 = rgb_to_ansi_16(r, g, b)
                if foreground:
                    return [f"\033[{COLORS_4BIT[ansi_16]}m"]
                else:
                    return [
                        f"\033[{BACKGROUND_COLORS_4BIT[f'on_{ansi_16}']}m"
                    ]
        return []


class StStr:
    def __init__(
            self,
            text: str | None = None,
            styles: Iterable | Style | None = None,
    ):
        if text is None:
            text = ""
        elif not isinstance(text, (StStr, str)):
            text = str(text)

        if styles is None:
            styles_iter: tuple[Style] | tuple = ()
        elif isinstance(styles, Style):
            styles_iter = (styles,)
        else:
            styles_iter = tuple(styles)

        self._segments = [(text, styles_iter)]

        self._styled_cache = None

        self._color: str | tuple | None = None
        self._background: str | tuple | None = None
        self._bold: bool | None = None
        self._dim: bool | None = None
        self._italic: bool | None = None
        self._underline: bool | None = None
        self._blink: bool | None = None
        self._inverted: bool | None = None
        self._hidden: bool | None = None
        self._strikethrough: bool | None = None

    def set_style(
            self,
            color: str | tuple | None = None,
            background: str | tuple | None = None,
            bold: bool | None = None,
            dim: bool | None = None,
            italic: bool | None = None,
            underline: bool | None = None,
            blink: bool | None = None,
            inverted: bool | None = None,
            hidden: bool | None = None,
            strikethrough: bool | None = None,
            color_mode: Literal["auto", "truecolor", "256", "16"] = "auto",
    ):
        style = Style(
            color=color if color is not None else self._color,
            background=background
            if background is not None
            else self._background,
            bold=bold if bold is not None else self._bold,
            dim=dim if dim is not None else self._dim,
            italic=italic if italic is not None else self._italic,
            underline=underline
            if underline is not None
            else self._underline,
            blink=blink if blink is not None else self._blink,
            inverted=inverted if inverted is not None else self._inverted,
            hidden=hidden if hidden is not None else self._hidden,
            strikethrough=strikethrough
            if strikethrough is not None
            else self._strikethrough,
            color_mode=color_mode,
        )

        self._color = style.color
        self._background = style.background
        self._bold = style.bold
        self._dim = style.dim
        self._italic = style.italic
        self._underline = style.underline
        self._blink = style.blink
        self._inverted = style.inverted
        self._hidden = style.hidden
        self._strikethrough = style.strikethrough

        self._add_styles(style.to_tuple())

        return self

    def _add_styles(self, new_styles):
        if self._segments:
            text, _ = self._segments[-1]
            self._segments[-1] = (text, new_styles)
            self._invalidate_cache()

    def _invalidate_cache(self):
        self._styled_cache = None

    def reset_style(self):
        if self._segments:
            text, _ = self._segments[-1]
            self._segments[-1] = (text, ())
            self._invalidate_cache()
        return self

    def _apply_styles(self):
        if self._styled_cache is None:
            styled_string = "".join(
                f"{''.join(styles)}{text}{STYLE_CODES['reset']}"
                for text, styles in self._segments
            )
            self._styled_cache = styled_string
        return self._styled_cache

    def __str__(self):
        return self._apply_styles()

    def __repr__(self):
        return self._apply_styles()

    def __add__(self, other):
        if isinstance(other, StStr):
            new_instance = StStr("")
            new_instance._segments = self._merge_segments(
                self._segments + other._segments
            )
            return new_instance
        elif isinstance(other, str):
            new_instance = StStr("")
            new_instance._segments = self._segments + [(other, ())]
            return new_instance
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, str):
            new_instance = StStr("")
            new_instance._segments = [(other, ())] + self._segments
            return new_instance
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, int):
            new_instance = StStr()
            new_instance._segments = self._segments * other
            return new_instance
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, int):
            new_instance = StStr()
            new_instance._segments = self._segments * other
            return new_instance
        return NotImplemented

    def __len__(self):
        # return sum(len(text) for text, _ in self._segments)
        return self.length(unicode=False)

    def length(self, *, unicode: bool = False):
        if not unicode:
            return sum(len(text) for text, _ in self._segments)
        else:
            return sum(
                get_display_width(text) for text, _ in self._segments
            )

    def actual_len(self):
        """Calculate the length of the string without ANSI escape codes."""
        return len(self._strip_styles())

    def __getitem__(self, key):
        if isinstance(key, slice):
            new_instance = StStr("")
            sliced_segments = self._slice_segments(key)
            new_instance._segments = sliced_segments
            return new_instance
        elif isinstance(key, int):
            combined_str = "".join(text for text, _ in self._segments)
            return combined_str[key]
        return NotImplemented

    def _slice_segments(self, key):
        start, stop, step = key.indices(len(self))
        combined_str = "".join(text for text, _ in self._segments)
        sliced_text = combined_str[start:stop:step]

        sliced_segments = []
        current_pos = 0
        slice_pos = 0

        for text, styles in self._segments:
            end_pos = current_pos + len(text)
            if current_pos >= stop:
                break

            if start < end_pos:
                if current_pos < start:
                    segment_start = start - current_pos
                else:
                    segment_start = 0

                if end_pos > stop:
                    segment_end = stop - current_pos
                else:
                    segment_end = len(text)

                sliced_segment = text[segment_start:segment_end:step]
                slice_pos += len(sliced_segment)
                sliced_segments.append((sliced_segment, styles))

            current_pos = end_pos

        return self._merge_segments(sliced_segments)

    def _merge_segments(self, segments):
        merged_segments = []
        from paguro.ashi.repr.string.utils import join_ststr

        for styles, group in groupby(segments, key=lambda x: x[1]):
            # merged_text = "".join(text for text, _ in group)
            merged_text: str | StStr = join_ststr(
                [text for text, _ in group], separator=""
            )

            merged_segments.append((merged_text, styles))
        return merged_segments

    def _strip_styles(self):
        """Return the string without ANSI escape codes."""
        return ANSI_ESCAPE_RE.sub("", self._apply_styles())

    def __eq__(self, other):
        if isinstance(other, StStr):
            return self._segments == other._segments
        return "".join(text for text, _ in self._segments) == str(other)

    def __hash__(self):
        # Combine the hash of the text content and the styles
        return hash(tuple(self._segments))

    def _get_styles_NEW(self, as_dict_of_dicts=False):
        """Extract styles from the segments with improved color handling."""
        style_dict = {}
        segments_dict = {}

        for text, segment_styles in self._segments:
            segment_style_dict = {}
            for style in segment_styles:
                # Handle basic styles
                if style in STYLE_CODES.values():
                    try:
                        style_name = [
                            name
                            for name, code in STYLE_CODES.items()
                            if code == style
                        ][0]
                        segment_style_dict[style_name] = True
                    except IndexError:
                        continue  # Skip unknown style codes

                # Handle 4-bit colors (16 colors)
                elif re.match(r"\033\[([34][0-7]|9[0-7]|10[0-7])m", style):
                    try:
                        color_code = style[2:-1]
                        if color_code.startswith(
                                "4"
                        ) or color_code.startswith("10"):
                            color_name = next(
                                (
                                    name
                                    for name, code in BACKGROUND_COLORS_4BIT.items()
                                    if code == color_code
                                ),
                                None,
                            )
                            if color_name:
                                segment_style_dict["background"] = (
                                    color_name.replace("on_", "")
                                )
                        else:
                            color_name = next(
                                (
                                    name
                                    for name, code in COLORS_4BIT.items()
                                    if code == color_code
                                ),
                                None,
                            )
                            if color_name:
                                segment_style_dict["color"] = color_name
                    except (IndexError, ValueError):
                        continue

                # Handle 8-bit and 24-bit colors
                elif re.match(r"\033\[([34]8;[25];\d+(?:;\d+)*)m", style):
                    try:
                        parts = style[2:-1].split(";")
                        color_type = parts[0]
                        key = (
                            "color" if color_type == "38" else "background"
                        )

                        if parts[1] == "5":  # 8-bit color
                            color_code = parts[2]
                            # Try to find color name, fall back to closest 4-bit color if not found
                            color_name = next(
                                (
                                    name
                                    for name, code in COLORS_256.items()
                                    if code == color_code
                                ),
                                None,
                            )
                            if color_name:
                                segment_style_dict[key] = color_name
                            else:
                                # Convert color code to RGB and find closest 4-bit color
                                r, g, b = _code256_to_rgb(int(color_code))
                                fallback_color = rgb_to_ansi_16(r, g, b)
                                segment_style_dict[key] = fallback_color

                        elif parts[1] == "2":  # 24-bit color
                            r, g, b = map(int, parts[2:5])
                            if (
                                    terminal_detector().color_system
                                    == "truecolor"
                            ):
                                segment_style_dict[key] = (r, g, b)
                            else:
                                # Fall back to closest 4-bit color for non-truecolor terminals
                                fallback_color = rgb_to_ansi_16(r, g, b)
                                segment_style_dict[key] = fallback_color
                    except (IndexError, ValueError):
                        continue

                if as_dict_of_dicts:
                    segments_dict[text] = segment_style_dict
                else:
                    style_dict.update(segment_style_dict)

        return segments_dict if as_dict_of_dicts else style_dict

    def _get_styles(self, as_dict_of_dicts=False):
        """Extract styles from the segments."""
        style_dict = {}
        segments_dict = {}

        for text, segment_styles in self._segments:
            segment_style_dict = {}

            for style in segment_styles:
                if style in STYLE_CODES.values():
                    style_name = [
                        name
                        for name, code in STYLE_CODES.items()
                        if code == style
                    ][0]
                    segment_style_dict[style_name] = True

                elif re.match(r"\033\[([34][0-7]|9[0-7]|10[0-7])m", style):
                    color_code = style[2:-1]
                    if color_code.startswith("4") or color_code.startswith(
                            "10"
                    ):
                        color_name = [
                            name
                            for name, code in BACKGROUND_COLORS_4BIT.items()
                            if code == color_code
                        ][0]
                        segment_style_dict["background"] = (
                            color_name.replace("on_", "")
                        )
                    else:
                        color_name = [
                            name
                            for name, code in COLORS_4BIT.items()
                            if code == color_code
                        ][0]
                        segment_style_dict["color"] = color_name

                elif re.match(r"\033\[([34]8;[25];\d+(?:;\d+)*)m", style):
                    parts = style[2:-1].split(";")
                    color_type = parts[0]

                    if parts[1] == "5":  # 8-bit color
                        color_code = parts[2]

                        color_name = [
                            name
                            for name, code in COLORS_256.items()
                            if code == color_code
                        ][0]
                        key = (
                            "color" if color_type == "38" else "background"
                        )
                        segment_style_dict[key] = color_name

                    elif parts[1] == "2":  # 24-bit color
                        r, g, b = map(int, parts[2:5])
                        key = (
                            "color" if color_type == "38" else "background"
                        )
                        segment_style_dict[key] = (r, g, b)

            if as_dict_of_dicts:
                segments_dict[text] = segment_style_dict
            else:
                style_dict.update(segment_style_dict)

        return segments_dict if as_dict_of_dicts else style_dict

    def __iter__(self):
        segment_styles_dict = self._get_styles(as_dict_of_dicts=True)

        for text, styles in self._segments:
            for char in text:
                yield StStr(char).set_style(**segment_styles_dict[text])

    def _pad(self, width, fillchar, align_func):
        stripped_content = self._strip_styles()

        padding_length = max(width - len(stripped_content), 0)
        # padding_length = max(width - get_display_width(stripped_content), 0)

        if padding_length == 0:
            return self

        left_padding = right_padding = 0
        if align_func == "center":
            left_padding = padding_length // 2
            right_padding = padding_length - left_padding
        elif align_func == "ljust":
            left_padding = 0
            right_padding = padding_length
        elif align_func == "rjust":
            left_padding = padding_length
            right_padding = 0

        left_pad_str = fillchar * left_padding
        right_pad_str = fillchar * right_padding

        left_padded_str = StStr(left_pad_str)
        right_padded_str = StStr(right_pad_str)

        out = left_padded_str + self + right_padded_str

        return out

    # String methods to be wrapped
    def _transform(self, method, *args, **kwargs):
        segments = [
            (getattr(text, method)(*args, **kwargs), styles)
            for text, styles in self._segments
        ]
        new_instance = StStr("")
        new_instance._segments = segments

        # return new_instance.set_style(**self._get_styles())
        return new_instance

    # def _transform(self, method, *args, **kwargs):
    #     segments = [(getattr(text, method)(*args, **kwargs), styles) for text, styles in self._segments]
    #     new_instance = StStr()
    #     new_instance._segments = segments
    #     segment_styles_dict = self._get_styles(as_dict_of_dicts=True)
    #     for i, (text, _) in enumerate(new_instance._segments):
    #         new_instance._segments[i] = (text, segment_styles_dict.get(text, ()))
    #     return new_instance

    def _apply_to_ends(self, method, chars=None):
        if not self._segments:
            return self

        new_segments = []

        if method == "lstrip" or method == "strip":
            # Apply lstrip to the left (first) segment
            first_text, first_styles = self._segments[0]
            new_first_text = first_text.lstrip(chars)
            new_segments.append((new_first_text, first_styles))
        else:
            # Keep the original first segment
            new_segments.append(self._segments[0])

        if len(self._segments) > 1:
            if method == "rstrip" or method == "strip":
                # Apply rstrip to the right (last) segment
                last_text, last_styles = self._segments[-1]
                new_last_text = last_text.rstrip(chars)
                new_segments.extend(self._segments[1:-1])
                new_segments.append((new_last_text, last_styles))
            else:
                # Keep the original last segment
                new_segments.extend(self._segments[1:])

        # Remove any empty segments that may have resulted from the strip
        new_segments = [
            (text, styles) for text, styles in new_segments if text
        ]

        new_instance = StStr("")
        new_instance._segments = self._merge_segments(new_segments)
        return new_instance

    def _apply_split_method(self, method, sep=None, maxsplit=-1):
        parts = getattr(self._strip_styles(), method)(sep, maxsplit)
        result = []
        current_pos = 0
        # combined_str = "".join(text for text, _ in self._segments)

        for part in parts:
            part_len = len(part)
            end_pos = current_pos + part_len
            segment_parts = []
            remaining_len = part_len

            for text, styles in self._segments:
                if remaining_len == 0:
                    break

                text_len = len(text)
                if current_pos >= text_len:
                    current_pos -= text_len
                    continue

                if current_pos + remaining_len <= text_len:
                    segment_parts.append(
                        (
                            text[
                            current_pos: current_pos + remaining_len
                            ],
                            styles,
                        )
                    )
                    current_pos = 0
                    remaining_len = 0
                else:
                    segment_parts.append((text[current_pos:], styles))
                    remaining_len -= text_len - current_pos
                    current_pos = 0

            styled_part = StStr(part)
            styled_part._segments = segment_parts
            result.append(styled_part)
            current_pos = end_pos + (
                len(sep) if sep else 1
            )  # Adjust for separator length or space

        return result

    def lstrip(self, chars=None):
        return self._apply_to_ends("lstrip", chars)

    def rstrip(self, chars=None):
        return self._apply_to_ends("rstrip", chars)

    def endswith(self, suffix, start=0, end=None):
        return self._strip_styles().endswith(suffix, start, end)

    def partition(self, sep):
        parts = self._strip_styles().partition(sep)
        segment_styles_dict = self._get_styles(as_dict_of_dicts=True)
        return tuple(
            StStr(part).set_style(**segment_styles_dict.get(part, {}))
            for part in parts
        )

    def rpartition(self, sep):
        parts = self._strip_styles().rpartition(sep)
        segment_styles_dict = self._get_styles(as_dict_of_dicts=True)
        return tuple(
            StStr(part).set_style(**segment_styles_dict.get(part, {}))
            for part in parts
        )

    def split(self, sep=None, maxsplit=-1):
        return self._apply_split_method("split", sep, maxsplit)

    def capitalize(self):
        return self._transform("capitalize")

    def casefold(self):
        return self._transform("casefold")

    def center(self, width, fillchar=" "):
        return self._pad(width, fillchar, "center")

    def count(self, sub, start=0, end=None):
        return self._strip_styles().count(sub, start, end)

    def encode(self, encoding="utf-8", errors="strict"):
        return self._strip_styles().encode(encoding, errors)

    def expandtabs(self, tabsize=8):
        return self._transform("expandtabs", tabsize)

    def find(self, sub, start=0, end=None):
        return self._strip_styles().find(sub, start, end)

    def format(self, *args, **kwargs):
        return self._transform("format", *args, **kwargs)

    def format_map(self, mapping):
        return self._transform("format_map", mapping)

    def index(self, sub, start=0, end=None):
        return self._strip_styles().index(sub, start, end)

    def isalnum(self):
        return self._strip_styles().isalnum()

    def isalpha(self):
        return self._strip_styles().isalpha()

    def isascii(self):
        return self._strip_styles().isascii()

    def isdecimal(self):
        return self._strip_styles().isdecimal()

    def isdigit(self):
        return self._strip_styles().isdigit()

    def isidentifier(self):
        return self._strip_styles().isidentifier()

    def islower(self):
        return self._strip_styles().islower()

    def isnumeric(self):
        return self._strip_styles().isnumeric()

    def isprintable(self):
        return self._strip_styles().isprintable()

    def isspace(self):
        return self._strip_styles().isspace()

    def istitle(self):
        return self._strip_styles().istitle()

    def isupper(self):
        return self._strip_styles().isupper()

    def join(self, iterable):
        new_segments = []
        first = True
        for item in iterable:
            if not first:
                new_segments.extend(self._segments)
            if isinstance(item, StStr):
                new_segments.extend(item._segments)
            elif isinstance(item, str):
                new_segments.append((item, ()))
            else:
                raise TypeError(
                    f"sequence item {iterable.index(item)}: expected str or StyledStr instance, {type(item).__name__} found"
                )
            first = False

        new_instance = StStr("")
        new_instance._segments = self._merge_segments(new_segments)
        return new_instance

    def ljust(self, width, fillchar=" "):
        return self._pad(width, fillchar, "ljust")

    def lower(self):
        return self._transform("lower")

    def maketrans(self, x, y=None, z=None):
        return self._strip_styles().maketrans(x, y, z)

    def replace(self, old, new, count=-1):
        segments = [
            (text.replace(old, new, count), styles)
            for text, styles in self._segments
        ]
        new_instance = StStr("")
        new_instance._segments = segments
        return new_instance

    def rfind(self, sub, start=0, end=None):
        return self._strip_styles().rfind(sub, start, end)

    def rindex(self, sub, start=0, end=None):
        return self._strip_styles().rindex(sub, start, end)

    def rjust(self, width, fillchar=" "):
        return self._pad(width, fillchar, "rjust")

    def rsplit(self, sep=None, maxsplit=-1):
        return self._apply_split_method("rsplit", sep, maxsplit)

    # def splitlines(self, keepends=False):
    #     parts = self._strip_styles().splitlines(keepends)
    #     return [StStr(part).set_style(**self._get_styles()) for part in parts]
    def splitlines(self, keepends=False):
        lines = self.split("\n")

        if not keepends:
            # Remove empty first and last elements if present
            if lines and not lines[0]:
                lines.pop(0)
            if lines and not lines[-1]:
                lines.pop()

        return lines

    def startswith(self, prefix, start=0, end=None):
        return self._strip_styles().startswith(prefix, start, end)

    def strip(self, chars=None):
        return self._apply_to_ends("strip", chars)

    def swapcase(self):
        return self._transform("swapcase")

    def title(self):
        return self._transform("title")

    def translate(self, table):
        return self._transform("translate", table)

    def upper(self):
        return self._transform("upper")

    def zfill(self, width):
        return self._transform("zfill", width)

    def wrap(self, width, **kwargs):
        # Combine all segments into one string while keeping track of styles

        combined_text = "".join(text for text, _ in self._segments)

        # Wrap the combined text
        wrapped_lines = textwrap.wrap(combined_text, width, **kwargs)

        # ---------
        # from paguro.ashi.repr.string.utils import join_ststr
        #
        # # combined_text = "".join(text for text, _ in self._segments)
        # combined_text: str | StStr = join_ststr([text for text, _ in self._segments], separator="")
        # # Wrap the combined text
        # if isinstance(combined_text, StStr):
        #     wrapped_lines = combined_text.wrap(width, **kwargs)
        # else:
        #     wrapped_lines = textwrap.wrap(combined_text, width, **kwargs)
        # ---------

        # Create a mapping of each character to its corresponding style
        first_style = ()
        for idx, (_, styles) in enumerate(self._segments):
            first_style = styles
            if idx == 0:
                break

        return [StStr(l, first_style) for l in wrapped_lines]


def _code256_to_rgb(code):
    """Convert 256-color code to RGB values."""
    if code < 16:
        # System colors 0-15: convert to 4-bit color
        if code < 8:
            return (
                (code & 1) * 205,
                ((code >> 1) & 1) * 205,
                (code >> 2) * 205,
            )
        else:
            return (
                (code & 1) * 255,
                ((code >> 1) & 1) * 255,
                (code >> 2) * 255,
            )
    elif code < 232:
        # 6×6×6 color cube (16-231): convert to RGB
        code -= 16
        return (
            ((code // 36) * 51),
            ((code // 6) % 6 * 51),
            (code % 6 * 51),
        )
    else:
        # Grayscale (232-255): convert to RGB
        gray = (code - 232) * 10 + 8
        return (gray, gray, gray)


# --------------------------- unicode chars ----------------------------


# 🔥todo: proper handling in StStr
def get_display_width(text):
    import unicodedata

    width = 0
    for char in text:
        # If the character is a combining mark, it doesn't add width.
        if unicodedata.combining(char):
            continue

        # Determine the East Asian width category.
        eaw = unicodedata.east_asian_width(char)
        if eaw in ("F", "W"):
            width += 2
        else:
            width += 1

    return width
