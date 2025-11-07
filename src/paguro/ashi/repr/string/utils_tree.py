from __future__ import annotations

from paguro.ashi.repr.string.styled.styled_str import StStr
from paguro.ashi.repr.string.utils import join_ststr


def nested_dict_to_str(
    d: dict, indent: int = 0, n_words: int = 10
) -> str | StStr:
    lines = _nested_dict_to_list(d=d, indent=indent, n_words=n_words)
    # return "\n".join(lines)

    return join_ststr(lines, separator="\n")


def _nested_dict_to_list(d: dict, indent: int, n_words: int) -> list:
    lines = []
    items = list(d.items())

    def adjust_indent(multiline_str, prefix):
        """Apply the same indentation to every line of a multiline string."""
        # return '\n'.join([prefix + line for line in multiline_str.split('\n')])
        return join_ststr(
            [prefix + line for line in multiline_str.split("\n")],
            separator="\n",
        )

    for i, (key, value) in enumerate(items):
        prefix = " " * indent

        if isinstance(value, dict):
            # lines.append(f"{prefix}{key}:")
            lines.append(prefix + key + ":")

            lines.extend(
                _nested_dict_to_list(
                    d=value, indent=indent + 3, n_words=n_words
                )
            )
        else:
            if isinstance(value, str):
                value = _limit_words_in_line(text=value, n_words=n_words)

            if not isinstance(value, StStr):
                value_str: str | StStr = str(value)  # Convert value to string
            else:
                value_str = value

            if "\n" in value_str:
                # lines.append(f"{prefix}{key}:")
                lines.append(prefix + key + ":")

                lines.append(adjust_indent(value_str, prefix + " " * 2))
            else:
                # lines.append(f"{prefix}{key}: {value_str}")
                lines.append(prefix + key + ": " + value_str)

    return lines


def _limit_words_in_line(text: str, n_words: int):
    words = text.split()
    reconstructed = []
    current: list = []
    for word in words:
        if len(current) == n_words:
            # reconstructed.append(' '.join(current))
            reconstructed.append(join_ststr(current, separator=" "))

            current = []
        current.append(word)
    if current:
        # reconstructed.append(' '.join(current))
        reconstructed.append(join_ststr(current, separator=" "))

    # return '\n'.join(reconstructed)
    return join_ststr(reconstructed, separator="\n")


def add_horizontal_line_to_list_of_strings(
    strings: list[str], title: str | None = None
):
    # this function is just used for expressions trees for now

    # Find the widest line
    max_width = max(len(line) for s in strings for line in s.split("\n"))

    # Adjust max_width to accommodate the title if it's longer
    if title and len(title) > max_width:
        max_width = len(title)

    result = []

    # Add title if provided
    if title:
        result.append("=" * max_width)
        result.append(title.center(max_width))
        result.append("=" * max_width)

    # Add horizontal line after each string
    for s in strings:
        result.append(s)
        result.append("-" * max_width)

    return result
