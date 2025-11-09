from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, cast
from types import EllipsisType
from io import StringIO
import polars as pl
from polars import selectors as cs
from polars._typing import IntoExpr

from paguro.utils.dependencies import _RICH_AVAILABLE

if TYPE_CHECKING:
    from paguro.utils.dependencies import pathlib


def insert_columns_where_ellipsis(
        exprs: tuple,
) -> tuple[IntoExpr, ...] | list[IntoExpr | cs.Selector]:
    # Normalize to a list we can iterate over
    if (
            len(exprs) == 1
            and isinstance(exprs[0], Iterable)
            and not isinstance(exprs[0], tuple)
    ):
        seq: list[IntoExpr | EllipsisType] = list(
            cast(Iterable[IntoExpr | EllipsisType], exprs[0])
        )
    else:
        seq = list(cast(Iterable[IntoExpr | EllipsisType], exprs))

    has_ellipsis = any(x is Ellipsis for x in seq)
    if not has_ellipsis:
        return cast(tuple[IntoExpr, ...], tuple(seq))

    missing_columns = _columns_where_ellipsis(seq)

    out: list[IntoExpr | cs.Selector] = []
    for expr in seq:
        if expr is Ellipsis:
            out.append(missing_columns)
        else:
            out.append(cast(IntoExpr, expr))
    return out


def _columns_where_ellipsis(
        exprs: Iterable[IntoExpr | cs.Selector | pl.Expr | str | EllipsisType],
) -> cs.Selector:
    columns = cs.all()

    for e in exprs:
        if e is Ellipsis:
            continue

        if isinstance(e, str):
            columns -= cs.matches(e)

        elif isinstance(e, cs.Selector):
            columns -= e

        elif isinstance(e, pl.Expr):
            for c in e.meta.root_names():
                columns -= cs.matches(c)

        else:
            # IntoExpr covers many atomic values (ints, floats, etc.). They don't
            # remove anything from `columns`, so we just ignore them.
            # Could  raise instead.
            pass

    return columns


# ----------------------------------------------------------------------


def write_text_to_svg(
        path: str | pathlib.Path | None,
        *,
        text: str,
        width: int,
        title: str | None,
        **kwargs: Any,
) -> None | str:
    if _RICH_AVAILABLE:
        from rich.console import Console
        from rich.text import Text
    else:
        msg = "'rich' must be installed to 'write_text_to_svg'"
        raise ImportError(msg)
    fake_stream = StringIO()

    console = Console(
        file=fake_stream,
        record=True,
        force_terminal=True,  # force color even if not a TTY
        color_system="truecolor",
        width=width,
    )
    rich_text = Text.from_ansi(str(text))
    console.print(rich_text)
    if title is None:
        title = ""

    if path is None:
        return console.export_svg(title=title, clear=True, **kwargs)
    else:
        console.save_svg(path=str(path), clear=True, title=title, **kwargs)
        return None


def write_text_to_html(
        path: str | pathlib.Path | None,
        *,
        text: str,
        width: int,
        color: str = "#000000",
        background: str = "#ffffff",
        **kwargs: Any,
) -> None | str:
    if _RICH_AVAILABLE:
        from rich.console import Console
        from rich.text import Text
    else:
        msg = "'rich' must be installed to 'write_text_to_html'"
        raise ImportError(msg)
    fake_stream = StringIO()

    console = Console(
        file=fake_stream,
        record=True,
        force_terminal=True,  # force color even if not a TTY
        color_system="truecolor",
        width=width,
    )
    rich_text = Text.from_ansi(str(text))
    console.print(rich_text)

    # if path is None:
    #     return console.export_html(clear=True, **kwargs)
    # else:
    #     console.save_html(path=str(path), clear=True, **kwargs)

    html = console.export_html(clear=True, **kwargs)
    html = html.replace(
        "</head>",
        "<style>body { " +
        f"color: {color} !important; background-color: {background} !important;" +
        " }</style></head>"
    )
    if path is not None:
        with open(path, "w", encoding="utf-8") as write_file:
            write_file.write(html)
        return None
    return html


def _write_data_repr_to_svg(
        data: pl.DataFrame,
        path: str | pathlib.Path | None = None,
        *,
        title: str | None = None,
        font_size: int = 20,
        line_height: int = 25,
) -> str | None:
    repr_ = data.__repr__()
    lines = repr_.strip("\n").splitlines()

    # Compute width/height dynamically
    max_line_length = max(len(line) for line in lines)
    extra_title_space = line_height if title else 0
    width = max_line_length * (font_size * 0.6) + 20
    height = len(lines) * line_height + 40 + extra_title_space

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg"',
        f' width="{int(width)}" '
        f' height="{int(height)}" '
        f' viewBox="0 0 {int(width)} {int(height)}">',
        f'  <text x="10" y="30"',
        f'        font-family="Menlo, SF Mono, Consolas, monospace"',
        f'        font-size="{font_size}"',
        f'        fill="#666"',
        f'        xml:space="preserve">'
    ]

    if title is not None:
        svg_lines.append(f'    <tspan x="10" dy="0" font-weight="bold">{title}</tspan>')
        svg_lines.append(f'    <tspan x="10" dy="{line_height / 2}"></tspan>')  # spacer

    # Then write the DataFrame lines
    for i, line in enumerate(lines):
        dy = 0 if (i == 0 and title is None) else line_height
        svg_lines.append(f'    <tspan x="10" dy="{dy}">{line}</tspan>')

    svg_lines.append("  </text>")
    svg_lines.append("</svg>")

    svg_string = "\n".join(svg_lines)

    if path is not None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg_string)
        return None
    return svg_string


# ----------------------------------------------------------------------

def pl_schema(data: pl.DataFrame | pl.LazyFrame) -> pl.Schema:
    if isinstance(data, pl.DataFrame):
        return data.schema
    return data.collect_schema()  # potentially expensive


def pl_column_names(data: pl.DataFrame | pl.LazyFrame) -> list[str]:
    if isinstance(data, pl.DataFrame):
        return data.columns
    return data.collect_schema().names()  # potentially expensive
