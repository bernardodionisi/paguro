"""
Module for formatting output data in HTML. FROM polars:
https://github.com/pola-rs/polars/blob/main/py-polars/polars/dataframe/_html.py
"""

from __future__ import annotations

from itertools import groupby
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from types import TracebackType


class Tag:
    """Class for representing an HTML tag."""

    def __init__(
        self,
        elements: list[str],
        tag: str,
        attributes: dict[str, str] | None = None,
        attributes_list: list[str] | str | None = None,
    ):
        self.tag = tag
        self.elements = elements
        self.attributes = attributes

        if isinstance(attributes_list, str):
            attributes_list = [attributes_list]

        self.attributes_list = attributes_list

    def __enter__(self) -> None:
        if self.attributes is None and self.attributes_list is None:
            self.elements.append(f"<{self.tag}>")
        else:
            s = f"<{self.tag} "

            if self.attributes_list is not None:
                for v in self.attributes_list:
                    s += f"{v} "

            if self.attributes is not None:
                for k, v in self.attributes.items():
                    s += f'{k}="{v}" '

            s = f"{s.rstrip()}>"
            self.elements.append(s)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.elements.append(f"</{self.tag}>")


# ----------------------------------------------------------------------


class HTMLTableFormatter:
    """Class for creating and styling HTML summaries in table format."""

    def __init__(
        self,
        header_data: pl.DataFrame,
        body_data: pl.DataFrame,
    ) -> None:
        self.elements: list[str] = []

        self.header_rows: list[tuple[str, ...]] = header_data.rows()
        self.body_rows: list[tuple[str, ...]] = body_data.rows()

    def write_header(self) -> None:
        with Tag(self.elements, "thead"):
            for header in self.header_rows:
                with Tag(self.elements, "tr"):
                    for content, group in groupby(header):
                        count = len(list(group))

                        attributes = (
                            {"colspan": str(count)} if count > 1 else {}
                        )

                        with Tag(self.elements, "th", attributes):
                            self.elements.append(content)

    def write_body(self) -> None:
        for row in self.body_rows:
            with Tag(self.elements, "tr"):
                for cell in row:
                    with Tag(self.elements, "td"):
                        self.elements.append(cell)

    def write_style(self) -> None: ...

    def render_table(
        self,
        title: str | None = None,
        caption: str | None = None,
    ) -> list[str]:
        """Render the HTML table with optional title and caption."""
        with Tag(self.elements, "table"):
            if title:
                self.elements.insert(
                    0,
                    f"<caption style='caption-side: top; text-align: center;'>{title}</caption>",
                )

            self.write_header()
            with Tag(self.elements, "tbody"):
                self.write_body()

            if caption:
                self.elements.append(
                    f"<caption style='caption-side: bottom; text-align: left;'>{caption}</caption>"
                )

        # insert style at 0
        self.write_style()

        return self.elements


# ----------------------------------------------------------------------


class HTMLRegTable(HTMLTableFormatter):
    def __init__(self, header_data, body_data):
        super().__init__(header_data, body_data)

    def write_body(self):
        for i, row in enumerate(self.body_rows):
            with Tag(self.elements, "tr"):
                with Tag(self.elements, "td", {"class": "coef"}):
                    self.elements.append(row[0])

                for cell in row[1:]:
                    with Tag(self.elements, "td"):
                        self.elements.append(cell)

    def write_style(self):
        style = """
            table {
              border-collapse: separate;
                  border-spacing: 2px;
            }
        
            td {
              border: 1px solid transparent;
              padding: 2px; 
              text-align: center;
            }
        
            th {
              border: 1px solid #dddddd;
              padding: 8px;
              background-color: #f2f2f2;
              border-bottom: 2px solid #000; 
            }

            th:first-child {
                border-bottom: none; 
            }

            thead {
              border-bottom: 5px solid #000;
            }


            .coef {
              text-align: left; 
            }
            """

        self.elements.insert(0, f"<style>{style}</style>")
