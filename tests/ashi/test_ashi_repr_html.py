from __future__ import annotations

import polars as pl
import pytest

from paguro.ashi.repr.html._html import HTMLRegTable, HTMLTableFormatter, Tag

# --------------------------
# Fixtures
# --------------------------

@pytest.fixture
def header_df():
    # Two header rows; first demonstrates consecutive-duplicate grouping for colspan
    # Row 1: A, A, B  -> <th colspan="2">A</th> then <th>B</th>
    # Row 2: X, Y, Y  -> <th>X</th> then <th colspan="2">Y</th>
    return pl.DataFrame(
        {
            "c1": ["A", "X"],
            "c2": ["A", "Y"],
            "c3": ["B", "Y"],
        }
    )


@pytest.fixture
def body_df():
    # Body of 3 rows by 3 cols
    return pl.DataFrame(
        {
            "c1": ["r1c1", "r2c1", "r3c1"],
            "c2": ["r1c2", "r2c2", "r3c2"],
            "c3": ["r1c3", "r2c3", "r3c3"],
        }
    )


# --------------------------
# Tag basics
# --------------------------

def test_tag_writes_open_and_close_with_attributes():
    elems = []
    with Tag(elems, "td", attributes={"class": "coef", "data-x": "1"}):
        elems.append("val")
    # Tag writes open/close as separate list entries
    assert elems[0].startswith('<td class="coef" data-x="1">')
    assert elems[-1] == "</td>"
    assert "".join(elems) == '<td class="coef" data-x="1">val</td>'


def test_tag_supports_attributes_list_and_string():
    elems1 = []
    with Tag(elems1, "th", attributes_list=["nowrap", "sortable"]):
        pass
    assert elems1 == ["<th nowrap sortable>", "</th>"]
    assert "".join(elems1) == "<th nowrap sortable></th>"

    elems2 = []
    with Tag(elems2, "th", attributes_list="nowrap"):
        pass
    assert elems2 == ["<th nowrap>", "</th>"]
    assert "".join(elems2) == "<th nowrap></th>"


def test_tag_without_attributes():
    elems = []
    with Tag(elems, "tr"):
        pass
    assert elems == ["<tr>", "</tr>"]


# --------------------------
# HTMLTableFormatter
# --------------------------

def test_write_header_colspan_grouping_only(header_df, body_df):
    fmt = HTMLTableFormatter(header_df, body_df)
    fmt.write_header()
    html = "".join(fmt.elements)

    # thead exists with two rows
    assert "<thead>" in html and "</thead>" in html
    assert html.count("<tr>") == 2  # only header rows written so far

    # Row 1 grouping: "A" occurs twice consecutively, then "B"
    assert '<th colspan="2">A</th>' in html
    assert "<th>B</th>" in html

    # Row 2 grouping: "X" once, "Y" twice consecutively
    assert "<th>X</th>" in html
    assert '<th colspan="2">Y</th>' in html


def test_write_body_rows_and_cells(header_df, body_df):
    fmt = HTMLTableFormatter(header_df, body_df)
    fmt.write_body()
    html = "".join(fmt.elements)

    # 3 body rows -> 3 <tr> entries
    assert html.count("<tr>") == 3
    # each row has 3 <td> cells
    assert html.count("<td>") == 9
    # spot-check a couple of cells
    assert "<td>r1c2</td>" in html
    assert "<td>r3c3</td>" in html


def test_render_table_wraps_sections_and_places_title_caption(header_df, body_df):
    fmt = HTMLTableFormatter(header_df, body_df)
    elements = fmt.render_table(title="My Table", caption="Notes go here")
    html = "".join(elements)

    # Table structure present
    assert "<table>" in html and "</table>" in html
    assert "<thead>" in html and "</thead>" in html
    assert "<tbody>" in html and "</tbody>" in html

    # Title (top caption) should be inserted at index 0
    assert elements[0].startswith("<caption")
    assert "caption-side: top" in elements[0]
    assert "My Table" in elements[0]

    # Bottom caption is appended (after tbody)
    assert "caption-side: bottom" in html
    assert "Notes go here" in html


def test_render_table_order_of_insertion(header_df, body_df):
    # Base class: write_style() is a no-op, so title stays at index 0.
    fmt = HTMLTableFormatter(header_df, body_df)
    elements = fmt.render_table(title="T", caption=None)

    # Title should be the very first element
    assert elements[0].startswith("<caption")
    assert "caption-side: top" in elements[0]
    assert "T" in elements[0]

    # Table markup should follow somewhere after
    html = "".join(elements)
    assert "<table>" in html and "</table>" in html
    assert "<thead>" in html and "</thead>" in html
    assert "<tbody>" in html and "</tbody>" in html

# --------------------------
# HTMLRegTable specialization
# --------------------------

def test_htmlregtable_first_cell_gets_coef_class(header_df, body_df):
    reg = HTMLRegTable(header_df, body_df)
    reg.write_body()
    html = "".join(reg.elements)

    # Each row should start with a <td class="coef">...</td>
    assert '<td class="coef">r1c1</td>' in html
    assert html.count('class="coef"') == 3  # 3 body rows

    # The rest of the row's cells are standard <td>
    assert "<td>r1c2</td>" in html and "<td>r1c3</td>" in html


def test_htmlregtable_style_is_inserted_first(header_df, body_df):
    reg = HTMLRegTable(header_df, body_df)
    elements = reg.render_table(title=None, caption=None)
    html = "".join(elements)

    # Style should be the very first element
    assert elements[0].startswith("<style>")
    assert "table {" in elements[0] and ".coef" in elements[0]

    # Ensure table and header/body are still present
    assert "<table>" in html and "</table>" in html
    assert "<thead>" in html and "</thead>" in html
    assert "<tbody>" in html and "</tbody>" in html


def test_htmlregtable_title_and_caption_with_style(header_df, body_df):
    reg = HTMLRegTable(header_df, body_df)
    elements = reg.render_table(title="TopTitle", caption="BottomNote")

    # Style should still be first
    assert elements[0].startswith("<style>")

    # Title should follow (index 1) because it was inserted before write_style ran
    # then table markup after that
    assert elements[1].startswith("<caption")
    assert "caption-side: top" in elements[1]
    assert "TopTitle" in elements[1]

    html = "".join(elements)
    assert "BottomNote" in html and "caption-side: bottom" in html

