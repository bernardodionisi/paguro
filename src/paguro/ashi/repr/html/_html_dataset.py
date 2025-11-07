from __future__ import annotations

from paguro.ashi.repr.html._html import Tag

DATASET_STYLE = """
<style>
.indented-content {
    margin-top: 0;
    padding-top: 0;
    padding-bottom: 1px;
    padding-left: 10px;
}

h4 {
    margin-bottom: 1px;
}
           
details {
    /*  border: 1px solid #aaa;*/
    /*  border-radius: 4px;*/
    padding: 0.5em 0.5em 0;
}

summary {
    font-weight: bold;
    margin: -0.5em -0.5em 0;
    padding: 0.5em;
}

details[open] summary {
    /*	  border-top: 1px solid #aaa;*/

    border-bottom: 1px solid #aaa;
    margin-bottom: 0.5em;
}

/* Base style for all table cells and headers */
table.dataframe th, table.dataframe td {
    background-color: white; /* White background for all cells and headers */
    padding: 8px; /* Adequate padding for content */
    text-align: left; /* Align text to the left */
}

/* Zebra striping for odd rows */
table.dataframe tr:nth-child(odd) td {
    background-color: #f9f9f9; /* Light gray background for cells in odd rows */
}

/* Hover effect for all rows */
table.dataframe tr:hover td {
    background-color: #ddd; /* Darker gray background on hover for cells in any row */
}

/* Ensure the dataframe is contained within the details window with scrolling */
.dataframe-container {
    width: 100%;
    overflow-x: auto;
}

.styled-box {
    background-color: transparent;
    border-radius: 15px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    padding: 10px;
    overflow-x: auto;
    text-align: left;
}
</style>
"""


def dict_to_html(rows: dict, top_name: str) -> str:
    elements: list = []

    with Tag(elements, "div", {"class": "styled-box"}):
        elements.append(DATASET_STYLE)

        with Tag(elements, "h4"):
            elements.append(top_name)

        # TODO: insert name of dataset, is modified ...

        with Tag(elements, "div", {"class": "indented-content"}):
            for i, (k, v) in enumerate(rows.items()):
                # TODO: add config to set the repr open or closed
                # attributes_list = "closed" if isinstance(v, dict) else "open"
                attributes_list = "closed"

                # just for documentation
                if isinstance(v, dict):
                    attributes_details = {
                        "class": "dataset-info admonition"
                    }
                else:
                    attributes_details = {
                        "class": "dataset-repr admonition"
                    }

                # a detail for each
                with Tag(
                    elements,
                    "details",
                    attributes_list=attributes_list,
                    attributes=attributes_details,
                ):
                    # just for documentation
                    attributes_summary = {"class": "admonition-title"}

                    with Tag(
                        elements, "summary", attributes=attributes_summary
                    ):
                        if k is None:
                            k = ""
                        elements.append(k)

                    with Tag(
                        elements, "div", {"class": "indented-content"}
                    ):
                        if isinstance(v, dict):
                            elements.append(_dict_to_html(v))

                        elif hasattr(v, "_repr_html_"):
                            # Wrap the HTML representation in the container div for scrolling
                            with Tag(
                                elements,
                                "div",
                                {"class": "dataframe-container"},
                            ):
                                elements.append(v._repr_html_())
                            # elements.append(v._repr_html_())

                        elif hasattr(v, "__repr__"):
                            elements.append(repr(v))

                        else:
                            elements.append(str(v))

    return "".join(elements)


def _dict_to_html(d, level=0, max_level=None):
    """
    Recursively converts a nested dictionary into an HTML list string.
    Dynamically generates CSS styles for indentation based on the maximum nesting level.

    :param d: The dictionary to convert.
    :param level: The current nesting level.
    :param max_level: The maximum nesting level found in the dictionary.
    :return: A string containing the HTML representation of the dictionary.
    """
    # Calculate max_level on the first call
    if max_level is None:
        max_level = find_max_depth(d)

    # Generate CSS styles
    css_styles = "<style>\n"
    for i in range(max_level):
        indent = 10 + i * 2
        css_styles += f".level-{i} {{ padding-left: {indent}px; }}\n"
    css_styles += "</style>\n"

    # Generate HTML
    # html = '<ul style="margin: 0;padding: 0;">'
    html = "<ul>"
    for key, value in d.items():
        html += f'<li><strong>{key}</strong><div class="level-{level}">'
        if isinstance(value, dict):
            # Recursive call for nested dictionaries
            html += _dict_to_html(value, level + 1, max_level)
        else:
            # Apply a class for indentation
            html += f"{value}</div>"
        html += "</li>"
    html += "</ul>"

    return css_styles + html if level == 0 else html


def find_max_depth(d, level=0):
    """
    Finds the maximum depth of a nested dictionary.

    :param d: The dictionary to check.
    :param level: The current level in the dictionary.
    :return: The maximum depth as an integer.
    """
    if not isinstance(d, dict) or not d:
        return level
    return max(find_max_depth(v, level + 1) for k, v in d.items())
