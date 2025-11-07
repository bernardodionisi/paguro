from __future__ import annotations

from typing import Any


class Tag:
    """Context manager for building HTML tags safely."""

    def __init__(
        self,
        elements: list[str],
        tag: str,
        attributes: dict[str, str | None] | None = None,
    ):
        self.elements = elements
        self.tag = tag
        self.attributes = attributes or {}

    def __enter__(self) -> None:
        """Build and append the opening tag."""
        attr_string = ""
        if self.attributes:
            parts = []
            for k, v in self.attributes.items():
                if v is None:
                    # e.g. `open` for <details open>
                    parts.append(k)
                else:
                    parts.append(f'{k}="{v}"')
            attr_string = " " + " ".join(parts)

        self.elements.append(f"<{self.tag}{attr_string}>")

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Append the closing tag."""
        self.elements.append(f"</{self.tag}>")


class DictHTML:
    """
    Renders a Python dict as an interactive HTML outline (<details>/<summary>).
    - Each nested dictionary is shown in a <details> block.
    - Lists/tuples are rendered as <ul> with bullets.
    - Scalar values are placed in <div>.
    """

    def __init__(
        self,
        data: dict[str, Any],
        *,
        open_by_default: bool = True,
        base_margin: int = 20,
        add_controls: bool = False,
        open_button_text: str = "Open All",
        close_button_text: str = "Close All",
        embed_css: bool = True,
    ):
        """
        :param data: The dictionary to render.
        :param open_by_default: If True, each <details> tag starts opened.
        :param base_margin: Indentation (pixels) per nesting level.
        :param add_controls: If True, shows "Open All"/"Close All" buttons at top.
        :param open_button_text: Text for "Open All" button.
        :param close_button_text: Text for "Close All" button.
        :param embed_css: If True, embed a small <style> block for basic styling.
        """
        if not isinstance(data, dict):
            raise TypeError("Input must be a dictionary.")

        self.data = data
        self.open_by_default = open_by_default
        self.base_margin = base_margin

        self.add_controls = add_controls
        self.open_button_text = open_button_text
        self.close_button_text = close_button_text
        self.embed_css = embed_css

    def _repr_html_(self) -> str:
        """For IPython/Jupyter display."""
        return self.to_html()

    def to_html(self) -> str:
        """Build and return the entire HTML representation."""
        elements: list[str] = []

        # Optionally embed CSS
        if self.embed_css:
            elements.append(self._css_block())

        # Add a container div
        elements.append('<div class="dh-container">')

        # Optionally add the open/close controls
        if self.add_controls:
            self._build_controls(elements)

        # Build the top-level dictionary
        self._build_dict(self.data, level=0, elements=elements)

        elements.append("</div>")
        return "".join(elements)

    def _css_block(self) -> str:
        """Minimal CSS to keep the layout neat."""
        return """
<style>
.dh-container {
    margin: 0;
    padding: 0;
}
.dh-details {
    margin: 8px 0;
}
.dh-summary {
    cursor: pointer;
    font-weight: bold;
    margin-bottom: 4px;
    margin-left: var(--indent, 0px);
}
.dh-div {
    margin: 4px 0;
    margin-left: var(--indent, 0px);
    white-space: normal;
    word-wrap: break-word;
}
.dh-ul {
    list-style-position: outside;
    margin: 6px 0;
    margin-left: var(--indent, 0px);
    padding-left: 20px;
}
.dh-li {
    margin: 2px 0;
}
.dh-controls {
    margin-bottom: 8px;
}
.dh-button {
    margin-right: 8px;
    padding: 4px 8px;
}
</style>
        """

    def _build_controls(self, elements: list[str]) -> None:
        """
        Render "Open All"/"Close All" buttons at the top.
        """
        controls_html = f"""
<div class="dh-controls">
  <button class="dh-button" onclick="document.querySelectorAll('details').forEach(d => d.open = true)">
    {self.open_button_text}
  </button>
  <button class="dh-button" onclick="document.querySelectorAll('details').forEach(d => d.open = false)">
    {self.close_button_text}
  </button>
</div>
        """
        elements.append(controls_html)

    def _build_dict(
        self, data: dict[str, Any], level: int, elements: list[str]
    ) -> None:
        """
        Each key/value pair in this dictionary is placed into a <details> block:
        - <summary> displays the key.
        - Then we render the value inside that block.
        """
        for key, value in data.items():
            # Calculate the indentation based on level
            indent = self.base_margin * level

            # Apply indent using CSS custom property
            details_attrs: dict[str, str | None] = {
                "class": "dh-details",
                "style": f"--indent: {indent}px;",
            }
            if self.open_by_default:
                details_attrs["open"] = None

            with Tag(elements, "details", details_attrs):
                with Tag(elements, "summary", {"class": "dh-summary"}):
                    elements.append(str(key))

                self._build_value(value, level + 1, elements)

    def _build_value(
        self, value: Any, level: int, elements: list[str]
    ) -> None:
        """
        Decide how to render a value:
        - dict => recursively build that dictionary
        - list/tuple => build a <ul>
        - object with _repr_html_ => use that
        - else => plain string in a <div>
        """
        indent = self.base_margin * level

        if isinstance(value, dict):
            self._build_dict(value, level, elements)
        elif isinstance(value, (list, tuple)):
            with Tag(
                elements,
                "ul",
                {"class": "dh-ul", "style": f"--indent: {indent}px;"},
            ):
                for item in value:
                    with Tag(elements, "li", {"class": "dh-li"}):
                        if isinstance(
                            item, (dict, list, tuple)
                        ) or hasattr(item, "_repr_html_"):
                            self._build_value(item, level, elements)
                        else:
                            elements.append(str(item))
        elif hasattr(value, "_repr_html_"):
            with Tag(
                elements,
                "div",
                {"class": "dh-div", "style": f"--indent: {indent}px;"},
            ):
                elements.append(value._repr_html_())
        else:
            with Tag(
                elements,
                "div",
                {"class": "dh-div", "style": f"--indent: {indent}px;"},
            ):
                elements.append(str(value))


def transform_nested_dict(obj: Any) -> dict[str, Any] | list[str] | Any:
    """
    Recursively transform a nested dictionary so that the deepest dictionaries
    (i.e. those that do not contain any sub-dictionaries) are converted to lists
    of formatted strings, while preserving the structure of intermediate dictionaries.

    Used for representing the info in html

    Parameters
    ----------
    obj : Any
        The object to transform. If it is a dictionary, the function will inspect
        its values; otherwise, it is returned unchanged.

    Returns
    -------
    Union[Dict[str, Any], List[str], Any]
        The transformed object. If a dictionary has no sub-dictionaries,
        it is converted to a list of formatted strings; otherwise, its structure is preserved.
    """
    if isinstance(obj, dict):
        # Check if any value in the current dictionary is itself a dictionary.
        if any(isinstance(value, dict) for value in obj.values()):
            # At least one value is a dict, so preserve the structure and recurse.
            return {
                key: transform_nested_dict(value)
                for key, value in obj.items()
            }
        else:
            # This is a leaf dictionary: convert it to a list of formatted strings.
            return [
                f"<strong>{key}</strong>: {value}"
                for key, value in obj.items()
            ]
    else:
        # Not a dictionary; return the object as-is.
        return obj
