from __future__ import annotations

import html


def html_repr_as_str(text: str):
    text = html.escape(text)
    return f'<pre style="white-space: pre;">{text}</pre>'
