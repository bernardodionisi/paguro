from __future__ import annotations

from io import StringIO
from typing import Any

import polars as pl

from paguro.utils.dependencies import json


def replace_expression_root(
        expr: pl.Expr,
        _new_root: str | pl.Expr,
) -> pl.Expr:
    json_str = _replace_expression_root(
        expr=expr,
        new_root=_new_root,
    )
    return pl.Expr.deserialize(StringIO(json_str), format="json")


# --------------------- pl.__version__ dependent -----------------------

def _replace_expression_root(expr: pl.Expr, new_root: str | pl.Expr, ) -> str:
    """
    Replace every *root reference* in `expr` with `new_root`.
    """
    expr_tree: Any = json.loads(expr.meta.serialize(format="json"))

    # Build subtree to graft in place of each root.
    replacement: Any = (
        {"Column": new_root}
        if isinstance(new_root, str)
        else json.loads(new_root.meta.serialize(format="json"))
    )

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            # Any "Selector" node is considered a root (Wildcard/ByName/ByDtype/ByRegex)
            if "Selector" in node:
                return replacement

            # A single-key {"Column": "..."} is a root
            if len(node) == 1 and "Column" in node:
                return replacement

            return {k: walk(v) for k, v in node.items()}

        if isinstance(node, list):
            return [walk(v) for v in node]

        if node == "Wildcard":
            return replacement

        return node

    replaced = walk(expr_tree)
    return json.dumps(replaced, separators=(",", ":"))
