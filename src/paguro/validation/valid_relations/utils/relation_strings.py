from __future__ import annotations

import re
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Final, Literal, TypedDict, cast

from collections.abc import Iterable

__all__ = [
    "parse_relationship_strings",
]

Relation = Literal["<", ">", "<>"]
TablePair = tuple[str, str]
ParsedPairsDict = dict[TablePair, list["RelationEntry"]]


class RelationshipParseError(ValueError):
    """Parsing error for relationship strings."""

    def __init__(
        self,
        message: str,
        *,
        pos: int | None = None,
        src: str | None = None,
    ) -> None:
        if pos is not None and src is not None:
            start = max(0, pos - 20)
            end = min(len(src), pos + 20)
            snippet = src[start:end]
            caret = " " * (pos - start) + "^"
            message = f"{message}\n…{snippet}\n…{caret}"
        super().__init__(message)


class RelationEntry(TypedDict):
    """Parsed relation entry oriented to the table-pair key.

    Notes
    -----
    - Dict is keyed by canonical pair ``(A, B)`` where ``A < B``.
    - ``left_on`` refers to columns on table ``A``; ``right_on`` on ``B``.
    - ``relation`` is oriented from ``A`` to ``B``. Opposite textual
      orientations (e.g., ``B[b] > A[a]``) are normalized.
    """

    left_on: list[str]
    right_on: list[str]
    relation: Relation


# Regexes (quoted & schema-qualified table names)

_IDENT: Final[str] = (
    r"(?:[A-Za-z_][A-Za-z0-9_]*|`[^`]+`|\"[^\"]+\"|'[^']+')"
)
_TABLE: Final[str] = rf"{_IDENT}(?:\.{_IDENT})*"

_NODE_RE: Final[re.Pattern[str]] = re.compile(
    rf"""
    \s*
    (?P<table>{_TABLE})
    \s*
    (?:
        \[
            \s*(?P<cols_bracket>[^\]]*?)\s*
        \]
      |
        \(
            \s*(?P<cols_paren>[^)]*?)\s*
        \)
    )
    \s*
    """,
    re.VERBOSE,
)

_OP_RE: Final[re.Pattern[str]] = re.compile(r"\s*(<>|<|>)\s*")
_ERR_SLICE: Final[int] = 40


# ----------------------------------------------------------------------


def _flip(rel: Relation) -> Relation:
    """Flip a relation (< ↔ >), keeping <> unchanged."""
    flipped: dict[Relation, Relation] = {"<": ">", ">": "<", "<>": "<>"}
    return flipped[rel]


def _unquote(name: str) -> str:
    if (
        len(name) >= 2
        and name[0] == name[-1]
        and name[0] in {"'", '"', "`"}
    ):
        return name[1:-1]
    return name


def _split_cols(
    src: str,
    *,
    allow_trailing_commas: bool = True,
) -> list[str]:
    cols: list[str] = []
    buf: list[str] = []
    q: str | None = None
    i = 0

    while i < len(src):
        ch = src[i]
        if q is None:
            if ch in ("'", '"'):
                q = ch
            elif ch == ",":
                token = "".join(buf).strip()
                cols.append(token)
                buf = []
            else:
                buf.append(ch)
        else:
            if ch == "\\" and i + 1 < len(src):
                buf.append(src[i + 1])
                i += 1
            elif ch == q:
                if i + 1 < len(src) and src[i + 1] == q:
                    buf.append(q)
                    i += 1
                else:
                    q = None
            else:
                buf.append(ch)
        i += 1

    if q is not None:
        raise RelationshipParseError("Unclosed quote in column list.")

    token = "".join(buf).strip()
    if token != "" or src.strip() != "":
        cols.append(token)

    cleaned: list[str] = []
    for idx, t in enumerate(cols):
        t = t.strip()
        if (
            t == ""
            and allow_trailing_commas
            and idx == len(cols) - 1
            and src.strip().endswith(",")
        ):
            # Allow a dangling comma at the end, e.g. "id,"
            continue
        if t and t[0] == t[-1] and t[0] in ("'", '"'):
            t = t[1:-1]
        if t == "":
            raise RelationshipParseError("Empty column name detected.")
        cleaned.append(t)
    return cleaned


def _parse_node(
    segment: str,
    *,
    allow_trailing_commas: bool,
) -> tuple[str, list[str]]:
    m = _NODE_RE.fullmatch(segment)
    if not m:
        raise RelationshipParseError(f"Malformed node: {segment!r}")
    table_raw = m.group("table")
    cols_raw = (
        m.group("cols_bracket")
        if m.group("cols_bracket") is not None
        else m.group("cols_paren")
    )
    table_parts = [
        _unquote(p) for p in re.findall(rf"{_IDENT}", table_raw)
    ]
    table = ".".join(table_parts)
    return table, _split_cols(
        cols_raw, allow_trailing_commas=allow_trailing_commas
    )


def _tokenize_chain(
    chain: str, *, allow_trailing_commas: bool
) -> tuple[list[tuple[str, list[str]]], list[Relation]]:
    nodes: list[tuple[str, list[str]]] = []
    ops: list[Relation] = []
    s = chain.strip()
    pos = 0

    m = _NODE_RE.match(s, pos)
    if not m:
        raise RelationshipParseError(
            f"Expected node at: {s[pos : pos + _ERR_SLICE]!r}",
            pos=pos,
            src=s,
        )
    nodes.append(
        _parse_node(
            m.group(0), allow_trailing_commas=allow_trailing_commas
        )
    )
    pos = m.end()

    while pos < len(s):
        mo = _OP_RE.match(s, pos)
        if not mo:
            raise RelationshipParseError(
                f"Expected relation operator "
                f"after position {pos}: {s[pos : pos + _ERR_SLICE]!r}",
                pos=pos,
                src=s,
            )
        # Regex guarantees one of "<", ">", "<>"
        ops.append(cast(Relation, mo.group(1)))
        pos = mo.end()

        mn = _NODE_RE.match(s, pos)
        if not mn:
            raise RelationshipParseError(
                f"Expected node after operator "
                f"at position {pos}: {s[pos : pos + _ERR_SLICE]!r}",
                pos=pos,
                src=s,
            )
        nodes.append(
            _parse_node(
                mn.group(0), allow_trailing_commas=allow_trailing_commas
            )
        )
        pos = mn.end()

    if len(ops) != len(nodes) - 1:
        raise RelationshipParseError("Operator/node count mismatch.")
    return nodes, ops


def _validate_pair_cols(
    left_cols: list[str],
    right_cols: list[str],
    left_tbl: str,
    right_tbl: str,
) -> None:
    if len(left_cols) != len(right_cols):
        raise RelationshipParseError(
            f"Column count mismatch between {left_tbl}[{', '.join(left_cols)}] "
            f"and {right_tbl}[{', '.join(right_cols)}]."
        )


# Coalescing policy that also considers textual orientation


@dataclass
class _Facts:
    rels: set[Relation] = field(default_factory=set)
    orient: set[bool] = field(default_factory=set)


def _finalize_bucket(
    entries_with_orient: list[tuple[RelationEntry, bool]],
) -> list[RelationEntry]:
    """
    Collapse duplicates and opposites per (left_on, right_on).

    We consider two signals for turning a mapping into '<>':
      1) Presence of both normalized relations '<' and '>' for the same columns.
      2) Presence of both textual orientations (left table was A at least once,
         and left table was B at least once) for the same columns — matching
         your test expectation.
      3) Presence of '<>' at all dominates everything.

    Parameters
    ----------
    entries_with_orient : list of (entry, left_is_a)
        left_is_a == True iff the original chain had the canonical left table
        (A in the (A,B) key) on the textual left side.
    """
    # Gather facts per column mapping
    facts: dict[tuple[tuple[str, ...], tuple[str, ...]], _Facts] = (
        defaultdict(_Facts)
    )

    for entry, left_is_a in entries_with_orient:
        key = (tuple(entry["left_on"]), tuple(entry["right_on"]))
        rec = facts[key]
        rec.rels.add(entry["relation"])
        rec.orient.add(bool(left_is_a))

    # Build collapsed list
    out: list[RelationEntry] = []
    order: dict[Relation, int] = {"<": 0, ">": 1, "<>": 2}

    for (lo, ro), rec in facts.items():
        lo_l, ro_l = list(lo), list(ro)

        if "<>" in rec.rels:
            out.append(
                {"left_on": lo_l, "right_on": ro_l, "relation": "<>"}
            )
            continue

        if "<" in rec.rels and ">" in rec.rels:
            out.append(
                {"left_on": lo_l, "right_on": ro_l, "relation": "<>"}
            )
            continue

        if len(rec.orient) == 2:
            # Saw both textual orientations (A … B and B … A) -> treat as bidirectional
            out.append(
                {"left_on": lo_l, "right_on": ro_l, "relation": "<>"}
            )
            continue

        # Otherwise keep the single remaining relation deterministically
        (only_rel,) = tuple(sorted(rec.rels, key=lambda r: order[r]))
        out.append(
            {"left_on": lo_l, "right_on": ro_l, "relation": only_rel}
        )
    return out


# ----------------------------------------------------------------------


def parse_relationship_strings(
    inputs: str | Iterable[str],
    *,
    allow_trailing_commas: bool = True,
    deterministic: bool = True,
) -> ParsedPairsDict:
    """Parse one or more relationship strings into canonical table pairs."""
    inputs_iter: Iterable[str] = (
        [inputs] if isinstance(inputs, str) else (inputs or [])
    )

    # Internally collect entries with an orientation flag:
    #   left_is_a == True iff textual left table == canonical A
    buckets: defaultdict[TablePair, list[tuple[RelationEntry, bool]]] = (
        defaultdict(list)
    )

    for raw in inputs_iter:
        s = raw.strip()
        if not s:
            continue

        nodes, ops = _tokenize_chain(
            s, allow_trailing_commas=allow_trailing_commas
        )

        for i, rel in enumerate(ops):
            left_tbl, left_cols = nodes[i]
            right_tbl, right_cols = nodes[i + 1]
            _validate_pair_cols(left_cols, right_cols, left_tbl, right_tbl)

            a, b = sorted((left_tbl, right_tbl))
            key: TablePair = (a, b)
            left_is_a = left_tbl == a

            # Normalize columns to (A,B) order; orient relation to A->B
            if left_is_a:
                entry: RelationEntry = {
                    "left_on": list(left_cols),  # columns on A
                    "right_on": list(right_cols),  # columns on B
                    "relation": rel,  # as seen (already A->B)
                }
            else:
                entry = {
                    "left_on": list(right_cols),  # columns on A
                    "right_on": list(left_cols),  # columns on B
                    "relation": _flip(rel),  # flip to A->B orientation
                }

            # DO NOT dedupe here; we need to know if we ever saw both orientations.
            buckets[key].append((entry, left_is_a))

    # Coalesce & dedupe per bucket, then (optionally) sort deterministically
    out: ParsedPairsDict = {}
    for k, v in buckets.items():
        collapsed = _finalize_bucket(v)
        if deterministic and collapsed:
            collapsed = sorted(
                collapsed,
                key=lambda e: (
                    e["relation"],
                    tuple(e["left_on"]),
                    tuple(e["right_on"]),
                ),
            )
        out[k] = collapsed

    # Sort keys deterministically if requested
    if deterministic and out:
        out = {k: out[k] for k in sorted(out.keys())}

    return out


if __name__ == "__main__":
    examples = [
        "customers['id'] < orders[id] <> shipments[id]",
        "orders[id] <> shipments[id]",
        "customers('id','name') < orders[\"id\", order_name] > shipments(id, 'n')",
        "A('last, first', code) <> B(\"last, first\", code)",
        "customers[id, name] < orders[id, order_name]",
        "B[b] < A[a]",  # reversed -> normalize to ('A','B') with '>'
        "`My Schema`.`Tbl`(k1) > 'Other'.\"Tbl\"(k1)",  # quoted + qualified
        "A[id,] < B[id, ]",  # trailing commas accepted
        "X[a] < Y[a]",
        "Y[a] < X[a]",
        "Y[a] <> X[a]",  # opposites collapse to '<>'
    ]
    from pprint import pprint
    from time import perf_counter

    start = perf_counter()
    out = parse_relationship_strings(examples)
    end = perf_counter()
    print(f"Time: {end - start}")
    pprint(out)
