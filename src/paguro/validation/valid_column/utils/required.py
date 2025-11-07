from __future__ import annotations

from paguro.utils.dependencies import re

from typing import Collection, Callable


def _normalize(s: str) -> str:
    s = s.casefold().strip()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def _suggest_columns(
        column_name: str,
        columns: Collection[str],
        *,
        n: int = 5,
        min_score: float | None = None,
) -> list[str]:
    # todo: precompute normalized column names, maybe
    cols = list(columns)
    if n <= 0 or not cols:
        return []

    q_norm = _normalize(column_name)
    for c in cols:
        if _normalize(c) == q_norm:
            return []

    try:
        from rapidfuzz import process, fuzz
        cutoff = 75.0 if min_score is None else float(min_score)
        processor: Callable[[str], str] = _normalize

        results = process.extract(  # type: ignore
            column_name,
            cols,
            scorer=fuzz.WRatio,
            processor=processor,
            limit=n,
            score_cutoff=cutoff,
            # applies as minimal similarity since WRatio is normalized
        )
        return [choice for (choice, _score, _idx) in results]

    except Exception:
        import difflib
        cutoff = 0.6 if min_score is None else float(min_score)

        norm_cols = [_normalize(c) for c in cols]
        first_idx = {}
        for i, v in enumerate(norm_cols):
            if v not in first_idx:
                first_idx[v] = i

        norm_matches = difflib.get_close_matches(
            q_norm,
            norm_cols,
            n=n,
            cutoff=cutoff,
        )
        return [
            cols[first_idx[nm]] for nm in norm_matches
        ]
