from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from polars._typing import JoinStrategy
    from polars import DataFrame, LazyFrame


def get_other_rename_mapping(
        on: str | Iterable[object] | None,
        how: JoinStrategy,
        # ---
        # polars
        right_on: str | Iterable[object] | None,
        suffix: str,
        coalesce: bool,
        *,
        _warn: bool = True,
        left_data: DataFrame,
        right_data: DataFrame,
        new_data: DataFrame | LazyFrame,
        **_,
        # so we can just pass **kwargs, but we need to be careful that out parameters have the correct name
) -> dict[str, str]:
    """
    Determine {right_col -> renamed_col} for a standard DataFrame.join.

    Notes:
      - Left columns are never renamed by Polars.
      - Only right non-key columns that collide with left names can be renamed,
        and only when coalesce=False.
      - Right key columns (plain string names in `right_on` or `on`) are excluded
        from consideration because Polars doesn't include right keys in the output.
    """
    if how in ("semi", "anti") or coalesce:
        return {}

    # todo: warn the user in the calling methods that
    # the schema needs to be collected, or find a better way
    # it is probably not that frequent but useful to make explicit

    # we are not using this function for now, lets
    # figure out what arguments to add to the join for lazyframes
    # to allow the user to opt out before adding this

    left_columns = left_data.collect_schema().names()
    right_columns = right_data.collect_schema().names()
    new_columns = new_data.collect_schema().names()

    # Collect right-side key columns (strings only)
    right_key_cols = _gather_right_keys_for_standard_join(
        right_columns, on=on, right_on=right_on, warn=_warn,
    )

    return _compute_renamed_mapping(
        left_columns=left_columns,
        right_columns=right_columns,
        new_columns=new_columns,
        right_key_columns=right_key_cols,
        suffix=suffix,
    )


def get_other_rename_mapping_asof(
        left_data: DataFrame,
        right_data: DataFrame,
        new_data: DataFrame | LazyFrame,
        *,
        on: str | Iterable[object] | None,
        right_on: str | Iterable[object] | None,
        by: str | Iterable[object] | None,
        by_right: str | Iterable[object] | None,
        suffix: str = "_right",
        coalesce: bool = True,  # Polars join_asof default
        warn: bool = True,
) -> dict[str, str]:
    """
    Determine {right_col -> renamed_col} for DataFrame.join_asof.

    Differences vs standard join:
      - `by` / `by_right` also act as join keys (and thus are treated as right keys).
      - Default coalesce=True: typically yields {} unless caller sets coalesce=False.
      - `strategy`, `tolerance`, etc. do not affect naming.
    """
    # coalesce=True no renames occur
    if coalesce:
        return {}

    # todo: warn the user in the calling methods that
    # the schema needs to be collected, or find a better way
    # it is probably not that frequent but useful to make explicit

    # we are not using this function for now, lets
    # figure out what arguments to add to the join for lazyframes
    # to allow the user to opt out before adding this
    left_columns = left_data.collect_schema().names()
    right_columns = right_data.collect_schema().names()
    new_columns = new_data.collect_schema().names()

    right_key_cols = _gather_right_keys_for_asof_join(
        right_columns,
        on=on,
        right_on=right_on,
        by=by,
        by_right=by_right,
        warn=warn,
    )

    return _compute_renamed_mapping(
        left_columns=left_columns,
        right_columns=right_columns,
        new_columns=new_columns,
        right_key_columns=right_key_cols,
        suffix=suffix,
    )


# ----------------------------------------------------------------------


def _iter_items(val: str | Iterable[object] | None) -> list[object]:
    if val is None:
        return []
    if isinstance(val, str):
        return [val]
    try:
        return list(val)
    except TypeError:
        return [val]


def _gather_right_keys_for_standard_join(
        right_columns: Iterable[str],
        *,
        on: str | Iterable[object] | None,
        right_on: str | Iterable[object] | None,
        warn: bool,
) -> set[str]:
    """
    For standard equality join:
      - Prefer `right_on` if provided; otherwise fall back to `on`.
      - Only string items that actually exist among right columns are treated as keys.
      - Non-string/Expr-like items are ignored (with a warning).
    """
    right_all = set(right_columns or [])
    right_key_cols: set[str] = set()
    ignored_exprs = False

    def _gather(label: str, value: str | Iterable[object] | None) -> None:
        nonlocal ignored_exprs
        for item in _iter_items(value):
            if isinstance(item, str):
                if item in right_all:
                    right_key_cols.add(item)
                else:
                    if warn:
                        warnings.warn(
                            f"`{label}` includes '{item}', "
                            f"not found in right columns; "
                            "ignoring for right-key dropping.",
                            stacklevel=2,
                        )
            else:
                ignored_exprs = True

    if right_on is not None:
        _gather("right_on", right_on)
    elif on is not None:
        _gather("on", on)

    if ignored_exprs and warn:
        warnings.warn(
            "Ignored non-string (expression-like) "
            "join keys when determining renamed columns; "
            "only plain string column names cause right keys "
            "to be dropped from the output.",
            stacklevel=2,
        )

    return right_key_cols


def _gather_right_keys_for_asof_join(
        right_columns: Iterable[str],
        *,
        on: str | Iterable[object] | None,
        right_on: str | Iterable[object] | None,
        by: str | Iterable[object] | None,
        by_right: str | Iterable[object] | None,
        warn: bool,
) -> set[str]:
    """
    For asof join:
      - Prefer `right_on` if provided; otherwise fall back to `on`.
      - Also treat `by_right`/`by` as right-side keys.
      - Only string items that exist among right columns are treated as keys.
      - Non-string/Expr-like items are ignored (with a warning).
    """
    right_all = set(right_columns or [])
    right_key_cols: set[str] = set()
    ignored_exprs = False

    def _gather(label: str, value: str | Iterable[object] | None, ) -> None:
        nonlocal ignored_exprs
        for item in _iter_items(value):
            if isinstance(item, str):
                if item in right_all:
                    right_key_cols.add(item)
                else:
                    if warn:
                        warnings.warn(
                            f"`{label}` includes '{item}', "
                            f"not found in right columns; "
                            "ignoring for right-key dropping.",
                            stacklevel=2,
                        )
            else:
                ignored_exprs = True

    # Primary asof key
    if right_on is not None:
        _gather("right_on", right_on)
    elif on is not None:
        _gather("on", on)

    # Grouping keys
    if by_right is not None:
        _gather("by_right", by_right)
    elif by is not None:
        _gather("by", by)

    if ignored_exprs and warn:
        warnings.warn(
            "Ignored non-string (expression-like) "
            "join keys when determining renamed columns; "
            "only plain string column names cause right "
            "keys to be dropped from the output.",
            stacklevel=2,
        )

    return right_key_cols


def _compute_renamed_mapping(
        *,
        left_columns: Iterable[str],
        right_columns: Iterable[str],
        new_columns: Iterable[str],
        right_key_columns: Iterable[str],
        suffix: str,
) -> dict[str, str]:
    """
    Core engine: return {right_col -> right_col+suffix} for right non-key columns
    that collided with left and appear suffixed in the output.
    """
    left = set(left_columns or [])
    right_all = set(right_columns or [])
    out = set(new_columns or [])
    right_non_keys = right_all - set(right_key_columns or [])
    overlaps = right_non_keys & left
    renamed = {r for r in overlaps if f"{r}{suffix}" in out}
    return {r: f"{r}{suffix}" for r in renamed}
