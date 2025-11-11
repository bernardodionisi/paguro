from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

from polars.selectors import Selector

if TYPE_CHECKING:
    from paguro.validation.validation import Validation
    from paguro.validation.valid_column.valid_column import ValidColumn
    from paguro.validation.valid_frame.valid_frame import ValidFrame


def _iter_validation_dfs(
        root: Validation | None,
        *,
        include_transformed_frames: bool,
        include_fields: bool,
):
    """
    Yield all Validation nodes reachable from `root`, walking through:
      - frame edges:   ValidFrame._validators
      - field edges:   ValidColumn._fields (if include_fields=True)

    Each Validation / ValidFrame visited at most once (cycle-safe).
    """
    if root is None:
        return
    stack: list[Validation] = [root]
    seen_v: set[int] = set()
    seen_vf: set[int] = set()

    while stack:
        v = stack.pop()
        vid = id(v)
        if vid in seen_v:
            continue
        seen_v.add(vid)
        yield v  # expose the node to callers

        # 1) Traverse via frames
        vfl = v._valid_frame_list
        if vfl:
            for vf in vfl:
                if not include_transformed_frames and vf._transform is not None:
                    continue
                vfid = id(vf)
                if vfid in seen_vf:
                    continue
                seen_vf.add(vfid)
                nxt = vf._validators
                if nxt is not None:
                    stack.append(nxt)

        # 2) Traverse via column fields (nested schemas)
        if include_fields:
            vcl = v._valid_column_list
            if vcl:
                for vc in vcl:
                    child = getattr(vc, "_fields", None)  # Validation | None
                    if child is not None:
                        stack.append(child)


def find_unique_vcol(
        root: Validation | None,
        *,
        name: str | None | Selector,
        include_transformed_frames: bool,
        include_fields: bool,
        return_first: bool,
) -> ValidColumn | None:
    """
    Search the graph (frames + optionally column fields) for columns named `name`.

    - return_first=False:
        * Return the column if exactly one is found; None if none.
        * Raise ValueError on the 2nd match (early exit).
    - return_first=True:
        * Return first match encountered; None if none. Never raises for duplicates.
    """
    found: ValidColumn | None = None
    for v in _iter_validation_dfs(
            root,
            include_transformed_frames=include_transformed_frames,
            include_fields=include_fields,
    ):
        vcl = v._valid_column_list
        if not vcl:
            continue
        for vc in vcl:
            if _names_equal(vc._name, name):
                if return_first:
                    return vc
                if found is None:
                    found = vc
                else:
                    raise ValueError(f"Multiple vcol(s) named {name!r} found.")
    return found


def find_all_vcols_multi(
        root: Validation | None,
        *,
        names: Iterable[str | None | Selector],
        include_transformed_frames: bool,
        include_fields: bool,
        dedupe: bool,
        group_by_name: bool,
) -> list[ValidColumn] | dict[str | None | Selector, list[ValidColumn]]:
    """
    Return all ValidColumns whose name matches ANY of `names`
    (traversing frames + optionally column fields).
    """
    if root is None:
        return {} if group_by_name else []

    # Preprocess targets for fast checks
    str_targets: set[str] = set()
    include_none = False
    selector_targets: list[Selector] = []
    original_order: list[str | None | Selector] = []

    for nm in names:
        original_order.append(nm)
        if isinstance(nm, str):
            str_targets.add(nm)
        elif nm is None:
            include_none = True
        else:
            selector_targets.append(nm)

    def _matches_any(col_name: str | None | Selector) -> bool:
        if isinstance(col_name, str):
            return col_name in str_targets
        if col_name is None:
            return include_none
        for tgt in selector_targets:
            if _names_equal(col_name, tgt):
                return True
        return False

    def _group_key_for(col_name: str | None | Selector) -> str | None | Selector:
        if isinstance(col_name, str) and col_name in str_targets:
            return col_name
        if col_name is None and include_none:
            return None
        for nm in original_order:
            if _names_equal(col_name, nm):
                return nm
        return col_name  # fallback (shouldn't happen)

    seen_cols: set[int] | None = set() if dedupe else None
    if group_by_name:
        grouped: dict[str | None | Selector, list[ValidColumn]] = {}
    else:
        out: list[ValidColumn] = []

    for v in _iter_validation_dfs(
            root,
            include_transformed_frames=include_transformed_frames,
            include_fields=include_fields,
    ):
        vcl = v._valid_column_list
        if not vcl:
            continue

        if group_by_name:
            if seen_cols is None:
                for vc in vcl:
                    cname = vc._name
                    if _matches_any(cname):
                        grouped.setdefault(_group_key_for(cname), []).append(vc)
            else:
                for vc in vcl:
                    cname = vc._name
                    if _matches_any(cname):
                        vcid = id(vc)
                        if vcid not in seen_cols:
                            seen_cols.add(vcid)
                            grouped.setdefault(_group_key_for(cname), []).append(vc)
        else:
            if seen_cols is None:
                for vc in vcl:
                    if _matches_any(vc._name):
                        out.append(vc)
            else:
                for vc in vcl:
                    if _matches_any(vc._name):
                        vcid = id(vc)
                        if vcid not in seen_cols:
                            seen_cols.add(vcid)
                            out.append(vc)

    return grouped if group_by_name else out


def find_unique_vframe(
        root: Validation | None,
        *,
        name: str | None,
        include_transformed_frames: bool,
        include_fields: bool,
        return_first: bool,
) -> ValidFrame | None:
    """
    Return the unique ValidFrame with `_name == name` across the graph.
    Traversal includes frames; if `include_fields=True`, also descends into
    column-field validations to reach more frames nested under structs.
    """
    found: ValidFrame | None = None
    # We must both *collect* matching frames and *traverse* through frames/fields.
    visited_frames: set[int] = set()

    for v in _iter_validation_dfs(
            root,
            include_transformed_frames=include_transformed_frames,
            include_fields=include_fields,
    ):
        vfl = v._valid_frame_list
        if not vfl:
            continue

        for vf in vfl:
            if not include_transformed_frames and vf._transform is not None:
                continue
            fid = id(vf)
            if fid in visited_frames:
                continue
            visited_frames.add(fid)

            if vf._name == name:
                if return_first:
                    return vf
                if found is None:
                    found = vf
                else:
                    raise ValueError(f"Multiple vframe(s) named {name!r} found.")

    return found


def find_all_vframes_multi(
        root: Validation | None,
        *,
        names: Iterable[str | None],
        include_transformed_frames: bool,
        include_fields: bool,
        dedupe: bool,
        group_by_name: bool,
) -> list[ValidFrame] | dict[str | None, list[ValidFrame]]:
    """
    Return all ValidFrames whose `_name` is in `names`, traversing frames and,
    if `include_fields=True`, nested column-field validations.
    """
    if root is None:
        return {} if group_by_name else []

    str_targets: set[str] = set()
    include_none = False
    for nm in names:
        if nm is None:
            include_none = True
        else:
            str_targets.add(nm)

    def _matches(frame_name: str | None) -> bool:
        return (frame_name is None and include_none) or (
                isinstance(frame_name, str) and frame_name in str_targets
        )

    seen_frames: set[int] | None = set() if dedupe else None
    if group_by_name:
        grouped: dict[str | None, list[ValidFrame]] = {}
    else:
        out: list[ValidFrame] = []

    for v in _iter_validation_dfs(
            root,
            include_transformed_frames=include_transformed_frames,
            include_fields=include_fields,
    ):
        vfl = v._valid_frame_list
        if not vfl:
            continue

        for vf in vfl:
            if not include_transformed_frames and vf._transform is not None:
                continue
            if not _matches(vf._name):
                # still traverse through it (already handled by DFS helper)
                continue

            if seen_frames is None:
                if group_by_name:
                    grouped.setdefault(vf._name, []).append(vf)
                else:
                    out.append(vf)
            else:
                fid = id(vf)
                if fid not in seen_frames:
                    seen_frames.add(fid)
                    if group_by_name:
                        grouped.setdefault(vf._name, []).append(vf)
                    else:
                        out.append(vf)

    return grouped if group_by_name else out


# ----------------------------------------------------------------------

def _names_equal(
        name_1: str | None | Selector,
        name_2: str | None | Selector,
) -> bool:
    if isinstance(name_1, str) and isinstance(name_2, str):
        return name_1 == name_2
    elif name_1 is None and name_2 is None:
        return True
    elif isinstance(name_1, Selector) and isinstance(name_2, Selector):
        return name_1.meta.eq(name_2)
    return False
