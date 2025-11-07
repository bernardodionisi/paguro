from __future__ import annotations

from typing import Callable, Mapping, TYPE_CHECKING, Union

from paguro.validation.shared.find_v import _iter_validation_dfs

if TYPE_CHECKING:
    from paguro.validation.validation import Validation

MappingOrFunc = Union[Mapping[str, str], Callable[[str], str]]

def rename_valid_columns(
    validation: Validation | None,
    mapping: MappingOrFunc,
    *,
    include_transformed_frames: bool = False,
    include_fields: bool = False,
) -> Validation | None:
    """
    In-place rename of ValidColumn._name (strings only) while traversing the graph.

    - If a rename is required but vc._allow_rename is False, raise immediately.
    - Mapping[str, str]: only keys present are candidates (old -> new).
    - Callable[[str], str]: called for every string name; returning the same
      name means "no change".
    """
    if validation is None:
        return None

    # Fast-path normalization
    if callable(mapping):
        map_fn: Callable[[str], str] = mapping  # type: ignore[assignment]
        use_func = True
        map_dict: Mapping[str, str] | None = None
    elif isinstance(mapping, Mapping):
        map_fn = None  # type: ignore[assignment]
        use_func = False
        map_dict = mapping
    else:
        raise TypeError("mapping must be a Mapping[str, str] or Callable[[str], str]")

    for v in _iter_validation_dfs(
        validation,
        include_transformed_frames=include_transformed_frames,
        include_fields=include_fields,
    ):
        vcl = v._valid_column_list
        if not vcl:
            continue

        for vc in vcl:
            old = vc._name
            if not isinstance(old, str):
                continue  # only rename explicit string names

            # Compute new name
            if use_func:
                new = map_fn(old)  # type: ignore[misc]
                if not isinstance(new, str):
                    raise TypeError(
                        f"mapping produced non-str for {old!r}: {type(new).__name__}"
                    )
                if new == old:
                    continue  # no-op
            else:
                # dict mapping: only rename if key exists
                # (also allows mapping to the same value -> no-op)
                if old not in map_dict:  # type: ignore[operator]
                    continue
                new = map_dict[old]  # type: ignore[index]
                if not isinstance(new, str):
                    raise TypeError(
                        f"mapping produced non-str for {old!r}: {type(new).__name__}"
                    )
                if new == old:
                    continue  # no-op

            if not getattr(vc, "_allow_rename", False):
                raise ValueError(f"Cannot rename column {old!r}: _allow_rename is False")

            if new == "":
                raise ValueError(f"mapping produced empty name for {old!r}")

            # Mutate in place
            vc._name = new

    return validation
