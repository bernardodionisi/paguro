from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from paguro.dataset.dataset import Dataset
    from paguro.dataset.lazydataset import LazyDataset

    from paguro.typing import IntoValidation


# ----------------------------------------------------------------------


def custom_serializer_for_repr(
        obj: Any,
) -> str:
    if not isinstance(obj, (int, str, float, list)):
        return str(obj)
    msg = f"Type {type(obj)} not serializable"
    raise TypeError(msg)


def data_to_lazyframe(
        data: IntoValidation,
) -> pl.LazyFrame:
    if isinstance(data, (pl.DataFrame, pl.LazyFrame)):
        return data.lazy()
    elif isinstance(data, pl.Series):
        return pl.LazyFrame(data)

    from paguro.dataset.dataset import Dataset
    from paguro.dataset.lazydataset import LazyDataset

    if isinstance(data, Dataset):
        return data.to_lazyframe()
    if isinstance(data, LazyDataset):
        return data.to_lazyframe()

    else:  # FrameInitTypes
        return pl.LazyFrame(data)


def data_to_frame_like(
        data: IntoValidation,
) -> pl.LazyFrame | pl.DataFrame | Dataset | LazyDataset:
    if isinstance(data, (pl.DataFrame, pl.LazyFrame)):
        return data
    elif isinstance(data, pl.Series):
        return pl.DataFrame(data)

    from paguro.dataset.dataset import Dataset
    from paguro.dataset.lazydataset import LazyDataset

    if isinstance(data, Dataset):
        return data
    if isinstance(data, LazyDataset):
        return data

    else:  # FrameInitTypes
        return pl.DataFrame(data)


# ----------------------------------------------------------------------

def _fixed_info_mapping(
        *,
        title: str | None,
        description: str | None,
        constraints: dict[str, Any] | None,
        mapping: dict[str, Any],
        valid_constraints: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Split incoming fields into:
      - fixed_mapping: {title?, description?, constraints?}
      - free_mapping: everything else that is not a constraint


    - Validates explicit `constraints` keys.
    - Places any keys from `mapping` that match a `valid_constraints` name
      into the `constraints` dict unless already provided there.
      (maybe should be configurable)
    - Returns a *new* (fixed_mapping, free_mapping) without mutating inputs.
    """
    # Work on copies to keep function pure
    free_mapping: dict[str, Any] = copy.deepcopy(mapping)
    cons: dict[str, Any] = dict(constraints) if constraints else {}

    # Validate that the info constraints keys are valid constraints
    invalid = [k for k in cons.keys() if k not in valid_constraints]
    if invalid:
        raise TypeError(
            "Invalid constraints names: "
            + ", ".join(sorted(invalid))
            + "\nValid constraints names you can use are: "
            + ", ".join(sorted(valid_constraints))
            + "\n"
        )

    # Place constraint keys from free mapping if not already set
    to_move = {
        k: free_mapping[k]
        for k in free_mapping.keys()
        if k in valid_constraints and k not in cons
    }
    if to_move:
        cons.update(to_move)
        for k in to_move:
            free_mapping.pop(k)

    # Build fixed section
    fixed_mapping: dict[str, Any] = {}
    if title is not None:
        fixed_mapping["title"] = title
    if description is not None:
        fixed_mapping["description"] = description
    if cons:
        fixed_mapping["constraints"] = cons

    return fixed_mapping, free_mapping

# ----------------------------------------------------------------------
