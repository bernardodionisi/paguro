from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

import paguro as pg
from paguro.ashi.info.info_collection import InfoCollection
from paguro.validation.validation import Validation

if TYPE_CHECKING:
    from paguro.dataset.dataset import Dataset
    from paguro.dataset.lazydataset import LazyDataset


def from_metadata_to_paguro(
        data: pl.DataFrame | pl.LazyFrame,
        *,
        source: Any,
        metadata: dict[str, Any] | None,  # already deserialized
        class_must_be: Literal["Dataset", "LazyDataset"] | None,
        default_class: Literal["Dataset", "LazyDataset"],
) -> Dataset | LazyDataset:
    if metadata is None:
        if isinstance(data, pl.DataFrame):
            return from_no_metadata_to_paguro_dataset(source=source, data=data)
        elif isinstance(data, pl.LazyFrame):
            return from_no_metadata_to_paguro_lazydataset(source=source, data=data)
        else:
            raise TypeError(f"Invalid type: {type(data)}")
    else:
        return _from_metadata_to_paguro(
            data=data,
            metadata=metadata,
            class_must_be=class_must_be,
            default_class=default_class,
        )


def _from_metadata_to_paguro(
        data: pl.DataFrame | pl.LazyFrame,
        *,
        metadata: dict[str, Any],  # already deserialized
        class_must_be: Literal["Dataset", "LazyDataset"] | None,
        default_class: Literal["Dataset", "LazyDataset"],
        **params_if_none: Any,
) -> Dataset | LazyDataset:
    name = metadata.get("paguro", {}).get("class", {}).get("name")

    if class_must_be is not None:
        # allow to read LazyDataset and to scan Dataset
        if class_must_be == "LazyDataset" and name == "Dataset":
            class_must_be = None

        elif class_must_be == "Dataset" and name == "LazyDataset":
            class_must_be = None

        if name is not None and name != class_must_be:
            msg = (
                f"The metadata is for the class: '{name}'\n"
                f"You are trying to initialize '{class_must_be}'"
            )
            raise ValueError(msg)
        name = class_must_be
    else:
        if name is None:
            name = default_class
            # raise ValueError("Unable to determine what paguro object to initialize.")

    obj = getattr(pg, name)  # class in paguro

    params = _get_params(metadata=metadata, **params_if_none)

    instance = obj(data=data, **params)  # must be a Data* class

    attrs = _get_attrs(metadata=metadata)

    for attr_name, attr_value in attrs.items():
        if hasattr(instance, attr_name):
            # only if the instance has an attribute with that name,
            # in case paguro updates change
            # the attribute name
            setattr(instance, attr_name, attr_value)
        else:
            warnings.warn(
                f"Attribute '{attr_name}' not found in metadata",
                stacklevel=2,
            )

    return instance


# ----------------------------------------------------------------------


def from_no_metadata_to_paguro_dataset(
        source: Any,
        data: pl.DataFrame,
) -> Dataset:
    from paguro.dataset.dataset import Dataset

    out: Dataset = Dataset(data)

    if isinstance(source, (str, Path)):
        name = f"...{str(source)[-20:]}"
        out._name = name
    return out


def from_no_metadata_to_paguro_lazydataset(
        source: Any,
        data: pl.LazyFrame,
) -> LazyDataset:
    from paguro.dataset.lazydataset import LazyDataset

    out: LazyDataset = LazyDataset(data)

    if isinstance(source, (str, Path)):
        name = f"...{str(source)[-20:]}"
        out._name = name
    return out


# ----------------------------------------------------------------------


def _get_params(
        metadata: dict[str, Any],
        **params_if_none: Any,
) -> dict:
    params = metadata.get("paguro", {}).get("class", {}).get("params", {})

    for k, v in params_if_none.items():
        if params.get(k) is None:
            params[k] = v
        # else:
        #     warnings.warn(
        #         f"Parameter {k}, passed as 'params_if_none', "
        #         f"not found in metadata"
        #     )

    return params


def _get_attrs(metadata: dict[str, Any]) -> dict[str, Any]:
    out = {}

    attrs = metadata.get("paguro", {}).get("class", {}).get("attrs", {})

    for name, attr in attrs.items():
        if name == "info_list":
            if attr:
                attr = InfoList._from_dict_with_attributes(info_list_dict=attr)
                name = "info"  # the name of info_list is 'info' to make it nicer

        if name == "validation":
            if attr:  # validation=None
                attr = Validation._from_dict(source=attr)

        if attr:
            out[name] = attr

    return out
