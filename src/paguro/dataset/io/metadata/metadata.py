from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import polars as pl

import paguro as pg
from paguro.dataset.io.metadata.serialize import serialize_dict_to_bytes, \
    serialize_dict_values_as_json
from paguro.shared.serialize import CustomJSONEncoder, CustomJSONDecoder
from paguro.utils.dependencies import json

if TYPE_CHECKING:
    from json import JSONEncoder


def to_paguro_dataset_metadata_serialized_key_value(
        class_name: Literal["Dataset", "LazyDataset"],
        attrs: dict[str, Any],  # all values must be serializable by json_encoder
        *,
        use_pyarrow_format: bool,
        json_encoder: type[JSONEncoder] | None = None,
) -> dict[str, str] | dict[bytes, bytes]:
    content = dataset_attrs_to_paguro_dict(
        class_name=class_name,
        attrs=attrs,
    )
    if use_pyarrow_format:
        return serialize_dict_to_bytes(
            data=content,
            json_encoder=json_encoder,
        )

    return serialize_dict_values_as_json(
        data=content,
        json_encoder=json_encoder,
    )


def dataset_attrs_to_paguro_dict(
        class_name: Literal["Dataset", "LazyDataset"],
        attrs: dict[str, Any],  # all values must be serializable by json_encoder
) -> dict[str, dict[str, Any]]:
    if class_name not in {"Dataset", "LazyDataset"}:
        msg = f"{class_name} must be a paguro.Dataset or paguro.LazyDataset."
        raise TypeError(msg)
    elif class_name == "LazyDataset":
        class_name = "Dataset"

    out = {
        "paguro": {
            "class": {
                "name": class_name,
                "attrs": attrs,
            },
            "versions": {
                "paguro": pg.__version__,
                "polars": pl.__version__,
            },
        }
    }
    return out


def _from_paguro_dataset_metadata_to_attrs(
        source: dict[str, str],
        *,
        json_decoder: type[JSONEncoder] | None = None,
) -> dict[str, Any]:
    if json_decoder is None:
        json_decoder = CustomJSONDecoder
    content = json.loads(source, cls=json_decoder)
    # validation and info need to be deserialized

    # out = {
    #     "paguro": {
    #         "class": {
    #             "name": class_name,
    #             "attrs": attrs,
    #         },
    #         "versions": {
    #             "paguro": pg.__version__,
    #             "polars": pl.__version__,
    #         },
    #     }
    # }
    content = content.get("paguro", {}).get("class", {})

    if not content:
        return {}

    class_name = content.get("name")
    if class_name is None:
        raise TypeError(f"'{content}' is not a paguro dataset.")

    attrs = content.get("attrs")
    if not attrs:
        return {}
    return attrs
