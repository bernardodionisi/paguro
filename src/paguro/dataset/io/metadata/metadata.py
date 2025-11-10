from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, IO, Iterable, TypedDict

import polars as pl

import paguro as pg
from paguro.ashi.info.info_collection import InfoCollection
from paguro.dataset.io.metadata.serialize import serialize_dict_to_bytes, \
    serialize_dict_values_as_json
from paguro.shared.serialize import CustomJSONDecoder
from paguro.utils.dependencies import json
from paguro.validation.validation import Validation

if TYPE_CHECKING:
    from json import JSONEncoder
    from pathlib import Path
    from polars import CredentialProviderFunction


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
) -> dict[str, Any]:
    if class_name not in {"Dataset", "LazyDataset"}:
        msg = f"{class_name} must be a paguro.Dataset or paguro.LazyDataset."
        raise TypeError(msg)
    elif class_name == "LazyDataset":
        class_name = "Dataset"

    out = {
        "paguro": {
            "_class": {
                "name": class_name,
                "attrs": attrs,
            },
            "version": {
                "paguro": pg.__version__,
                "polars": pl.__version__,
            },
        }
    }
    return out


class DeserializedPaguroInnerMetadata(TypedDict, total=False):
    name: str | None
    validation: Validation | None
    info: InfoCollection | None


def _deserialize_paguro_metadata(
        paguro_metadata: dict[str, str],
        _schema_keys_for_info: Iterable[str] | None,
) -> DeserializedPaguroInnerMetadata:
    _meta: str | None = paguro_metadata.get("paguro", None)

    if not _meta:
        return DeserializedPaguroInnerMetadata()

    content: dict[str, Any] = json.loads(_meta, cls=CustomJSONDecoder)

    _class: dict[str, Any] = content.get("_class", {})

    if (
            class_name := _class.get("name")
    ) not in ("Dataset", "LazyDataset"):
        raise TypeError(f"The metadata is for {class_name} and not a paguro dataset.")

    # validation and info still need to be deserialized

    # info/validation/name

    _attrs: dict[str, Any] = _class.get("attrs", {})

    _name: str | None = _attrs.pop("name")

    if not _attrs:
        return DeserializedPaguroInnerMetadata()

    # ---------

    # validation and info still need deserialization

    # ---------
    _validation: str | None = _attrs.pop("validation", None)
    if _validation is not None:
        validation = Validation.deserialize(_validation)
    else:
        validation = None
    # ---------

    _info: str | None = _attrs.pop("info")
    if _info is not None:
        info = InfoCollection._deserialize(
            _info,
            schema_keys=_schema_keys_for_info
        )
    else:
        info = None

    return DeserializedPaguroInnerMetadata(
        name=_name,
        info=info,
        validation=validation,
    )


def read_parquet_paguro_metadata(
        source: str | Path | IO[bytes] | bytes,
        storage_options: dict[str, Any] | None = None,
        credential_provider: (
                CredentialProviderFunction
                | Literal['auto'] | None
        ) = 'auto',
        retries: int = 2,
) -> dict[str, str] | None:
    metadata = pl.read_parquet_metadata(
        source=source,
        storage_options=storage_options,
        credential_provider=credential_provider,
        retries=retries
    )
    paguro_metadata = metadata.get("paguro")
    if paguro_metadata is None:
        return None
    return {"paguro": paguro_metadata}
