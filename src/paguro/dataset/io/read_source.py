from __future__ import annotations

import polars as pl

from typing import TYPE_CHECKING, IO, Literal
from typing import Any, TypeVar, overload

from polars import CredentialProviderFunction

from paguro.models.vfm import VFrameModel
from paguro.dataset.dataset import Dataset
from paguro.dataset.lazydataset import LazyDataset

if TYPE_CHECKING:
    from polars._typing import FileSource, ParallelStrategy, SchemaDict

    from pathlib import Path
    from io import IOBase

    from paguro.utils.dependencies import pyarrow as pa
    from pyiceberg.table import Table

    from deltalake import DeltaTable

__all__ = [
    "read_parquet",
    "scan_parquet",
    "scan_ipc",
    "read_ipc",
    "read_avro",
    "read_csv",
    "scan_csv",
    "read_database_uri",
    "read_delta",
    "scan_delta",
    "read_excel",
    "read_json",
    "read_ndjson",
    "scan_ndjson",
    "read_ods",
    "scan_iceberg",
    "scan_pyarrow_dataset",
]

M = TypeVar("M", bound=VFrameModel)


# -------------------------- parquet -----------------------------------


@overload
def read_parquet(
        source: FileSource,
        *,
        model: type[M],
        paguro_metadata: dict[str, str] | bool,
        storage_options: dict[str, Any] | None,
        credential_provider: (
                CredentialProviderFunction | Literal['auto'] | None
        ),
        retries: int,
        **kwargs: Any,
) -> Dataset[M]: ...


@overload
def read_parquet(
        source: FileSource,
        *,
        model: None,
        paguro_metadata: dict[str, str] | bool,
        storage_options: dict[str, Any] | None,
        credential_provider: (
                CredentialProviderFunction | Literal['auto'] | None
        ),
        retries: int,
        **kwargs: Any,
) -> Dataset[Any]: ...


def read_parquet(
        source: FileSource,
        *,
        model: type[VFrameModel] | None = None,
        paguro_metadata: dict[str, str] | bool = False,
        storage_options: dict[str, Any] | None = None,
        credential_provider: (
                CredentialProviderFunction | Literal['auto'] | None
        ) = 'auto',
        retries: int = 2,
        **kwargs: Any,
):
    """
    ..

    Polars `.read_parquet() <https://docs.pola.rs/py-polars/html/reference/api/polars.read_parquet.html>`_
    with the option to pass a model or read paguro metadata.

    Group
    -----
        read_source
    """
    # https://docs.pola.rs/py-polars/html/reference/api/polars.read_parquet.html
    # https://docs.pola.rs/api/python/dev/reference/api/polars.read_parquet_metadata.html

    data = pl.read_parquet(
        source=source,
        storage_options=storage_options,
        credential_provider=credential_provider,
        retries=retries,
        **kwargs
    )

    _metadata = _read_metadata(
        source=source,
        paguro_metadata=paguro_metadata,
        storage_options=storage_options,
        credential_provider=credential_provider,
        retries=retries,
    )

    dataset = _from_metadata_to_paguro_eager(
        data=data,
        model=model,
        metadata=_metadata,
    )
    return dataset


@overload
def scan_parquet(
        source: FileSource,
        *,
        model: type[M],
        paguro_metadata: dict[str, str] | bool,
        storage_options: dict[str, Any] | None,
        credential_provider: (
                CredentialProviderFunction | Literal['auto'] | None
        ),
        retries: int,
        **kwargs: Any,
) -> LazyDataset[M]: ...


@overload
def scan_parquet(
        source: FileSource,
        *,
        model: None,
        paguro_metadata: dict[str, str] | bool,
        storage_options: dict[str, Any] | None,
        credential_provider: (
                CredentialProviderFunction | Literal['auto'] | None
        ),
        retries: int,
        **kwargs: Any,
) -> LazyDataset[Any]: ...


def scan_parquet(
        source: FileSource,
        *,
        model: type[VFrameModel] | None = None,
        paguro_metadata: dict[str, str] | bool = False,
        storage_options: dict[str, Any] | None = None,
        credential_provider: (
                CredentialProviderFunction | Literal['auto'] | None
        ) = 'auto',
        retries: int = 2,
        **kwargs: Any,
):
    """
    ..

    Polars `.scan_parquet() <https://docs.pola.rs/py-polars/html/reference/api/polars.scan_parquet.html>`_
    with the option to pass a model or read paguro metadata.

    Group
    -----
        scan_source
    """
    # https://docs.pola.rs/py-polars/html/reference/api/polars.scan_parquet.html
    # https://docs.pola.rs/api/python/dev/reference/api/polars.read_parquet_metadata.html
    data = pl.scan_parquet(
        source=source,
        storage_options=storage_options,
        credential_provider=credential_provider,
        retries=retries,
        **kwargs,
    )
    _metadata = _read_metadata(
        source=source,
        paguro_metadata=paguro_metadata,
        storage_options=storage_options,
        credential_provider=credential_provider,
        retries=retries,
    )
    dataset = _from_metadata_to_paguro_lazy(
        data=data,
        model=model,
        metadata=_metadata,
    )
    return dataset


def _read_metadata(
        source: FileSource,
        *,
        paguro_metadata: dict[str, str] | bool,
        storage_options: dict[str, Any] | None,
        credential_provider: (
                CredentialProviderFunction | Literal['auto'] | None
        ),
        retries: int,
) -> dict[str, str] | None:
    _metadata: dict[str, str] | None = None
    if isinstance(paguro_metadata, bool):
        if paguro_metadata:

            try:
                _metadata = pl.read_parquet_metadata(
                    source=source,  # type: ignore[arg-type]
                    storage_options=storage_options,
                    credential_provider=credential_provider,
                    retries=retries,
                )
            except Exception as e:
                raise type(e)(
                    "Unable to read the parquet metadata, "
                    "try calling polars.read_parquet_metadata and passing "
                    "the metadata directly.") from e
    else:
        _metadata = paguro_metadata

    return _metadata


# ---------------------------- ipc -------------------------------------

@overload
def read_ipc(
        source: (
                str | Path | IO[bytes] | bytes

        ),
        *,
        model: type[M],
        paguro_metadata: dict[str, str] | None,
        **kwargs: Any,
) -> Dataset[M]: ...


@overload
def read_ipc(
        source: (
                str | Path | IO[bytes] | bytes

        ),
        *,
        model: None,
        paguro_metadata: dict[str, str] | None,
        **kwargs: Any,
) -> Dataset[Any]: ...


def read_ipc(
        source: (
                str | Path | IO[bytes] | bytes
        ),
        *,
        model: type[VFrameModel] | None = None,
        paguro_metadata: dict[str, str] | None = None,
        **kwargs: Any,
):
    """
    ..

    Wrapper for Polars `.read_ipc() <https://docs.pola.rs/py-polars/html/reference/api/polars.read_ipc.html>`_

    Group
    -----
        read_source
    """
    # https://docs.pola.rs/py-polars/html/reference/api/polars.read_ipc.html
    data = pl.read_ipc(source=source, **kwargs)
    dataset = _from_metadata_to_paguro_eager(
        data=data,
        model=model,
        metadata=paguro_metadata,
    )
    return dataset


@overload
def scan_ipc(
        source: (
                str | Path | IO[bytes] | bytes | list[str] | list[Path] | list[
            IO[bytes]] |
                list[bytes]
        ),
        *,
        model: type[M],
        paguro_metadata: dict[str, str] | None,
        **kwargs: Any,
) -> LazyDataset[M]: ...


@overload
def scan_ipc(
        source: (
                str | Path | IO[bytes] | bytes | list[str] | list[Path] | list[
            IO[bytes]] |
                list[bytes]
        ),
        *,
        model: None,
        paguro_metadata: dict[str, str] | None,
        **kwargs: Any,
) -> LazyDataset[Any]: ...


def scan_ipc(
        source: (
                str | Path | IO[bytes] | bytes | list[str] | list[Path] | list[
            IO[bytes]] |
                list[bytes]
        ),
        *,
        model: type[VFrameModel] | None = None,
        paguro_metadata: dict[str, str] | None = None,
        **kwargs: Any,
):
    """
    ..

    Wrapper for Polars `.scan_ipc() <https://docs.pola.rs/py-polars/html/reference/api/polars.scan_ipc.html>`_

    Group
    -----
        scan_source
    """
    # https://docs.pola.rs/py-polars/html/reference/api/polars.scan_ipc.html
    data = pl.scan_ipc(source=source, **kwargs)
    dataset = _from_metadata_to_paguro_lazy(
        data=data,
        model=model,
        metadata=paguro_metadata,
    )
    return dataset


# -------------------------- csv ---------------------------------------

@overload
def read_csv(
        source: str | Path | IO[str] | IO[bytes] | bytes,
        *,
        model: type[M],
        paguro_metadata: dict[str, str] | None,
        **kwargs: Any,
) -> Dataset[M]: ...


@overload
def read_csv(
        source: str | Path | IO[str] | IO[bytes] | bytes,
        *,
        model: None,
        paguro_metadata: dict[str, str] | None,
        **kwargs: Any,
) -> Dataset[Any]: ...


def read_csv(
        source: str | Path | IO[str] | IO[bytes] | bytes,
        *,
        model: type[VFrameModel] | None = None,
        paguro_metadata: dict[str, str] | None = None,
        **kwargs: Any,
):
    """
    ..

    Wrapper for Polars `.read_csv() <https://docs.pola.rs/py-polars/html/reference/api/polars.read_csv.html>`_

    Group
    -----
        read_source
    """
    # https://docs.pola.rs/py-polars/html/reference/api/polars.read_csv.html
    data = pl.read_csv(source=source, **kwargs)

    dataset: Dataset[Any] = _from_metadata_to_paguro_eager(
        data=data,
        model=model,
        metadata=paguro_metadata,
    )
    return dataset


# ----------------------------------------------------------------------


@overload
def scan_csv(
        source: (str | Path | IO[str] | IO[bytes] | bytes | list[str] | list[Path] |
                 list[IO[str]] | list[IO[bytes]] | list[bytes]),
        *,
        model: type[VFrameModel],
        paguro_metadata: dict[str, str] | None,
        **kwargs: Any,
) -> LazyDataset[M]: ...


@overload
def scan_csv(
        source: (str | Path | IO[str] | IO[bytes] | bytes | list[str] | list[Path] |
                 list[IO[str]] | list[IO[bytes]] | list[bytes]),
        *,
        model: None,
        paguro_metadata: dict[str, str] | None,
        **kwargs: Any,
) -> LazyDataset[Any]: ...


def scan_csv(
        source: (str | Path | IO[str] | IO[bytes] | bytes | list[str] | list[Path] |
                 list[IO[str]] | list[IO[bytes]] | list[bytes]),
        *,
        model: type[VFrameModel] | None = None,
        paguro_metadata: dict[str, str] | None = None,
        **kwargs: Any,
):
    """
    ..

    Wrapper for Polars `.scan_csv() <https://docs.pola.rs/py-polars/html/reference/api/polars.scan_csv.html>`_

    Group
    -----
        scan_source
    """
    # https://docs.pola.rs/py-polars/html/reference/api/polars.scan_csv.html
    data = pl.scan_csv(source=source, **kwargs)
    dataset = _from_metadata_to_paguro_lazy(
        data=data,
        model=model,
        metadata=paguro_metadata,
    )

    return dataset


# ----------------------------------------------------------------------

@overload
def read_avro(
        source: str | Path | IO[bytes] | bytes,
        *,
        model: type[M],
        paguro_metadata: dict[str, str] | None = None,
        **kwargs: Any,
) -> Dataset[M]: ...


@overload
def read_avro(
        source: str | Path | IO[bytes] | bytes,
        *,
        model: None = None,
        paguro_metadata: dict[str, str] | None = None,
        **kwargs: Any,
) -> Dataset[Any]: ...


def read_avro(
        source: str | Path | IO[bytes] | bytes,
        *,
        model: type[M] | None = None,
        paguro_metadata: dict[str, str] | None = None,
        **kwargs: Any,
):
    """
    ..

    Wrapper for Polars `.read_avro() <https://docs.pola.rs/py-polars/html/reference/api/polars.read_avro.html>`_

    Group
    -----
        read_source
    """
    # https://docs.pola.rs/py-polars/html/reference/api/polars.read_avro.html
    data = pl.read_avro(source=source, **kwargs)
    dataset = _from_metadata_to_paguro_eager(
        data=data,
        model=model,
        metadata=paguro_metadata
    )

    return dataset


# ----------------------------------------------------------------------


@overload
def read_database_uri(
        query: str,
        *,
        model: type[M],
        paguro_metadata: dict[str, str] | None,
        **kwargs: Any,
) -> Dataset[M]: ...


@overload
def read_database_uri(
        query: str,
        *,
        model: None,
        paguro_metadata: dict[str, str] | None,
        **kwargs: Any,
) -> Dataset[Any]: ...


def read_database_uri(
        query: str,
        *,
        model: type[VFrameModel] | None = None,
        paguro_metadata: dict[str, str] | None = None,
        **kwargs: Any,
):
    """
    ..

    Wrapper for Polars `.read_database_uri() <https://docs.pola.rs/py-polars/html/reference/api/polars.read_database_uri.html>`_

    Group
    -----
        read_source
    """
    data = pl.read_database_uri(query=query, **kwargs)
    dataset = _from_metadata_to_paguro_eager(
        data=data,
        model=model,
        metadata=paguro_metadata
    )

    return dataset


# ----------------------------------------------------------------------


@overload
def read_delta(
        source: str | Path | DeltaTable,
        *,
        model: type[M],
        paguro_metadata: dict[str, str] | None,
        **kwargs: Any,
) -> Dataset[M]: ...


@overload
def read_delta(
        source: str | Path | DeltaTable,
        *,
        model: None,
        paguro_metadata: dict[str, str] | None,
        **kwargs: Any,
) -> Dataset[Any]: ...


def read_delta(
        source: str | Path | DeltaTable,
        *,
        model: type[VFrameModel] | None = None,
        paguro_metadata: dict[str, str] | None = None,
        **kwargs: Any,
) -> Dataset:
    """
    ..

    Wrapper for Polars `.read_delta() <https://docs.pola.rs/py-polars/html/reference/api/polars.read_delta.html>`_

    Group
    -----
        read_source
    """
    data = pl.read_delta(source=source, **kwargs)
    dataset = _from_metadata_to_paguro_eager(
        data=data,
        model=model,
        metadata=paguro_metadata,
    )
    return dataset


# ------------

@overload
def scan_delta(
        source: str | Path | DeltaTable,
        *,
        model: type[M],
        paguro_metadata: dict[str, str] | None,
        **kwargs: Any,
) -> LazyDataset[M]: ...


@overload
def scan_delta(
        source: str | Path | DeltaTable,
        *,
        model: None,
        paguro_metadata: dict[str, str] | None,
        **kwargs: Any,
) -> LazyDataset[Any]: ...


def scan_delta(
        source: str | Path | DeltaTable,
        *,
        model: type[VFrameModel] | None = None,
        paguro_metadata: dict[str, str] | None = None,
        **kwargs: Any,
):
    """
    ..

    Wrapper for Polars `.scan_delta() <https://docs.pola.rs/py-polars/html/reference/api/polars.scan_delta.html>`_

    Group
    -----
        scan_source
    """
    # https://docs.pola.rs/py-polars/html/reference/api/polars.scan_delta.html
    data = pl.scan_delta(source=source, **kwargs)
    dataset = _from_metadata_to_paguro_lazy(
        data=data,
        model=model,
        metadata=paguro_metadata
    )
    return dataset


# ----------------------------------------------------------------------

@overload
def read_excel(
        source: FileSource,
        *,
        model: type[M],
        paguro_metadata: dict[str, str] | None,
        **kwargs: Any,
) -> Dataset[M]: ...


@overload
def read_excel(
        source: FileSource,
        *,
        model: None,
        paguro_metadata: dict[str, str] | None,
        **kwargs: Any,
) -> Dataset[Any]: ...


def read_excel(
        source: FileSource,
        *,
        model: type[VFrameModel] | None = None,
        paguro_metadata: dict[str, str] | None = None,
        **kwargs: Any,
):
    """
    ..

    Wrapper for Polars `.read_excel() <https://docs.pola.rs/py-polars/html/reference/api/polars.read_excel.html>`_

    Group
    -----
        read_source
    """
    # https://docs.pola.rs/py-polars/html/reference/api/polars.read_excel.html
    data = pl.read_excel(source=source, **kwargs)
    dataset = _from_metadata_to_paguro_eager(
        data=data,
        model=model,
        metadata=paguro_metadata,
    )
    return dataset


# ----------------------------------------------------------------------


@overload
def read_json(
        source: str | Path | IOBase | bytes,
        *,
        model: type[M],
        paguro_metadata: dict[str, str] | None,
        **kwargs: Any,
) -> Dataset[M]: ...


@overload
def read_json(
        source: str | Path | IOBase | bytes,
        *,
        model: None,
        paguro_metadata: dict[str, str] | None,
        **kwargs: Any,
) -> Dataset[Any]: ...


def read_json(
        source: (str | Path | IOBase | bytes),
        *,
        model: type[VFrameModel] | None = None,
        paguro_metadata: dict[str, str] | None = None,
        **kwargs: Any,
):
    """
    ..

    Wrapper for Polars `.read_json() <https://docs.pola.rs/py-polars/html/reference/api/polars.read_json.html>`_

    Group
    -----
        read_source
    """
    # https://docs.pola.rs/py-polars/html/reference/api/polars.read_json.html
    data = pl.read_json(source=source, **kwargs)
    dataset = _from_metadata_to_paguro_eager(
        data=data,
        model=model,
        metadata=paguro_metadata,
    )
    return dataset


# ----------------------------------------------------------------------

@overload
def read_ndjson(
        source: (str | Path | IO[str] | IO[bytes] | bytes | list[str] | list[Path] |
                 list[IO[str]] | list[IO[bytes]]),
        *,
        model: type[M],
        paguro_metadata: dict[str, str] | None,
        **kwargs: Any,
) -> Dataset[M]: ...


@overload
def read_ndjson(
        source: (str | Path | IO[str] | IO[bytes] | bytes | list[str] | list[Path] |
                 list[IO[str]] | list[IO[bytes]]),
        *,
        model: None,
        paguro_metadata: dict[str, str] | None,
        **kwargs: Any,
) -> Dataset[Any]: ...


def read_ndjson(
        source: (
                str | Path | IO[str] | IO[bytes] | bytes | list[str] | list[Path] |
                list[IO[str]] | list[IO[bytes]]),
        *,
        model: type[VFrameModel] | None = None,
        paguro_metadata: dict[str, str] | None = None,
        **kwargs: Any,
):
    """
    ..

    Wrapper for Polars `.read_ndjson() <https://docs.pola.rs/py-polars/html/reference/api/polars.read_ndjson.html>`_

    Group
    -----
        read_source
    """
    # https://docs.pola.rs/py-polars/html/reference/api/polars.read_ndjson.html

    data = pl.read_ndjson(source=source, **kwargs)
    dataset = _from_metadata_to_paguro_eager(
        data=data,
        model=model,
        metadata=paguro_metadata,
    )
    return dataset


# ----------

@overload
def scan_ndjson(
        source: (str | Path | IO[str] | IO[bytes] | bytes | list[str] | list[Path] |
                 list[IO[str]] | list[IO[bytes]]),
        *,
        model: type[M],
        paguro_metadata: dict[str, str] | None,
        **kwargs: Any,
) -> LazyDataset[M]: ...


@overload
def scan_ndjson(
        source: (str | Path | IO[str] | IO[bytes] | bytes | list[str] | list[Path] |
                 list[IO[str]] | list[IO[bytes]]),
        *,
        model: None,
        paguro_metadata: dict[str, str] | None,
        **kwargs: Any,
) -> LazyDataset[Any]: ...


def scan_ndjson(
        source: (
                str | Path | IO[str] | IO[bytes] | bytes | list[str] | list[Path] |
                list[IO[str]] | list[IO[bytes]]),
        *,
        model: type[VFrameModel] | None = None,
        paguro_metadata: dict[str, str] | None = None,
        **kwargs: Any,
) -> LazyDataset:
    """
    ..

    Wrapper for Polars `.scan_ndjson() <https://docs.pola.rs/py-polars/html/reference/api/polars.scan_ndjson.html>`_

    Group
    -----
        scan_source
    """
    # https://docs.pola.rs/py-polars/html/reference/api/polars.scan_ndjson.html
    data = pl.scan_ndjson(source=source, **kwargs)
    dataset = _from_metadata_to_paguro_lazy(
        data=data,
        model=model,
        metadata=paguro_metadata,
    )

    return dataset


# ----------------------------------------------------------------------

@overload
def read_ods(
        source: FileSource,
        *,
        model: type[M],
        paguro_metadata: dict[str, str] | None,
        **kwargs: Any,
) -> Dataset[M]: ...


@overload
def read_ods(
        source: FileSource,
        *,
        model: None,
        paguro_metadata: dict[str, str] | None,
        **kwargs: Any,
) -> Dataset[Any]: ...


def read_ods(
        source: FileSource,
        *,
        model: type[VFrameModel] | None = None,
        paguro_metadata: dict[str, str] | None = None,
        **kwargs: Any,
):
    """
    ..

    Wrapper for Polars `.read_ods() <https://docs.pola.rs/py-polars/html/reference/api/polars.read_ods.html>`_

    Group
    -----
        read_source
    """
    # https://docs.pola.rs/py-polars/html/reference/api/polars.read_ods.html
    data = pl.read_ods(source=source, **kwargs)
    dataset = _from_metadata_to_paguro_eager(
        data=data,
        model=model,
        metadata=paguro_metadata,
    )
    return dataset


# ----------------------------------------------------------------------

@overload
def scan_iceberg(
        source: str | Table,
        *,
        model: type[M],
        paguro_metadata: dict[str, str] | None,
        **kwargs: Any,
) -> LazyDataset[M]: ...


@overload
def scan_iceberg(
        source: str | Table,
        *,
        model: None,
        paguro_metadata: dict[str, str] | None,
        **kwargs: Any,
) -> LazyDataset[Any]: ...


def scan_iceberg(
        source: str | Table,
        *,
        model: type[VFrameModel] | None = None,
        paguro_metadata: dict[str, str] | None = None,
        **kwargs: Any,
):
    """
    ..

    Wrapper for Polars `.scan_iceberg() <https://docs.pola.rs/py-polars/html/reference/api/polars.scan_iceberg.html>`_

    Group
    -----
        scan_source
    """
    # https://docs.pola.rs/py-polars/html/reference/api/polars.scan_iceberg.html
    data = pl.scan_iceberg(source=source, **kwargs)
    dataset = _from_metadata_to_paguro_lazy(
        data=data,
        model=model,
        metadata=paguro_metadata,
    )
    return dataset


# ----------------------------------------------------------------------


@overload
def scan_pyarrow_dataset(
        source: pa.dataset.Dataset,
        *,
        model: type[M],
        paguro_metadata: dict[str, str] | None,
        **kwargs: Any,
) -> LazyDataset[M]: ...


@overload
def scan_pyarrow_dataset(
        source: pa.dataset.Dataset,
        *,
        model: None,
        paguro_metadata: dict[str, str] | None,
        **kwargs: Any,
) -> LazyDataset[Any]: ...


def scan_pyarrow_dataset(
        source: pa.dataset.Dataset,
        *,
        model: type[VFrameModel] | None = None,
        paguro_metadata: dict[str, str] | None = None,
        **kwargs: Any,
):
    """
    ..

    Wrapper for Polars `.scan_pyarrow_dataset() <https://docs.pola.rs/py-polars/html/reference/api/polars.scan_pyarrow_dataset.html>`_

    Group
    -----
        scan_source
    """
    # https://docs.pola.rs/py-polars/html/reference/api/polars.scan_pyarrow_dataset.html
    data = pl.scan_pyarrow_dataset(source, **kwargs)
    dataset = _from_metadata_to_paguro_lazy(
        data=data,
        model=model,
        metadata=paguro_metadata,
    )
    return dataset


# ----------------------------------------------------------------------

@overload
def _from_metadata_to_paguro_eager(
        data: pl.DataFrame,
        *,
        model: type[M],
        metadata: dict[str, str] | None,
) -> Dataset[M]: ...


@overload
def _from_metadata_to_paguro_eager(
        data: pl.DataFrame,
        *,
        model: None,
        metadata: dict[str, str] | None,
) -> Dataset[Any]: ...


def _from_metadata_to_paguro_eager(
        data: pl.DataFrame,
        *,
        model: type[M] | None,
        metadata: dict[str, str] | None,
):
    ds: Dataset = Dataset._from_paguro_metadata_dict(  # type:ignore[return-type]
        frame=data,
        paguro_metadata=metadata,
    )

    if model is not None:
        # Callers who passed a concrete model match the first overload,
        # so they *see* Dataset[T] at the call site.
        return ds.with_model(model=model)
    return ds


# ----------------------------------------------------------------------


@overload
def _from_metadata_to_paguro_lazy(
        data: pl.LazyFrame,
        *,
        model: type[M],
        metadata: dict[str, str] | None,
) -> LazyDataset[M]: ...


@overload
def _from_metadata_to_paguro_lazy(
        data: pl.LazyFrame,
        *,
        model: None,
        metadata: dict[str, str] | None,
) -> LazyDataset[Any]: ...


def _from_metadata_to_paguro_lazy(
        data: pl.LazyFrame,
        *,
        model: type[M] | None,
        metadata: dict[str, str] | None,
):
    ds: LazyDataset = LazyDataset._from_paguro_metadata_dict(
        # type:ignore[return-type]
        frame=data,
        paguro_metadata=metadata,
    )
    if model is not None:
        # Callers who passed a concrete model match the first overload,
        # so they *see* Dataset[T] at the call site.
        return ds.with_model(model=model)
    return ds
