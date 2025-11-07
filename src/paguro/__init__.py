from __future__ import annotations

try:
    from ._version import __version__, __version_tuple__
except ModuleNotFoundError:  # pragma: no cover
    import warnings

    __version__ = "0.0.0"
    __version_tuple__ = (0, 0, 0)
    warnings.warn(
        "\nAn error occurred during package install "
        "where setuptools_scm failed to create a _version.py file."
        "\nDefaulting version to 0.0.0.",
        stacklevel=2,
    )

# structures
from paguro.dataset.dataset import Dataset
from paguro.dataset.lazydataset import LazyDataset
from paguro.collection.collection import Collection
from paguro.collection.lazycollection import LazyCollection

# utilities
from paguro.utils.config import Config
from paguro.utils.show_versions import show_versions

# validation
from paguro.validation import vcol, vframe, vrelations
from paguro.validation.valid_relations import vpair

from paguro.validation.validation import Validation

# other
from paguro import exceptions
from paguro.defer import deferred
from paguro.shared.functions import collect_all, concat

from paguro.dataset.io._read_source import (
    read_parquet,
    scan_parquet,
    read_ipc,
    scan_ipc,
    read_avro,
    read_csv,
    scan_csv,
    read_database_uri,
    read_delta,
    scan_delta,
    read_json,
    read_excel,
    read_ndjson,
    scan_ndjson,
    read_ods,
    scan_iceberg,
    scan_pyarrow_dataset,
)

__all__ = [
    # utilities
    "show_versions",
    "Config",

    "exceptions",

    # deferred pipeline
    "LazyFrameExpr",
    "defer",

    # structures
    "Dataset",
    "LazyDataset",

    "Collection",
    "LazyCollection",

    "collect_all",
    "concat",

    # validation dsl

    "vcol",
    "vframe",
    "vrelations",

    # validation utilities
    "vpair",
    "Validation",
    # "with_validation",

    # io

    "read_parquet",
    "read_ipc",
    "read_csv",
    "read_avro",
    "read_delta",
    "read_excel",
    "read_json",
    "read_ndjson",
    "read_database_uri",
    "read_ods",

    "scan_parquet",
    "scan_ipc",
    "scan_csv",
    "scan_delta",
    "scan_ndjson",
    "scan_iceberg",
    "scan_pyarrow_dataset",

]
