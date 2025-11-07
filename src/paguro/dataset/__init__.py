from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import polars as pl

from paguro.dataset.dataset import Dataset
from paguro.dataset.lazydataset import LazyDataset
from paguro.validation.validation import Validation

if TYPE_CHECKING:
    from paguro.typing import (
        OnFailure,
        ValidatorOrExpr,
        ValidationMode,
        IntoKeepColumns
    )

__all__ = [
    "Dataset",
    "LazyDataset",
]


# --------------------------- validate ---------------------------------

@pl.api.register_dataframe_namespace("validate")
class ValidateDataFrame:
    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def __call__(
            self,
            *validators: (
                    ValidatorOrExpr
                    | Iterable[ValidatorOrExpr]
                    | Validation
            ),
            mode: ValidationMode = "all",
            keep_columns: IntoKeepColumns = False,
            on_failure: OnFailure = "raise",
    ) -> pl.LazyFrame:
        validation = None
        if validators:
            validation = Validation(*validators)
        if validation is None:
            raise TypeError("Please provide one or more validators.")

        return validation.validate(
            data=self._df,
            mode=mode,
            keep_columns=keep_columns,
            on_success="return_data",
            on_failure=on_failure,
        )


@pl.api.register_lazyframe_namespace("validate")
class ValidateLazyFrame:
    def __init__(self, df: pl.LazyFrame) -> None:
        self._df = df

    def __call__(
            self,
            *validators: (
                    ValidatorOrExpr
                    | Iterable[ValidatorOrExpr]
                    | Validation
            ),
            mode: ValidationMode = "all",
            keep_columns: IntoKeepColumns = False,
            on_failure: OnFailure = "raise",
    ) -> pl.LazyFrame:
        validation = None
        if validators:
            validation = Validation(*validators)
        if validation is None:
            raise TypeError("Please provide one or more validators.")

        return validation.validate(
            data=self._df,
            mode=mode,
            keep_columns=keep_columns,
            on_success="return_data",
            on_failure=on_failure,
        )


# --------------------------- to_paguro --------------------------------

@pl.api.register_dataframe_namespace("to_paguro")
class DataFrameToDataset:
    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def __call__(self, name: str | None = None) -> Dataset:
        return Dataset(data=self._df, name=name)


@pl.api.register_lazyframe_namespace("to_paguro")
class LazyFrameToLazyDataset:
    def __init__(self, df: pl.LazyFrame) -> None:
        self._df = df

    def __call__(self, name: str | None = None) -> LazyDataset:
        return LazyDataset(data=self._df, name=name)
