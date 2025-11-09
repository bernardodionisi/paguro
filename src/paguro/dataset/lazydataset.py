from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, TypeVar, Generic

import polars as pl

from paguro.dataset._dataset import _Dataset
from paguro.shared._getattr._polars._lazyframe import _LazyFrame
from paguro.shared.serialize.encoder import CustomJSONEncoder
from paguro.models.vfm import VFrameModel, VFM

if TYPE_CHECKING:
    import sys
    from collections.abc import Iterable, Mapping, Sequence
    from json import JSONEncoder
    from pathlib import Path
    from typing import IO
    from paguro.dataset.utils._group_by import _GroupBy

    from polars._plr import Label
    from polars._typing import StartBy

    from polars import DataFrame, LazyFrame
    from polars._typing import (
        FrameInitTypes,
        IntoExpr,
        PartitioningScheme, JoinStrategy, JoinValidation, MaintainOrderJoin,
        AsofJoinStrategy, ClosedInterval, StartBy,
    )
    from datetime import timedelta

    from paguro import Validation
    from paguro.dataset.dataset import Dataset
    from paguro.typing import ValidatorOrExpr

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

U = TypeVar("U", bound=VFrameModel)


class LazyDataset(_Dataset[pl.LazyFrame], _LazyFrame, Generic[VFM]):
    """
    A LazyFrame like structure with validation and other extensions.

    Constructors
    ------------

    Validation + Model
    ------------------

    Validation
    ==========

    Model
    =====

    Information
    -----------

    EDA
    ---

    Export
    ------

    Polars Methods
    --------------

    Adapted
    =======

    Some polars methods have been adapted to manage model/validation/info or
    to accept `paguro`'s types as arguments. The computation over the data is still
    handled with `polars`.

    Delegated
    =========

    """

    _data: pl.LazyFrame
    _model: VFM | None  # type: ignore[assignment]

    def __init__(
            self,
            data: (
                    DataFrame
                    | LazyFrame
                    | FrameInitTypes
                    | Dataset
                    | None
            ) = None,
            *,
            name: str | None = None,
            **kwargs: Any,
    ) -> None:
        """
        Initialize the lazydataset.

        Parameters
        ----------
        data
        name
        kwargs

        See Also
        --------
        :class:`paguro.Dataset`
        """
        if isinstance(data, pl.DataFrame):
            data = data.lazy()

        if not isinstance(data, (pl.LazyFrame, _Dataset)):
            data = pl.LazyFrame(data, **kwargs)

        super().__init__(data=data, name=name, **kwargs)

    # -------------------------- model ---------------------------------

    @property
    def vcol(self) -> VFM:
        if self._model is None:
            raise RuntimeError(
                "Model has not been set. Please set a model using .with_model() to access vcol."
            )
        return self._model

    @property
    def model(self) -> VFM | None:
        return self._model

    def with_model(
            self,
            model: type[U],
            *,
            overwrite: bool = False,
    ) -> LazyDataset[U]:
        return super()._with_model(  # type: ignore[return-value]
            model=model,
            overwrite=overwrite
        )

    def without_model(
            self,
    ) -> LazyDataset[Any]:
        """
        ..

        Group
        -----
            Model
        """
        return super()._without_model()  # type: ignore

    # ------------------------------------------------------------------

    def to_polars(self) -> pl.LazyFrame:
        return self.to_lazyframe()

    # ------------------------------------------------------------------

    def sink_parquet(
            self,
            path: str | Path | IO[bytes] | IO[str] | PartitioningScheme,
            *,
            write_paguro_metadata: bool = True,
            **kwargs: Any,
    ) -> None:
        """
        Sink parquet.

        Group
        -----
            Adapted
        """

        metadata = self._metadata_for_polars_parquet(
            write_paguro_metadata=write_paguro_metadata,
            kwargs=kwargs,
        )
        if metadata:
            self._getattr("sink_parquet")(
                path=path,
                metadata=metadata,
                **kwargs,
            )
        else:
            self._getattr("sink_parquet")(
                path=path,
                **kwargs
            )

    # ------------------------------------------------------------------

    def collect(  # type: ignore[override]
            self,
            **kwargs: Any,
    ) -> Dataset[VFM]:
        """
        Polars' .lazy().

        Group
        -----
            Adapted
        """
        return self._getattr("collect")(**kwargs)

    def lazy(
            self,
    ) -> Self:
        """
        Polars' .lazy.

        Group
        -----
            Delegated
        """
        return self._getattr("lazy")()

    def group_by(
            self,
            *by: IntoExpr | Iterable[IntoExpr],
            maintain_order: bool = False,
            **named_by: IntoExpr,
    ) -> _GroupBy[LazyDataset]:
        """
        Polars' group_by.

        Group
        -----
            Adapted
        """
        return super()._group_by(
            *by,
            maintain_order=maintain_order,
            **named_by,
        )

    def group_by_dynamic(
            self,
            index_column: IntoExpr,
            *,
            every: str | timedelta,
            period: str | timedelta | None = None,
            offset: str | timedelta | None = None,
            include_boundaries: bool = False,
            closed: ClosedInterval = 'left',
            label: Label = 'left',
            group_by: IntoExpr | Iterable[IntoExpr] | None = None,
            start_by: StartBy = 'window',
    ) -> _GroupBy[LazyDataset]:
        """
        Polars' group_by_dynamic.

        Group
        -----
            Adapted
        """
        return super()._group_by_dynamic(
            index_column=index_column,
            every=every,
            period=period,
            offset=offset,
            include_boundaries=include_boundaries,
            closed=closed,
            label=label,
            group_by=group_by,
            start_by=start_by,
        )

    def rolling(
            self,
            index_column: IntoExpr,
            *,
            period: str | timedelta,
            offset: str | timedelta | None = None,
            closed: ClosedInterval = 'right',
            group_by: IntoExpr | Iterable[IntoExpr] | None = None,
    ) -> _GroupBy[LazyDataset]:
        """
        Polars' rolling.

        Group
        -----
            Adapted
        """
        return super()._rolling(
            index_column=index_column,
            period=period,
            offset=offset,
            closed=closed,
            group_by=group_by,
        )

    # -------------------------- other ---------------------------------

    def join(
            self,
            other: LazyDataset[U] | LazyFrame,
            on: str | pl.Expr | Sequence[str | pl.Expr] | None = None,
            how: JoinStrategy = 'inner',
            *,
            left_on: str | pl.Expr | Sequence[str | pl.Expr] | None = None,
            right_on: str | pl.Expr | Sequence[str | pl.Expr] | None = None,
            suffix: str = "_right",
            validate: JoinValidation = "m:m",
            nulls_equal: bool = False,
            coalesce: bool | None = None,
            maintain_order: MaintainOrderJoin | None = None,
            allow_parallel: bool = True,
            force_parallel: bool = False,
    ) -> Self:
        """
        Polars' join.

        Group
        -----
            Adapted
        """
        if not isinstance(other, (LazyDataset, pl.LazyFrame)):
            raise TypeError("other must be a LazyDataset or pl.LazyFrame")

        return super()._join(
            other=other,  # type: ignore[arg-type]
            on=on,
            how=how,
            left_on=left_on,
            right_on=right_on,
            suffix=suffix,
            validate=validate,
            nulls_equal=nulls_equal,
            coalesce=coalesce,
            maintain_order=maintain_order,
            allow_parallel=allow_parallel,
            force_parallel=force_parallel,
        )

    def join_asof(
            self,
            other: LazyDataset[U] | LazyFrame,
            *,
            left_on: str | None | pl.Expr = None,
            right_on: str | None | pl.Expr = None,
            on: str | None | pl.Expr = None,
            by_left: str | Sequence[str] | None = None,
            by_right: str | Sequence[str] | None = None,
            by: str | Sequence[str] | None = None,
            strategy: AsofJoinStrategy = 'backward',
            suffix: str = '_right',
            tolerance: str | int | float | timedelta | None = None,
            allow_parallel: bool = True,
            force_parallel: bool = False,
            coalesce: bool = True,
            allow_exact_matches: bool = True,
            check_sortedness: bool = True,
    ) -> Self:
        """
        Polars' join_asof.

        Group
        -----
            Adapted
        """
        if not isinstance(other, (LazyDataset, pl.LazyFrame)):
            raise TypeError("other must be a LazyDataset or pl.LazyFrame")
        return super()._join_asof(
            other=other,  # type: ignore[arg-type]
            left_on=left_on,
            right_on=right_on,
            on=on,
            by_left=by_left,
            by_right=by_right,
            by=by,
            strategy=strategy,
            suffix=suffix,
            tolerance=tolerance,
            allow_parallel=allow_parallel,
            force_parallel=force_parallel,
            coalesce=coalesce,
            allow_exact_matches=allow_exact_matches,
            check_sortedness=check_sortedness,

        )

    def join_where(
            self,
            other: LazyDataset[U] | pl.LazyFrame,
            *predicates: pl.Expr | Iterable[pl.Expr],
            suffix: str = '_right',
    ) -> Self:
        """
        Polars' join_where.

        Group
        -----
            Adapted
        """
        if not isinstance(other, (LazyDataset, pl.LazyFrame)):
            raise TypeError("other must be a LazyDataset or pl.LazyFrame")
        return super()._join_where(
            other,  # type: ignore[arg-type]
            *predicates,
            suffix=suffix,
        )

    def merge_sorted(
            self,
            other: LazyDataset[U] | pl.LazyFrame,
            key: str,
    ) -> Self:
        """
        Polars' merge_sorted.

        Group
        -----
            Adapted
        """
        if not isinstance(other, (LazyDataset, pl.LazyFrame)):
            raise TypeError("other must be a LazyDataset or pl.LazyFrame")

        return super()._merge_sorted(
            other=other,  # type: ignore[arg-type]
            key=key,
        )

    def rename(
            self,
            mapping: Mapping[str, str] | Callable[[str], str],
            *,
            strict: bool = True,
    ) -> Self:
        """
        Polars' rename.

        Group
        -----
            Adapted
        """
        return super()._rename(mapping=mapping, strict=strict)

    # ------------------------------------------------------------------

    def with_validation(
            self,
            *validators: ValidatorOrExpr
                         | Iterable[ValidatorOrExpr]
                         | Validation,
            overwrite: bool = False,
            **named_validators: ValidatorOrExpr,
    ) -> Self:
        return super().with_validation(
            *validators,
            overwrite=overwrite,
            **named_validators,
        )

    def with_info(
            self,
            k: str,
            /,
            **mapping: Any,
    ) -> Self:
        return super().with_info(k, **mapping)

    def with_name(
            self,
            name: str | None,
    ) -> Self:
        return super().with_name(name=name)
