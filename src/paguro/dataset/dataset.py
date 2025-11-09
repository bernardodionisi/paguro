from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, TypeVar, Generic, IO

import polars as pl
from collections.abc import Sequence
from paguro.dataset._dataset import _Dataset
from paguro.shared._getattr._polars._dataframe import _DataFrame
from paguro.shared.serialize.encoder import CustomJSONEncoder

from paguro.shared.various import _write_data_repr_to_svg

from paguro.models.vfm import VFrameModel, VFM

if TYPE_CHECKING:
    import sys
    from collections.abc import Iterable, Mapping
    from pathlib import Path

    from polars import DataFrame
    from polars._typing import (
        FrameInitTypes,
        IntoExpr, ClosedInterval,
    )

    from polars._plr import Label
    from polars._typing import StartBy

    from paguro import Validation
    from paguro.dataset.lazydataset import LazyDataset
    from paguro.typing import ValidatorOrExpr
    from paguro.dataset.utils._group_by import _GroupBy
    from datetime import timedelta
    from polars._typing import (
        FrameInitTypes,
        IntoExpr,
        JoinStrategy, JoinValidation, MaintainOrderJoin,
        AsofJoinStrategy
    )

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

U = TypeVar("U", bound=VFrameModel)


class Dataset(_Dataset[pl.DataFrame], _DataFrame, Generic[VFM]):
    """
    A DataFrame like structure with validation and other extensions.

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

    _data: pl.DataFrame

    def __init__(
            self,
            data: (
                    DataFrame
                    | FrameInitTypes
                    | Dataset
                    | None
            ) = None,
            *,
            name: str | None = None,
            **kwargs: Any,
    ) -> None:
        """
        Initializes the Dataset class.

        Parameters
        ----------
        data
            - Polars DataFrame/LazyFrame
            - FrameInitTypes: Initializable format compatible with a Polars' DataFrame
            - LazyFrame (collected at initialization)
            - Dataset/LazyDataset
        name
            An optional name for identifying the dataset.
        kwargs
            Additional keyword arguments passed during initialization:
            in case data is an
            initializable format compatible with with a Polars' DataFrame
            `kwargs` can contain any of the parameters that can be used when
            initializing a Polars DataFrame.

        See Also
        --------

            :class:`paguro.LazyDataset`
        """
        if not isinstance(data, (pl.DataFrame, _Dataset)):
            data = pl.DataFrame(data, **kwargs)

        super().__init__(data=data, name=name, **kwargs)

        self._model: VFM | None = None  # type: ignore[assignment]

    # --------------------------- IO -----------------------------------

    def write_parquet(
            self,
            file: str | Path | IO[bytes],
            *,
            write_paguro_metadata: bool = True,
            **kwargs: Any,
    ) -> None:
        """
        Write parquet.

        Group
        -----
            Adapted
        """

        metadata = self._metadata_for_polars_parquet(
            write_paguro_metadata=write_paguro_metadata,
            kwargs=kwargs,
        )

        if metadata:
            self._getattr("write_parquet")(
                file=file,
                metadata=metadata,
                **kwargs,
            )
        else:
            self._getattr("write_parquet")(
                file=file,
                **kwargs
            )

    def _write_repr_svg(
            self,
            path: str | Path | None = None,
            *,
            name: str | None = None,
            font_size: int = 20,
            line_height: int = 25,
    ) -> str | None:
        if name is None:
            name = self._name
        return _write_data_repr_to_svg(
            data=self._data,
            title=name,
            path=path,
            font_size=font_size,
            line_height=line_height,

        )

    def to_polars(self) -> pl.DataFrame:
        return self.to_dataframe()

    # -------------------------- polars ---------------------------------

    def lazy(
            self,
    ) -> LazyDataset[VFM]:
        """
        Polars' .lazy.

        Group
        -----
            Adapted
        """
        return self._getattr("lazy")()

    # ---------------

    def group_by(
            self,
            *by: IntoExpr | Iterable[IntoExpr],
            maintain_order: bool = False,
            **named_by: IntoExpr,
    ) -> _GroupBy[Dataset]:
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
    ) -> _GroupBy[Dataset]:
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
    ) -> _GroupBy[Dataset]:
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

    # ---------------

    def join(
            self,
            other: Dataset[U] | pl.DataFrame,
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
    ) -> Self:
        """
        Polars' join.

        Group
        -----
            Adapted
        """
        if not isinstance(other, (Dataset, pl.DataFrame)):
            raise TypeError("other must be a Dataset or polars.DataFrame")

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
        )

    def join_asof(
            self,
            other: Dataset[U] | pl.DataFrame,
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
        if not isinstance(other, (Dataset, pl.DataFrame)):
            raise TypeError("other must be a Dataset or polars.DataFrame")

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
            other: Dataset[U] | pl.DataFrame,
            *predicates: pl.Expr | Iterable[pl.Expr],
            suffix: str = '_right',
    ) -> Self:
        """
        Polars' join_where.

        Group
        -----
            Adapted
        """
        if not isinstance(other, (Dataset, pl.DataFrame)):
            raise TypeError("other must be a Dataset or polars.DataFrame")

        return super()._join_where(
            other,  # type: ignore[arg-type]
            *predicates,
            suffix=suffix,
        )

    def merge_sorted(
            self,
            other: Dataset[U] | pl.DataFrame,
            key: str,
    ) -> Self:
        """
        Polars' merge_sorted.

        Group
        -----
            Adapted
        """
        if not isinstance(other, (Dataset, pl.DataFrame)):
            raise TypeError("other must be a Dataset or polars.DataFrame")

        return super()._merge_sorted(
            other=other,  # type: ignore[arg-type]
            key=key,
        )

    def vstack(
            self,
            other: Dataset[VFM] | pl.DataFrame,
            *,
            in_place: bool = False,
    ) -> Self:
        """
        Polars' vstack.

        Group
        -----
            Adapted
        """
        if not isinstance(other, (Dataset, pl.DataFrame)):
            raise TypeError("other must be a Dataset or polars.DataFrame")
        return super()._vstack(
            other=other,  # type: ignore[arg-type]
            in_place=in_place,
        )

    # ---------------

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

    # --------------------------- paguro -------------------------------

    @property
    def vcol(self) -> VFM:
        """
        Retrieve the dataset model if it is not None.

        Group
        -----
            Model
        """
        if self._model is None:
            raise RuntimeError(
                "Model has not been set. Please set a model using .with_model() to access vcol."
            )
        return self._model

    @property
    def model(self) -> VFM | None:
        """
        Retrieve the dataset model.

        Group
        -----
            Model
        """
        return self._model

    def with_model(
            self,
            model: type[U],
            *,
            overwrite: bool = False,
    ) -> Dataset[U]:
        """
        Add a model and validation to the dataset.

        Group
        -----
            Model
        """
        return super()._with_model(  # type: ignore [return-value]
            model=model,
            overwrite=overwrite,
        )

    def without_model(
            self,
    ) -> Dataset[Any]:
        """
        Remove the model and validation from the dataset.

        Group
        -----
            Model
        """
        return super()._without_model()  # type: ignore

    def with_validation(
            self,
            *validators: (
                    ValidatorOrExpr
                    | Iterable[ValidatorOrExpr]
                    | Validation
            ),
            overwrite: bool = False,
            **named_validators: ValidatorOrExpr,
    ) -> Self:
        """
        Add validation to the dataset.

        Group
        -----
            Validation
        """
        return super().with_validation(
            *validators,
            overwrite=overwrite,
            **named_validators
        )

    def with_info(
            self,
            k: str,
            /,
            **mapping: Any,
    ) -> Self:
        """
        Add information to the dataset.

        Group
        -----
            Information
        """
        return super().with_info(k, **mapping)

    def with_name(
            self,
            name: str | None,
    ) -> Self:
        """
        Add a name to the dataset.

        Group
        -----
            Information
        """
        return super().with_name(name=name)
