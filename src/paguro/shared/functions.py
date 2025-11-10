from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING, Any, overload, Union

import polars as pl

from paguro.dataset.dataset import Dataset
from paguro.dataset.lazydataset import LazyDataset
from paguro.typing import FrameLike
from paguro.validation.validation import Validation
from paguro.validation.shared.preprocessing.duplicates import DuplicateNameError

if TYPE_CHECKING:
    from collections.abc import Iterable
    from paguro.ashi.info.info_collection import InfoCollection


# from typing import TypeVarTuple, Unpack

# Ts = TypeVarTuple("Ts")
#
# @overload
# def collect_all(
#     ls: tuple[LazyDataset[Unpack[Ts]] | pl.LazyFrame, ...],
#     **kwargs: Any,
# ) -> tuple[Dataset[Unpack[Ts]], ...]: ...
#
# @overload
# def collect_all(
#     ls: Iterable[LazyDataset[Any] | pl.LazyFrame],
#     **kwargs: Any,
# ) -> list[Dataset[Any]]: ...
#
# def collect_all(
#     ls: Iterable[LazyDataset[Any] | pl.LazyFrame],
#     **kwargs: Any,
# ):
#

# ideally we would type hint [*Ts]
def collect_all(
        ls: Iterable[LazyDataset | pl.LazyFrame],
        **kwargs: Any,
):
    dataframes = pl.collect_all(
        [
            i.to_polars() if isinstance(i, LazyDataset)
            else i for i in ls
        ],
        **kwargs
    )

    out: list[Dataset[Any]] = []  # todo: each Dataset should preserve typing
    for idx, lazy_ in enumerate(ls):
        if isinstance(lazy_, pl.LazyFrame):
            out.append(
                Dataset(dataframes[idx])
            )
        else:
            out.append(
                Dataset._from_object(  # type: ignore
                    base_object=lazy_,
                    frame=dataframes[idx])
            )
    return out


# ----------------------------------------------------------------------

EagerLike = Union[pl.DataFrame, Dataset]
LazyLike = Union[pl.LazyFrame, LazyDataset]


@overload
def concat(frame_likes: Iterable[EagerLike], **kwargs) -> Dataset: ...


@overload
def concat(frame_likes: Iterable[LazyLike], **kwargs) -> LazyDataset: ...


def concat(
        frame_likes: Iterable[EagerLike | LazyLike],
        **kwargs,
) -> Dataset | LazyDataset:
    frames: list[pl.LazyFrame] = []
    validations: list[Validation] = []
    infos: list[InfoCollection] = []  # todo
    is_eager: bool = False

    for fl in frame_likes:
        if isinstance(fl, (pl.DataFrame, pl.LazyFrame)):
            if not is_eager and isinstance(fl, pl.DataFrame):
                is_eager = True
            frames.append(
                fl.lazy()
            )

        elif isinstance(fl, (Dataset, LazyDataset)):

            if not is_eager and isinstance(fl, Dataset):
                is_eager = True

            frames.append(
                fl.to_polars().lazy()
            )

            if fl._validation is not None:
                validations.append(
                    fl._validation
                )

            if fl._info is not None:
                infos.append(
                    fl._info
                )

        else:
            raise TypeError(f"Invalid frame type: {type(fl)}")

    frame = pl.concat(frames, **kwargs)

    dataset: Dataset | LazyDataset
    if is_eager:
        dataset = Dataset(frame.collect())
    else:
        dataset = LazyDataset(frame)

    if validations:
        # this is going to fail if there are common validation
        try:
            validation = Validation(*validations)
            dataset._validation = copy.deepcopy(validation)
        except DuplicateNameError as e:
            warnings.warn(
                f"Concatenating validation failed -- {e}"
                f"\nWarning: The concatenated dataset does not contain validation."
            )
            pass

    return dataset
